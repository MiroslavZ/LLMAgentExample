"""Сценарии чата, общие для пользовательских интерфейсов.

Каждый успешный этап агента сохраняет рабочий контекст и полную ленту одной
атомарной заменой JSON. Синхронный send следует запускать вне потока UI.
"""

import threading
import time
from contextlib import contextmanager, suppress
from copy import deepcopy
from pathlib import Path
from typing import Iterator
from uuid import uuid4

from openai import (
    APIConnectionError, APIStatusError, APITimeoutError, AuthenticationError,
    BadRequestError, RateLimitError,
)

from .agent import Agent
from .context_strategy import FactsStrategy, WindowStrategy
from .history import HistoryManager, TokenUsage
from .models import ContextSettings, Conversation, RequestOptions, Turn, utc_now
from .memory import MemorySnapshot, MemoryStorageError, MemoryStore
from .profile import ProfileStorageError, ProfileStore, UserProfile
from .storage import ConversationBusyError, ConversationStorageError, ConversationStore
from .task_state import TaskStage, TaskState, TaskStateError


class _ConversationHistory(HistoryManager):
    """Адаптер существующего ядра к единому состоянию веб-диалога."""

    def __init__(self, store: ConversationStore, conversation: Conversation, started_at: float) -> None:
        self.store = store
        self.conversation = conversation
        self.started_at = started_at
        self.pending_snapshot: Conversation | None = None
        self._turn_updates: dict = {}
        settings = conversation.settings
        strategy = None
        if settings.strategy == "window":
            strategy = WindowStrategy(settings.window_size)
        elif settings.strategy == "facts":
            strategy = FactsStrategy(settings.window_size)
        super().__init__(store.path(conversation.id), strategy=strategy)

    def _read_data(self) -> object:
        return deepcopy(self.conversation.working_context)

    def _write_data(self, data: object) -> None:
        snapshot = deepcopy(self.conversation)
        snapshot.working_context = deepcopy(data)
        snapshot.updated_at = utc_now()
        for key, value in self._turn_updates.items():
            setattr(snapshot.turns[-1], key, value)
        self.pending_snapshot = snapshot
        self.store.save(snapshot)
        self.conversation = snapshot
        self.pending_snapshot = None

    def add_exchange(
        self, user: str, assistant: str, usage: TokenUsage | None, *,
        task_state: TaskState | None = None,
    ) -> None:
        turn = self.conversation.turns[-1]
        is_meta_stage = turn.options.meta_prompt and turn.meta_prompt is None
        self._turn_updates = (
            {"meta_prompt": assistant} if is_meta_stage else {
                "answer": assistant,
                "status": "completed",
                "elapsed_seconds": time.perf_counter() - self.started_at,
            }
        )
        try:
            super().add_exchange(user, assistant, usage, task_state=task_state)
        finally:
            self._turn_updates = {}

    def update_facts(self, facts: dict[str, str], usage: TokenUsage | None) -> None:
        self._turn_updates = {"memory_updated": True}
        try:
            super().update_facts(facts, usage)
        finally:
            self._turn_updates = {}

    def compress(self, count: int, summary: str, usage: TokenUsage | None) -> None:
        self._turn_updates = {"memory_updated": True}
        try:
            super().compress(count, summary, usage)
        finally:
            self._turn_updates = {}


def _friendly_error(error: Exception) -> str:
    # Никогда не выводим str(error) из SDK: там могут быть тело ответа и секреты.
    if isinstance(error, ConversationStorageError):
        return "Не удалось подтвердить сохранение результата. Проверьте доступ к каталогу данных."
    if isinstance(error, MemoryStorageError):
        return "Не удалось загрузить память. Проверьте файл memory.sqlite3 и доступ к каталогу данных."
    if isinstance(error, ProfileStorageError):
        return "Не удалось загрузить профиль. Проверьте файл profiles.sqlite3 и доступ к каталогу данных."
    if isinstance(error, TaskStateError):
        return "Модель вернула некорректный или незавершённый ответ задачи. Этап и шаг сохранены; повторите запрос."
    if isinstance(error, AuthenticationError):
        return "Сервис модели отклонил API-ключ. Проверьте API_KEY на сервере."
    if isinstance(error, RateLimitError):
        return "Сервис модели временно ограничил запросы. Повторите позже."
    if isinstance(error, APITimeoutError):
        return "Сервис модели не ответил за отведённое время. Можно отправить запрос повторно."
    if isinstance(error, APIConnectionError):
        return "Не удалось подключиться к сервису модели. Проверьте соединение и повторите запрос."
    if isinstance(error, BadRequestError):
        return "Сервис модели отклонил запрос. Проверьте параметры и размер контекста."
    if isinstance(error, APIStatusError):
        return "Сервис модели вернул ошибку. Повторите запрос позже."
    if isinstance(error, ValueError):
        return "Модель вернула некорректный результат обработки контекста. Можно повторить запрос."
    return "Не удалось выполнить запрос. Сохранённые этапы доступны в диалоге."


class ConversationService:
    """Потокобезопасный сервис локальных диалогов.

    Изменения одного диалога отклоняются, пока выполняется другая операция.
    Чтение атомарных снимков не ждёт ответа модели. Разные диалоги независимы.
    """

    def __init__(self, data_dir: Path, token: str | None) -> None:
        self.store = ConversationStore(data_dir)
        self.memory = MemoryStore(self.store.data_dir / "memory.sqlite3")
        self.profiles = ProfileStore(self.store.data_dir / "profiles.sqlite3")
        self._token = token
        self._state_lock = threading.Lock()
        self._locks: dict[str, threading.Lock] = {}
        self._storage_errors: dict[str, str] = {}
        # При отказе диска сохраняем полученный результат в памяти до следующей
        # успешной записи; интерфейс показывает явную ошибку сохранения.
        self._unsaved: dict[str, Conversation] = {}
        self._recover_interrupted()

    @property
    def token_available(self) -> bool:
        return bool(self._token and self._token.strip())

    @property
    def storage_errors(self) -> list[str]:
        with self._state_lock:
            return list(self._storage_errors.values()) + [
                f"Диалог {conversation.id}: последний результат пока сохранён только в памяти сервера."
                for conversation in self._unsaved.values()
            ]

    def _save(self, conversation: Conversation) -> None:
        self.store.save(conversation)
        with self._state_lock:
            self._unsaved.pop(conversation.id, None)

    @contextmanager
    def _operation(self, conversation_id: str) -> Iterator[None]:
        self.store.path(conversation_id)
        with self._state_lock:
            lock = self._locks.setdefault(conversation_id, threading.Lock())
        if not lock.acquire(blocking=False):
            raise ConversationBusyError("В этом диалоге уже выполняется операция")
        try:
            with self.store.lock(conversation_id):
                yield
        finally:
            lock.release()

    def _recover_interrupted(self) -> None:
        for conversation in self.list_conversations():
            if not conversation.busy:
                continue
            try:
                with self._operation(conversation.id):
                    snapshot = self.store.load(conversation.id)
                    for turn in snapshot.turns:
                        if turn.status == "running":
                            turn.status = "interrupted"
                            turn.error = "Запрос был прерван перезапуском сервера; его итог неизвестен. Сохранённые этапы доступны ниже."
                    snapshot.updated_at = utc_now()
                    self.store.save(snapshot)
            except (ConversationBusyError, KeyError):
                # Другой живой сервер продолжает запрос либо диалог уже удалён.
                continue
            except ConversationStorageError as error:
                with self._state_lock:
                    self._storage_errors[conversation.id] = str(error)

    def list_conversations(self) -> list[Conversation]:
        conversations = []
        errors = {}
        for conversation_id in self.store.ids():
            try:
                conversations.append(self.store.load(conversation_id))
            except KeyError:
                continue
            except (ConversationStorageError, ValueError):
                errors[conversation_id] = (
                    f"Диалог {conversation_id}.json не удалось прочесть. Файл сохранён без изменений."
                )
        with self._state_lock:
            self._storage_errors = errors
            pending = deepcopy(self._unsaved)
        conversations = [pending.pop(item.id, item) for item in conversations]
        conversations.extend(pending.values())
        return sorted(conversations, key=lambda item: (item.updated_at, item.id), reverse=True)

    def create(self) -> Conversation:
        conversation = Conversation(id=uuid4().hex)
        with self._operation(conversation.id):
            self._save(conversation)
        return deepcopy(conversation)

    def get(self, conversation_id: str) -> Conversation:
        self.store.path(conversation_id)
        with self._state_lock:
            pending = self._unsaved.get(conversation_id)
            if pending is not None:
                return deepcopy(pending)
        return self.store.load(conversation_id)

    def get_memory(self, conversation_id: str) -> MemorySnapshot:
        self.get(conversation_id)
        return self.memory.snapshot(conversation_id)

    def _task_operation(self, conversation_id: str, action: str, title: str | None = None) -> Conversation:
        with self._operation(conversation_id):
            conversation = self.get(conversation_id)
            if conversation.busy:
                raise ConversationBusyError("Дождитесь завершения запроса перед изменением задачи")
            history = _ConversationHistory(self.store, conversation, time.perf_counter())
            if action == "start":
                history.start_task(title)
            elif action == "pause":
                history.pause_task()
            elif action == "resume":
                history.resume_task()
            with self._state_lock:
                self._unsaved.pop(conversation_id, None)
            return deepcopy(history.conversation)

    def start_task(self, conversation_id: str, title: str) -> Conversation:
        return self._task_operation(conversation_id, "start", title)

    def pause_task(self, conversation_id: str) -> Conversation:
        return self._task_operation(conversation_id, "pause")

    def resume_task(self, conversation_id: str) -> Conversation:
        return self._task_operation(conversation_id, "resume")

    def get_profile(self, conversation_id: str) -> UserProfile | None:
        self.get(conversation_id)
        return self.profiles.selected(conversation_id)

    @contextmanager
    def _profile_operation(self, conversation_id: str) -> Iterator[None]:
        with self._operation(conversation_id):
            if self.get(conversation_id).busy:
                raise ConversationBusyError("Дождитесь завершения запроса перед изменением профиля")
            yield

    def select_profile(self, conversation_id: str, profile_id: str | None) -> None:
        with self._profile_operation(conversation_id):
            self.profiles.select(conversation_id, profile_id)

    def save_profile(
        self, conversation_id: str, profile: UserProfile, *, expected: UserProfile | None = None,
    ) -> None:
        with self._profile_operation(conversation_id):
            self.profiles.save(profile, expected=expected)

    def delete_profile(self, conversation_id: str, profile_id: str) -> None:
        with self._profile_operation(conversation_id):
            self.profiles.delete(profile_id)

    @contextmanager
    def _memory_operation(self, conversation_id: str) -> Iterator[None]:
        with self._operation(conversation_id):
            if self.get(conversation_id).busy:
                raise ConversationBusyError("Дождитесь завершения запроса перед изменением памяти")
            yield

    def remember_memory(
        self, conversation_id: str, layer: str, key: str, value: str,
    ) -> None:
        with self._memory_operation(conversation_id):
            self.memory.remember(conversation_id, layer, key, value)

    def forget_memory(
        self, conversation_id: str, layer: str, key: str,
    ) -> None:
        with self._memory_operation(conversation_id):
            self.memory.forget(conversation_id, layer, key)

    def clear_working_memory(self, conversation_id: str) -> None:
        with self._memory_operation(conversation_id):
            self.memory.clear_working(conversation_id)

    @staticmethod
    def _validate_settings(settings: ContextSettings) -> None:
        if not isinstance(settings, ContextSettings):
            raise ValueError("Некорректные настройки контекста")
        settings.validate()

    @staticmethod
    def _apply_settings(
        conversation: Conversation, settings: ContextSettings,
        expected_settings: ContextSettings | None,
    ) -> None:
        if expected_settings is not None and conversation.settings != expected_settings:
            raise ValueError("Настройки изменились в другой вкладке. Обновите их перед сохранением.")
        if conversation.started and settings.strategy != conversation.settings.strategy:
            raise ValueError("Тип стратегии фиксируется при первой отправке. Создайте новый диалог.")
        conversation.settings = settings

    def update_settings(
        self, conversation_id: str, settings: ContextSettings, *,
        expected_settings: ContextSettings | None = None,
    ) -> Conversation:
        self._validate_settings(settings)
        with self._operation(conversation_id):
            conversation = self.get(conversation_id)
            if conversation.busy:
                raise ConversationBusyError("Дождитесь завершения запроса")
            self._apply_settings(conversation, settings, expected_settings)
            conversation.updated_at = utc_now()
            # Изменение параметров не сокращает контекст до следующего запроса.
            self._save(conversation)
            return deepcopy(conversation)

    def delete(self, conversation_id: str) -> None:
        with self._operation(conversation_id):
            conversation = self.get(conversation_id)
            if conversation.busy:
                raise ConversationBusyError("Нельзя удалить диалог, пока выполняется запрос")
            profile = self.profiles.selected(conversation_id)
            deleted = False
            try:
                with self.memory.deleting_task(conversation_id), self.profiles.deleting_task(conversation_id):
                    self.store.delete(conversation_id)
                    deleted = True
            except (MemoryStorageError, ProfileStorageError):
                if deleted:
                    # Если commit SQLite не удался, возвращаем JSON. При отказе
                    # записи сохраняем снимок в существующем механизме _unsaved.
                    try:
                        self._save(conversation)
                    except ConversationStorageError:
                        with self._state_lock:
                            self._unsaved[conversation_id] = deepcopy(conversation)
                    # Вложенная транзакция профилей могла завершиться до отказа
                    # памяти; восстанавливаем выбор вместе с возвращённым JSON.
                    if profile is not None:
                        with suppress(ProfileStorageError, ValueError):
                            self.profiles.select(conversation_id, profile.id)
                raise
            with self._state_lock:
                self._unsaved.pop(conversation_id, None)

    def send(
        self, conversation_id: str, user: str, *,
        system_prompt: str = "", options: RequestOptions = RequestOptions(),
        settings: ContextSettings | None = None,
        expected_settings: ContextSettings | None = None,
    ) -> Conversation:
        if not isinstance(user, str) or not user.strip():
            raise ValueError("Введите сообщение")
        if not isinstance(system_prompt, str):
            raise ValueError("Системный промпт должен быть текстом")
        if not isinstance(options, RequestOptions):
            raise ValueError("Некорректные параметры запроса")
        options.validate()
        if settings is not None:
            self._validate_settings(settings)
        with self._operation(conversation_id):
            conversation = self.get(conversation_id)
            if conversation.busy:
                raise ConversationBusyError("В этом диалоге уже выполняется запрос")
            task = conversation.task_state
            if task is not None:
                if task.paused:
                    raise TaskStateError("Задача на паузе. Сначала возобновите её.")
                if task.stage != TaskStage.DONE and options.meta_prompt:
                    raise TaskStateError("Отключите мета-промпт для работы с активной задачей")
            if settings is not None:
                self._apply_settings(conversation, settings, expected_settings)
            if conversation.started and system_prompt and system_prompt != conversation.system_prompt:
                raise ValueError("Системный промпт фиксируется при первой отправке. Создайте новый диалог.")
            if not conversation.started:
                conversation.system_prompt = system_prompt
                conversation.title = " ".join(user.split())[:56]
                conversation.started = True
            conversation.turns.append(Turn(user=user.strip(), options=options))
            conversation.updated_at = utc_now()
            self._save(conversation)
            started_at = time.perf_counter()
            history = None
            agent = None
            try:
                if not self.token_available:
                    conversation.turns[-1].status = "error"
                    conversation.turns[-1].error = "API_KEY не задан. Укажите ключ в окружении сервера или файле .env и перезапустите приложение."
                    conversation.turns[-1].elapsed_seconds = time.perf_counter() - started_at
                    conversation.updated_at = utc_now()
                    self._save(conversation)
                    return deepcopy(conversation)
                memory = self.memory.snapshot(conversation_id)
                profile = self.profiles.selected(conversation_id)
                history = _ConversationHistory(self.store, conversation, started_at)
                settings = conversation.settings
                agent_options = {
                    "history": history, "timeout": 60.0, "max_retries": 0,
                    "memory": memory,
                    "profile": profile,
                }
                if settings.strategy in ("window", "facts"):
                    agent_options.update(strategy=settings.strategy, window_size=settings.window_size)
                elif settings.strategy == "summary":
                    agent_options.update(last_messages=settings.last_messages, compress_every=settings.compress_every)
                agent = Agent(self._token, **agent_options)
                request = agent.request_with_meta_prompt if options.meta_prompt else agent.request
                request(
                    user.strip(), system=conversation.system_prompt,
                    temperature=options.temperature, max_tokens=options.max_tokens,
                )
                return deepcopy(history.conversation)
            except Exception as error:
                if history is not None:
                    conversation = deepcopy(history.pending_snapshot or history.conversation)
                turn = conversation.turns[-1]
                turn.status = "partial" if (
                    turn.meta_prompt is not None or turn.answer is not None or turn.memory_updated
                ) else "error"
                turn.error = _friendly_error(error)
                turn.elapsed_seconds = time.perf_counter() - started_at
                conversation.updated_at = utc_now()
                try:
                    self._save(conversation)
                except ConversationStorageError:
                    turn.error = "Не удалось сохранить результат на диск. Он доступен только в памяти сервера; не перезапускайте приложение до восстановления записи."
                    with self._state_lock:
                        self._unsaved[conversation.id] = deepcopy(conversation)
                return deepcopy(conversation)
            finally:
                if agent is not None:
                    with suppress(Exception):
                        agent.close()
