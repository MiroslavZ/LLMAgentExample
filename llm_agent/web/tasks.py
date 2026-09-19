"""Управление формализованной задачей и просмотр сохранённого прогресса."""

from collections.abc import Callable

from nicegui import ui

from ..models import Conversation
from ..service import ConversationBusyError, ConversationService, ConversationStorageError

TASK_STAGES = {
    "planning": "Планирование",
    "execution": "Выполнение",
    "validation": "Проверка",
    "done": "Завершена",
}
TASK_ERRORS = (ValueError, OSError, KeyError, ConversationBusyError, ConversationStorageError)


class TaskPanel:
    def __init__(
        self, service: ConversationService, *, snapshot: Callable[[], Conversation | None],
        busy: Callable[[], bool], token_available: bool, refresh_page: Callable[[], None],
        continue_task: Callable[[], None],
    ) -> None:
        self.service = service
        self.snapshot = snapshot
        self.busy = busy
        self.token_available = token_available
        self.refresh_page = refresh_page
        self.continue_task = continue_task
        self._scope: str | None = None
        self._drafts: dict[str, str] = {}
        self._details_key: object = None

    def build(self) -> None:
        with ui.column().classes("w-full gap-3 mt-6 pb-5 border-b border-gray-200") as self.root:
            with ui.row().classes("items-center gap-2"):
                ui.icon("checklist", size="20px")
                ui.label("Состояние задачи").classes("font-medium")
            self.status = ui.label().classes("setting-note")
            with ui.column().classes("w-full gap-2") as self.new_task:
                self.title_input = ui.textarea(
                    "Новая задача", placeholder="Опишите цель и желаемый результат",
                ).props("outlined autogrow rows=2").classes("w-full")
                self.start_button = ui.button(
                    "Начать задачу", icon="add_task", on_click=self.start,
                ).props("outline no-caps").classes("w-full")
            with ui.column().classes("w-full gap-2") as self.progress:
                self.title = ui.label().classes("font-medium break-words")
                self.stage = ui.label()
                self.current_step = ui.label().classes("memory-value")
                self.expected_action = ui.label().classes("memory-value")
                self.details = ui.column().classes("w-full gap-2")
                self.pause_button = ui.button(
                    "Пауза", icon="pause", on_click=self.pause,
                ).props("outline no-caps").classes("w-full")
                self.resume_button = ui.button(
                    "Возобновить", icon="play_arrow", on_click=self.resume,
                ).props("outline no-caps").classes("w-full")
                self.continue_button = ui.button(
                    "Продолжить шаг", icon="skip_next", on_click=self.continue_task,
                ).props("unelevated no-caps").classes("w-full")

    def refresh(self) -> None:
        conversation = self.snapshot()
        if conversation is None:
            return
        if self._scope != conversation.id:
            if self._scope is not None:
                self._drafts[self._scope] = self.title_input.value or ""
            self._scope = conversation.id
            self.title_input.set_value(self._drafts.get(conversation.id, ""))
        task = conversation.task_state
        active = task is not None and task.stage != "done"
        paused = bool(task and task.paused)
        busy = self.busy()
        self.new_task.set_visibility(not active)
        self.progress.set_visibility(task is not None)
        self.title_input.set_enabled(not busy and not active)
        self.start_button.set_enabled(not busy and not active)
        self.pause_button.set_visibility(task is not None and not paused)
        self.resume_button.set_visibility(paused)
        self.continue_button.set_visibility(active)
        self.pause_button.set_enabled(task is not None and not paused and not busy)
        self.resume_button.set_enabled(paused and not busy)
        self.continue_button.set_enabled(active and not paused and not busy and self.token_available)
        if busy:
            self.status.set_text(
                "Дождитесь ответа. После него можно поставить задачу на паузу." if active else
                "Дождитесь ответа, чтобы начать новую задачу."
            )
        elif paused:
            self.status.set_text("На паузе. Возобновите задачу, чтобы продолжить с сохранённого шага.")
        elif active:
            self.status.set_text("Каждая отправка выполняет один шаг. Прогресс сохраняется автоматически.")
        elif task:
            self.status.set_text("Задача завершена. Можно общаться в чате или начать новую задачу.")
        else:
            self.status.set_text("Обычный чат. Начните задачу, чтобы сохранять её этапы и прогресс.")
        if task is None:
            return
        self.title.set_text(task.title)
        self.stage.set_text(f"Этап: {TASK_STAGES[task.stage]}")
        self.current_step.set_text(f"Текущий шаг: {task.current_step}")
        self.expected_action.set_text(f"Ожидаемое действие: {task.expected_action}")
        key = (conversation.id, task.to_dict())
        if key == self._details_key:
            return
        self._details_key = key
        self.details.clear()
        with self.details:
            if task.plan:
                with ui.expansion(f"План · выполнено {task.step} из {len(task.plan)}").classes("w-full"):
                    for index, step in enumerate(task.plan):
                        prefix = "✓" if index < task.step else str(index + 1) + "."
                        ui.label(f"{prefix} {step}").classes("memory-value")
            if task.results:
                with ui.expansion("Результаты шагов").classes("w-full"):
                    for index, result in enumerate(task.results, start=1):
                        ui.label(f"{index}. {result}").classes("memory-value")
            if task.notes:
                with ui.expansion("Уточнения задачи").classes("w-full"):
                    for note in task.notes:
                        ui.label(note).classes("memory-value")
            if task.validation:
                with ui.expansion("Итог проверки").classes("w-full"):
                    ui.label(task.validation).classes("memory-value")

    def change(self, action: Callable[[str], Conversation]) -> bool:
        conversation = self.snapshot()
        if conversation is None or self.busy():
            return False
        try:
            action(conversation.id)
            self.refresh_page()
            return True
        except TASK_ERRORS as error:
            ui.notify(str(error) if isinstance(error, (ValueError, ConversationBusyError))
                      else "Не удалось сохранить состояние задачи.", type="negative")
            self.refresh_page()
            return False

    def start(self) -> None:
        if self.change(lambda conversation_id: self.service.start_task(
            conversation_id, self.title_input.value or "",
        )):
            self.title_input.set_value("")
            self._drafts.pop(self._scope, None)

    def pause(self) -> None:
        self.change(self.service.pause_task)

    def resume(self) -> None:
        self.change(self.service.resume_task)
