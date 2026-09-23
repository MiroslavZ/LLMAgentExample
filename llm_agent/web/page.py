"""Состояние одной страницы: три панели, черновики и обновление снимков."""

import json
from collections.abc import Callable
from dataclasses import dataclass

from nicegui import ui

from ..memory import MemoryStorageError
from ..models import ContextSettings, Conversation, RequestOptions
from ..profile import ProfileStorageError
from ..service import ConversationBusyError, ConversationService, ConversationStorageError
from ..task_state import CONTINUE_TASK
from .components import STRATEGIES, STRATEGY_HELP, empty_chat, render_turn
from .jobs import RequestRunner
from .invariants import InvariantPanel
from .mcp import MCPPanel
from .profiles import ProfilePanel
from .tasks import TaskPanel

UI_ERRORS = (ValueError, OSError, KeyError, ConversationBusyError, ConversationStorageError,
             MemoryStorageError, ProfileStorageError)


@dataclass
class MemoryEditor:
    key: ui.input
    value: ui.textarea


@dataclass
class Draft:
    user: str = ""
    system: str = ""
    meta_prompt: bool = False
    temperature: float | None = None
    max_tokens: float | None = None


class ChatPage:
    def __init__(
        self, service: ConversationService, runner: RequestRunner, *,
        token_available: bool, conversation_id: str | None = None,
    ) -> None:
        self.service = service
        self.runner = runner
        self.token_available = token_available
        self.conversation_id = conversation_id
        self.snapshot: Conversation | None = None
        self.drafts: dict[str, Draft] = {}
        self._submitted: dict[str, str] = {}
        self._list_key: object = None
        self._settings_key: object = None
        self._composer_key: object = None
        self._turns_key: object = None
        self._banner_key: object = None
        self._near_bottom = True
        self.memory_dialog: ui.dialog | None = None
        self._memory_key: object = None
        self._memory_available = False
        self._memory_controls: list = []
        self.memory_editors: dict[str, MemoryEditor] = {}
        self.invariant_panel = InvariantPanel(service, on_save=self.refresh)
        self.mcp_panel = MCPPanel(service.mcp_servers)
        self.profile_panel = ProfilePanel(
            service, conversation_id=lambda: self.conversation_id, busy=lambda: self.busy,
        )
        self.task_panel = TaskPanel(
            service, snapshot=lambda: self.snapshot, busy=lambda: self.busy,
            token_available=token_available, refresh_page=self.refresh, continue_task=self.continue_task,
        )

    def build(self) -> None:
        conversations = self.service.list_conversations()
        known_ids = {conversation.id for conversation in conversations}
        missing = self.conversation_id is not None and self.conversation_id not in known_ids
        if self.conversation_id not in known_ids:
            self.conversation_id = conversations[0].id if conversations else self.service.create().id

        with ui.element("div").classes("chat-app"):
            with ui.element("aside").classes("sidebar") as self.sidebar:
                with ui.row().classes("brand"):
                    with ui.element("div").classes("brand-icon"):
                        ui.icon("auto_awesome", size="23px")
                    with ui.column().classes("gap-0"):
                        ui.label("Агент").classes("brand-name")
                        ui.label("ПРОСТРАНСТВО ДИАЛОГОВ").classes("brand-caption")
                    ui.button(icon="close", on_click=lambda: self.sidebar.classes(remove="panel-open")).props(
                        'flat round dense aria-label="Закрыть список диалогов"'
                    ).classes("mobile-close")
                ui.button("Новый диалог", icon="add", on_click=self.create).props("unelevated no-caps").classes("new-chat")
                ui.label("ВАШИ ДИАЛОГИ").classes("section-caption")
                self.conversation_list = ui.column().classes("conversation-list")
                with ui.column().classes("sidebar-footer"):
                    with ui.row().classes("items-center gap-2"):
                        ui.element("span").classes("status-dot")
                        ui.label("Локальное хранилище")
                    ui.label("Диалоги сохраняются автоматически").classes("text-xs muted")

            with ui.element("main").classes("chat-main"):
                with ui.row().classes("chat-header"):
                    ui.button(icon="menu", on_click=lambda: self.sidebar.classes(add="panel-open")).props(
                        'flat round aria-label="Открыть список диалогов"'
                    ).classes("mobile-nav")
                    with ui.column().classes("header-text"):
                        self.title = ui.label().classes("chat-title")
                        self.subtitle = ui.label().classes("chat-subtitle")
                    ui.badge("DeepSeek", color="white", text_color="grey-8").props("outline").classes("model-badge")
                    ui.button(icon="person_outline", on_click=self.profile_panel.open).props(
                        'flat round aria-label="Профили пользователя"'
                    ).tooltip("Профили пользователя")
                    ui.button(icon="memory", on_click=self.open_memory).props(
                        'flat round aria-label="Память агента"'
                    ).tooltip("Память агента")
                    ui.button("MCP", icon="hub", on_click=self.mcp_panel.open).props(
                        'flat no-caps aria-label="MCP-серверы"'
                    ).tooltip("MCP-серверы")
                    ui.button(icon="tune", on_click=lambda: self.settings_panel.classes(add="panel-open")).props(
                        'flat round aria-label="Открыть настройки диалога"'
                    ).classes("mobile-settings")
                self.banner = ui.column().classes("banner-area")
                with ui.scroll_area(on_scroll=self.on_scroll).classes("message-scroll") as self.scroll:
                    self.transcript = ui.column().classes("transcript")
                self.composer = ui.column().classes("composer-area")
                self.mcp_panel.build()

            with ui.element("aside").classes("settings-panel") as self.settings_panel:
                with ui.row().classes("settings-heading"):
                    ui.icon("tune", size="21px")
                    ui.label("Контекст диалога")
                    ui.button(icon="close", on_click=lambda: self.settings_panel.classes(remove="panel-open")).props(
                        'flat round dense aria-label="Закрыть настройки"'
                    ).classes("mobile-settings ml-auto")
                self.task_panel.build()
                self.profile_panel.build()
                self.invariant_panel.build()
                self.settings_content = ui.column().classes("settings-content")
        self.refresh(force=True)
        ui.navigate.history.replace(f"/?conversation={self.conversation_id}")
        if missing:
            ui.notify("Диалог недоступен. Открыт другой диалог.", type="warning")
        ui.timer(0.7, self.refresh, immediate=False)

    @property
    def busy(self) -> bool:
        return bool(self.snapshot and (self.snapshot.busy or self.runner.is_running(self.snapshot.id)))

    def remember_draft(self) -> None:
        if self.snapshot and hasattr(self, "user_input"):
            self.drafts[self.snapshot.id] = Draft(
                user=self.user_input.value or "",
                system=self.system_input.value if self.system_input else self.snapshot.system_prompt,
                meta_prompt=self.meta_input.value,
                temperature=self.temperature_input.value,
                max_tokens=self.max_tokens_input.value,
            )

    def select(self, conversation_id: str) -> None:
        self.remember_draft()
        if self.memory_dialog:
            self.memory_dialog.close()
        self.profile_panel.close()
        self.conversation_id = conversation_id
        self.snapshot = None
        self._near_bottom = True
        self._settings_key = self._composer_key = self._turns_key = None
        self.refresh(force=True)
        self.sidebar.classes(remove="panel-open")
        ui.navigate.history.replace(f"/?conversation={conversation_id}")

    def create(self) -> None:
        try:
            self.select(self.service.create().id)
        except UI_ERRORS:
            ui.notify("Не удалось создать диалог. Проверьте доступ к хранилищу.", type="negative")

    def confirm_delete(self, conversation: Conversation) -> None:
        def confirm() -> None:
            dialog.close()
            try:
                if self.runner.is_running(conversation.id):
                    raise ValueError("Дождитесь завершения запроса перед удалением.")
                self.service.delete(conversation.id)
                self.drafts.pop(conversation.id, None)
                self.runner.errors.pop(conversation.id, None)
                if self.conversation_id == conversation.id:
                    remaining = self.service.list_conversations()
                    self.select(remaining[0].id if remaining else self.service.create().id)
                else:
                    self.refresh(force=True)
            except UI_ERRORS as error:
                ui.notify(str(error) if isinstance(error, (ValueError, ConversationBusyError)) else "Не удалось удалить диалог.", type="negative")

        # Список пересоздаётся при refresh; окно должно переживать очистку его строк.
        with self.sidebar, ui.dialog() as dialog, ui.card().classes("delete-dialog"):
            ui.label("Удалить диалог?").classes("text-lg font-semibold")
            ui.label(conversation.title).classes("break-words")
            ui.label(
                "Переписка, настройки и рабочая память этого диалога будут удалены. "
                "Долговременная память сохранится."
            ).classes("muted")
            with ui.row().classes("w-full justify-end"):
                ui.button("Отмена", on_click=dialog.close).props("flat no-caps")
                ui.button("Удалить", on_click=confirm, color="negative").props("unelevated no-caps")
        dialog.on("hide", dialog.delete)
        dialog.open()

    def refresh(self, *, force: bool = False) -> None:
        try:
            conversations = self.service.list_conversations()
            current = next((item for item in conversations if item.id == self.conversation_id), None)
            if current is None:
                self.select(conversations[0].id if conversations else self.service.create().id)
                return
            # Перед перестройкой виджетов сохраняем локальный незавершённый ввод.
            self.remember_draft()
            self.snapshot = current
            if not self.busy and current.id in self._submitted:
                submitted = self._submitted.pop(current.id)
                failed = self.runner.errors.get(current.id) or (
                    current.turns and current.turns[-1].user == submitted.strip()
                    and current.turns[-1].status != "completed"
                )
                if failed and not self.drafts[current.id].user:
                    self.drafts[current.id].user = submitted
                    self.user_input.set_value(submitted)
            self.title.set_text(current.title)
            self.subtitle.set_text("Агент готовит ответ…" if self.busy else "Чат с вашим ИИ-агентом")
            list_key = [(item.id, item.title, item.updated_at, item.busy, self.runner.is_running(item.id)) for item in conversations]
            list_key.append(self.conversation_id)
            if force or list_key != self._list_key:
                self._list_key = list_key
                self.render_list(conversations)
            turns_key = (current.id, current.turns, self.busy)
            if force or turns_key != self._turns_key:
                self._turns_key = turns_key
                self.render_transcript()
            settings_key = (current.id, current.settings, current.started, self.busy)
            if force or settings_key != self._settings_key:
                self._settings_key = settings_key
                self.render_settings()
            self.invariant_panel.refresh()
            composer_key = (current.id, current.started, self.busy, current.task_state, self.invariant_panel.available)
            if force or composer_key != self._composer_key:
                self._composer_key = composer_key
                self.render_composer()
            error = self.runner.errors.get(current.id)
            banner_key = (error, tuple(self.service.storage_errors))
            if force or banner_key != self._banner_key:
                self._banner_key = banner_key
                self.render_banner(error)
            self.refresh_memory()
            self.profile_panel.refresh()
            self.task_panel.refresh()
            if not self.invariant_panel.available:
                self.task_panel.continue_button.set_enabled(False)
        except UI_ERRORS:
            self.subtitle.set_text("Не удалось прочитать диалоги. Проверьте хранилище.")

    def open_memory(self) -> None:
        if self.memory_dialog and self.memory_dialog.value:
            return
        self._memory_key = None
        self._memory_controls = []
        self.memory_editors = {}
        self._memory_conversation_id = self.snapshot.id
        # Окно не должно исчезать при обновлении настроек или списка диалогов.
        with self.sidebar, ui.dialog() as self.memory_dialog, ui.card().classes("memory-dialog"):
            with ui.row().classes("w-full items-center no-wrap"):
                ui.icon("memory", size="24px")
                ui.label("Память агента").classes("text-lg font-semibold")
                ui.space()
                ui.button(icon="close", on_click=self.memory_dialog.close).props(
                    'flat round dense aria-label="Закрыть память"'
                )
            ui.label(
                "Переписка автоматически сохраняется в краткосрочной памяти. "
                "Рабочую и долговременную память вы заполняете явно; "
                "обычные сообщения не добавляют в них записи."
            ).classes("memory-description")
            self.memory_status = ui.label().classes("memory-status")
            with ui.tabs().classes("w-full") as tabs:
                short_tab = ui.tab("Краткосрочная", icon="chat_bubble_outline")
                working_tab = ui.tab("Рабочая", icon="work_outline")
                long_tab = ui.tab("Долговременная", icon="inventory_2")
            with ui.tab_panels(tabs, value=short_tab).classes("w-full memory-panels"):
                with ui.tab_panel(short_tab):
                    ui.label(
                        "Только этот диалог. Стратегия контекста определяет, какая часть "
                        "истории попадёт в следующий запрос. Полная переписка доступна в чате."
                    ).classes("memory-description")
                    self.short_memory_content = ui.column().classes("w-full")
                with ui.tab_panel(working_tab):
                    ui.label(
                        "Данные текущей задачи в этом диалоге: цель, ограничения, промежуточный результат. "
                        "Все записи передаются агенту со следующим запросом."
                    ).classes("memory-description")
                    self.working_memory_records = ui.column().classes("memory-records")
                    self.render_memory_editor("working")
                    clear = ui.button(
                        "Очистить рабочую память", icon="delete_sweep", on_click=self.clear_working_memory,
                    ).props("flat no-caps color=negative")
                    self._memory_controls.append(clear)
                with ui.tab_panel(long_tab):
                    ui.label(
                        "Общие для всех диалогов записи. "
                        "Сохраняются между запусками и передаются агенту со следующим запросом."
                    ).classes("memory-description")
                    self.long_memory_records = ui.column().classes("memory-records")
                    self.render_memory_editor("long_term")
        self.memory_dialog.on("hide", self.memory_dialog.delete)
        self.memory_dialog.open()
        self.refresh_memory()

    def render_memory_editor(self, layer: str) -> None:
        with ui.column().classes("memory-editor"):
            ui.label("Добавить или заменить запись").classes("font-medium")
            key = ui.input("Ключ", placeholder="Например: цель или язык").props("outlined dense").classes("w-full")
            value = ui.textarea("Значение", placeholder="Что агенту нужно учитывать").props("outlined autogrow rows=2").classes("w-full")
            self.memory_editors[layer] = MemoryEditor(key, value)
            ui.label("Запись с тем же ключом в выбранном слое будет заменена.").classes("memory-description")
            save = ui.button("Сохранить запись", icon="save", on_click=lambda: self.save_memory(layer)).props("unelevated no-caps")
            self._memory_controls.extend((key, value, save))

    def refresh_memory(self) -> None:
        if not self.memory_dialog or not self.memory_dialog.value:
            return
        try:
            memory = self.service.get_memory(self._memory_conversation_id)
            memory_key = (self.snapshot.working_context, self.snapshot.turns, self.snapshot.settings,
                          memory.working, memory.long_term, self.busy)
            self._memory_available = True
            self.memory_status.set_text("Дождитесь завершения запроса, чтобы редактировать память." if self.busy else "")
            if memory_key != self._memory_key:
                self._memory_key = memory_key
                self.short_memory_content.clear()
                with self.short_memory_content:
                    ui.label(f"Реплик пользователя: {len(self.snapshot.turns)} · Стратегия: {STRATEGIES[self.snapshot.settings.strategy]}")
                    with ui.expansion("Сохранённый контекст стратегии", icon="data_object").classes("w-full"):
                        ui.label(json.dumps(self.snapshot.working_context, ensure_ascii=False, indent=2)).classes("memory-context")
                    ui.label("Факты и сжатое содержание стратегии относятся только к этому диалогу.").classes("memory-description")
                self.render_memory_records(self.working_memory_records, "working", memory.working)
                self.render_memory_records(self.long_memory_records, "long_term", memory.long_term)
        except UI_ERRORS:
            self._memory_available = False
            self.memory_status.set_text("Не удалось прочитать память. Проверьте хранилище; редактирование временно недоступно.")
        for control in self._memory_controls:
            control.set_enabled(self._memory_available and not self.busy)
        for container in (self.working_memory_records, self.long_memory_records):
            for control in container.descendants():
                if isinstance(control, ui.button):
                    control.set_enabled(self._memory_available and not self.busy)

    def render_memory_records(self, container, layer: str, records: dict[str, str]) -> None:
        container.clear()
        with container:
            if not records:
                ui.label("Пока нет записей").classes("memory-description")
            for key, value in records.items():
                with ui.row().classes("memory-record"):
                    with ui.column().classes("memory-record-text"):
                        ui.label(key).classes("font-medium")
                        ui.label(value).classes("memory-value")
                    ui.button(
                        icon="edit", on_click=lambda k=key, v=value: self.edit_memory(layer, k, v),
                    ).props('flat round dense aria-label="Редактировать запись"').tooltip("Редактировать запись")
                    ui.button(
                        icon="delete_outline", on_click=lambda k=key: self.delete_memory(layer, k),
                    ).props('flat round dense aria-label="Удалить запись"').tooltip("Удалить запись")

    def edit_memory(self, layer: str, key: str, value: str) -> None:
        if self.busy or not self._memory_available:
            return
        editor = self.memory_editors[layer]
        editor.key.set_value(key)
        editor.value.set_value(value)

    def change_memory(self, action: Callable[[], None]) -> bool:
        if self.busy or not self._memory_available or self._memory_conversation_id != self.conversation_id:
            return False
        try:
            action()
            self.refresh_memory()
            ui.notify("Память обновлена", type="positive")
            return True
        except UI_ERRORS as error:
            ui.notify(str(error) if isinstance(error, (ValueError, ConversationBusyError)) else "Не удалось сохранить память.", type="negative")
            return False

    def save_memory(self, layer: str) -> None:
        editor = self.memory_editors[layer]
        if self.change_memory(lambda: self.service.remember_memory(
            self._memory_conversation_id, layer, editor.key.value or "", editor.value.value or "",
        )):
            editor.key.set_value("")
            editor.value.set_value("")

    def delete_memory(self, layer: str, key: str) -> None:
        self.change_memory(lambda: self.service.forget_memory(self._memory_conversation_id, layer, key))

    def clear_working_memory(self) -> None:
        self.change_memory(lambda: self.service.clear_working_memory(self._memory_conversation_id))

    def render_list(self, conversations: list[Conversation]) -> None:
        self.conversation_list.clear()
        with self.conversation_list:
            for conversation in conversations:
                selected = conversation.id == self.conversation_id
                with ui.row().classes("conversation-item" + (" selected" if selected else "")):
                    ui.button(conversation.title, icon="chat_bubble_outline", on_click=lambda c=conversation: self.select(c.id)).props(
                        "flat no-caps align=left"
                    ).classes("conversation-link").tooltip(conversation.title)
                    if conversation.busy or self.runner.is_running(conversation.id):
                        ui.spinner(size="16px").classes("mr-2")
                    else:
                        ui.button(icon="delete_outline", on_click=lambda c=conversation: self.confirm_delete(c)).props(
                            'flat round dense aria-label="Удалить диалог"'
                        ).classes("delete-conversation").tooltip("Удалить диалог")

    def render_banner(self, error: str | None) -> None:
        self.banner.clear()
        with self.banner:
            if not self.token_available:
                ui.label("Для отправки сообщений задайте API_KEY в .env и перезапустите приложение.").classes("info-banner")
            for message in self.service.storage_errors:
                ui.label(message).classes("warning-banner")
            if error:
                ui.label(error).classes("warning-banner")

    def on_scroll(self, event) -> None:
        self._near_bottom = event.vertical_percentage > 0.9

    def render_transcript(self) -> None:
        self.transcript.clear()
        with self.transcript:
            if not self.snapshot.turns:
                empty_chat()
            for turn in self.snapshot.turns:
                render_turn(turn, restore=lambda text=turn.user: self.restore_text(text), busy=self.busy)
            if self._near_bottom:
                ui.timer(0.05, lambda: self.scroll.scroll_to(percent=1, duration=0.2), once=True)

    def restore_text(self, text: str) -> None:
        if self.user_input.value and self.user_input.value != text:
            ui.notify("Сначала сохраните или очистите текущий черновик.", type="warning")
            return
        self.user_input.set_value(text)

    def render_composer(self) -> None:
        draft = self.drafts.get(self.snapshot.id, Draft(system=self.snapshot.system_prompt))
        task = self.snapshot.task_state
        active_task = task is not None and task.stage != "done"
        paused = task is not None and task.paused
        self.composer.clear()
        with self.composer:
            self.system_input = None
            if not self.snapshot.started:
                with ui.column().classes("system-prompt-card"):
                    with ui.row().classes("items-center gap-2"):
                        ui.icon("psychology", size="18px")
                        ui.label("Инструкции для агента").classes("font-medium")
                        ui.label("необязательно").classes("text-xs muted")
                    self.system_input = ui.textarea(
                        "Системный промпт", value=draft.system,
                        placeholder="Например: ты опытный Python-разработчик. Отвечай кратко и по-русски.",
                    ).props("outlined autogrow rows=2").classes("w-full system-input")
            with ui.column().classes("message-composer"):
                self.user_input = ui.textarea(
                    "Ваше сообщение", value=draft.user, placeholder="Напишите сообщение агенту…",
                ).props("borderless autogrow rows=2").classes("user-input")
                self.user_input.on("keydown", self.send, js_handler=(
                    "e => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey) && !e.isComposing) "
                    "{ e.preventDefault(); emit(); } }"
                ))
                with ui.row().classes("composer-toolbar"):
                    self.meta_input = ui.checkbox("Мета-промпт", value=draft.meta_prompt and not active_task).props("dense size=sm").tooltip(
                        "Недоступен во время активной задачи и на паузе." if active_task or paused else
                        "Сначала агент улучшит ваш запрос, затем выполнит его. Два запроса к модели."
                    )
                    ui.space()
                    self.send_button = ui.button(
                        "Отправить", icon="arrow_upward", on_click=self.send,
                    ).props("unelevated no-caps").classes("send-button")
                with ui.expansion("Параметры запроса", icon="tune").classes("request-options"):
                    with ui.row().classes("w-full gap-4"):
                        self.temperature_input = ui.number(
                            "Температура", value=draft.temperature, min=0, max=2, step=0.1,
                            placeholder="По умолчанию",
                        ).props("outlined dense clearable").classes("option-field")
                        self.max_tokens_input = ui.number(
                            "Максимум токенов ответа", value=draft.max_tokens, min=1, step=1,
                            placeholder="По умолчанию",
                        ).props("outlined dense clearable").classes("option-field")
                    ui.label("Пустое значение использует настройки модели. Параметры применяются к этой отправке.").classes("text-xs muted")
            with ui.row().classes("composer-hint"):
                ui.label("Ctrl / ⌘ + Enter — отправить · Enter — новая строка")
                ui.space()
                ui.label("Ответы могут содержать ошибки")
        for widget in (self.user_input, self.meta_input, self.temperature_input, self.max_tokens_input):
            widget.set_enabled(not self.busy)
        self.meta_input.set_enabled(not self.busy and not active_task and not paused)
        if self.system_input:
            self.system_input.set_enabled(not self.busy)
        self.send_button.set_enabled(
            not self.busy and not paused and self.token_available and self.invariant_panel.available,
        )
        if self.busy:
            self.send_button.props("loading")

    def render_settings(self) -> None:
        settings = self.snapshot.settings
        self.settings_content.clear()
        with self.settings_content:
            ui.label("УПРАВЛЕНИЕ КОНТЕКСТОМ").classes("section-caption mt-1")
            self.strategy_input = ui.select(
                STRATEGIES, value=settings.strategy, label="Стратегия",
                on_change=lambda: self.render_strategy_fields(),
            ).props("outlined").classes("w-full")
            self.strategy_input.set_enabled(not self.snapshot.started and not self.busy)
            if self.snapshot.started:
                with ui.row().classes("setting-note"):
                    ui.icon("lock_outline", size="15px")
                    ui.label("Стратегия закреплена за диалогом")
            else:
                ui.label("Выберите перед первым сообщением. Позже можно менять параметры стратегии.").classes("setting-note")
            self.strategy_fields = ui.column().classes("w-full gap-4")
            self.render_strategy_fields()
            self.save_settings_button = ui.button("Сохранить настройки", icon="check", on_click=self.save_settings).props(
                "outline no-caps"
            ).classes("w-full")
            self.save_settings_button.set_enabled(not self.busy)
            ui.separator().classes("my-3")
            with ui.row().classes("context-tip"):
                ui.icon("history", size="19px")
                ui.label("Полная переписка остаётся в чате при любой стратегии.")
            if self.snapshot.started:
                with ui.expansion("Системный промпт", icon="psychology").classes("w-full system-preview"):
                    ui.label(self.snapshot.system_prompt or "Не задан").classes("system-preview-text")

    def render_strategy_fields(self) -> None:
        strategy = self.strategy_input.value
        settings = self.snapshot.settings
        self.strategy_fields.clear()
        self.window_input = self.last_messages_input = self.compress_every_input = None
        with self.strategy_fields:
            ui.label(STRATEGY_HELP[strategy]).classes("strategy-description")
            if strategy in {"window", "facts"}:
                self.window_input = ui.number(
                    "Размер окна", value=settings.window_size, min=1, step=1,
                ).props("outlined dense suffix=сообщений").classes("w-full")
                ui.label("Считаются отдельные сообщения, включая текущий запрос, а не пары вопрос–ответ.").classes("setting-note")
            elif strategy == "summary":
                self.last_messages_input = ui.number(
                    "Оставлять последних сообщений", value=settings.last_messages, min=0, step=1,
                ).props("outlined dense").classes("w-full")
                self.compress_every_input = ui.number(
                    "Порог сжатия старых сообщений", value=settings.compress_every, min=1, step=1,
                ).props("outlined dense").classes("w-full")
                ui.label("Сжатие начинается, когда число сообщений перед сохраняемым хвостом достигает порога.").classes("setting-note")
            if strategy != "full":
                ui.label("Изменения применятся со следующим запросом.").classes("setting-note")
        for widget in (self.window_input, self.last_messages_input, self.compress_every_input):
            if widget:
                widget.set_enabled(not self.busy)

    @staticmethod
    def integer(value: float | None, label: str, minimum: int = 1) -> int:
        if value is None or not float(value).is_integer() or value < minimum:
            raise ValueError(f"{label}: укажите целое число не меньше {minimum}.")
        return int(value)

    def collect_settings(self) -> ContextSettings:
        current = self.snapshot.settings
        return ContextSettings(
            strategy=self.strategy_input.value,
            window_size=self.integer(self.window_input.value, "Размер окна") if self.window_input else current.window_size,
            last_messages=self.integer(self.last_messages_input.value, "Количество последних сообщений", 0) if self.last_messages_input else current.last_messages,
            compress_every=self.integer(self.compress_every_input.value, "Порог сжатия") if self.compress_every_input else current.compress_every,
        )

    def save_settings(self) -> None:
        if self.busy:
            return
        try:
            self.service.update_settings(
                self.snapshot.id, self.collect_settings(), expected_settings=self.snapshot.settings,
            )
            self.refresh()
            ui.notify("Настройки сохранены", type="positive")
        except UI_ERRORS as error:
            ui.notify(str(error) if isinstance(error, (ValueError, ConversationBusyError)) else "Не удалось сохранить настройки.", type="negative")

    def send(self) -> None:
        if self.busy or not self.token_available:
            return
        self.invariant_panel.refresh()
        if not self.invariant_panel.available:
            ui.notify("Исправьте файл инвариантов перед отправкой сообщения.", type="warning")
            self.refresh()
            return
        task = self.snapshot.task_state
        if task is not None and task.paused:
            ui.notify("Задача на паузе. Сначала возобновите её.", type="warning")
            return
        self.remember_draft()
        draft = self.drafts[self.snapshot.id]
        if not draft.user.strip():
            ui.notify("Введите сообщение", type="warning")
            return
        try:
            options = RequestOptions(
                meta_prompt=draft.meta_prompt, temperature=draft.temperature,
                max_tokens=self.integer(draft.max_tokens, "Максимум токенов") if draft.max_tokens is not None else None,
            )
            options.validate()
            settings = self.collect_settings()
            self.runner.submit(
                self.snapshot.id, draft.user, system_prompt=draft.system if not self.snapshot.started else "",
                options=options,
                settings=settings if settings != self.snapshot.settings else None,
                expected_settings=self.snapshot.settings,
            )
            self._submitted[self.snapshot.id] = draft.user
            self.user_input.set_value("")
            self._near_bottom = True
            self.refresh()
        except UI_ERRORS as error:
            ui.notify(str(error) if isinstance(error, (ValueError, ConversationBusyError)) else "Не удалось отправить сообщение.", type="negative")

    def continue_task(self) -> None:
        task = self.snapshot.task_state
        if self.busy or not self.token_available or task is None or task.stage == "done" or task.paused:
            return
        if task.awaiting_approval:
            ui.notify("Сначала утвердите план или отправьте правки в чат.", type="warning")
            return
        self.invariant_panel.refresh()
        if not self.invariant_panel.available:
            self.refresh()
            return
        if (self.user_input.value or "").strip():
            ui.notify("Сначала отправьте или очистите черновик сообщения.", type="warning")
            return
        self.user_input.set_value(CONTINUE_TASK)
        self.send()
