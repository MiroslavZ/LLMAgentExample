"""Состояние одной страницы: три панели, черновики и обновление снимков."""

from dataclasses import dataclass

from nicegui import ui

from ..models import ContextSettings, Conversation, RequestOptions
from ..service import ConversationBusyError, ConversationService, ConversationStorageError
from .components import STRATEGIES, STRATEGY_HELP, empty_chat, render_turn
from .jobs import RequestRunner

UI_ERRORS = (ValueError, OSError, KeyError, ConversationBusyError, ConversationStorageError)


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
                    ui.button(icon="tune", on_click=lambda: self.settings_panel.classes(add="panel-open")).props(
                        'flat round aria-label="Открыть настройки диалога"'
                    ).classes("mobile-settings")
                self.banner = ui.column().classes("banner-area")
                with ui.scroll_area(on_scroll=self.on_scroll).classes("message-scroll") as self.scroll:
                    self.transcript = ui.column().classes("transcript")
                self.composer = ui.column().classes("composer-area")

            with ui.element("aside").classes("settings-panel") as self.settings_panel:
                with ui.row().classes("settings-heading"):
                    ui.icon("tune", size="21px")
                    ui.label("Контекст диалога")
                    ui.button(icon="close", on_click=lambda: self.settings_panel.classes(remove="panel-open")).props(
                        'flat round dense aria-label="Закрыть настройки"'
                    ).classes("mobile-settings ml-auto")
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
        async def confirm() -> None:
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

        with ui.dialog() as dialog, ui.card().classes("delete-dialog"):
            ui.label("Удалить диалог?").classes("text-lg font-semibold")
            ui.label(conversation.title).classes("break-words")
            ui.label("Вся переписка и настройки этого диалога будут удалены.").classes("muted")
            with ui.row().classes("w-full justify-end"):
                ui.button("Отмена", on_click=dialog.close).props("flat no-caps")
                ui.button("Удалить", on_click=confirm, color="negative").props("unelevated no-caps")
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
            composer_key = (current.id, current.started, self.busy)
            if force or composer_key != self._composer_key:
                self._composer_key = composer_key
                self.render_composer()
            error = self.runner.errors.get(current.id)
            banner_key = (error, tuple(self.service.storage_errors))
            if force or banner_key != self._banner_key:
                self._banner_key = banner_key
                self.render_banner(error)
        except UI_ERRORS:
            self.subtitle.set_text("Не удалось прочитать диалоги. Проверьте хранилище.")

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
                    self.meta_input = ui.checkbox("Мета-промпт", value=draft.meta_prompt).props("dense size=sm").tooltip(
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
        if self.system_input:
            self.system_input.set_enabled(not self.busy)
        self.send_button.set_enabled(not self.busy and self.token_available)
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
