"""Выбор общего профиля для диалога и редактор его предпочтений."""

from collections.abc import Callable
from uuid import uuid4

from nicegui import ui

from ..profile import PROFILE_FIELDS, ProfileStorageError, UserProfile
from ..service import ConversationBusyError, ConversationService, ConversationStorageError

PROFILE_ERRORS = (ValueError, OSError, KeyError, ConversationBusyError,
                  ConversationStorageError, ProfileStorageError)


class ProfilePanel:
    def __init__(
        self, service: ConversationService, *, conversation_id: Callable[[], str],
        busy: Callable[[], bool],
    ) -> None:
        self.service = service
        self.conversation_id = conversation_id
        self.busy = busy
        self.dialog: ui.dialog | None = None
        self.confirm_dialog: ui.dialog | None = None
        self._available = False
        self._updating_selection = False
        self._selected: UserProfile | None = None
        self._editing: UserProfile | None = None
        self._details_key: object = None

    def build(self) -> None:
        with ui.column().classes("profile-summary") as self.root:
            with ui.row().classes("w-full items-center no-wrap"):
                ui.icon("person_outline", size="20px")
                self.summary = ui.label("Профиль не выбран").classes("profile-summary-name")
            ui.button("Профили пользователя", on_click=self.open).props("outline no-caps").classes("w-full")
            ui.label("Выбранный профиль учитывается в каждом запросе.").classes("setting-note")

    def close(self) -> None:
        for dialog in (self.confirm_dialog, self.dialog):
            if dialog:
                dialog.close()

    def open(self) -> None:
        if self.dialog and self.dialog.value:
            return
        self._scope = self.conversation_id()
        self._editing = None
        self._details_key = object()
        with self.root, ui.dialog() as self.dialog, ui.card().classes("profile-dialog"):
            with ui.row().classes("w-full items-center no-wrap"):
                ui.icon("person_outline", size="24px")
                ui.label("Профили пользователя").classes("text-lg font-semibold")
                ui.space()
                ui.button(icon="close", on_click=self.close).props(
                    'flat round dense aria-label="Закрыть профили"'
                )
            ui.label(
                "Выбор сохраняется для этого диалога. Предпочтения автоматически передаются "
                "агенту с каждым запросом. Профиль можно менять между сообщениями."
            ).classes("memory-description")
            self.status = ui.label().classes("memory-status")
            self.selector = ui.select(
                {}, label="Профиль для этого диалога", clearable=True,
                on_change=lambda event: self.select(event.value),
            ).props("outlined").classes("w-full")
            ui.label("Очистите выбор, чтобы отключить персонализацию профилем.").classes("memory-description")
            self.details = ui.column().classes("profile-details")
            with ui.row().classes("w-full gap-2"):
                self.new_button = ui.button("Создать профиль", icon="add", on_click=self.create).props("unelevated no-caps")
                self.edit_button = ui.button("Редактировать", icon="edit", on_click=self.edit).props("outline no-caps")
                self.delete_button = ui.button("Удалить профиль", icon="delete_outline", on_click=self.confirm_delete).props("flat no-caps color=negative")
            with ui.column().classes("profile-editor") as self.editor:
                self.editor_title = ui.label().classes("font-medium")
                ui.label(
                    "Профили общие для всех диалогов. Изменения применятся со следующего запроса "
                    "во всех диалогах с этим профилем. Сообщения не изменяют профиль автоматически."
                ).classes("memory-description")
                self.fields: dict[str, ui.input | ui.textarea] = {}
                for key, label in PROFILE_FIELDS.items():
                    if key in {"name", "address", "language"}:
                        field = ui.input(label + (" *" if key == "name" else ""))
                    else:
                        field = ui.textarea(label).props("autogrow rows=2")
                    self.fields[key] = field.props("outlined dense").classes("w-full")
                ui.label("Обязательно только название. После создания выберите профиль в списке выше.").classes("memory-description")
                with ui.row().classes("w-full justify-end"):
                    ui.button("Отмена", on_click=lambda: self.editor.set_visibility(False)).props("flat no-caps")
                    self.save_button = ui.button("Сохранить профиль", icon="save", on_click=self.save).props("unelevated no-caps")
            self.editor.set_visibility(False)
        self.dialog.on("hide", self.dialog.delete)
        self.dialog.open()
        self.refresh()

    def refresh(self) -> None:
        opened = bool(self.dialog and self.dialog.value)
        try:
            selected = self.service.get_profile(self.conversation_id())
            profiles = self.service.profiles.list() if opened else []
            self._available = True
            self._selected = selected
            self.summary.set_text(selected.name if selected else "Профиль не выбран")
            if opened:
                self.status.set_text("Дождитесь завершения запроса, чтобы менять профиль." if self.busy() else "")
                if self.editor.visible and self._editing:
                    latest = next((profile for profile in profiles if profile.id == self._editing.id), None)
                    if latest != self._editing:
                        self.status.set_text(
                            "Профиль изменён или удалён в другой вкладке. Черновик сохранён в форме; "
                            "повторно откройте редактор, чтобы загрузить актуальные данные."
                        )
                self._updating_selection = True
                try:
                    options = {profile.id: profile.name for profile in profiles}
                    value = selected.id if selected else None
                    if self.selector.options != options or self.selector.value != value:
                        self.selector.set_options(options, value=value)
                finally:
                    self._updating_selection = False
                if selected != self._details_key:
                    self._details_key = selected
                    self.render_details(selected)
        except PROFILE_ERRORS:
            self._available = False
            self.summary.set_text("Профиль недоступен")
            if opened:
                self.status.set_text("Не удалось прочитать профили. Проверьте хранилище; редактирование временно недоступно.")
        if opened:
            enabled = self.can_change()
            for control in (self.selector, self.new_button, self.save_button, *self.fields.values()):
                control.set_enabled(enabled)
            self.edit_button.set_enabled(enabled and self._selected is not None)
            self.delete_button.set_enabled(enabled and self._selected is not None)

    def render_details(self, profile: UserProfile | None) -> None:
        self.details.clear()
        with self.details:
            if profile is None:
                ui.label("Профиль не выбран. Используются настройки диалога и память.").classes("memory-description")
                return
            ui.label(profile.name).classes("font-medium")
            for key, label in PROFILE_FIELDS.items():
                value = getattr(profile, key)
                if key != "name" and value:
                    ui.label(label).classes("profile-field-label")
                    ui.label(value).classes("memory-value")

    def can_change(self) -> bool:
        return bool(self.dialog and self.dialog.value and self._available
                    and self._scope == self.conversation_id() and not self.busy())

    def change(self, action: Callable[[], None]) -> bool:
        if not self.can_change():
            return False
        try:
            action()
            self.refresh()
            return True
        except PROFILE_ERRORS as error:
            ui.notify(str(error) if isinstance(error, (ValueError, ConversationBusyError))
                      else "Не удалось сохранить профиль.", type="negative")
            self.refresh()
            return False

    def select(self, profile_id: str | None) -> None:
        if not self._updating_selection:
            self.change(lambda: self.service.select_profile(self._scope, profile_id))

    def create(self) -> None:
        if self.can_change():
            self._editing = None
            self.editor_title.set_text("Новый профиль")
            for field in self.fields.values():
                field.set_value("")
            self.editor.set_visibility(True)

    def edit(self) -> None:
        if self.can_change() and self._selected:
            self._editing = self._selected
            self.editor_title.set_text(f"Редактирование: {self._editing.name}")
            for key, field in self.fields.items():
                field.set_value(getattr(self._editing, key))
            self.editor.set_visibility(True)

    def save(self) -> None:
        if not self.can_change() or not self.editor.visible:
            return
        def save() -> None:
            profile = UserProfile(
                id=self._editing.id if self._editing else uuid4().hex,
                **{key: field.value or "" for key, field in self.fields.items()},
            )
            self.service.save_profile(self._scope, profile, expected=self._editing)

        if self.change(save):
            self.editor.set_visibility(False)
            self._editing = None
            self.refresh()
            ui.notify("Профиль сохранён", type="positive")

    def confirm_delete(self) -> None:
        if not self.can_change() or not self._selected:
            return
        profile = self._selected
        scope, owner = self._scope, self.dialog
        with self.root, ui.dialog() as dialog, ui.card().classes("delete-dialog"):
            self.confirm_dialog = dialog
            ui.label("Удалить профиль?").classes("text-lg font-semibold")
            ui.label(profile.name).classes("break-words")
            ui.label("Профиль будет отключён во всех диалогах. Переписка и память сохранятся.").classes("memory-description")

            def confirm() -> None:
                if self.dialog is not owner or self._scope != scope:
                    return
                if self.change(lambda: self.service.delete_profile(scope, profile.id)):
                    if self._editing and self._editing.id == profile.id:
                        self.editor.set_visibility(False)
                        self._editing = None
                    dialog.close()
                    ui.notify("Профиль удалён", type="positive")

            with ui.row().classes("w-full justify-end"):
                ui.button("Отмена", on_click=dialog.close).props("flat no-caps")
                ui.button("Удалить", on_click=confirm, color="negative").props("unelevated no-caps")
        dialog.on("hide", dialog.delete)
        dialog.open()
