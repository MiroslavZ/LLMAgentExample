"""Каталог LLM: ручная настройка и импорт из OpenAI-совместимого API."""

import asyncio
from collections.abc import Callable
from uuid import uuid4

from nicegui import run, ui

from ..llm_models import LLMModel, ModelConnectionError, ModelStorageError, ModelStore, discover_models


class ModelsPanel:
    def __init__(self, store: ModelStore, *, on_change: Callable[[], None]) -> None:
        self.store = store
        self.on_change = on_change
        self.dialog: ui.dialog | None = None
        self._selected: LLMModel | None = None
        self._editing: LLMModel | None = None
        self._available = False
        self._busy = False
        self._refreshing = False
        self._request: asyncio.Task | None = None
        self._discovery_credentials: tuple[str, str] | None = None

    def build(self) -> None:
        self.root = ui.element("div")

    def open(self) -> None:
        if self.dialog is None:
            self._build_dialog()
        self.refresh()
        self.dialog.open()

    def _build_dialog(self) -> None:
        with self.root, ui.dialog() as self.dialog, ui.card().classes("models-dialog"):
            with ui.row().classes("w-full items-center no-wrap"):
                ui.icon("view_in_ar", size="24px")
                ui.label("Менеджер моделей").classes("text-lg font-semibold")
                ui.space()
                ui.button(icon="close", on_click=self.dialog.close).props(
                    'flat round dense aria-label="Закрыть менеджер моделей"'
                )
            ui.label(
                "Добавьте модель вручную или получите список с сервера. "
                "Затем выберите модель в шапке диалога. Изменения настроек действуют со следующего запроса."
            ).classes("memory-description")
            self.status = ui.label().classes("memory-status")
            self.selector = ui.select({}, label="Модели в каталоге", on_change=self.select).props(
                "outlined dense"
            ).classes("w-full")
            self.address = ui.label().classes("memory-description break-all")
            with ui.row().classes("w-full gap-2"):
                self.new_button = ui.button("Добавить вручную", icon="add", on_click=self.create).props("unelevated no-caps")
                self.edit_button = ui.button("Редактировать", icon="edit", on_click=self.edit).props("outline no-caps")
                self.delete_button = ui.button("Удалить", icon="delete_outline", on_click=self.delete).props("flat no-caps color=negative")
            with ui.column().classes("model-editor") as self.editor:
                self.editor_title = ui.label().classes("font-medium")
                self.name_input = ui.input("Отображаемое имя", placeholder="По умолчанию — ID модели").props("outlined dense").classes("w-full")
                self.id_input = ui.input("ID модели в API *").props("outlined dense").classes("w-full")
                self.url_input = ui.input("Base URL *", placeholder="http://192.168.1.188:1234/v1").props("outlined dense").classes("w-full")
                self.token_input = ui.input("API-токен (необязательно)", password=True, password_toggle_button=True).props(
                    'outlined dense autocomplete="new-password"'
                ).classes("w-full")
                self.token_hint = ui.label().classes("memory-description")
                self.remove_token = ui.checkbox("Удалить сохранённый токен", value=False)
                self.timeout_input = ui.number("Таймаут запроса, секунд", value=180, min=1).props("outlined dense").classes("w-full")
                self.json_input = ui.checkbox("Использовать JSON mode (response_format)", value=True)
                self.tools_input = ui.checkbox("Модель поддерживает инструменты (tool calling)", value=True)
                ui.label(
                    "Если сервер не поддерживает JSON mode или инструменты, отключите соответствующую возможность. "
                    "Для локального сервера без авторизации токен не нужен."
                ).classes("memory-description")
                with ui.row().classes("w-full justify-end"):
                    ui.button("Отмена", on_click=self.close_editor).props("flat no-caps")
                    self.save_button = ui.button("Сохранить модель", icon="save", on_click=self.save).props("unelevated no-caps")
            self.editor.set_visibility(False)
            with ui.expansion("Получить модели с сервера", icon="cloud_download", value=True).classes("w-full model-discovery"):
                self.server_url = ui.input("Base URL сервера *", placeholder="http://192.168.1.188:1234/v1").props("outlined dense").classes("w-full")
                ui.label("Например: http://192.168.1.188:1234/v1 или https://api.deepseek.com/v1").classes("memory-description break-all")
                self.server_token = ui.input("Токен сервера (необязательно)", password=True, password_toggle_button=True).props(
                    'outlined dense autocomplete="new-password"'
                ).classes("w-full")
                self.discover_button = ui.button("Получить список моделей", icon="sync", on_click=self.discover).props("outline no-caps")
                self.result_status = ui.label().classes("memory-status")
                self.discovered = ui.select([], label="Модели для добавления", multiple=True, on_change=self.update_controls).props(
                    "outlined dense use-chips"
                ).classes("w-full")
                self.import_button = ui.button("Добавить выбранные модели", icon="playlist_add", on_click=self.import_selected).props("unelevated no-caps")
            ui.label(
                "Токены сохраняются в локальном файле каталога моделей. "
                "Сохранённые токены не отображаются в интерфейсе."
            ).classes("memory-description")
        self.dialog.on("hide", self.on_close)

    def on_close(self) -> None:
        if self._request is not None:
            self._request.cancel()
        self.close_editor()
        self.server_token.set_value("")
        self.clear_result()

    def refresh(self, selected_id: str | None = None) -> None:
        if self.dialog is None:
            return
        self._refreshing = True
        try:
            models = self.store.list()
            selected_id = selected_id or self.selector.value
            if selected_id not in {model.id for model in models}:
                selected_id = models[0].id if models else None
            self._selected = next((model for model in models if model.id == selected_id), None)
            options = {model.id: model.name for model in models}
            if self.selector.options != options or self.selector.value != selected_id:
                self.selector.set_options(options, value=selected_id)
            self._available = True
            self.address.set_text(
                f"{self._selected.model_id} · {self._selected.base_url}" if self._selected else ""
            )
            self.status.set_text("" if models else "Моделей пока нет. Добавьте модель вручную или получите список с сервера.")
        except ModelStorageError as error:
            self._available = False
            self.status.set_text(str(error))
        finally:
            self._refreshing = False
        self.update_controls()

    def select(self) -> None:
        if self._refreshing or self._busy:
            return
        self.close_editor()
        self.refresh()

    def update_controls(self) -> None:
        enabled = self._available and not self._busy
        for control in (self.selector, self.new_button, self.save_button, self.name_input,
                        self.id_input, self.url_input, self.token_input, self.remove_token,
                        self.timeout_input, self.json_input, self.tools_input,
                        self.server_url, self.server_token, self.discover_button, self.discovered):
            control.set_enabled(enabled)
        for control in (self.edit_button, self.delete_button):
            control.set_enabled(enabled and self._selected is not None)
        self.import_button.set_enabled(enabled and bool(self.discovered.value) and self._discovery_credentials is not None)
        self.discover_button.props("loading" if self._busy else "loading=false")

    def close_editor(self) -> None:
        self.editor.set_visibility(False)
        self.token_input.set_value("")
        self._editing = None

    def create(self) -> None:
        if self._busy or not self._available:
            return
        self._editing = None
        self.editor_title.set_text("Новая модель")
        for field in (self.name_input, self.id_input, self.url_input, self.token_input):
            field.set_value("")
        self.token_hint.set_text("Токен необязателен для сервера без авторизации.")
        self.remove_token.set_value(False)
        self.remove_token.set_visibility(False)
        self.timeout_input.set_value(180)
        self.json_input.set_value(True)
        self.tools_input.set_value(True)
        self.editor.set_visibility(True)

    def edit(self) -> None:
        if self._busy or not self._available or self._selected is None:
            return
        self._editing = self._selected
        self.editor_title.set_text("Редактирование модели")
        self.name_input.set_value(self._editing.name)
        self.id_input.set_value(self._editing.model_id)
        self.url_input.set_value(self._editing.base_url)
        # Секрет остаётся на сервере; пустой ввод означает «не менять».
        self.token_input.set_value("")
        self.token_hint.set_text(
            "Токен сохранён. Оставьте поле пустым, чтобы сохранить его, или введите новый."
            if self._editing.token else "Сохранённого токена нет."
        )
        self.remove_token.set_value(False)
        self.remove_token.set_visibility(bool(self._editing.token))
        self.timeout_input.set_value(self._editing.timeout)
        self.json_input.set_value(self._editing.json_mode)
        self.tools_input.set_value(self._editing.tools_enabled)
        self.editor.set_visibility(True)

    def save(self) -> None:
        if self._busy or not self._available or not self.editor.visible:
            return
        try:
            model_id = (self.id_input.value or "").strip()
            token = (self.token_input.value or "").strip()
            if self.remove_token.value and token:
                raise ValueError("Введите новый токен или выберите удаление сохранённого токена.")
            if self._editing and not token and not self.remove_token.value:
                token = self._editing.token
            model = LLMModel(
                id=self._editing.id if self._editing else uuid4().hex,
                name=(self.name_input.value or "").strip() or model_id,
                model_id=model_id, base_url=(self.url_input.value or "").strip(), token=token,
                timeout=self.timeout_input.value, json_mode=self.json_input.value,
                tools_enabled=self.tools_input.value,
            )
            self.store.save(model, expected=self._editing)
        except (ValueError, ModelStorageError) as error:
            ui.notify(str(error), type="negative")
            return
        self.close_editor()
        self.refresh(model.id)
        self.on_change()
        ui.notify("Модель сохранена. Выберите её в шапке нужного диалога.", type="positive")

    def delete(self) -> None:
        if self._busy or not self._available or self._selected is None:
            return
        selected = self._selected

        def confirm() -> None:
            dialog.close()
            try:
                self.store.delete(selected.id, expected=selected)
            except (ValueError, KeyError, ModelStorageError) as error:
                ui.notify(str(error), type="negative")
                self.refresh()
                return
            self.close_editor()
            self.refresh()
            self.on_change()
            ui.notify("Модель удалена", type="positive")

        with self.root, ui.dialog() as dialog, ui.card().classes("delete-dialog"):
            ui.label("Удалить модель?").classes("text-lg font-semibold")
            ui.label(selected.name).classes("break-words")
            ui.label(
                "Переписка сохранится. В диалогах с этой моделью потребуется выбрать другую. "
                "Уже выполняющийся запрос завершится с прежними настройками."
            ).classes("memory-description")
            with ui.row().classes("w-full justify-end"):
                ui.button("Отмена", on_click=dialog.close).props("flat no-caps")
                ui.button("Удалить", on_click=confirm, color="negative").props("unelevated no-caps")
        dialog.on("hide", dialog.delete)
        dialog.open()

    def clear_result(self) -> None:
        self._discovery_credentials = None
        self.discovered.set_options([], value=[])
        self.result_status.set_text("")
        self.update_controls()

    async def discover(self) -> None:
        if self._busy or not self._available:
            return
        base_url = (self.server_url.value or "").strip()
        token = (self.server_token.value or "").strip()
        self._busy = True
        self._request = asyncio.current_task()
        self.clear_result()
        self.result_status.set_text("Подключение к серверу и получение списка моделей…")
        try:
            models = await run.io_bound(discover_models, base_url, token)
            self._discovery_credentials = (base_url, token)
            self.discovered.set_options(models, value=[])
            self.result_status.set_text(
                f"Найдено моделей: {len(models)}. Выберите нужные и добавьте в каталог."
                if models else "Сервер вернул пустой список моделей."
            )
        except (ValueError, ModelConnectionError, ModelStorageError) as error:
            self.result_status.set_text(str(error))
        except asyncio.CancelledError:
            self.clear_result()
            raise
        finally:
            self._busy = False
            self._request = None
            self.update_controls()

    def import_selected(self) -> None:
        if self._busy or not self._available or self._discovery_credentials is None:
            return
        selected = self.discovered.value or []
        if not selected:
            return
        base_url, token = self._discovery_credentials
        if (base_url, token) != ((self.server_url.value or "").strip(), (self.server_token.value or "").strip()):
            self.clear_result()
            ui.notify("Адрес или токен изменились. Получите список моделей повторно.", type="warning")
            return
        try:
            known = {model.id for model in self.store.list()}
            imported = self.store.import_models(base_url, token, selected)
            added = sum(model.id not in known for model in imported)
        except (ValueError, ModelStorageError) as error:
            ui.notify(str(error), type="negative")
            return
        self.result_status.set_text(f"Добавлено моделей: {added}. Уже в каталоге: {len(selected) - added}.")
        self.refresh(imported[0].id if imported else None)
        self.on_change()
