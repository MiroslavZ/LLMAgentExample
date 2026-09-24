"""Общие MCP-подключения, включение инструментов в чате и проверка сервера."""

import asyncio
import json
from uuid import uuid4

from nicegui import ui

from ..mcp_client import MCPConnectionError, MCPDiscovery, get_tools
from ..mcp_config import LOCAL_GITHUB_MCP_URL, MCPServer, MCPServerStore, MCPStorageError


class MCPPanel:
    def __init__(self, store: MCPServerStore) -> None:
        self.store = store
        self.dialog: ui.dialog | None = None
        self._selected: MCPServer | None = None
        self._editing: MCPServer | None = None
        self._available = False
        self._busy = False
        self._request: asyncio.Task | None = None

    def build(self) -> None:
        self.root = ui.element("div")

    def open(self) -> None:
        if self.dialog is None:
            self._build_dialog()
        self.refresh()
        self.dialog.open()

    def _build_dialog(self) -> None:
        with self.root, ui.dialog() as self.dialog, ui.card().classes("mcp-dialog"):
            with ui.row().classes("w-full items-center no-wrap"):
                ui.icon("hub", size="24px")
                ui.label("MCP-серверы").classes("text-lg font-semibold")
                ui.space()
                ui.button(icon="close", on_click=self.dialog.close).props('flat round dense aria-label="Закрыть MCP"')
            ui.label(
                "Добавьте сервер, проверьте его инструменты и включите «Использовать в чате». "
                "После сохранения агент сможет сам выбирать инструменты для ваших запросов "
                "и использовать результаты в ответе. Настройка действует для всех диалогов."
            ).classes("memory-description")
            self.status = ui.label().classes("memory-status")
            self.selector = ui.select({}, label="MCP-сервер", on_change=self.select).props("outlined").classes("w-full")
            self.address = ui.label().classes("memory-description break-all")
            with ui.row().classes("w-full gap-2"):
                self.new_button = ui.button("Добавить MCP", icon="add", on_click=self.create).props("unelevated no-caps")
                self.edit_button = ui.button("Редактировать", icon="edit", on_click=self.edit).props("outline no-caps")
                self.delete_button = ui.button("Удалить", icon="delete_outline", on_click=self.delete).props("flat no-caps color=negative")
            with ui.column().classes("mcp-editor") as self.editor:
                self.editor_title = ui.label().classes("font-medium")
                self.name_input = ui.input("Название *").props("outlined dense").classes("w-full")
                self.url_input = ui.input("URL MCP-сервера *").props("outlined dense").classes("w-full")
                ui.label("Streamable HTTP: HTTPS или локальный HTTP для разработки.").classes("memory-description")
                self.token_input = ui.input("Переменная окружения с токеном").props("outlined dense").classes("w-full")
                ui.label(
                    "Для собственного сервера укажите имя переменной с токеном доступа к MCP, "
                    "например MCP_ACCESS_TOKEN. Сам токен добавьте в .env агента и перезапустите приложение. "
                    "GitHub-токен хранится на вашем MCP-сервере. "
                    "Для локального сервера без авторизации оставьте поле пустым."
                ).classes("memory-description")
                self.enabled_input = ui.checkbox("Использовать в чате", value=False)
                with ui.row().classes("w-full justify-end"):
                    ui.button("Отмена", on_click=lambda: self.editor.set_visibility(False)).props("flat no-caps")
                    self.save_button = ui.button("Сохранить MCP", icon="save", on_click=self.save).props("unelevated no-caps")
            self.editor.set_visibility(False)
            self.tools_button = ui.button("Get tools", icon="sync", on_click=self.discover).props("unelevated no-caps")
            self.result_status = ui.label().classes("memory-status")
            self.tools_container = ui.column().classes("mcp-tools")
        self.dialog.on("hide", self.cancel_request)

    def cancel_request(self) -> None:
        if self._request is not None:
            self._request.cancel()

    def refresh(self, selected_id: str | None = None) -> None:
        try:
            servers = self.store.list()
            selected_id = selected_id or self.selector.value
            if selected_id not in {server.id for server in servers}:
                selected_id = servers[0].id if servers else None
            selected = next((server for server in servers if server.id == selected_id), None)
            if selected != self._selected:
                self.clear_result()
            self._selected = selected
            self._available = True
            self.selector.set_options({server.id: server.name for server in servers}, value=selected_id)
            self.address.set_text(
                f"{selected.url} · {'Включён в чате' if selected.enabled else 'Выключен в чате'}"
                if selected else ""
            )
            self.status.set_text("" if servers else "Серверов пока нет. Нажмите «Добавить MCP» — форма уже настроена для GitHub.")
        except MCPStorageError as error:
            self._available = False
            self.status.set_text(str(error))
            self.clear_result()
        self.update_controls()

    def select(self) -> None:
        if not self._busy:
            self.editor.set_visibility(False)
            self.refresh()

    def update_controls(self) -> None:
        enabled = self._available and not self._busy
        for control in (self.selector, self.new_button, self.save_button,
                        self.name_input, self.url_input, self.token_input, self.enabled_input):
            control.set_enabled(enabled)
        for control in (self.edit_button, self.delete_button, self.tools_button):
            control.set_enabled(enabled and self._selected is not None)
        self.tools_button.props("loading" if self._busy else "loading=false")

    def clear_result(self) -> None:
        self.tools_container.clear()
        self.result_status.set_text("")

    def create(self) -> None:
        if self._busy or not self._available:
            return
        self._editing = None
        self.editor_title.set_text("Новый MCP-сервер")
        self.name_input.set_value("GitHub")
        self.url_input.set_value(LOCAL_GITHUB_MCP_URL)
        self.token_input.set_value("")
        self.enabled_input.set_value(False)
        self.editor.set_visibility(True)

    def edit(self) -> None:
        if self._busy or not self._available or self._selected is None:
            return
        self._editing = self._selected
        self.editor_title.set_text("Редактирование MCP-сервера")
        self.name_input.set_value(self._editing.name)
        self.url_input.set_value(self._editing.url)
        self.token_input.set_value(self._editing.token_env)
        self.enabled_input.set_value(self._editing.enabled)
        self.editor.set_visibility(True)

    def save(self) -> None:
        if self._busy or not self._available or not self.editor.visible:
            return
        try:
            server = MCPServer(
                id=self._editing.id if self._editing else uuid4().hex,
                name=(self.name_input.value or "").strip(), url=(self.url_input.value or "").strip(),
                token_env=(self.token_input.value or "").strip(),
                enabled=self.enabled_input.value,
            )
            self.store.save(server, expected=self._editing)
        except (ValueError, MCPStorageError) as error:
            ui.notify(str(error), type="negative")
            return
        self.editor.set_visibility(False)
        self.refresh(server.id)
        ui.notify("MCP-сервер сохранён", type="positive")

    def delete(self) -> None:
        if self._busy or not self._available or self._selected is None:
            return
        try:
            self.store.delete(self._selected.id)
        except (ValueError, MCPStorageError) as error:
            ui.notify(str(error), type="negative")
        self.editor.set_visibility(False)
        self.refresh()

    async def discover(self) -> None:
        if self._busy or not self._available or self._selected is None:
            return
        self._busy = True
        self._request = asyncio.current_task()
        self.editor.set_visibility(False)
        self.clear_result()
        self.result_status.set_text("Подключение к MCP-серверу и получение инструментов…")
        self.update_controls()
        try:
            # Берём актуальную запись: её могли изменить в другой вкладке.
            server = self.store.get(self._selected.id)
            result = await get_tools(server)
            if self.store.get(server.id) != server:
                raise ValueError("Настройки сервера изменены в другой вкладке. Повторите Get tools.")
            self.render_result(result)
        except (MCPConnectionError, MCPStorageError, ValueError) as error:
            self.result_status.set_text(str(error))
        except asyncio.CancelledError:
            self.result_status.set_text("Получение инструментов отменено.")
            raise
        finally:
            self._busy = False
            self._request = None
            self.update_controls()

    def render_result(self, result: MCPDiscovery) -> None:
        self.result_status.set_text(
            f"Соединение проверено · {result.server_name} {result.server_version} · "
            f"MCP {result.protocol_version} · Инструментов: {len(result.tools)}"
        )
        with self.tools_container:
            if not result.tools:
                ui.label("Сервер вернул пустой список инструментов.").classes("memory-description")
            for tool in result.tools:
                with ui.expansion(tool.name, icon="build").classes("mcp-tool"):
                    ui.label(tool.description or "Описание отсутствует.").classes("memory-value")
                    ui.label("Входные параметры (JSON Schema)").classes("font-medium text-sm")
                    ui.code(json.dumps(tool.input_schema, ensure_ascii=False, indent=2), language="json").classes("w-full")
