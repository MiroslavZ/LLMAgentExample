"""MCP в NiceGUI: настройки, асинхронная загрузка, ошибки и отмена."""

import asyncio
import inspect
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock, patch

from mcp import Tool
from nicegui import Client, core, ui

from llm_agent.mcp_client import MCPConnectionError, MCPDiscovery
from llm_agent.mcp_config import GITHUB_MCP_URL, GITHUB_TOKEN_ENV, MCPServer
from llm_agent.service import ConversationService
from llm_agent.web.jobs import RequestRunner
from llm_agent.web.page import ChatPage


class MCPPageTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        directory = self.enterContext(tempfile.TemporaryDirectory())
        self.service = ConversationService(Path(directory), token=None)
        self.enterContext(patch("nicegui.background_tasks.create_or_defer",
                                side_effect=lambda coroutine, **_: coroutine.close()))
        self.enterContext(patch("llm_agent.web.page.ui.timer"))
        self.enterContext(patch("llm_agent.web.page.ui.navigate.history.replace"))
        self.notify = self.enterContext(patch("llm_agent.web.mcp.ui.notify"))
        self.enterContext(patch("llm_agent.agent.OpenAI", side_effect=AssertionError("LLM не нужен для MCP")))
        self.enterContext(patch.object(core, "loop", asyncio.get_running_loop()))
        self.client = Client(ui.page("/mcp-unit-test"))
        self.addCleanup(self.client.delete)
        with self.client:
            self.page = ChatPage(self.service, RequestRunner(self.service), token_available=False)
            self.page.build()
        self.panel = self.page.mcp_panel
        self.result = MCPDiscovery("Test server", "1.0", "2025-03-26", (
            Tool(name="get_issue", description="Прочитать issue", input_schema={
                "type": "object", "properties": {"number": {"type": "integer"}}, "required": ["number"],
            }),
        ))

    @staticmethod
    def labels(container):
        return [element.text for element in container.descendants() if isinstance(element, ui.label)]

    async def click(self, button):
        handler = next(listener.handler for listener in button._event_listeners.values() if listener.type == "click")
        with patch("nicegui.elements.button.handle_event") as dispatch:
            handler(None)
        with button.parent_slot:
            result = dispatch.call_args.args[0]()
            if inspect.isawaitable(result):
                await result

    def add_server(self, server_id="github"):
        server = MCPServer(server_id, server_id, GITHUB_MCP_URL, GITHUB_TOKEN_ENV)
        self.service.mcp_servers.save(server)
        with self.client:
            self.panel.open()
        return server

    async def test_header_add_edit_delete_and_reopen_without_llm_key(self):
        button = next(element for element in self.client.elements.values()
                      if isinstance(element, ui.button) and element.text == "MCP")
        await self.click(button)
        self.assertTrue(self.panel.dialog.value)
        self.assertFalse(self.panel.tools_button.enabled)
        await self.click(self.panel.new_button)
        self.assertEqual(self.panel.url_input.value, GITHUB_MCP_URL)
        self.assertEqual(self.panel.token_input.value, GITHUB_TOKEN_ENV)
        await self.click(self.panel.save_button)
        saved, = self.service.mcp_servers.list()
        self.assertEqual(saved.name, "GitHub")
        self.assertEqual(self.panel.selector.value, saved.id)
        self.assertTrue(self.panel.tools_button.enabled)
        await self.click(self.panel.edit_button)
        self.panel.name_input.set_value("GitHub рабочий")
        await self.click(self.panel.save_button)
        reopened = ConversationService(self.service.store.data_dir, token=None)
        self.assertEqual(reopened.mcp_servers.get(saved.id).name, "GitHub рабочий")
        await self.click(self.panel.delete_button)
        self.assertEqual(reopened.mcp_servers.list(), [])
        self.assertFalse(self.panel.tools_button.enabled)

    async def test_get_tools_renders_metadata_and_schema_then_clears_on_error(self):
        server = self.add_server()
        with patch("llm_agent.web.mcp.get_tools", new=AsyncMock(return_value=self.result)) as discover:
            await self.click(self.panel.tools_button)
        discover.assert_awaited_once_with(server)
        self.assertIn("Инструментов: 1", self.panel.result_status.text)
        self.assertIn("Прочитать issue", self.labels(self.panel.tools_container))
        expansions = [element for element in self.panel.tools_container.descendants() if isinstance(element, ui.expansion)]
        self.assertEqual(expansions[0].text, "get_issue")
        code = next(element for element in self.panel.tools_container.descendants() if isinstance(element, ui.code))
        self.assertIn('"number"', code.content)
        with patch("llm_agent.web.mcp.get_tools", new=AsyncMock(side_effect=MCPConnectionError("Ошибка авторизации"))):
            await self.click(self.panel.tools_button)
        self.assertEqual(self.panel.result_status.text, "Ошибка авторизации")
        self.assertEqual(list(self.panel.tools_container), [])
        self.assertTrue(self.panel.tools_button.enabled)

    async def test_empty_tools_and_switch_server_clear_previous_result(self):
        self.add_server()
        self.add_server("other")
        with patch("llm_agent.web.mcp.get_tools", new=AsyncMock(return_value=replace(self.result, tools=()))):
            await self.click(self.panel.tools_button)
        self.assertIn("Инструментов: 0", self.panel.result_status.text)
        self.assertIn("Сервер вернул пустой список инструментов.", self.labels(self.panel.tools_container))
        with self.client:
            self.panel.selector.set_value("other")
        self.assertEqual(self.panel.result_status.text, "")
        self.assertEqual(list(self.panel.tools_container), [])

    async def test_pending_request_does_not_block_loop_and_close_cancels(self):
        self.add_server()
        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def wait_for_server(_):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        with patch("llm_agent.web.mcp.get_tools", side_effect=wait_for_server) as discover:
            with self.client:
                request = asyncio.create_task(self.panel.discover())
            await asyncio.wait_for(started.wait(), 1)
            self.assertFalse(self.panel.tools_button.enabled)
            self.assertFalse(self.panel.selector.enabled)
            await self.panel.discover()
            self.assertEqual(discover.call_count, 1)
            self.panel.cancel_request()
            with self.assertRaises(asyncio.CancelledError):
                await request
        self.assertTrue(cancelled.is_set())
        self.assertTrue(self.panel.tools_button.enabled)
        self.assertFalse(self.panel._busy)

    async def test_settings_changed_while_loading_do_not_publish_stale_tools(self):
        server = self.add_server()

        async def change_settings(_):
            self.service.mcp_servers.save(replace(server, url="https://example.com/mcp"))
            return self.result

        with patch("llm_agent.web.mcp.get_tools", side_effect=change_settings):
            await self.click(self.panel.tools_button)
        self.assertIn("изменены в другой вкладке", self.panel.result_status.text)
        self.assertEqual(list(self.panel.tools_container), [])

    async def test_invalid_url_keeps_editor_and_does_not_save(self):
        with self.client:
            self.panel.open()
        await self.click(self.panel.new_button)
        self.panel.url_input.set_value("https://example.com/?token=secret")
        await self.click(self.panel.save_button)
        self.assertTrue(self.panel.editor.visible)
        self.assertEqual(self.service.mcp_servers.list(), [])
        self.notify.assert_called_once()


if __name__ == "__main__":
    unittest.main()
