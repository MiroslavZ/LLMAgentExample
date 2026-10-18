import json
import socket
import tempfile
import threading
import time
import unittest
from pathlib import Path

import uvicorn

from llm_agent.mcp_client import call_tool, get_tools
from llm_agent.mcp_config import MCPServer
from SupportMCPServer.server import create_server
from SupportMCPServer.store import DEFAULT_DATA_PATH, SupportDataError, SupportStore


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.store = SupportStore()

    def test_user_list_is_only_a_short_card(self):
        users = self.store.list_users()["users"]
        self.assertEqual({user["id"] for user in users}, {"u-alex", "u-maria", "u-ivan"})
        self.assertTrue(all(set(user) == {"id", "name", "summary"} for user in users))

    def test_ticket_list_is_scoped_to_user(self):
        tickets = self.store.list_user_tickets("u-alex")["tickets"]
        self.assertEqual({ticket["id"] for ticket in tickets}, {"t-rag-off", "t-setup-resolved"})
        self.assertTrue(all(ticket["user_id"] == "u-alex" for ticket in tickets))
        with self.assertRaises(SupportDataError):
            self.store.list_user_tickets("u-unknown")

    def test_context_without_ticket_and_different_causes(self):
        alex = self.store.get_support_context("u-alex")
        maria = self.store.get_support_context("u-maria", "t-index-missing")
        self.assertTrue(alex["demo"])
        self.assertIsNone(alex["ticket"])
        self.assertFalse(alex["user"]["settings"]["rag_enabled"])
        self.assertTrue(maria["user"]["settings"]["rag_enabled"])
        self.assertEqual(maria["user"]["settings"]["index_status"], "missing")
        self.assertEqual(maria["ticket"]["status"], "waiting_user")

    def test_foreign_unknown_and_invalid_tickets_are_rejected(self):
        for ticket in ("t-index-missing", "t-unknown", "../demo_data.json", ""):
            with self.subTest(ticket=ticket), self.assertRaises(SupportDataError):
                self.store.get_support_context("u-alex", ticket)
        with self.assertRaises(SupportDataError):
            self.store.get_support_context("../demo_data.json")

    def test_json_is_reloaded_and_corruption_is_not_empty_success(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "support.json"
            data = json.loads(DEFAULT_DATA_PATH.read_text(encoding="utf-8"))
            path.write_text(json.dumps(data), encoding="utf-8")
            store = SupportStore(path)
            self.assertFalse(store.get_support_context("u-alex")["user"]["settings"]["rag_enabled"])
            data["users"][0]["settings"]["rag_enabled"] = True
            path.write_text(json.dumps(data), encoding="utf-8")
            self.assertTrue(store.get_support_context("u-alex")["user"]["settings"]["rag_enabled"])
            path.write_text("broken json", encoding="utf-8")
            with self.assertRaises(SupportDataError):
                store.list_users()

    def test_invalid_data_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "support.json"
            store = SupportStore(path)
            with self.assertRaises(SupportDataError):
                store.list_users()
            original = json.loads(DEFAULT_DATA_PATH.read_text(encoding="utf-8"))
            for mutation in ("duplicate", "unknown_owner", "secret", "boolean"):
                data = json.loads(json.dumps(original))
                if mutation == "duplicate":
                    data["users"].append(data["users"][0])
                elif mutation == "unknown_owner":
                    data["tickets"][0]["user_id"] = "u-unknown"
                elif mutation == "secret":
                    data["users"][0]["api_key"] = "not-a-real-secret"
                else:
                    data["users"][0]["settings"]["rag_enabled"] = "false"
                path.write_text(json.dumps(data), encoding="utf-8")
                with self.subTest(mutation=mutation), self.assertRaises(SupportDataError):
                    store.list_users()


class HttpIntegrationTests(unittest.IsolatedAsyncioTestCase):
    """Официальный SDK-сервер и существующий клиент общаются через localhost TCP."""

    @classmethod
    def setUpClass(cls):
        app = create_server().streamable_http_app(stateless_http=True, json_response=True)
        cls.socket = socket.socket()
        cls.socket.bind(("127.0.0.1", 0))
        cls.socket.listen(128)
        port = cls.socket.getsockname()[1]
        cls.server = uvicorn.Server(uvicorn.Config(app, log_level="critical"))
        cls.thread = threading.Thread(target=cls.server.run, kwargs={"sockets": [cls.socket]}, daemon=True)
        cls.thread.start()
        deadline = time.monotonic() + 10
        while not cls.server.started:
            if not cls.thread.is_alive() or time.monotonic() >= deadline:
                cls.server.should_exit = True
                cls.thread.join(timeout=2)
                cls.socket.close()
                raise RuntimeError("MCP HTTP-сервер не запустился")
            time.sleep(0.01)
        cls.connection = MCPServer(id="support", name="Support", url=f"http://127.0.0.1:{port}/mcp")

    @classmethod
    def tearDownClass(cls):
        cls.server.should_exit = True
        cls.thread.join(timeout=10)
        cls.socket.close()
        if cls.thread.is_alive():
            raise RuntimeError("MCP HTTP-сервер не завершился")

    async def test_discovery_and_all_tools_with_existing_client(self):
        before = DEFAULT_DATA_PATH.read_bytes()
        discovery = await get_tools(self.connection)
        self.assertEqual({tool.name for tool in discovery.tools}, {
            "list_users", "list_user_tickets", "get_support_context",
        })
        self.assertTrue(all(tool.annotations.read_only_hint for tool in discovery.tools))
        for tool, arguments, key in (
            ("list_users", {}, "users"),
            ("list_user_tickets", {"user_id": "u-alex"}, "tickets"),
            ("get_support_context", {"user_id": "u-ivan", "ticket_id": "t-mcp-disabled"}, "user"),
        ):
            with self.subTest(tool=tool):
                result = await call_tool(self.connection, tool, arguments)
                self.assertFalse(result.is_error)
                self.assertIn(key, result.structured_content)
        self.assertEqual(DEFAULT_DATA_PATH.read_bytes(), before)

    async def test_foreign_ticket_is_mcp_error_and_has_no_foreign_context(self):
        result = await call_tool(self.connection, "get_support_context", {
            "user_id": "u-alex", "ticket_id": "t-index-missing",
        })
        self.assertTrue(result.is_error)
        self.assertIsNone(result.structured_content)
        text = " ".join(block.text for block in result.content if hasattr(block, "text"))
        self.assertIn("недоступен", text)
        self.assertNotIn("Мария", text)


if __name__ == "__main__":
    unittest.main()
