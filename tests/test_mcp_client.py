"""Проверка MCP SDK на уровне протокола без GitHub, LLM и настоящих токенов."""

import asyncio
import io
import json
import socket
import unittest
from contextlib import contextmanager, nullcontext, redirect_stderr, redirect_stdout
from unittest.mock import AsyncMock, patch

import anyio
import httpx2
import uvicorn
from mcp import Tool, types
from mcp.server import MCPServer as SDKServer

from llm_agent.mcp_client import MCPConnectionError, MCPDiscovery, get_tools, main
from llm_agent.mcp_config import MCPServer


_ASYNC_CLIENT = httpx2.AsyncClient
_TOKEN_ENV = "MCP_TEST_ACCESS_TOKEN"
_TOKEN = "test-only-secret"


class WireServer:
    """HTTP-ответы тестового MCP; сам клиент и сериализация SDK настоящие."""

    def __init__(self, *, pages=None, capabilities=None, status=None, invalid=False,
                 rpc_error=False, wait=False):
        self.pages = pages if pages is not None else [{"tools": []}]
        self.capabilities = capabilities if capabilities is not None else {"tools": {}}
        self.status = status
        self.invalid = invalid
        self.rpc_error = rpc_error
        self.wait = wait
        self.requests = []
        self.messages = []
        self.list_started = asyncio.Event()

    async def __call__(self, request):
        self.requests.append(request)
        if request.method == "GET":
            return httpx2.Response(405)
        if request.method == "DELETE":
            return httpx2.Response(204)
        message = json.loads(request.content)
        self.messages.append(message)
        if self.status is not None:
            return httpx2.Response(self.status, text=f"private response {_TOKEN}")
        method = message["method"]
        if method in {"notifications/initialized", "notifications/cancelled"}:
            return httpx2.Response(202)
        if method == "initialize":
            result = {
                "protocolVersion": message["params"]["protocolVersion"],
                "capabilities": self.capabilities,
                "serverInfo": {"name": "wire-test", "version": "1.2.3"},
            }
        elif method == "tools/list":
            self.list_started.set()
            if self.wait:
                await asyncio.Event().wait()
            if self.invalid:
                return httpx2.Response(200, text="invalid JSON", headers={"content-type": "application/json"})
            if self.rpc_error:
                return httpx2.Response(200, json={
                    "jsonrpc": "2.0", "id": message["id"],
                    "error": {"code": -32603, "message": f"private RPC error {_TOKEN}"},
                })
            index = sum(item["method"] == "tools/list" for item in self.messages) - 1
            result = self.pages[min(index, len(self.pages) - 1)]
        else:
            raise AssertionError(f"Неожиданный вызов MCP: {method}")
        return httpx2.Response(200, json={"jsonrpc": "2.0", "id": message["id"], "result": result},
                               headers={"Mcp-Session-Id": "test-session"})


@contextmanager
def mock_http(server):
    clients = []

    def create_client(**kwargs):
        client = _ASYNC_CLIENT(transport=httpx2.MockTransport(server), trust_env=False, **kwargs)
        clients.append(client)
        return client

    with patch("llm_agent.mcp_client.httpx2.AsyncClient", side_effect=create_client):
        yield clients


def settings(*, authenticated=False):
    return MCPServer("test", "Тест", "https://mcp.example.test/mcp",
                     _TOKEN_ENV if authenticated else "")


class MCPClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_initialize_notification_paginated_tools_and_cleanup(self):
        wire = WireServer(pages=[
            {"tools": [{"name": "read_issue", "description": "Читает задачу",
                        "inputSchema": {"type": "object", "properties": {"number": {"type": "integer"}},
                                        "required": ["number"]}}], "nextCursor": "page-two"},
            {"tools": [{"name": "list_repos", "inputSchema": {"type": "object"}}]},
        ])
        with patch.dict("os.environ", {_TOKEN_ENV: _TOKEN}), mock_http(wire) as clients:
            result = await get_tools(settings(authenticated=True), timeout=2)

        self.assertEqual((result.server_name, result.server_version, result.protocol_version),
                         ("wire-test", "1.2.3", wire.messages[0]["params"]["protocolVersion"]))
        self.assertIsInstance(result.tools, tuple)
        self.assertEqual([tool.name for tool in result.tools], ["read_issue", "list_repos"])
        self.assertEqual(result.tools[0].input_schema["required"], ["number"])
        self.assertEqual([message["method"] for message in wire.messages],
                         ["initialize", "notifications/initialized", "tools/list", "tools/list"])
        self.assertEqual(wire.messages[-1]["params"]["cursor"], "page-two")
        self.assertTrue(all(request.headers.get("Authorization") == f"Bearer {_TOKEN}"
                            for request in wire.requests))
        self.assertEqual(wire.requests[-1].method, "DELETE")
        self.assertTrue(clients[0].is_closed)

    async def test_empty_catalog_is_valid_without_authentication(self):
        wire = WireServer()
        with mock_http(wire) as clients:
            result = await get_tools(settings(), timeout=2)
        self.assertEqual(result.tools, ())
        self.assertTrue(all("Authorization" not in request.headers for request in wire.requests))
        self.assertTrue(clients[0].is_closed)

    async def test_missing_or_invalid_token_does_not_connect(self):
        for token in ("", "\n", "secret\nheader", "секрет"):
            with self.subTest(token=token), patch.dict("os.environ", {_TOKEN_ENV: token}), \
                    patch("llm_agent.mcp_client.httpx2.AsyncClient") as client:
                with self.assertRaises(MCPConnectionError):
                    await get_tools(settings(authenticated=True))
                client.assert_not_called()

    async def test_invalid_timeout_does_not_connect(self):
        for timeout in (0, -1, float("nan"), float("inf")):
            with self.subTest(timeout=timeout), patch("llm_agent.mcp_client.httpx2.AsyncClient") as client:
                with self.assertRaises(ValueError):
                    await get_tools(settings(), timeout=timeout)
                client.assert_not_called()

    async def test_http_errors_are_classified_and_do_not_expose_response(self):
        for status, message in ((401, "авторизацию"), (403, "авторизацию"),
                                (404, "404"), (500, "500")):
            wire = WireServer(status=status)
            with self.subTest(status=status), mock_http(wire) as clients:
                with self.assertRaisesRegex(MCPConnectionError, message) as caught:
                    await get_tools(settings(), timeout=2)
                self.assertNotIn(_TOKEN, str(caught.exception))
                self.assertTrue(clients[0].is_closed)

    async def test_invalid_json_and_rpc_error_are_sanitized(self):
        for options in ({"invalid": True}, {"rpc_error": True}):
            wire = WireServer(**options)
            logs = self.assertLogs("mcp.client.streamable_http", level="ERROR") if options.get("invalid") else nullcontext()
            with self.subTest(options=options), mock_http(wire) as clients, logs:
                with self.assertRaises(MCPConnectionError) as caught:
                    await get_tools(settings(), timeout=2)
                self.assertNotIn(_TOKEN, str(caught.exception))
                self.assertTrue(clients[0].is_closed)

    async def test_network_error_is_classified_and_sanitized(self):
        async def offline(request):
            raise httpx2.ConnectError(f"private network details {_TOKEN}", request=request)

        with mock_http(offline) as clients:
            with self.assertRaisesRegex(MCPConnectionError, "адрес и сеть") as caught:
                await get_tools(settings(), timeout=2)
        self.assertNotIn(_TOKEN, str(caught.exception))
        self.assertTrue(clients[0].is_closed)

    async def test_redirect_does_not_send_token_to_another_origin(self):
        requests = []

        async def redirect(request):
            requests.append(request)
            return httpx2.Response(307, headers={"Location": "https://other.example.test/mcp"})

        with patch.dict("os.environ", {_TOKEN_ENV: _TOKEN}), mock_http(redirect) as clients, \
                patch("mcp.client.streamable_http.logger.warning"):
            with self.assertRaises(MCPConnectionError):
                await get_tools(settings(authenticated=True), timeout=2)
        self.assertEqual(len(requests), 1)
        self.assertEqual(str(requests[0].url), settings().url)
        self.assertTrue(clients[0].is_closed)

    async def test_server_without_tools_capability_stops_after_initialize(self):
        wire = WireServer(capabilities={})
        with mock_http(wire) as clients:
            with self.assertRaisesRegex(MCPConnectionError, "не поддерживает инструменты"):
                await get_tools(settings(), timeout=2)
        self.assertNotIn("tools/list", [message["method"] for message in wire.messages])
        self.assertTrue(clients[0].is_closed)

    async def test_repeated_pagination_cursor_is_rejected(self):
        wire = WireServer(pages=[{"tools": [], "nextCursor": "loop"}])
        with mock_http(wire) as clients:
            with self.assertRaisesRegex(MCPConnectionError, "повторил курсор"):
                await get_tools(settings(), timeout=2)
        self.assertEqual(sum(message["method"] == "tools/list" for message in wire.messages), 2)
        self.assertTrue(clients[0].is_closed)

    async def test_timeout_closes_http_client(self):
        wire = WireServer(wait=True)
        with mock_http(wire) as clients:
            with self.assertRaisesRegex(MCPConnectionError, "отведённое время"):
                await get_tools(settings(), timeout=0.1)
        self.assertTrue(wire.list_started.is_set())
        self.assertTrue(clients[0].is_closed)

    async def test_cancellation_propagates_and_closes_http_client(self):
        wire = WireServer(wait=True)
        with mock_http(wire) as clients:
            task = asyncio.create_task(get_tools(settings(), timeout=2))
            try:
                await asyncio.wait_for(wire.list_started.wait(), timeout=1)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
            finally:
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        self.assertTrue(clients[0].is_closed)

    async def test_real_sdk_server_over_loopback_http(self):
        with patch("mcp.server.mcpserver.server.configure_logging"):
            sdk_server = SDKServer("local-integration-test", version="1.0.0")
        invocations = []

        @sdk_server.tool()
        def echo(text: str) -> str:
            """Возвращает переданный текст."""
            invocations.append(text)
            return text

        app = sdk_server.streamable_http_app(json_response=True)
        http_server = uvicorn.Server(uvicorn.Config(app, log_level="error", log_config=None, lifespan="on"))
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.setblocking(False)
            port = listener.getsockname()[1]
            serving = asyncio.create_task(http_server.serve(sockets=[listener]))
            try:
                with anyio.fail_after(5):
                    while not http_server.started:
                        if serving.done():
                            await serving
                            self.fail("Тестовый MCP-сервер завершился до запуска")
                        await asyncio.sleep(0.01)
                server = MCPServer("local", "Локальный тест", f"http://127.0.0.1:{port}/mcp")
                with patch.dict("os.environ", {"NO_PROXY": "127.0.0.1", "no_proxy": "127.0.0.1"}):
                    result = await get_tools(server, timeout=5)
            finally:
                http_server.should_exit = True
                await asyncio.wait_for(serving, timeout=5)
        self.assertEqual((result.server_name, result.server_version), ("local-integration-test", "1.0.0"))
        self.assertEqual([tool.name for tool in result.tools], ["echo"])
        self.assertEqual(result.tools[0].input_schema["required"], ["text"])
        self.assertEqual(invocations, [])


class MCPClientCLITests(unittest.TestCase):
    def test_cli_outputs_json_and_passes_connection_settings(self):
        discovery = MCPDiscovery("Проверка", "1.0", types.LATEST_PROTOCOL_VERSION, (
            Tool(name="echo", input_schema={"type": "object"}),
        ))
        output = io.StringIO()
        with patch("sys.argv", ["mcp_client", "--url", "http://localhost:8000/mcp", "--token-env", ""]), \
                patch("llm_agent.mcp_client.load_dotenv"), \
                patch("llm_agent.mcp_client.get_tools", new_callable=AsyncMock, return_value=discovery) as discover, \
                redirect_stdout(output):
            self.assertEqual(main(), 0)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["server"], "Проверка")
        self.assertEqual(payload["tools"][0]["inputSchema"], {"type": "object"})
        self.assertEqual(discover.await_args.args[0].url, "http://localhost:8000/mcp")
        self.assertEqual(discover.await_args.args[0].token_env, "")

    def test_cli_connection_error_exits_without_json_or_traceback(self):
        output, errors = io.StringIO(), io.StringIO()
        with patch("sys.argv", ["mcp_client"]), patch("llm_agent.mcp_client.load_dotenv"), \
                patch("llm_agent.mcp_client.get_tools", new_callable=AsyncMock,
                      side_effect=MCPConnectionError("Не удалось подключиться")), \
                redirect_stdout(output), redirect_stderr(errors):
            with self.assertRaises(SystemExit) as caught:
                main()
        self.assertEqual(caught.exception.code, 1)
        self.assertEqual(output.getvalue(), "")
        self.assertIn("Ошибка: Не удалось подключиться", errors.getvalue())
        self.assertNotIn("Traceback", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
