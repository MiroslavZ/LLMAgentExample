"""Чат → function calling → реальный GitHub MCP по HTTP → ответ и сохранение."""

import asyncio
import importlib
import json
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import anyio
import httpx2
import uvicorn
from openai import OpenAI
from openai.resources.chat.completions import Completions
from openai.types.chat import ChatCompletion

from MCPServerExample import github_api, github_results
from MCPServerExample.http_server import Settings, create_app
from llm_agent.history import DialogueUsage, HistoryManager
from llm_agent.mcp_config import MCPServer
from llm_agent.service import ConversationService


class MCPIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_chat_uses_github_mcp_over_http_and_persists_result(self):
        # Сервер рассчитан также на самостоятельный запуск из своего каталога.
        with patch.dict("sys.modules", {"github_api": github_api, "github_results": github_results}), \
                patch("mcp.server.mcpserver.server.configure_logging"):
            mcp = importlib.import_module("MCPServerExample.server").mcp

        token_env = "MCP_INTEGRATION_ACCESS_TOKEN"
        token = "test-only-mcp-bearer"
        call_id = "call_github_repository_17"
        fixture = {
            "full_name": "octocat/Hello-World", "stargazers_count": 1742,
            "html_url": "https://github.com/octocat/Hello-World", "default_branch": "main",
            "events_url": "https://api.github.com/repos/octocat/Hello-World/events",
        }
        expected_data = {key: value for key, value in fixture.items() if key != "events_url"}
        expected_answer = "octocat/Hello-World: 1742 звёзд, ветка main. https://github.com/octocat/Hello-World"
        app = create_app(mcp, Settings(access_token=token))
        http_requests = []
        rpc_messages = []
        active_requests = set()

        async def observe_http(scope, receive, send):
            if scope["type"] != "http":
                return await app(scope, receive, send)
            request_id = len(http_requests)
            http_requests.append(scope)
            active_requests.add(request_id)
            body = bytearray()

            async def observe_receive():
                message = await receive()
                if message["type"] == "http.request":
                    body.extend(message.get("body", b""))
                    if not message.get("more_body", False) and body:
                        rpc_messages.append(json.loads(body))
                        body.clear()
                return message

            try:
                await app(scope, observe_receive, send)
            finally:
                active_requests.remove(request_id)

        http_server = uvicorn.Server(uvicorn.Config(
            observe_http, log_level="error", log_config=None, lifespan="on",
        ))
        mcp_clients = []
        llm_clients = []
        real_http_client = httpx2.AsyncClient

        def tracked_mcp_client(**kwargs):
            client = real_http_client(trust_env=False, **kwargs)
            mcp_clients.append(client)
            return client

        def tracked_llm_client(**kwargs):
            client = OpenAI(**kwargs)
            llm_clients.append(client)
            return client

        with tempfile.TemporaryDirectory() as directory, socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.setblocking(False)
            port = listener.getsockname()[1]
            service = ConversationService(Path(directory), "test-only-llm-token")
            conversation = service.create()
            connection = MCPServer(
                "own-github", "Мой GitHub MCP", f"http://127.0.0.1:{port}/mcp",
                token_env, enabled=True,
            )
            service.mcp_servers.save(connection)
            llm_requests = []

            def respond_to_model(**request):
                llm_requests.append(request)
                self.assertEqual(request["tool_choice"], "auto")
                if len(llm_requests) == 1:
                    self.assertEqual(len(request["tools"]), 7)
                    tool = next(item["function"] for item in request["tools"]
                                if "get_repository." in item["function"]["description"])
                    schema = tool["parameters"]
                    self.assertEqual(schema["required"], ["owner", "repo"])
                    self.assertTrue(schema["properties"]["owner"]["description"])
                    message = {"role": "assistant", "content": None, "tool_calls": [{
                        "id": call_id, "type": "function", "function": {
                            "name": tool["name"],
                            "arguments": json.dumps({"owner": "octocat", "repo": "Hello-World"}),
                        },
                    }]}
                    prompt, output, reason = 100, 15, "tool_calls"
                else:
                    self.assertEqual(len(llm_requests), 2)
                    messages = request["messages"]
                    self.assertEqual(messages[-2]["tool_calls"][0]["id"], call_id)
                    self.assertEqual(messages[-1]["role"], "tool")
                    self.assertEqual(messages[-1]["tool_call_id"], call_id)
                    result = json.loads(messages[-1]["content"])
                    self.assertEqual(result, {"is_error": False, "data": expected_data})
                    # Результат уже записан до финального обращения к модели.
                    pending = service.get(conversation.id).turns[-1]
                    self.assertEqual(pending.status, "running")
                    self.assertEqual(pending.tool_calls[0].call_id, call_id)
                    self.assertEqual(json.loads(pending.tool_calls[0].result), result)
                    repo = result["data"]
                    message = {"role": "assistant", "content": (
                        f"{repo['full_name']}: {repo['stargazers_count']} звёзд, "
                        f"ветка {repo['default_branch']}. {repo['html_url']}"
                    )}
                    prompt, output, reason = 120, 20, "stop"
                return ChatCompletion.model_validate({
                    "id": f"completion-{len(llm_requests)}", "created": 0,
                    "model": "deepseek-chat", "object": "chat.completion",
                    "choices": [{"index": 0, "finish_reason": reason, "message": message}],
                    "usage": {"prompt_tokens": prompt, "completion_tokens": output,
                              "total_tokens": prompt + output},
                })

            serving = asyncio.create_task(http_server.serve(sockets=[listener]))
            try:
                with anyio.fail_after(5):
                    while not http_server.started:
                        if serving.done():
                            await serving
                            self.fail("Тестовый MCP-сервер завершился до запуска")
                        await asyncio.sleep(0.01)
                with patch.dict("os.environ", {token_env: token}), \
                        patch.object(github_api, "get", new_callable=AsyncMock,
                                     return_value=(fixture, False)) as github, \
                        patch.object(Completions, "create", side_effect=respond_to_model), \
                        patch("llm_agent.agent.OpenAI", side_effect=tracked_llm_client), \
                        patch("llm_agent.mcp_client.httpx2.AsyncClient", side_effect=tracked_mcp_client):
                    snapshot = await asyncio.wait_for(asyncio.to_thread(
                        service.send, conversation.id,
                        "Получи через MCP сведения о репозитории octocat/Hello-World: звёзды, ветку и ссылку.",
                    ), timeout=15)
                    github.assert_awaited_once_with("/repos/octocat/Hello-World")
            finally:
                http_server.should_exit = True
                await asyncio.wait_for(serving, timeout=5)

            turn = snapshot.turns[-1]
            self.assertEqual(turn.status, "completed", turn.error)
            self.assertEqual(turn.answer, expected_answer)
            self.assertEqual(len(turn.tool_calls), 1)
            record = turn.tool_calls[0]
            self.assertEqual((record.call_id, record.server_name, record.tool_name),
                             (call_id, "Мой GitHub MCP", "get_repository"))
            self.assertEqual(json.loads(record.arguments), {"owner": "octocat", "repo": "Hello-World"})
            self.assertEqual(json.loads(record.result), {"is_error": False, "data": expected_data})
            self.assertFalse(record.is_error)
            self.assertGreaterEqual(record.elapsed_seconds, 0)
            restored_service = ConversationService(Path(directory), "replacement-test-key")
            restored = restored_service.get(conversation.id)
            self.assertEqual(restored, snapshot)
            self.assertEqual(restored_service.mcp_servers.get(connection.id), connection)
            history = HistoryManager(Path(directory) / "unused-context.json")
            # Рабочий контекст читается из уже сохранённого и восстановленного диалога.
            history._restore_data(restored.working_context)
            self.assertEqual(history.get_usage(), DialogueUsage(220, 35, 255))
            saved = service.store.path(conversation.id).read_text(encoding="utf-8")
            self.assertNotIn(token, saved)
            self.assertNotIn("test-only-llm-token", saved)

        methods = [message["method"] for message in rpc_messages]
        self.assertEqual(methods.count("initialize"), 2)
        self.assertEqual(methods.count("notifications/initialized"), 2)
        self.assertEqual(methods.count("tools/call"), 1)
        self.assertGreaterEqual(methods.count("tools/list"), 1)
        self.assertTrue(all(dict(request["headers"])[b"authorization"] == f"Bearer {token}".encode()
                            for request in http_requests))
        self.assertEqual(len(mcp_clients), 2)
        self.assertTrue(all(client.is_closed for client in mcp_clients))
        self.assertEqual(len(llm_clients), 1)
        self.assertTrue(llm_clients[0].is_closed())
        self.assertFalse(active_requests)
        self.assertFalse(http_server.server_state.connections)
        self.assertTrue(serving.done())


if __name__ == "__main__":
    unittest.main()
