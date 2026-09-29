"""Один запрос → три настоящих MCP HTTP-сервера → журнал и восстановление.

Подменены только LLM и внешние GitHub/NVD/Telegram API. Выбор следующего
инструмента делает детерминированная модель на основании предыдущих результатов.
"""

import asyncio
import importlib
import json
import socket
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import AsyncMock, patch

import anyio
import httpx2
import requests
import uvicorn
from openai.resources.chat.completions import Completions
from openai.types.chat import ChatCompletion

from MCPServerExample import github_api, http_server as github_http
from TelegramMCPServerExample import http_server as telegram_http
from VulnSearchMCPServerExample import http_server as nvd_http
from llm_agent.mcp_config import MCPServer
from llm_agent.service import ConversationService


class MCPOrchestrationIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_seven_dependent_calls_cross_three_servers_and_survive_restart(self):
        await self._run_workflow()

    async def test_nvd_error_is_not_no_results_and_does_not_trigger_delivery(self):
        await self._run_workflow(nvd_fails=True)

    async def test_partial_telegram_delivery_is_reported_as_error_without_retry(self):
        await self._run_workflow(partial_delivery=True)

    async def _run_workflow(self, *, nvd_fails=False, partial_delivery=False):
        with patch("mcp.server.mcpserver.server.configure_logging"):
            github_server = importlib.import_module("MCPServerExample.server")
            nvd_server = importlib.import_module("VulnSearchMCPServerExample.server")
            telegram_server = importlib.import_module("TelegramMCPServerExample.server")
        nvd_api = importlib.import_module("VulnSearchMCPServerExample.nvd_api")
        telegram_api = importlib.import_module("TelegramMCPServerExample.telegram_api")

        projects = [
            ("pytorch", "pytorch", "CVE-2025-10001"),
            ("microsoft", "vscode", "CVE-2025-10002"),
            ("example", "zero-cves", None),
        ]
        repositories = {
            f"/repos/{owner}/{name}": {
                "name": name, "full_name": f"{owner}/{name}",
                "description": f"Описание {name}. " + ("x" * 1900 if partial_delivery else "Python/IDE project"),
                "html_url": f"https://github.com/{owner}/{name}",
                "language": "Python" if name == "pytorch" else "TypeScript",
                "stargazers_count": 1234,
            }
            for owner, name, _ in projects
        }
        nvd_fixtures = {
            name: [{
                "id": cve_id,
                "descriptions": [{"lang": "en", "value": f"A test issue affecting {name}."}],
                "published": "2025-01-01T00:00:00.000",
                "lastModified": "2025-01-02T00:00:00.000",
                "vulnStatus": "Analyzed",
                "metrics": {"cvssMetricV31": [{
                    "type": "Primary",
                    "cvssData": {"version": "3.1", "baseScore": 8.8, "baseSeverity": "HIGH"},
                }]},
            }] if cve_id else []
            for _, name, cve_id in projects
        }
        server_specs = [
            ("github", "GitHub integration", github_server.mcp),
            ("nvd", "NVD integration", nvd_server.mcp),
            ("telegram", "Telegram integration", telegram_server.mcp),
        ]
        tokens = {name: f"integration-{name}-bearer" for name, _, _ in server_specs}
        environment = {
            **{f"ORCHESTRATION_{name.upper()}_TOKEN": token for name, token in tokens.items()},
            "NIST_API_TOKEN": "integration-nist-token",
            "API_TOKEN": "12345:integration-telegram-token",
            "CHAT_ID": "987654321",
        }
        agent_token = "integration-llm-token"
        expected_answer = (
            "NVD недоступен: проверка vscode не завершена. Отчёт в Telegram не отправлен."
            if nvd_fails else
            "Telegram доставил только 1 из 2 частей отчёта. Повторная отправка не выполнялась."
            if partial_delivery else
            "Проверены три проекта; полный отчёт отправлен в Telegram."
        )
        rpc_messages = {name: [] for name, _, _ in server_specs}
        http_headers = {name: [] for name, _, _ in server_specs}
        routed_calls = []
        telegram_requests = []
        llm_requests = []
        results = {}
        report = ""
        real_http_client = httpx2.AsyncClient

        def observe_server(name, app):
            async def observe_http(scope, receive, send):
                if scope["type"] != "http":
                    return await app(scope, receive, send)
                http_headers[name].append(dict(scope["headers"]))
                body = bytearray()

                async def observe_receive():
                    message = await receive()
                    if message["type"] == "http.request":
                        body.extend(message.get("body", b""))
                        if not message.get("more_body", False) and body:
                            rpc = json.loads(body)
                            rpc_messages[name].append(rpc)
                            if rpc.get("method") == "tools/call":
                                routed_calls.append((name, rpc["params"]))
                            body.clear()
                    return message

                await app(scope, observe_receive, send)

            return observe_http

        async def github_response(path, **kwargs):
            self.assertFalse(kwargs)
            return repositories[path], False

        def nvd_response(**kwargs):
            self.assertEqual(kwargs["key"], environment["NIST_API_TOKEN"])
            self.assertTrue(kwargs["asDict"])
            self.assertEqual(kwargs["limit"], 5)
            project_name = kwargs["keywordSearch"]
            if nvd_fails and project_name == "vscode":
                response = requests.Response()
                response.status_code = 503
                raise requests.HTTPError("NVD service unavailable", response=response)
            return nvd_fixtures[project_name]

        def telegram_response(request):
            self.assertEqual(request.method, "POST")
            self.assertEqual(request.url.host, "api.telegram.org")
            self.assertEqual(request.url.path, f"/bot{environment['API_TOKEN']}/sendMessage")
            data = json.loads(request.content)
            self.assertEqual(str(data["chat_id"]), environment["CHAT_ID"])
            telegram_requests.append(data)
            if partial_delivery and len(telegram_requests) == 2:
                return httpx2.Response(429, json={
                    "ok": False, "error_code": 429, "description": "Too Many Requests",
                    "parameters": {"retry_after": 3},
                })
            return httpx2.Response(200, json={
                "ok": True, "result": {"message_id": 700 + len(telegram_requests)},
            })

        def telegram_client():
            return real_http_client(transport=httpx2.MockTransport(telegram_response), trust_env=False)

        def local_mcp_client(**kwargs):
            return real_http_client(trust_env=False, **kwargs)

        def tool_message(request, server_name, tool_name, call_id, arguments):
            functions = [item["function"] for item in request["tools"]
                         if item["function"]["description"].startswith(f"{server_name}: {tool_name}.")]
            self.assertEqual(len(functions), 1)
            return {"role": "assistant", "content": None, "tool_calls": [{
                "id": call_id, "type": "function", "function": {
                    "name": functions[0]["name"],
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                },
            }]}

        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            service = ConversationService(root / "conversations", agent_token)
            conversation = service.create()
            connections = []
            http_servers = []
            listeners = []
            for name, display_name, mcp in server_specs:
                listener = stack.enter_context(socket.socket())
                listener.bind(("127.0.0.1", 0))
                listener.setblocking(False)
                listeners.append(listener)
                connection = MCPServer(
                    name, display_name, f"http://127.0.0.1:{listener.getsockname()[1]}/mcp",
                    f"ORCHESTRATION_{name.upper()}_TOKEN", enabled=True,
                )
                connections.append(connection)
                service.mcp_servers.save(connection)
                transport = {"github": github_http, "nvd": nvd_http, "telegram": telegram_http}[name]
                app = transport.create_app(mcp, transport.Settings(access_token=tokens[name]))
                http_servers.append(uvicorn.Server(uvicorn.Config(
                    observe_server(name, app), log_level="error", log_config=None, lifespan="on",
                )))

            def respond_to_model(**request):
                nonlocal report
                llm_requests.append(request)
                self.assertEqual(request["tool_choice"], "auto")
                round_number = len(llm_requests)
                if round_number > 1:
                    previous = request["messages"][-1]
                    self.assertEqual(previous["role"], "tool")
                    result = json.loads(previous["content"])
                    results[previous["tool_call_id"]] = result
                    pending = service.get(conversation.id).turns[-1]
                    self.assertEqual(pending.status, "running")
                    self.assertEqual(pending.tool_calls[-1].call_id, previous["tool_call_id"])
                    self.assertEqual(json.loads(pending.tool_calls[-1].result), result)

                if round_number <= 3:
                    owner, name, _ = projects[round_number - 1]
                    message = tool_message(request, "GitHub integration", "get_repository",
                                           f"github-{name}", {"owner": owner, "repo": name})
                elif round_number == 6 and nvd_fails:
                    self.assertTrue(result["is_error"])
                    self.assertNotIn("result_ref", result)
                    self.assertNotIn("no_results", json.dumps(result))
                    self.assertIn("NVD", json.dumps(result, ensure_ascii=False))
                    message = {"role": "assistant", "content": expected_answer}
                elif round_number <= 6:
                    _, name, _ = projects[round_number - 4]
                    repository_result = results[f"github-{name}"]
                    self.assertFalse(repository_result["is_error"])
                    self.assertEqual(repository_result["data"]["name"], name)
                    source = repository_result["result_ref"]
                    message = tool_message(request, "NVD integration", "search_vulnerabilities", f"nvd-{name}", {
                        "project_name": {**source, "pointer": source["pointer"] + "/name"}, "limit": 5,
                    })
                elif round_number == 7:
                    sections = []
                    for _, name, cve_id in projects:
                        repository = results[f"github-{name}"]["data"]
                        nvd_result = results[f"nvd-{name}"]
                        self.assertFalse(nvd_result["is_error"])
                        search = nvd_result["data"]
                        self.assertEqual(search["project_name"], repository["name"])
                        self.assertEqual(search["query"], repository["name"])
                        self.assertEqual(search["status"], "ok" if cve_id else "no_results")
                        self.assertEqual(search["returned_count"], 1 if cve_id else 0)
                        details = [f"{item['id']} — {item['description']} — {item['url']}"
                                   for item in search["vulnerabilities"]]
                        if not details:
                            details = ["По запросу NVD совпадений не найдено; это не доказывает отсутствие уязвимостей."]
                        sections.append("\n".join([
                            repository["full_name"], repository["html_url"], repository["description"], *details,
                        ]))
                    # Текст составляется из фактических ответов MCP, а не из fixtures.
                    report = "Отчёт по проектам GitHub и результатам поиска NVD\n\n" + "\n\n".join(sections)
                    message = tool_message(request, "Telegram integration", "send_message",
                                           "telegram-report", {"text": report})
                else:
                    self.assertEqual(round_number, 8)
                    self.assertEqual(previous["tool_call_id"], "telegram-report")
                    self.assertEqual(result["is_error"], partial_delivery)
                    if partial_delivery:
                        self.assertNotIn("result_ref", result)
                        details = result.get("data") or json.loads(result["content"][0]["text"])
                        self.assertEqual(details["status"], "partial")
                        self.assertEqual(details["sent_parts"], 1)
                        self.assertEqual(details["total_parts"], 2)
                        self.assertEqual(details["message_ids"], [701])
                        self.assertEqual(details["failed_part"], 2)
                        self.assertEqual(details["retry_after"], 3)
                    else:
                        self.assertEqual(result["data"]["status"], "sent")
                        self.assertEqual(result["data"]["message_ids"], [701])
                        self.assertEqual(result["data"]["sent_parts"], result["data"]["total_parts"])
                    message = {"role": "assistant", "content": expected_answer}

                return ChatCompletion.model_validate({
                    "id": f"orchestration-{round_number}", "created": 0,
                    "model": "deepseek-chat", "object": "chat.completion",
                    "choices": [{"index": 0, "message": message,
                                 "finish_reason": "tool_calls" if "tool_calls" in message else "stop"}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
                })

            serving = [asyncio.create_task(server.serve(sockets=[listener]))
                       for server, listener in zip(http_servers, listeners)]
            try:
                with anyio.fail_after(5):
                    while not all(server.started for server in http_servers):
                        for task in serving:
                            if task.done():
                                await task
                                self.fail("MCP-сервер завершился до запуска")
                        await asyncio.sleep(0.01)
                with patch.dict("os.environ", environment), \
                        patch.object(github_api, "get", new_callable=AsyncMock, side_effect=github_response) as github, \
                        patch.object(nvd_api.nvdlib, "searchCVE", side_effect=nvd_response) as nvd, \
                        patch.object(telegram_api, "create_client", side_effect=telegram_client), \
                        patch.object(telegram_api, "PART_DELAY_SECONDS", 0), \
                        patch.object(Completions, "create", side_effect=respond_to_model), \
                        patch("llm_agent.mcp_client.httpx2.AsyncClient", side_effect=local_mcp_client):
                    links = ", ".join(f"https://github.com/{owner}/{name}" for owner, name, _ in projects)
                    snapshot = await asyncio.wait_for(asyncio.to_thread(
                        service.send, conversation.id,
                        f"Получи информацию о {links}, найди для каждого CVE в NVD и отправь общий отчёт мне в Telegram.",
                    ), timeout=30)
                    self.assertEqual(github.await_count, 3)
                    self.assertEqual(nvd.call_count, 2 if nvd_fails else 3)
            finally:
                for server in http_servers:
                    server.should_exit = True
                await asyncio.wait_for(asyncio.gather(*serving), timeout=5)

            turn = snapshot.turns[-1]
            self.assertEqual(turn.status, "completed", turn.error)
            self.assertEqual(turn.answer, expected_answer)
            self.assertEqual(len(llm_requests), 6 if nvd_fails else 8)
            expected_calls = [("github", "get_repository")] * 3 + [
                ("nvd", "search_vulnerabilities")
            ] * (2 if nvd_fails else 3)
            if not nvd_fails:
                expected_calls.append(("telegram", "send_message"))
            self.assertEqual([(server, call["name"]) for server, call in routed_calls], expected_calls)
            display_names = {name: display_name for name, display_name, _ in server_specs}
            self.assertEqual([(record.server_name, record.tool_name) for record in turn.tool_calls],
                             [(display_names[server], tool) for server, tool in expected_calls])
            self.assertEqual([record.is_error for record in turn.tool_calls],
                             [False] * (len(expected_calls) - 1) + [nvd_fails or partial_delivery])
            for index, (owner, name, _) in enumerate(projects):
                self.assertEqual(routed_calls[index][1]["arguments"], {"owner": owner, "repo": name})
            for index, (_, name, _) in enumerate(projects[:2 if nvd_fails else 3], start=3):
                self.assertEqual(routed_calls[index][1]["arguments"], {"project_name": name, "limit": 5})
                self.assertEqual(json.loads(turn.tool_calls[index].arguments), {
                    "project_name": {"$mcp_result": f"github-{name}", "pointer": "/data/name"}, "limit": 5,
                })
            if nvd_fails:
                self.assertFalse(telegram_requests)
            else:
                self.assertEqual(routed_calls[-1][1]["arguments"], {"text": report})
                self.assertEqual(len(telegram_requests), 2 if partial_delivery else 1)
                self.assertEqual("".join(request["text"] for request in telegram_requests), report)
                for owner, name, cve_id in projects:
                    self.assertIn(f"{owner}/{name}", report)
                    if cve_id:
                        self.assertIn(cve_id, report)
                self.assertIn("совпадений не найдено", report)

            restored_service = ConversationService(root / "conversations", "replacement-llm-token")
            self.assertEqual(restored_service.get(conversation.id), snapshot)
            for connection in connections:
                self.assertEqual(restored_service.mcp_servers.get(connection.id), connection)
            saved = service.store.path(conversation.id).read_text(encoding="utf-8")
            for token in (*tokens.values(), environment["NIST_API_TOKEN"], environment["API_TOKEN"], agent_token):
                self.assertNotIn(token, saved)

        for name, _, _ in server_specs:
            self.assertIn("tools/list", [message.get("method") for message in rpc_messages[name]])
            self.assertTrue(http_headers[name])
            self.assertTrue(all(headers.get(b"authorization") == f"Bearer {tokens[name]}".encode()
                                for headers in http_headers[name]))
        self.assertTrue(all(task.done() for task in serving))
        self.assertTrue(all(not server.server_state.connections for server in http_servers))


if __name__ == "__main__":
    unittest.main()
