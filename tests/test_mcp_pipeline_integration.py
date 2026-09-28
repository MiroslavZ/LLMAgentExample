"""Один запрос чата → поиск → LLM-сводка → файл через настоящий MCP HTTP."""

import asyncio
import hashlib
import importlib
import json
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import anyio
import httpx
import httpx2
import uvicorn
from openai import APIConnectionError
from openai.resources.chat.completions import AsyncCompletions, Completions
from openai.types.chat import ChatCompletion

from MCPServerExample import github_api, github_results, pipeline_tools
from MCPServerExample.http_server import Settings, create_app
from llm_agent.mcp_config import MCPServer
from llm_agent.mcp_tools import MAX_RESULT_CHARS
from llm_agent.service import ConversationService


class MCPCompositionIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_pipeline_preserves_full_results_and_downloadable_file_after_restart(self):
        await self._run_pipeline()

    async def test_function_name_reference_saves_once_without_repeating_summarization(self):
        await self._run_pipeline(save_reference="tool_name")

    async def test_unknown_reference_is_corrected_from_available_results_without_repeating_summarization(self):
        await self._run_pipeline(save_reference="unknown")

    async def test_summary_error_reaches_model_without_saving_a_file(self):
        await self._run_pipeline(summary_fails=True)

    async def _run_pipeline(self, *, summary_fails: bool = False, save_reference: str = "result_ref"):
        with patch.dict("sys.modules", {"github_api": github_api, "github_results": github_results}), \
                patch("mcp.server.mcpserver.server.configure_logging"):
            mcp = importlib.import_module("MCPServerExample.server").mcp

        query = "language:python stars:>100"
        # Последний фрагмент не помещается в preview для модели. Ссылка обязана
        # передать его следующему инструменту вместе со всем исходным объектом.
        description = 'Python: "данные", перенос\n🐍 ' * 1100 + "КОНЕЦ ПОЛНЫХ ДАННЫХ"
        repository = {
            "full_name": "example/python-agent",
            "description": description,
            "html_url": "https://github.com/example/python-agent",
            "stargazers_count": 1742,
        }
        fixture = {"items": [repository], "total_count": 1, "incomplete_results": False}
        expected_data = {
            "repositories": [repository], "total_count": 1, "incomplete_results": False,
            "page": 1, "next_page": None,
        }
        self.assertGreater(len(json.dumps(expected_data, ensure_ascii=False)), MAX_RESULT_CHARS)
        summary = 'Репозиторий example/python-agent — 1742 звезды 🐍.\nВывод: "подходит для изучения".\n'
        filename = "python-summary.txt"
        expected_answer = (
            "Суммаризация не удалась; файл не создан." if summary_fails
            else "Поиск и суммаризация завершены. Сводка сохранена в python-summary.txt."
        )
        token_env = "MCP_PIPELINE_INTEGRATION_TOKEN"
        mcp_token = "pipeline-test-mcp-token"
        agent_token = "pipeline-test-agent-token"
        summary_token = "pipeline-test-summary-token"
        app = create_app(mcp, Settings(access_token=mcp_token))
        rpc_messages = []
        http_headers = []

        async def observe_http(scope, receive, send):
            if scope["type"] != "http":
                return await app(scope, receive, send)
            http_headers.append(dict(scope["headers"]))
            body = bytearray()

            async def observe_receive():
                message = await receive()
                if message["type"] == "http.request":
                    body.extend(message.get("body", b""))
                    if not message.get("more_body", False) and body:
                        rpc_messages.append(json.loads(body))
                        body.clear()
                return message

            await app(scope, observe_receive, send)

        http_server = uvicorn.Server(uvicorn.Config(
            observe_http, log_level="error", log_config=None, lifespan="on",
        ))
        real_http_client = httpx2.AsyncClient

        def local_mcp_client(**kwargs):
            return real_http_client(trust_env=False, **kwargs)

        with tempfile.TemporaryDirectory() as directory, socket.socket() as listener:
            root = Path(directory)
            output_dir = root / "mcp-output"
            listener.bind(("127.0.0.1", 0))
            listener.setblocking(False)
            service = ConversationService(root / "conversations", agent_token)
            conversation = service.create()
            service.mcp_servers.save(MCPServer(
                "pipeline", "MCP композиция", f"http://127.0.0.1:{listener.getsockname()[1]}/mcp",
                token_env, enabled=True,
            ))
            llm_requests = []

            def tool_function(request, name):
                return next(item["function"] for item in request["tools"]
                            if f": {name}." in item["function"]["description"])

            def tool_message(request, name, call_id, arguments):
                function = tool_function(request, name)
                return {"role": "assistant", "content": None, "tool_calls": [{
                    "id": call_id, "type": "function", "function": {
                        "name": function["name"],
                        "arguments": json.dumps(arguments, ensure_ascii=False),
                    },
                }]}

            def respond_to_agent(**request):
                llm_requests.append(request)
                self.assertEqual(request["tool_choice"], "auto")
                round_number = len(llm_requests)
                if round_number == 1:
                    message = tool_message(request, "search_repositories", "search-call", {
                        "query": query, "per_page": 1,
                    })
                else:
                    tool_result = request["messages"][-1]
                    self.assertEqual(tool_result["role"], "tool")
                    result = json.loads(tool_result["content"])
                    if round_number == 2:
                        self.assertEqual(tool_result["tool_call_id"], "search-call")
                        self.assertFalse(result["is_error"])
                        self.assertTrue(result["truncated"])
                        self.assertNotIn("КОНЕЦ ПОЛНЫХ ДАННЫХ", result["preview"])
                        self.assertEqual(result["result_ref"], {
                            "$mcp_result": "search-call", "pointer": "/data",
                        })
                        message = tool_message(request, "summarize", "summary-call", {
                            "data": result["result_ref"],
                            "instructions": "Сделай краткую сводку на русском языке.",
                        })
                    elif round_number == 3:
                        self.assertEqual(tool_result["tool_call_id"], "summary-call")
                        if summary_fails:
                            self.assertTrue(result["is_error"])
                            self.assertTrue(result["content"])
                            self.assertNotIn("result_ref", result)
                            message = {"role": "assistant", "content": expected_answer}
                        else:
                            self.assertEqual(result, {
                                "is_error": False, "data": {"summary": summary},
                                "result_ref": {"$mcp_result": "summary-call", "pointer": "/data"},
                            })
                            content_reference = {
                                **result["result_ref"], "pointer": result["result_ref"]["pointer"] + "/summary",
                            }
                            if save_reference == "tool_name":
                                content_reference["$mcp_result"] = tool_function(request, "summarize")["name"]
                            elif save_reference == "unknown":
                                content_reference["$mcp_result"] = "unknown-summary-result"
                            message = tool_message(request, "save_to_file", "save-call", {
                                "content": content_reference,
                                "filename": filename,
                            })
                    elif save_reference == "unknown" and round_number == 4:
                        self.assertEqual(tool_result["tool_call_id"], "save-call")
                        self.assertTrue(result["is_error"])
                        self.assertNotIn("result_ref", result)
                        available = result["available_results"]
                        self.assertEqual(available, [
                            {"tool_name": tool_function(request, "search_repositories")["name"],
                             "result_ref": {"$mcp_result": "search-call", "pointer": "/data"}},
                            {"tool_name": tool_function(request, "summarize")["name"],
                             "result_ref": {"$mcp_result": "summary-call", "pointer": "/data"}},
                        ])
                        # Модель исправляет только ссылку сохранения по подсказке
                        # ошибки, не повторяя ни поиск, ни вызов LLM-суммаризатора.
                        source = next(item["result_ref"] for item in available
                                      if item["tool_name"] == tool_function(request, "summarize")["name"])
                        message = tool_message(request, "save_to_file", "save-retry-call", {
                            "content": {**source, "pointer": source["pointer"] + "/summary"},
                            "filename": filename,
                        })
                    else:
                        self.assertEqual(round_number, 5 if save_reference == "unknown" else 4)
                        final_call_id = "save-retry-call" if save_reference == "unknown" else "save-call"
                        self.assertEqual(tool_result["tool_call_id"], final_call_id)
                        self.assertFalse(result["is_error"])
                        self.assertEqual(result["result_ref"], {"$mcp_result": final_call_id, "pointer": "/data"})
                        self.assertEqual(result["data"]["filename"], filename)
                        self.assertEqual(result["data"]["size_bytes"], len(summary.encode("utf-8")))
                        # Запись результата происходит до завершающего ответа LLM.
                        pending = service.get(conversation.id).turns[-1]
                        self.assertEqual(pending.status, "running")
                        self.assertEqual(pending.tool_calls[-1].attachments[0].text, summary)
                        message = {"role": "assistant", "content": expected_answer}
                return ChatCompletion.model_validate({
                    "id": f"pipeline-completion-{round_number}", "created": 0,
                    "model": "deepseek-chat", "object": "chat.completion",
                    "choices": [{"index": 0, "message": message,
                                 "finish_reason": "tool_calls" if "tool_calls" in message else "stop"}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
                })

            async def respond_to_summarizer(**request):
                self.assertEqual(request["model"], "pipeline-summary-model")
                self.assertEqual(request["messages"][-2], {
                    "role": "user", "content": "Сделай краткую сводку на русском языке.",
                })
                self.assertEqual(json.loads(request["messages"][-1]["content"]), expected_data)
                if summary_fails:
                    raise APIConnectionError(request=httpx.Request("POST", "https://summary.invalid/v1"))
                return ChatCompletion.model_validate({
                    "id": "summary-completion", "created": 0,
                    "model": "pipeline-summary-model", "object": "chat.completion",
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant", "content": summary}}],
                })

            serving = asyncio.create_task(http_server.serve(sockets=[listener]))
            try:
                with anyio.fail_after(5):
                    while not http_server.started:
                        if serving.done():
                            await serving
                            self.fail("Тестовый MCP-сервер завершился до запуска")
                        await asyncio.sleep(0.01)
                with patch.dict("os.environ", {
                    token_env: mcp_token, "LLM_API_KEY": summary_token,
                    "LLM_BASE_URL": "https://summary.invalid/v1", "LLM_MODEL": "pipeline-summary-model",
                    "MCP_OUTPUT_DIR": str(output_dir),
                }), patch.object(github_api, "get", new_callable=AsyncMock,
                                 return_value=(fixture, False)) as github, \
                        patch.object(pipeline_tools, "load_dotenv"), \
                        patch.object(Completions, "create", side_effect=respond_to_agent), \
                        patch.object(AsyncCompletions, "create", new_callable=AsyncMock,
                                     side_effect=respond_to_summarizer) as summary_api, \
                        patch("llm_agent.mcp_client.httpx2.AsyncClient", side_effect=local_mcp_client):
                    snapshot = await asyncio.wait_for(asyncio.to_thread(
                        service.send, conversation.id,
                        "Найди Python-репозитории, суммаризуй результаты и сохрани сводку в python-summary.txt.",
                    ), timeout=20)
                    github.assert_awaited_once_with("/search/repositories", q=query, page=1, per_page=1)
                    summary_api.assert_awaited_once()
            finally:
                http_server.should_exit = True
                await asyncio.wait_for(serving, timeout=5)

            turn = snapshot.turns[-1]
            self.assertEqual(turn.status, "completed", turn.error)
            self.assertEqual(turn.answer, expected_answer)
            expected_rounds = 3 if summary_fails else 5 if save_reference == "unknown" else 4
            self.assertEqual(len(llm_requests), expected_rounds)
            expected_tools = ["search_repositories", "summarize"]
            if not summary_fails:
                expected_tools.append("save_to_file")
            expected_records = [*expected_tools]
            if save_reference == "unknown":
                expected_records.append("save_to_file")
            self.assertEqual([record.tool_name for record in turn.tool_calls], expected_records)
            calls = [message["params"] for message in rpc_messages if message.get("method") == "tools/call"]
            self.assertEqual([call["name"] for call in calls], expected_tools)
            self.assertEqual(calls[1]["arguments"]["data"], expected_data)
            self.assertEqual(json.loads(turn.tool_calls[1].arguments)["data"], {
                "$mcp_result": "search-call", "pointer": "/data",
            })
            if summary_fails:
                self.assertTrue(turn.tool_calls[-1].is_error)
                self.assertFalse(list(output_dir.glob("**/*.txt")))
            else:
                self.assertEqual([record.is_error for record in turn.tool_calls],
                                 [False, False, True, False] if save_reference == "unknown" else [False] * 3)
                self.assertEqual(calls[2]["arguments"], {"content": summary, "filename": filename})
                summary_source = (
                    tool_function(llm_requests[0], "summarize")["name"]
                    if save_reference == "tool_name" else "summary-call"
                )
                self.assertEqual(json.loads(turn.tool_calls[-1].arguments)["content"], {
                    "$mcp_result": summary_source, "pointer": "/data/summary",
                })
                if save_reference == "unknown":
                    self.assertEqual(json.loads(turn.tool_calls[2].arguments)["content"]["$mcp_result"],
                                     "unknown-summary-result")
                    self.assertEqual(turn.tool_calls[2].attachments, ())
                saved_files = list(output_dir.glob("**/*.txt"))
                self.assertEqual(len(saved_files), 1)
                self.assertEqual(saved_files[0].read_bytes(), summary.encode("utf-8"))
                metadata = json.loads(turn.tool_calls[-1].result)["data"]
                self.assertEqual(metadata["sha256"], hashlib.sha256(summary.encode("utf-8")).hexdigest())
                self.assertEqual(len(turn.tool_calls[-1].attachments), 1)
                attachment = turn.tool_calls[-1].attachments[0]
                self.assertEqual((attachment.filename, attachment.text, attachment.uri),
                                 (filename, summary, metadata["uri"]))

            restored = ConversationService(root / "conversations", "replacement-test-key").get(conversation.id)
            self.assertEqual(restored, snapshot)
            saved_history = service.store.path(conversation.id).read_text(encoding="utf-8")
            for token in (mcp_token, agent_token, summary_token):
                self.assertNotIn(token, saved_history)

        self.assertTrue(all(headers[b"authorization"] == f"Bearer {mcp_token}".encode()
                            for headers in http_headers))
        self.assertFalse(http_server.server_state.connections)
        self.assertTrue(serving.done())


if __name__ == "__main__":
    unittest.main()
