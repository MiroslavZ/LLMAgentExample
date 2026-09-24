"""Выбор инструмента моделью, результаты, ограничения и сохранение без внешних API."""

import json
import io
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock, patch

from mcp import Tool, types
from openai.types.chat import ChatCompletion
from rich.console import Console

from llm_agent.agent import Agent
from llm_agent.cli import main as cli_main, run as cli_run
from llm_agent.history import DialogueUsage, HistoryManager, TokenUsage
from llm_agent.invariants import Invariant, InvariantSet
from llm_agent.mcp_client import MCPConnectionError, MCPDiscovery
from llm_agent.mcp_config import MCPServer
from llm_agent.mcp_tools import MAX_RESULT_CHARS, MAX_TOOL_CALLS, MAX_TOOL_ROUNDS, MCPToolCatalog, MCPToolError
from llm_agent.models import RequestOptions
from llm_agent.service import ConversationService
from llm_agent.task_state import TaskStage, TaskState
from tests.helpers import completion


SERVER = MCPServer("github", "GitHub", "http://127.0.0.1:8000/mcp", enabled=True)
SCHEMA = {"type": "object", "properties": {
    "owner": {"type": "string"}, "repo": {"type": "string"},
}, "required": ["owner", "repo"], "additionalProperties": False}
ARGS = '{"owner":"octocat","repo":"Hello-World"}'
TOOL = Tool(name="get_repository", description="Получить репозиторий GitHub", input_schema=SCHEMA)


def answer(text="Репозиторий octocat/Hello-World: Python"):
    result = completion()
    result.choices[0].message.content = text
    return result


def tool_response(name, *, arguments=ARGS, call_id="call-1", count=1, finish_reason="tool_calls"):
    result = completion().model_dump()
    result["choices"][0].update(finish_reason=finish_reason, message={
        "role": "assistant", "content": None, "tool_calls": [{
            "id": call_id if count == 1 else f"{call_id}-{index}", "type": "function",
            "function": {"name": name, "arguments": arguments},
        } for index in range(count)],
    })
    return ChatCompletion.model_validate(result)


def verdict(status="pass"):
    return answer(json.dumps({"checks": [{"id": "rule", "status": status, "reason": "Проверено"}]}))


class MCPAgentTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.path = self.directory / "history.json"
        self.create = self.enterContext(patch("llm_agent.agent.OpenAI")).return_value.chat.completions.create
        self.discover = self.enterContext(patch("llm_agent.mcp_tools.get_tools", new_callable=AsyncMock,
            return_value=MCPDiscovery("github", "1.0", "test", (TOOL,))))
        self.call = self.enterContext(patch("llm_agent.mcp_tools.call_tool", new_callable=AsyncMock,
            return_value=types.CallToolResult(content=[], structured_content={
                "full_name": "octocat/Hello-World", "language": "Python",
            })))

    def agent(self, **options):
        options.setdefault("mcp_servers", (SERVER,))
        agent = Agent("test-key", history_path=self.path, **options)
        self.addCleanup(agent.close)
        return agent

    def choose(self, **request):
        return tool_response(request["tools"][0]["function"]["name"])

    def sequence(self, *steps):
        steps = iter(steps)

        def generate(**request):
            step = next(steps)
            return step(**request) if callable(step) else step
        self.create.side_effect = generate

    def test_schema_choice_tool_result_final_and_usage(self):
        records = []
        self.sequence(self.choose, answer())
        agent = self.agent(on_tool_call=records.append)
        result = agent.request("Расскажи о репозитории octocat/Hello-World")
        first, final = [call.kwargs for call in self.create.call_args_list]
        self.assertEqual(first["tools"][0]["function"]["parameters"], SCHEMA)
        self.assertIn(TOOL.description, first["tools"][0]["function"]["description"])
        self.assertEqual(first["tool_choice"], "auto")
        self.assertEqual(first["messages"][-1]["role"], "user")
        self.assertEqual(final["messages"][-2]["tool_calls"][0]["id"], "call-1")
        self.assertEqual(final["messages"][-1]["tool_call_id"], "call-1")
        payload = json.loads(final["messages"][-1]["content"])
        self.assertEqual(payload["data"]["language"], "Python")
        self.call.assert_awaited_once_with(SERVER, "get_repository", json.loads(ARGS))
        self.assertEqual(result.tool_calls, tuple(records))
        self.assertEqual(result.dialogue_usage, DialogueUsage(200, 60, 260))
        self.assertEqual([item["role"] for item in agent.history.get_messages()], ["user", "assistant"])
        self.assertEqual(agent.history.get_messages()[-1]["content"], result.content)

    def test_disabled_connection_does_not_discover_or_send_tools(self):
        self.create.return_value = answer("Привет")
        self.agent(mcp_servers=(replace(SERVER, enabled=False),)).request("Привет")
        self.discover.assert_not_awaited()
        self.call.assert_not_awaited()
        self.assertNotIn("tools", self.create.call_args.kwargs)

    def test_empty_catalog_allows_plain_answer(self):
        self.discover.return_value = MCPDiscovery("empty", "1", "test", ())
        self.create.return_value = answer("Нет инструментов")
        self.agent().request("Привет")
        self.assertNotIn("tools", self.create.call_args.kwargs)

    def test_invalid_arguments_are_returned_to_model_without_call(self):
        for arguments in ("broken", "[]", "{}", '{"owner":1,"repo":"x"}',
                          '{"owner":"x","repo":"x","other":1}', '{"owner":NaN,"repo":"x"}'):
            with self.subTest(arguments=arguments):
                self.sequence(lambda **r: tool_response(r["tools"][0]["function"]["name"], arguments=arguments), answer())
                result = self.agent().request("Запрос")
                self.assertTrue(result.tool_calls[0].is_error)
                self.assertTrue(json.loads(self.create.call_args.kwargs["messages"][-1]["content"])["is_error"])
        self.call.assert_not_awaited()

    def test_unknown_tool_is_not_dispatched(self):
        self.sequence(tool_response("invented"), answer("Инструмент недоступен"))
        result = self.agent().request("Запрос")
        self.call.assert_not_awaited()
        self.assertTrue(result.tool_calls[0].is_error)

    def test_model_can_correct_arguments_in_next_round(self):
        self.sequence(lambda **r: tool_response(r["tools"][0]["function"]["name"], arguments="{}"),
                      lambda **r: tool_response(r["tools"][0]["function"]["name"], call_id="corrected"), answer())
        result = self.agent().request("Запрос")
        self.assertEqual([call.is_error for call in result.tool_calls], [True, False])
        self.call.assert_awaited_once()
        self.assertEqual(result.dialogue_usage.total_tokens, 390)

    def test_multiple_calls_and_rounds_preserve_each_result_id(self):
        self.sequence(lambda **r: tool_response(r["tools"][0]["function"]["name"], count=2),
                      lambda **r: tool_response(r["tools"][0]["function"]["name"], call_id="next"), answer())
        result = self.agent().request("Запрос")
        tool_messages = [m for m in self.create.call_args.kwargs["messages"] if m["role"] == "tool"]
        self.assertEqual([m["tool_call_id"] for m in tool_messages], ["call-1-0", "call-1-1", "next"])
        self.assertEqual(len(result.tool_calls), 3)

    def test_transport_and_tool_errors_are_data_for_final_answer(self):
        for failure in (MCPConnectionError("Сервер недоступен"),
                        types.CallToolResult(is_error=True, content=[types.TextContent(type="text", text="HTTP 404")])):
            with self.subTest(failure=failure):
                self.call.side_effect = failure if isinstance(failure, Exception) else None
                if not isinstance(failure, Exception):
                    self.call.return_value = failure
                self.sequence(self.choose, answer("Не удалось получить репозиторий"))
                result = self.agent().request("Запрос")
                self.assertTrue(result.tool_calls[0].is_error)
                self.assertIn("Не удалось", result.content)

    def test_large_result_is_marked_and_valid_json(self):
        self.call.return_value = types.CallToolResult(content=[], structured_content={"text": "я" * 100_000})
        self.sequence(self.choose, answer())
        result = self.agent().request("Запрос")
        payload = json.loads(result.tool_calls[0].result)
        self.assertTrue(payload["truncated"])
        self.assertEqual(len(payload["preview"]), MAX_RESULT_CHARS)
        self.assertFalse(payload["is_error"])

    def test_round_limit_requests_final_without_more_tools(self):
        def generate(**request):
            if request["tool_choice"] == "none":
                return answer("Получена только часть данных")
            return tool_response(request["tools"][0]["function"]["name"], call_id=f"call-{self.create.call_count}")
        self.create.side_effect = generate
        result = self.agent().request("Запрос")
        self.assertEqual(self.call.await_count, MAX_TOOL_ROUNDS)
        self.assertEqual(self.create.call_count, MAX_TOOL_ROUNDS + 1)
        self.assertEqual(len(result.tool_calls), MAX_TOOL_ROUNDS)

    def test_model_ignoring_round_limit_stops(self):
        self.create.side_effect = lambda **r: tool_response(r["tools"][0]["function"]["name"], call_id=str(self.create.call_count))
        with self.assertRaisesRegex(MCPToolError, "лимит"):
            self.agent().request("Запрос")
        self.assertEqual(self.call.await_count, MAX_TOOL_ROUNDS)
        self.assertEqual(HistoryManager(self.path).get_usage().total_tokens, 130 * (MAX_TOOL_ROUNDS + 1))

    def test_oversized_batch_is_rejected_before_execution(self):
        self.create.side_effect = lambda **r: tool_response(r["tools"][0]["function"]["name"], count=MAX_TOOL_CALLS + 1)
        with self.assertRaises(MCPToolError):
            self.agent().request("Запрос")
        self.call.assert_not_awaited()

    def test_incomplete_tool_call_is_never_executed(self):
        self.create.side_effect = lambda **r: tool_response(r["tools"][0]["function"]["name"], finish_reason="length")
        with self.assertRaises(MCPToolError):
            self.agent().request("Запрос")
        self.call.assert_not_awaited()

    def test_repeated_call_id_is_not_executed_twice(self):
        self.create.side_effect = self.choose
        with self.assertRaisesRegex(MCPToolError, "идентификаторы"):
            self.agent().request("Запрос")
        self.call.assert_awaited_once()

    def test_meta_prompt_generation_has_no_tools_but_execution_does(self):
        self.sequence(answer("Проверь GitHub"), self.choose, answer())
        meta, result = self.agent().request_with_meta_prompt("Запрос")
        self.assertEqual(meta.tool_calls, ())
        self.assertNotIn("tools", self.create.call_args_list[0].kwargs)
        self.assertEqual(len(result.tool_calls), 1)

    def test_task_final_json_repair_uses_results_without_repeating_tools(self):
        task = TaskState("Проверить GitHub", stage=TaskStage.EXECUTION, plan=("Прочитать репозиторий",))
        history = HistoryManager(self.path)
        history.add_exchange("План", "Утверждён", TokenUsage(), task_state=task)
        self.sequence(self.choose, answer('{"action":"complete_step","answer":broken}'),
                      answer('{"action":"complete_step","answer":"Python"}'))
        result = self.agent().request("Выполни текущий пункт")
        repair = self.create.call_args.kwargs
        self.assertEqual(repair["tool_choice"], "none")
        self.assertTrue(any(m["role"] == "tool" for m in repair["messages"]))
        self.call.assert_awaited_once()
        self.assertEqual(result.content, "Python")
        self.assertEqual(HistoryManager(self.path).task_state.results, ("Python",))

    def test_planning_never_executes_before_plan_approval(self):
        HistoryManager(self.path).start_task("Проверить GitHub")
        self.create.return_value = answer('{"action":"clarify","answer":"Какой репозиторий?"}')
        self.agent().request("Спланируй")
        self.discover.assert_not_awaited()
        self.assertNotIn("tools", self.create.call_args.kwargs)

    def test_invariants_reject_proposed_call_before_external_action(self):
        rules = InvariantSet((Invariant("rule", "Не читать закрытые данные"),))
        self.sequence(verdict(), self.choose, verdict("conflict"))
        result = self.agent(invariants=rules).request("Проверь репозиторий")
        self.assertTrue(result.refused)
        self.call.assert_not_awaited()
        self.assertEqual(result.tool_calls, ())
        self.assertNotIn("tools", self.create.call_args.kwargs)

    def test_service_preserves_call_when_final_llm_request_fails(self):
        service = ConversationService(self.directory / "conversations", token="test-key")
        service.mcp_servers.save(SERVER)
        conversation = service.create()
        def generate(**request):
            if request["messages"][-1]["role"] == "tool":
                raise RuntimeError("private SDK failure")
            return self.choose(**request)
        self.create.side_effect = generate
        result = service.send(conversation.id, "Проверь GitHub", options=RequestOptions())
        self.assertEqual(result.turns[-1].status, "partial")
        self.assertNotIn("private", result.turns[-1].error)
        restored = service.store.load(conversation.id)
        self.assertEqual(restored.turns[-1].tool_calls[0].tool_name, "get_repository")
        self.assertEqual(restored.turns[-1].tool_calls, result.turns[-1].tool_calls)

    def run_cli(self, *, entrypoint=cli_main):
        output = io.StringIO()
        arguments = [
            "agent", "--mcp-url", SERVER.url, "--user", "Получи сведения о репозитории octocat/Hello-World",
            "--history", str(self.path), "--memory-db", str(self.directory / "memory.sqlite3"),
            "--invariants-file", str(self.directory / "invariants.json"),
        ]
        with patch("sys.argv", arguments), patch("llm_agent.cli.load_env"), \
                patch.dict("os.environ", {"API_KEY": "test-key"}), \
                patch("llm_agent.cli.console", Console(file=output, width=160, color_system=None)):
            entrypoint()
        return output.getvalue()

    def test_cli_executes_tool_and_prints_result(self):
        self.sequence(self.choose, answer())
        output = self.run_cli()
        self.assertIn("get_repository", output)
        self.assertIn("результат получен", output)
        self.assertIn("octocat/Hello-World", output)
        self.call.assert_awaited_once()
        self.assertTrue(self.call.await_args.args[0].enabled)

    def test_cli_connection_failure_is_clear_without_traceback(self):
        self.discover.side_effect = MCPConnectionError("MCP-сервер недоступен")
        with self.assertRaisesRegex(SystemExit, "MCP-сервер недоступен"):
            self.run_cli(entrypoint=cli_run)
        self.create.assert_not_called()


class MCPCatalogTests(unittest.TestCase):
    def discover(self, tools, servers=(SERVER,)):
        with patch("llm_agent.mcp_tools.get_tools", new_callable=AsyncMock,
                   return_value=MCPDiscovery("test", "1", "test", tuple(tools))):
            return MCPToolCatalog.discover(servers)

    def test_names_are_distinct_for_dots_long_names_and_different_servers(self):
        tools = [TOOL.model_copy(update={"name": name}) for name in ("a.b", "a_b", "x" * 100)]
        catalog = self.discover(tools, (SERVER, replace(SERVER, id="second")))
        names = [tool["function"]["name"] for tool in catalog.functions]
        self.assertEqual(len(set(names)), 6)
        for name in names:
            self.assertRegex(name, r"^[A-Za-z0-9_-]{1,64}$")

    def test_invalid_schema_fails_without_calling_model(self):
        with self.assertRaises(MCPToolError):
            self.discover([TOOL.model_copy(update={"input_schema": {"type": 15}})])

    def test_remote_schema_reference_does_not_fetch_url_or_execute(self):
        catalog = self.discover([TOOL.model_copy(update={"input_schema": {
            "type": "object", "$ref": "https://example.invalid/schema.json",
        }})])
        with patch("llm_agent.mcp_tools.call_tool", new_callable=AsyncMock) as call:
            record = catalog.execute("1", catalog.functions[0]["function"]["name"], ARGS)
        self.assertTrue(record.is_error)
        call.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
