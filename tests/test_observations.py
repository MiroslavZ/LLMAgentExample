"""Архив MCP переживает перезапуск, сокращение контекста и сбой ответа."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from mcp import types

from llm_agent.models import Conversation, ContextSettings, Turn
from llm_agent.observations import MAX_PAGE_CHARS, TOOL_NAME, ObservationReader
from llm_agent.service import ConversationService
from tests.helpers import create_selected_conversation
from llm_agent.tool_events import ToolCallRecord
from tests.test_mcp_agent import SERVER, TOOL, answer, tool_response
from llm_agent.mcp_client import MCPDiscovery


def record(value="sha-1", *, error=False, server="GitHub", tool="commits"):
    return ToolCallRecord("call", server, tool, '{"repo":"user/project"}',
                          json.dumps({"is_error": error, "data": value}), error, 0.1)


class ObservationReaderTests(unittest.TestCase):
    def test_failures_before_first_tool_remain_visible_with_source_filter(self):
        reader = ObservationReader(Conversation("a" * 32, turns=[
            Turn("collect", status="completed", tool_calls=[record()]),
            Turn("collect", status="error", error="MCP недоступен"),
            Turn("summary", status="running"),
        ]))
        page = reader.read(server_name="GitHub")
        self.assertEqual(page["total"], 2)
        self.assertEqual(page["items"][1]["kind"], "failed_request")
        self.assertEqual(page["items"][1]["run_error"], "MCP недоступен")
        self.assertIsNone(page["items"][1]["server_name"])

    def test_pages_filters_boundaries_and_errors(self):
        conversation = Conversation("a" * 32, turns=[
            Turn("collect", created_at=f"2026-09-27T0{hour}:00:00+00:00",
                 status="partial" if hour == 2 else "completed",
                 tool_calls=[record(str(hour), error=hour == 2)]) for hour in range(4)
        ])
        reader = ObservationReader(conversation)
        page = reader.read(limit=1, since="2026-09-27T06:00:00+05:00", until="2026-09-27T03:00:00Z")
        self.assertEqual(page["total"], 2)
        self.assertEqual(page["next_offset"], 1)
        self.assertEqual(page["items"][0]["turn_index"], 1)
        last = reader.read(offset=page["next_offset"], limit=1,
                           since="2026-09-27T01:00:00Z", until="2026-09-27T03:00:00Z")
        self.assertIsNone(last["next_offset"])
        self.assertTrue(last["items"][0]["is_error"])
        self.assertEqual(last["items"][0]["run_status"], "partial")
        self.assertEqual(reader.read(server_name="other")["total"], 0)
        self.assertEqual(reader.read(tool_name="other")["items"], [])
        self.assertEqual(reader.read(offset=100)["items"], [])

    def test_byte_budget_keeps_cursor_and_oversized_records_are_explicit(self):
        reader = ObservationReader(Conversation("a" * 32, turns=[
            Turn("collect", tool_calls=[record("X" * 30_000) for _ in range(3)]),
        ]))
        page = reader.read()
        self.assertLess(len(json.dumps(page, ensure_ascii=False)), MAX_PAGE_CHARS)
        self.assertEqual(page["total"], 3)
        self.assertEqual(page["next_offset"], 1)
        self.assertEqual(len(reader.read(offset=1)["items"]), 1)
        huge = ObservationReader(Conversation("b" * 32, turns=[
            Turn("collect", tool_calls=[record("X" * 100_000)]),
        ])).read()
        self.assertTrue(huge["items"][0]["truncated"])
        self.assertIsNone(huge["next_offset"])

    def test_invalid_arguments_are_tool_errors(self):
        reader = ObservationReader(Conversation("a" * 32))
        for arguments in ("{", "[]", '{"offset":true}', '{"limit":21}',
                          '{"offset":-1}', '{"since":"2026-09-27"}',
                          '{"until":null,"unknown":1}', '{"since":12}',
                          '{"since":"2026-09-28T00:00:00Z","until":"2026-09-27T00:00:00Z"}'):
            with self.subTest(arguments=arguments):
                self.assertTrue(reader.execute("call", arguments).is_error)

    def test_snapshot_and_read_results_do_not_become_observations(self):
        conversation = Conversation("a" * 32, turns=[Turn("collect", tool_calls=[record()])])
        reader = ObservationReader(conversation)
        conversation.turns.append(Turn("summary", tool_calls=[reader.execute("read", "{}")]))
        conversation.turns.append(Turn("collect", tool_calls=[record("sha-2")]))
        self.assertEqual(reader.read()["total"], 1)
        self.assertEqual(ObservationReader(conversation).read()["total"], 2)


class ObservationIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.create = self.enterContext(patch("llm_agent.agent.OpenAI")).return_value.chat.completions.create
        self.discovery = self.enterContext(patch("llm_agent.mcp_tools.get_tools", new_callable=AsyncMock,
            return_value=MCPDiscovery("github", "1.0", "test", (TOOL,))))
        self.call = self.enterContext(patch("llm_agent.mcp_tools.call_tool", new_callable=AsyncMock,
            return_value=types.CallToolResult(content=[], structured_content={"sha": "sha-1"})))

    def service(self):
        return ConversationService(self.root, "test-key")

    def test_model_can_choose_archive_while_external_mcp_remains_available(self):
        service = self.service()
        conversation = create_selected_conversation(service)
        service.mcp_servers.save(SERVER)
        steps = iter([
            lambda request: tool_response(request["tools"][0]["function"]["name"]),
            lambda request: answer("Собрано"),
        ])
        self.create.side_effect = lambda **request: next(steps)(request)
        collected = service.send(conversation.id, "Получи коммиты",
                                 settings=ContextSettings(strategy="window", window_size=1))
        self.assertEqual(collected.turns[-1].status, "completed")
        self.discovery.reset_mock()
        self.call.reset_mock()
        self.create.side_effect = [tool_response(TOOL_NAME, arguments="{}"), answer("Исходный снимок: sha-1")]
        reopened = self.service()
        summary = reopened.send(conversation.id, "Сводка только по архиву, не запрашивай новые данные")
        self.assertEqual(summary.turns[-1].status, "completed")
        self.discovery.assert_awaited_once()
        self.call.assert_not_called()
        functions = self.create.call_args.kwargs["tools"]
        self.assertEqual(len(functions), 2)
        self.assertIn(TOOL_NAME, [item["function"]["name"] for item in functions])
        messages = self.create.call_args.kwargs["messages"]
        page = json.loads(messages[-1]["content"])["data"]
        self.assertEqual(page["total"], 1)
        self.assertIn("sha-1", page["items"][0]["result"])
        self.assertEqual(ObservationReader(reopened.get(conversation.id)).read()["total"], 1)
        self.assertEqual(ObservationReader(reopened.create()).read()["total"], 0)

    def test_tool_result_survives_failed_final_and_can_be_read_after_restart(self):
        service = self.service()
        conversation = create_selected_conversation(service)
        service.mcp_servers.save(SERVER)
        count = 0

        def complete(**request):
            nonlocal count
            count += 1
            if count == 1:
                return tool_response(request["tools"][0]["function"]["name"])
            raise RuntimeError("Final unavailable")

        self.create.side_effect = complete
        failed = service.send(conversation.id, "Собери данные")
        self.assertEqual(failed.turns[-1].status, "partial")
        archived = ObservationReader(self.service().get(conversation.id)).read()
        self.assertEqual(archived["total"], 1)
        self.assertEqual(archived["items"][0]["run_status"], "partial")
        self.assertIn("sha-1", archived["items"][0]["result"])

    def test_empty_archive_is_available_alongside_configured_mcp(self):
        service = self.service()
        conversation = create_selected_conversation(service)
        service.mcp_servers.save(SERVER)
        self.create.side_effect = [tool_response(TOOL_NAME, arguments="{}"), answer("Наблюдений пока нет")]
        result = service.send(conversation.id, "Сводка")
        self.assertEqual(result.turns[-1].status, "completed")
        self.assertEqual(json.loads(result.turns[-1].tool_calls[0].result)["data"]["total"], 0)
        self.discovery.assert_awaited_once()
        self.call.assert_not_called()
