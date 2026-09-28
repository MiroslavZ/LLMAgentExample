"""Точная передача между MCP-вызовами, ошибки ссылок и проверка исходных типов."""

import json
import unittest
from unittest.mock import AsyncMock, patch

from mcp import Tool, types

from llm_agent.mcp_client import MCPDiscovery
from llm_agent.mcp_results import MAX_TRANSFER_CHARS
from llm_agent.mcp_tools import MCPToolCatalog
from tests.test_mcp_agent import SERVER


def reference(call_id="source", pointer="/data/summary"):
    return {"$mcp_result": call_id, "pointer": pointer}


class MCPResultTests(unittest.TestCase):
    def setUp(self):
        tool = Tool(name="process", input_schema={
            "type": "object", "properties": {"content": {"type": "string"}},
            "required": ["content"], "additionalProperties": False,
        })
        self.discovery = MCPDiscovery("test", "1", "test", (tool,))
        self.enterContext(patch("llm_agent.mcp_tools.get_tools", new_callable=AsyncMock,
                                return_value=self.discovery))
        self.call = self.enterContext(patch("llm_agent.mcp_tools.call_tool", new_callable=AsyncMock))
        self.catalog = MCPToolCatalog.discover((SERVER,))
        self.name = self.catalog.functions[0]["function"]["name"]

    def seed(self, data, *, error=False):
        self.call.return_value = types.CallToolResult(content=[], structured_content=data, is_error=error)
        return self.catalog.execute("source", self.name, '{"content":"исходные данные"}')

    def execute(self, value, **kwargs):
        return self.catalog.execute("next", self.name, json.dumps({"content": value}), **kwargs)

    def test_full_text_survives_truncation_without_copying_by_model(self):
        summary = 'Сводка "точная"\r\n' * 3000
        source = self.seed({"summary": summary})
        self.assertTrue(json.loads(source.result)["truncated"])
        self.assertEqual(json.loads(source.result)["result_ref"], reference(pointer="/data"))
        self.call.return_value = types.CallToolResult(content=[], structured_content={"saved": True})
        record = self.execute(reference())
        self.assertFalse(record.is_error)
        self.assertEqual(self.call.await_args.args[2], {"content": summary})
        self.assertEqual(json.loads(record.arguments), {"content": reference()})

    def test_unique_tool_name_resolves_to_its_only_call(self):
        self.seed({"summary": "Точная сводка\r\n"})
        self.call.reset_mock()
        record = self.execute(reference(call_id=self.name))
        self.assertFalse(record.is_error)
        self.call.assert_awaited_once_with(SERVER, "process", {"content": "Точная сводка\r\n"})
        self.assertEqual(json.loads(record.arguments)["content"]["$mcp_result"], self.name)

    def test_repeated_tool_name_is_ambiguous_and_error_lists_concrete_calls(self):
        self.seed({"summary": "первая сводка"})
        self.call.return_value = types.CallToolResult(content=[], structured_content={"summary": "вторая сводка"})
        self.catalog.execute("second", self.name, '{"content":"second input"}')
        self.call.reset_mock()
        record = self.execute(reference(call_id=self.name))
        self.assertTrue(record.is_error)
        self.assertIn("неоднозначно", record.result)
        self.call.assert_not_awaited()
        available = json.loads(record.result)["available_results"]
        self.assertEqual([entry["result_ref"]["$mcp_result"] for entry in available], ["source", "second"])
        self.assertFalse(self.execute({**available[1]["result_ref"], "pointer": "/data/summary"}).is_error)
        self.assertEqual(self.call.await_args.args[2], {"content": "вторая сводка"})

    def test_tool_name_cannot_reuse_old_success_after_failed_or_oversized_call(self):
        for data, error in (({"summary": "failed"}, True), ({"summary": "x" * MAX_TRANSFER_CHARS}, False)):
            with self.subTest(error=error):
                self.catalog = MCPToolCatalog.discover((SERVER,))
                self.seed({"summary": "устаревшая сводка"})
                self.call.return_value = types.CallToolResult(content=[], structured_content=data, is_error=error)
                second = self.catalog.execute("second", self.name, '{"content":"second input"}')
                self.assertNotIn("result_ref", json.loads(second.result))
                self.call.reset_mock()
                record = self.execute(reference(call_id=self.name))
                self.assertTrue(record.is_error)
                self.call.assert_not_awaited()
                self.assertEqual(json.loads(record.result)["available_results"], [
                    {"tool_name": self.name, "result_ref": reference(pointer="/data")},
                ])

    def test_pending_tool_name_cannot_bypass_round_boundary(self):
        self.seed({"summary": "current batch"})
        self.call.reset_mock()
        record = self.execute(reference(call_id=self.name), pending_call_ids=["source", "next"])
        self.assertTrue(record.is_error)
        self.assertNotIn("available_results", json.loads(record.result))
        self.call.assert_not_awaited()

    def test_tool_name_and_reference_are_scoped_to_current_request(self):
        self.seed({"summary": "old request"})
        self.catalog = MCPToolCatalog.discover((SERVER,))
        self.call.reset_mock()
        self.assertTrue(self.execute(reference(call_id=self.name)).is_error)
        self.call.assert_not_awaited()

    def test_pointer_escapes_and_array_indexes(self):
        self.seed({"a/b": {"~key": ["точный текст"]}, "": "пустой ключ"})
        for pointer, expected in (("/data/a~1b/~0key/0", "точный текст"), ("/data/", "пустой ключ")):
            with self.subTest(pointer=pointer):
                self.assertFalse(self.execute(reference(pointer=pointer)).is_error)
                self.assertEqual(self.call.await_args.args[2], {"content": expected})

    def test_unavailable_or_malformed_references_never_execute(self):
        self.seed({"summary": "ok", "items": ["first"]})
        references = [reference("missing"), reference(pointer="/data/missing"),
                      reference(pointer="data/summary"), reference(pointer="/data/~2"),
                      reference(pointer="/data/items/-1"), reference(pointer="/data/items/01"),
                      reference(pointer="/data/items/9"), reference(pointer="/data/items/" + "9" * 5000),
                      {"$mcp_result": []}, {"$mcp_result": "source", "pointer": []},
                      {**reference(), "extra": True}]
        self.call.reset_mock()
        for value in references:
            with self.subTest(value=str(value)[:100]):
                self.assertTrue(self.execute(value).is_error)
        self.call.assert_not_awaited()

    def test_original_schema_rejects_wrong_type_after_resolution(self):
        self.seed({"summary": ["список вместо текста"]})
        self.call.reset_mock()
        self.assertTrue(self.execute(reference()).is_error)
        self.call.assert_not_awaited()

    def test_failed_and_oversized_results_cannot_feed_downstream_tools(self):
        for data, error in (({"summary": "failed"}, True), ({"summary": "x" * MAX_TRANSFER_CHARS}, False)):
            with self.subTest(error=error):
                self.catalog = MCPToolCatalog.discover((SERVER,))
                self.seed(data, error=error)
                self.call.reset_mock()
                self.assertTrue(self.execute(reference()).is_error)
                self.call.assert_not_awaited()

    def test_reference_does_not_leak_to_new_user_request(self):
        self.seed({"summary": "old"})
        self.catalog = MCPToolCatalog.discover((SERVER,))
        self.call.reset_mock()
        self.assertTrue(self.execute(reference()).is_error)
        self.call.assert_not_awaited()

    def test_dependent_call_in_same_batch_waits_for_next_round(self):
        self.seed({"summary": "cached during batch"})
        self.call.reset_mock()
        record = self.execute(reference(), pending_call_ids=["source", "next"])
        self.assertTrue(record.is_error)
        self.assertIn("следующем раунде", record.result)
        self.call.assert_not_awaited()
        self.assertFalse(self.execute(reference()).is_error)

    def test_source_data_is_not_interpreted_as_another_reference(self):
        tool = Tool(name="json_input", input_schema={"type": "object", "properties": {"content": {"type": "object"}}})
        self.catalog._register(SERVER, tool)
        name = self.catalog.functions[-1]["function"]["name"]
        data = {"$mcp_result": "invented", "pointer": "/secrets"}
        self.seed({"summary": data})
        result = self.catalog.execute("next", name, json.dumps({"content": reference()}))
        self.assertFalse(result.is_error)
        self.assertEqual(self.call.await_args.args[2], {"content": data})

    def test_text_content_can_be_referenced_without_structured_result(self):
        self.call.return_value = types.CallToolResult(content=[types.TextContent(type="text", text="исходный текст")])
        source = self.catalog.execute("source", self.name, '{"content":"input"}')
        self.assertEqual(json.loads(source.result)["result_ref"], reference(pointer="/content/0/text"))
        self.assertFalse(self.execute(reference(pointer="/content/0/text")).is_error)
        self.assertEqual(self.call.await_args.args[2], {"content": "исходный текст"})


if __name__ == "__main__":
    unittest.main()
