"""История MCP-вызовов: строгая загрузка и совместимость старых диалогов."""

import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError, asdict
from pathlib import Path

from llm_agent.models import Conversation, Turn
from llm_agent.storage import ConversationStorageError, ConversationStore
from llm_agent.tool_events import ToolCallRecord


class ToolCallRecordTests(unittest.TestCase):
    def setUp(self):
        self.record = ToolCallRecord(
            call_id="call_1", server_name="GitHub", tool_name="get_issue",
            arguments='{"number": 17}', result='{"title": "Задача"}',
            is_error=False, elapsed_seconds=0.25,
        )

    def test_record_roundtrip_preserves_raw_arguments_and_result(self):
        restored = ToolCallRecord.from_dict(json.loads(json.dumps(asdict(self.record))))
        self.assertEqual(restored, self.record)
        with self.assertRaises(FrozenInstanceError):
            restored.result = "changed"

    def test_loading_rejects_wrong_types_unknown_fields_and_nonfinite_time(self):
        data = asdict(self.record)
        invalid = [None, [], {}, {**data, "extra": "unknown"}]
        for name in ("call_id", "server_name", "tool_name", "arguments", "result"):
            invalid.extend({**data, name: value} for value in (None, 1, {}, []))
        invalid.extend({**data, "is_error": value} for value in (None, 0, 1, "false"))
        invalid.extend({**data, "elapsed_seconds": value} for value in (
            None, True, "0.2", -1, float("nan"), float("inf"), 10 ** 500,
        ))
        for item in invalid:
            with self.subTest(data=item), self.assertRaises(ValueError):
                ToolCallRecord.from_dict(item)


class ToolCallStorageTests(unittest.TestCase):
    def setUp(self):
        directory = self.enterContext(tempfile.TemporaryDirectory())
        self.store = ConversationStore(Path(directory))
        self.record = ToolCallRecord("call_1", "GitHub", "get_issue", "{}", "Результат", False, 0.1)
        self.conversation = Conversation(
            id="a" * 32, started=True,
            turns=[Turn(user="Найди issue", answer="Нашёл", status="completed", tool_calls=[self.record])],
        )

    def test_tool_records_survive_save_and_reload_as_dataclasses(self):
        self.store.save(self.conversation)
        restored = ConversationStore(self.store.data_dir).load(self.conversation.id)
        self.assertEqual(restored, self.conversation)
        self.assertIsInstance(restored.turns[0].tool_calls[0], ToolCallRecord)

    def test_old_turn_without_tool_calls_loads_and_new_turns_do_not_share_list(self):
        self.store.save(self.conversation)
        path = self.store.path(self.conversation.id)
        data = json.loads(path.read_text(encoding="utf-8"))
        del data["turns"][0]["tool_calls"]
        path.write_text(json.dumps(data), encoding="utf-8")
        self.assertEqual(self.store.load(self.conversation.id).turns[0].tool_calls, [])
        first, second = Turn(user="Первый"), Turn(user="Второй")
        first.tool_calls.append(self.record)
        self.assertEqual(second.tool_calls, [])

    def test_corrupt_tool_calls_are_rejected_without_changing_file(self):
        self.store.save(self.conversation)
        path = self.store.path(self.conversation.id)
        data = json.loads(path.read_text(encoding="utf-8"))
        for calls in (None, {}, "calls", [None], [{**asdict(self.record), "is_error": 1}]):
            with self.subTest(calls=calls):
                data["turns"][0]["tool_calls"] = calls
                path.write_text(json.dumps(data), encoding="utf-8")
                original = path.read_bytes()
                with self.assertRaises(ConversationStorageError):
                    self.store.load(self.conversation.id)
                self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
