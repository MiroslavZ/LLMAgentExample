"""Краткая Windows-блокировка не должна терять снимок или обрывать запрос."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from llm_agent.models import Conversation
from llm_agent.service import _friendly_error
from llm_agent.storage import ConversationStorageError, ConversationStore


class StorageRetryTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.store = ConversationStore(self.directory)
        self.conversation = Conversation(id="a" * 32, title="Старый снимок")
        self.store.save(self.conversation)
        self.path = self.store.path(self.conversation.id)
        self.original = self.path.read_bytes()
        self.conversation.title = "Новый снимок"

    @unittest.skipUnless(os.name == "nt", "Чтение блокирует замену файла только в Windows")
    def test_actual_windows_reader_is_closed_and_save_succeeds(self):
        with self.path.open(encoding="utf-8") as reader:
            reader.read()
            with patch("llm_agent.storage.time.sleep", side_effect=lambda _: reader.close()) as delay:
                self.store.save(self.conversation)
        delay.assert_called_once_with(0.02)
        self.assertEqual(self.store.load(self.conversation.id), self.conversation)
        self.assertFalse(list(self.directory.glob("*.tmp")))

    def test_persistent_windows_lock_has_bounded_retries_and_preserves_original(self):
        for code in (5, 32, 33):
            with self.subTest(code=code):
                error = PermissionError("private path")
                error.winerror = code
                with patch.object(Path, "replace", side_effect=error) as replace, \
                        patch("llm_agent.storage.time.sleep") as delay:
                    with self.assertRaises(ConversationStorageError) as caught:
                        self.store.save(self.conversation)
                self.assertIs(caught.exception.__cause__, error)
                self.assertEqual(replace.call_count, 6)
                self.assertEqual(delay.call_count, 5)
                self.assertLessEqual(sum(call.args[0] for call in delay.call_args_list), 0.31)
                self.assertEqual(self.path.read_bytes(), self.original)
                self.assertFalse(list(self.directory.glob("*.tmp")))

    def test_unrelated_errors_are_not_retried(self):
        for error in (PermissionError("no access"), OSError("disk full")):
            with self.subTest(error=type(error).__name__):
                with patch.object(Path, "replace", side_effect=error) as replace, \
                        patch("llm_agent.storage.time.sleep") as delay:
                    with self.assertRaises(ConversationStorageError):
                        self.store.save(self.conversation)
                replace.assert_called_once()
                delay.assert_not_called()
                self.assertEqual(self.path.read_bytes(), self.original)

    def test_error_identifies_local_history_and_windows_code_without_raw_details(self):
        cause = PermissionError("private path or data")
        cause.winerror = 5
        error = ConversationStorageError("save failed")
        error.__cause__ = cause
        message = _friendly_error(error)
        self.assertIn("локальной истории агента", message)
        self.assertIn("WinError 5", message)
        self.assertNotIn("private", message)


if __name__ == "__main__":
    unittest.main()
