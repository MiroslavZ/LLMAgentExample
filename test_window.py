import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent import Agent, META_PROMPT_SYSTEM
from history import DialogueUsage, HistoryManager, TokenUsage
from main import parse_args
from test_token_usage import completion


class WindowTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "history.json"
        client = patch("agent.OpenAI")
        self.create = client.start().return_value.chat.completions.create
        self.addCleanup(client.stop)
        self.create.return_value = completion()

    def agent(self, size):
        return Agent("test", self.path, strategy="window", window_size=size)

    def test_request_and_saved_history_are_bounded_and_usage_survives_restart(self):
        agent = self.agent(3)
        agent.request("Первый", system="Правила")
        agent.request("Второй")
        result = agent.request("Третий")
        self.assertEqual(self.create.call_count, 3)
        self.assertEqual(self.create.call_args.kwargs["messages"], [
            {"role": "system", "content": "Правила"},
            {"role": "user", "content": "Второй"},
            {"role": "assistant", "content": "Ответ"},
            {"role": "user", "content": "Третий"},
        ])
        restored = HistoryManager(self.path)
        self.assertEqual(restored.get_messages(), [
            {"role": "system", "content": "Правила"},
            {"role": "assistant", "content": "Ответ"},
            {"role": "user", "content": "Третий"},
            {"role": "assistant", "content": "Ответ"},
        ])
        self.assertEqual(result.dialogue_usage, DialogueUsage(300, 90, 390))
        self.assertEqual(restored.get_usage(), result.dialogue_usage)

    def test_one_message_window_applies_to_both_meta_stages(self):
        agent = self.agent(1)
        agent.request_with_meta_prompt("Задача", system="Правила")
        for call, system in zip(self.create.call_args_list, (META_PROMPT_SYSTEM, "Правила")):
            messages = call.kwargs["messages"]
            self.assertEqual(len(messages), 2)
            self.assertEqual(messages[0], {"role": "system", "content": system})
            self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(len(HistoryManager(self.path).get_messages()), 2)

    def test_loading_compressed_history_discards_summary_and_preserves_usage(self):
        history = HistoryManager(self.path)
        history.add_exchange("Старый вопрос", "Старый ответ", None)
        history.compress(2, "Старое summary", TokenUsage(10, 5, 15))
        history.add_exchange("Новый вопрос", "Новый ответ", TokenUsage(20, 5, 25))
        total = history.get_usage()
        agent = self.agent(1)
        self.assertEqual(agent.history.summary, "")
        self.assertEqual(agent.history.get_messages(), [{"role": "assistant", "content": "Новый ответ"}])
        self.assertNotIn("Старое summary", self.path.read_text(encoding="utf-8"))
        self.assertEqual(HistoryManager(self.path).get_usage(), total)
        agent.history.clear()
        self.assertEqual(HistoryManager(self.path).get_usage(), DialogueUsage())

    def test_api_failure_does_not_add_messages(self):
        agent = self.agent(2)
        agent.request("Первый")
        original = self.path.read_bytes()
        self.create.side_effect = RuntimeError("API error")
        with self.assertRaises(RuntimeError):
            agent.request("Второй")
        self.assertEqual(self.path.read_bytes(), original)

    def test_invalid_agent_options_do_not_touch_history(self):
        for size in (None, 0, -1, True, 1.5):
            with self.subTest(size=size), self.assertRaises(ValueError):
                self.agent(size)
        with self.assertRaises(ValueError):
            Agent("test", self.path, strategy="window", window_size=2, last_messages=1)
        self.assertFalse(self.path.exists())

    def test_cli_validation(self):
        with patch("sys.argv", ["main.py", "--user", "Запрос", "--strategy", "window", "--window-size", "3"]):
            args = parse_args()
        self.assertEqual((args.strategy, args.window_size), ("window", 3))
        for options in (
            ["--strategy", "window"], ["--window-size", "2"],
            ["--strategy", "facts"],
            ["--strategy", "unknown"],
            ["--strategy", "window", "--window-size", "0"],
            ["--strategy", "window", "--window-size", "2", "--compress-every", "3"],
        ):
            with self.subTest(options=options), patch("sys.argv", ["main.py", "--user", "Запрос", *options]):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    parse_args()
                self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
