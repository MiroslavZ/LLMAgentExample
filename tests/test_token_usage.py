import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rich.console import Console

from llm_agent.agent import Agent
from llm_agent.history import DialogueUsage, HistoryManager, TokenUsage
from llm_agent.cli import print_response
from tests.helpers import completion


class TokenUsageTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "llm_agent.history.json"
        client = patch("llm_agent.agent.OpenAI")
        self.create = client.start().return_value.chat.completions.create
        self.addCleanup(client.stop)
        self.agent = Agent("test-key", history_path=self.path)

    def test_api_usage_includes_cache_and_reasoning_without_double_counting(self):
        self.create.return_value = completion()
        result = self.agent.request("Вопрос", system="Система")
        self.assertEqual(result.dialogue_usage, DialogueUsage(100, 30, 130))
        saved = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(saved[-1]["usage"], {
            "prompt_tokens": 100, "completion_tokens": 30, "total_tokens": 130,
        })

    def test_repeated_context_is_counted_and_totals_survive_restart(self):
        self.create.side_effect = [completion(), completion(150, 40)]
        first = self.agent.request("Первый", system="Система")
        restarted = Agent("test-key", history_path=self.path)
        second = restarted.request("Второй")
        self.assertEqual(first.dialogue_usage, DialogueUsage(100, 30, 130))
        self.assertEqual(second.dialogue_usage, DialogueUsage(250, 70, 320))
        self.assertEqual(HistoryManager(self.path).get_usage(), second.dialogue_usage)
        self.assertEqual(self.create.call_args.kwargs["messages"], [
            {"role": "system", "content": "Система"},
            {"role": "user", "content": "Первый"},
            {"role": "assistant", "content": "Ответ"},
            {"role": "user", "content": "Второй"},
        ])

    def test_meta_prompt_counts_both_requests_once(self):
        self.create.side_effect = [completion(), completion(150, 40)]
        meta, result = self.agent.request_with_meta_prompt("Задача")
        self.assertEqual(self.create.call_count, 2)
        self.assertEqual(meta.dialogue_usage, DialogueUsage(100, 30, 130))
        self.assertEqual(result.dialogue_usage, DialogueUsage(250, 70, 320))
        self.assertEqual(HistoryManager(self.path).get_usage(), result.dialogue_usage)

    def test_failed_second_stage_keeps_only_first_stage_usage(self):
        self.create.side_effect = [completion(), RuntimeError("API unavailable")]
        with self.assertRaises(RuntimeError):
            self.agent.request_with_meta_prompt("Задача")
        self.assertEqual(HistoryManager(self.path).get_usage(), DialogueUsage(100, 30, 130))
        self.assertEqual(len(self.agent.history.get_messages()), 2)

    def test_missing_usage_and_legacy_responses_are_marked_incomplete(self):
        self.agent.history.add_message("assistant", "Старый ответ")
        self.create.side_effect = [completion(missing=True), completion()]
        self.agent.request("Без статистики")
        result = self.agent.request("Со статистикой")
        self.assertEqual(result.dialogue_usage, DialogueUsage(100, 30, 130, 2))
        self.assertEqual(HistoryManager(self.path).get_usage(), result.dialogue_usage)
        output = io.StringIO()
        with patch("llm_agent.cli.console", Console(file=output, width=160, color_system=None)):
            print_response(result, "text")
        rendered = output.getvalue()
        self.assertIn("неполные данные", rendered)
        self.assertIn("Ответов без статистики", rendered)
        self.assertIn("вход 100 → выход 30 (всего 130)", rendered)

    def test_zero_usage_is_known_and_clear_resets_totals(self):
        self.agent.history.add_exchange("Вопрос", "Ответ", TokenUsage())
        self.assertEqual(HistoryManager(self.path).get_usage(), DialogueUsage())
        self.agent.history.add_exchange("Ещё", "Ответ", TokenUsage(100, 30, 130))
        self.agent.history.add_message("assistant", "Без статистики")
        self.agent.history.clear()
        self.assertEqual(self.agent.history.get_usage(), DialogueUsage())
        self.assertEqual(HistoryManager(self.path).get_usage(), DialogueUsage())

    def test_invalid_usage_does_not_overwrite_history(self):
        self.agent.history.add_exchange("Вопрос", "Ответ", TokenUsage(100, 30, 130))
        original = self.path.read_bytes()
        for invalid in (
            TokenUsage(100, 30, 131),
            TokenUsage(-1, 30, 29),
            TokenUsage(True, 30, 31),
            TokenUsage(1.5, 30, 31.5),
        ):
            with self.subTest(usage=invalid):
                with self.assertRaises(ValueError):
                    self.agent.history.add_exchange("Вопрос", "Ответ", invalid)
                self.assertEqual(self.path.read_bytes(), original)
                self.assertEqual(self.agent.history.get_usage(), DialogueUsage(100, 30, 130))

    def test_cli_reports_current_request_separately_from_dialogue(self):
        self.create.side_effect = [completion(), completion(150, 40)]
        self.agent.request("Первый")
        result = self.agent.request("Второй")
        output = io.StringIO()
        with patch("llm_agent.cli.console", Console(file=output, width=160, color_system=None)):
            print_response(result, "text")
        rendered = output.getvalue()
        self.assertIn("вход 150 → выход 40 (всего 190)", rendered)
        self.assertIn("вход 250 → выход 70 (всего 320)", rendered)


    def test_large_cumulative_usage_does_not_trigger_context_warning(self):
        from llm_agent.agent import RequestResult

        result = RequestResult(completion(), 0.1, DialogueUsage(2_000_000, 30, 2_000_030))
        output = io.StringIO()
        with patch("llm_agent.cli.console", Console(file=output, width=180, color_system=None)):
            print_response(result, "text", context_limit=1000, max_tokens=200)
        rendered = output.getvalue()
        self.assertIn("100 / 1000 (10.0%)", rendered)
        self.assertIn("700", rendered)
        self.assertNotIn("превышает", rendered)
        self.assertIn("Из входа: кэш", rendered)
        self.assertIn("Из выхода: рассуждения", rendered)
        self.assertIn("2000030", rendered)

    def test_context_reserve_overflow_is_reported(self):
        from llm_agent.agent import RequestResult

        output = io.StringIO()
        with patch("llm_agent.cli.console", Console(file=output, width=180, color_system=None)):
            print_response(RequestResult(completion(), 0), "text", context_limit=120, max_tokens=30)
        self.assertIn("превышает заданный лимит", output.getvalue())
        self.assertIn("-10", output.getvalue())


if __name__ == "__main__":
    unittest.main()
