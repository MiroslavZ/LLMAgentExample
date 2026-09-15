import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from rich.console import Console

from llm_agent.agent import Agent, CompressionResult, META_PROMPT_SYSTEM
from llm_agent.history import DialogueUsage, HistoryManager, TokenUsage
from llm_agent.cli import parse_args, print_compression
from tests.helpers import completion


def summary_response(content="Краткая история", finish_reason="stop"):
    response = completion()
    response.choices[0].message.content = content
    response.choices[0].finish_reason = finish_reason
    return response


class CompressionTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "llm_agent.history.json"
        client = patch("llm_agent.agent.OpenAI")
        self.create = client.start().return_value.chat.completions.create
        self.addCleanup(client.stop)
        self.agent = Agent("test", self.path, last_messages=2, compress_every=2)
        self.agent.history.set_system_prompt("Правила")
        self.agent.history.add_exchange("Первый", "Первый ответ", TokenUsage(10, 5, 15))

    def test_threshold_and_exact_tail(self):
        self.create.return_value = completion()
        self.agent.request("Второй")
        self.assertEqual(self.create.call_count, 1)
        tail = self.agent.history.get_messages()[-2:]
        self.create.reset_mock()
        self.create.side_effect = [summary_response(), completion()]
        result = self.agent.request("Третий", response_format="object", stop_sequences=["STOP"])
        compression, request = self.create.call_args_list
        payload = json.loads(compression.kwargs["messages"][1]["content"])
        self.assertEqual(payload, {"previous_summary": "", "messages": [
            {"role": "user", "content": "Первый"},
            {"role": "assistant", "content": "Первый ответ"},
        ]})
        self.assertEqual(compression.kwargs["response_format"], {"type": "text"})
        self.assertNotIn("stop", compression.kwargs)
        sent = request.kwargs["messages"]
        self.assertEqual(sent[0], {"role": "system", "content": "Правила"})
        self.assertIn("Краткая история", sent[1]["content"])
        self.assertEqual(sent[2:-1], tail)
        self.assertEqual(result.dialogue_usage, DialogueUsage(310, 95, 405))
        saved = HistoryManager(self.path)
        self.assertEqual(saved.get_messages(), self.agent.history.get_messages())
        self.assertEqual(saved.get_usage(), result.dialogue_usage)

    def test_incremental_summary_after_restart_and_clear(self):
        self.agent.history.add_exchange("Второй", "Ответ", None)
        self.create.side_effect = [summary_response("Первое summary"), completion()]
        self.agent.request("Третий")
        restarted = Agent("test", self.path, last_messages=1, compress_every=2)
        self.create.side_effect = [summary_response("Обновлённое summary"), completion()]
        restarted.request("Четвёртый")
        payload = json.loads(self.create.call_args_list[-2].kwargs["messages"][1]["content"])
        self.assertEqual(payload["previous_summary"], "Первое summary")
        self.assertEqual([m["content"] for m in payload["messages"]], ["Второй", "Ответ", "Третий"])
        self.assertEqual(restarted.history.get_usage(), DialogueUsage(410, 125, 535, 1))
        restarted.history.clear()
        self.assertEqual(HistoryManager(self.path).get_messages(), [])
        self.assertEqual(restarted.history.summary, "")
        self.assertEqual(restarted.history.get_usage(), DialogueUsage())

    def test_failed_or_incomplete_compression_keeps_history(self):
        self.agent.history.add_exchange("Второй", "Ответ", None)
        original = self.path.read_bytes()
        for response in [RuntimeError("API error"), summary_response(" "), summary_response("Часть", "length")]:
            with self.subTest(response=response):
                self.create.side_effect = [response]
                with self.assertRaises((RuntimeError, ValueError)):
                    self.agent.request("Третий")
                self.assertEqual(self.path.read_bytes(), original)
                self.assertEqual(self.agent.history.summary, "")

    def test_save_failure_keeps_memory_and_disk(self):
        original = self.path.read_bytes()
        with patch("llm_agent.history.os.fsync", side_effect=OSError("Disk error")):
            with self.assertRaises(OSError):
                self.agent.history.compress(2, "Summary", None)
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(self.agent.history.summary, "")
        self.assertEqual(self.agent.history.get_usage(), DialogueUsage(10, 5, 15))

    def test_meta_prompt_receives_summary_at_both_stages(self):
        self.agent.history.add_exchange("Второй", "Ответ", None)
        self.create.side_effect = [summary_response(), completion(), summary_response("Новое summary"), completion()]
        self.agent.request_with_meta_prompt("Задача")
        first = self.create.call_args_list[1].kwargs["messages"]
        second = self.create.call_args_list[3].kwargs["messages"]
        self.assertEqual(first[0]["content"], META_PROMPT_SYSTEM)
        self.assertIn("Краткая история", first[1]["content"])
        self.assertEqual(second[0]["content"], "Правила")
        self.assertIn("Новое summary", second[1]["content"])

    def test_zero_tail_and_failed_main_request(self):
        agent = Agent("test", self.path, last_messages=0, compress_every=1)
        self.create.side_effect = [completion(missing=True), RuntimeError("Main error")]
        with self.assertRaises(RuntimeError):
            agent.request("Новый вопрос")
        saved = HistoryManager(self.path)
        self.assertEqual(len(saved.get_messages()), 2)  # system и summary
        self.assertEqual(saved.get_usage(), DialogueUsage(10, 5, 15, 1))

    def test_argument_validation(self):
        for options in ({"last_messages": -1}, {"compress_every": 0}, {"last_messages": True}):
            with self.assertRaises(ValueError):
                Agent("test", self.path, **options)
        with patch("sys.argv", ["main.py", "--user", "Тест", "--last-messages", "0", "--compress-every", "3"]):
            args = parse_args()
        self.assertEqual((args.last_messages, args.compress_every), (0, 3))
        for option, value in [("--last-messages", "-1"), ("--compress-every", "0")]:
            with patch("sys.argv", ["main.py", "--user", "Тест", option, value]), patch("sys.stderr"):
                with self.assertRaises(SystemExit):
                    parse_args()

    def test_cli_compression_options_and_warning(self):
        for options, expected, warning in [
            ([], (None, None), False),
            (["--last-messages", "0"], (0, None), True),
            (["--compress-every", "1"], (None, 1), True),
            (["--last-messages", "0", "--compress-every", "1"], (0, 1), False),
        ]:
            with self.subTest(options=options):
                output = io.StringIO()
                with patch("sys.argv", ["main.py", "--user", "Тест", *options]), patch(
                    "llm_agent.cli.console", Console(file=output, width=240, color_system=None)
                ):
                    args = parse_args()
                self.assertEqual((args.last_messages, args.compress_every), expected)
                if warning:
                    self.assertIn("необходимо указать оба аргумента", output.getvalue())
                    self.assertIn("--last-messages и --compress-every", output.getvalue())
                    self.assertIn("продолжит работу без сжатия", output.getvalue())
                else:
                    self.assertEqual(output.getvalue(), "")

    def test_disabled_compression_preserves_history_at_all_stages(self):
        for options in ({}, {"last_messages": 0}, {"compress_every": 1}):
            for meta_prompt in (False, True):
                with self.subTest(options=options, meta_prompt=meta_prompt):
                    notify = Mock()
                    agent = Agent("test", self.path, on_compression=notify, **options)
                    agent.history.clear()
                    agent.history.set_system_prompt("Правила")
                    for index in range(12):
                        agent.history.add_exchange(f"Вопрос {index}", f"Ответ {index}", None)
                    original = agent.history.get_messages()
                    self.create.reset_mock()
                    self.create.return_value = completion()
                    if meta_prompt:
                        agent.request_with_meta_prompt("Новый вопрос")
                    else:
                        agent.request("Новый вопрос")
                    self.assertEqual(self.create.call_count, 2 if meta_prompt else 1)
                    for call in self.create.call_args_list:
                        self.assertEqual(call.kwargs["messages"][1:len(original)], original[1:])
                    saved = HistoryManager(self.path)
                    self.assertEqual(saved.summary, "")
                    self.assertEqual(saved.get_messages()[:len(original)], original)
                    self.assertEqual(len(saved.get_messages()), len(original) + (4 if meta_prompt else 2))
                    notify.assert_not_called()

    def test_notification_after_save_even_if_main_request_fails(self):
        self.agent.history.add_exchange("Второй", "Ответ", None)
        observed = []

        def on_compression(result):
            self.assertEqual(HistoryManager(self.path).summary, "Краткая история")
            observed.append(result)

        self.agent.on_compression = on_compression
        self.create.side_effect = [summary_response(), RuntimeError("Main error")]
        with self.assertRaises(RuntimeError):
            self.agent.request("Третий")
        self.assertEqual(observed, [CompressionResult(2, TokenUsage(100, 30, 130))])

    def test_no_notification_without_successful_compression(self):
        notify = Mock()
        self.agent.on_compression = notify
        self.create.return_value = completion()
        self.agent.request("Второй")  # Порог ещё не достигнут.
        notify.assert_not_called()
        self.create.return_value = summary_response("")
        with self.assertRaises(ValueError):
            self.agent.request("Третий")
        notify.assert_not_called()
        self.create.return_value = summary_response()
        with patch("llm_agent.history.os.fsync", side_effect=OSError("Disk error")):
            with self.assertRaises(OSError):
                self.agent.request("Третий")
        notify.assert_not_called()

    def test_cli_compression_reports_api_counts_or_missing_usage(self):
        for usage in (TokenUsage(100, 30, 130), TokenUsage(), None):
            with self.subTest(usage=usage):
                output = io.StringIO()
                with patch("llm_agent.cli.console", Console(file=output, width=180, color_system=None)):
                    print_compression(CompressionResult(10, usage))
                rendered = output.getvalue()
                self.assertIn("Сжатие истории выполнено", rendered)
                self.assertIn("сообщений заменено на summary — 10", rendered)
                if usage is None:
                    self.assertIn("API не вернул статистику", rendered)
                    self.assertNotIn("→", rendered)
                else:
                    self.assertIn(
                        f"до (вход сжатия) {usage.prompt_tokens} → после (выход summary) {usage.completion_tokens}",
                        rendered,
                    )
                    self.assertIn("это не размер всей истории", rendered)


if __name__ == "__main__":
    unittest.main()
