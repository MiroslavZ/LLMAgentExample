import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent import Agent, FACTS_SYSTEM, META_PROMPT_SYSTEM
from history import DialogueUsage, HistoryManager, TokenUsage
from main import parse_args
from test_token_usage import completion


def facts_response(updates, *, missing=False):
    response = completion(missing=missing)
    response.choices[0].message.content = json.dumps(updates, ensure_ascii=False)
    return response


class FactsTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "history.json"
        client = patch("agent.OpenAI")
        self.create = client.start().return_value.chat.completions.create
        self.addCleanup(client.stop)

    def agent(self, size=3):
        return Agent("test", self.path, strategy="facts", window_size=size)

    def update_payload(self, index=-2):
        return json.loads(self.create.call_args_list[index].kwargs["messages"][1]["content"])

    def test_facts_survive_window_restart_and_incremental_changes(self):
        agent = self.agent()
        self.create.side_effect = [
            facts_response({"goal": "ИИ-агент", "language": "русский", "deadline": "завтра"}),
            completion(),
            facts_response({"language": "английский", "deadline": None}),
            completion(),
        ]
        agent.request("Создаю ИИ-агента. Отвечай по-русски. Срок — завтра", system="Правила")
        agent.request("Теперь отвечай по-английски, срок отменяется")
        restarted = self.agent()
        expected_facts = {"goal": "ИИ-агент", "language": "английский"}
        self.assertEqual(restarted.history.facts, expected_facts)
        self.create.side_effect = [facts_response({}), completion()]
        result = restarted.request("Продолжим")
        self.assertEqual(self.create.call_count, 6)
        self.assertEqual(self.update_payload()["previous_facts"], expected_facts)
        self.assertFalse(self.update_payload()["initialize"])
        self.assertEqual(len(self.update_payload()["messages"]), 3)
        sent = self.create.call_args.kwargs["messages"]
        self.assertEqual(sent[0], {"role": "system", "content": "Правила"})
        self.assertEqual(sent[1]["role"], "system")
        self.assertIn(json.dumps(expected_facts, ensure_ascii=False), sent[1]["content"])
        self.assertEqual(sent[2:], [
            {"role": "user", "content": "Теперь отвечай по-английски, срок отменяется"},
            {"role": "assistant", "content": "Ответ"},
            {"role": "user", "content": "Продолжим"},
        ])
        saved = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(saved["facts"], expected_facts)
        self.assertEqual(saved["summary"], "")
        self.assertEqual(len(saved["messages"]), 4)  # system + N сообщений
        self.assertNotIn("Срок — завтра", self.path.read_text(encoding="utf-8"))
        self.assertEqual(result.dialogue_usage, DialogueUsage(600, 180, 780))
        self.assertEqual(HistoryManager(self.path).get_usage(), result.dialogue_usage)

    def test_update_is_saved_before_answer_and_has_its_own_request_options(self):
        agent = self.agent(1)

        def respond(**kwargs):
            if kwargs["messages"][0]["content"] == FACTS_SYSTEM:
                return facts_response({"name": "Мирослав"})
            self.assertEqual(HistoryManager(self.path).facts, {"name": "Мирослав"})
            return completion()

        self.create.side_effect = respond
        agent.request(
            "Меня зовут Мирослав", model="selected-model", system="Правила",
            response_format="object", temperature=0.8, max_tokens=200, stop_sequences=["STOP"],
        )
        update, answer = self.create.call_args_list
        self.assertEqual(update.kwargs["model"], "selected-model")
        self.assertEqual(update.kwargs["response_format"], {"type": "json_object"})
        for parameter in ("temperature", "max_tokens", "stop"):
            self.assertNotIn(parameter, update.kwargs)
        self.assertEqual(self.update_payload()["user_message"], "Меня зовут Мирослав")
        self.assertTrue(self.update_payload()["initialize"])
        self.assertEqual(answer.kwargs["response_format"], {"type": "json_object"})
        self.assertEqual(answer.kwargs["stop"], ["STOP"])
        self.assertEqual(len(answer.kwargs["messages"]), 3)  # system + facts + текущий user
        self.assertEqual(answer.kwargs["messages"][-1]["role"], "user")
        for call in self.create.call_args_list:
            self.assertTrue(all(set(message) == {"role", "content"} for message in call.kwargs["messages"]))

    def test_legacy_history_and_summary_are_read_before_trimming(self):
        for compressed in (False, True):
            with self.subTest(compressed=compressed):
                history = HistoryManager(self.path)
                history.clear()
                history.add_exchange("Старая цель", "Старый ответ", None)
                if compressed:
                    history.compress(2, "Старая цель в summary", TokenUsage(10, 5, 15))
                history.add_exchange("Ограничение", "Договорились", TokenUsage(20, 5, 25))
                history.add_exchange("Последний вопрос", "Последний ответ", None)
                previous_messages = history.get_compression_messages(0)
                previous_usage = history.get_usage()
                original = self.path.read_bytes()
                agent = self.agent(1)
                self.assertEqual(self.path.read_bytes(), original)
                self.create.side_effect = [facts_response({"goal": "Старая цель"}), completion()]
                result = agent.request("Продолжим", system="Правила")
                payload = self.update_payload()
                self.assertEqual(payload["messages"], previous_messages)
                self.assertEqual(payload["previous_summary"], "Старая цель в summary" if compressed else "")
                self.assertTrue(payload["initialize"])
                self.assertEqual(len(self.create.call_args.kwargs["messages"]), 3)
                self.assertEqual(agent.history.summary, "")
                self.assertEqual(result.dialogue_usage, DialogueUsage(
                    previous_usage.prompt_tokens + 200,
                    previous_usage.completion_tokens + 60,
                    previous_usage.total_tokens + 260,
                    previous_usage.missing_responses,
                ))

    def test_failed_initial_extraction_keeps_full_legacy_history(self):
        history = HistoryManager(self.path)
        history.add_exchange("Важная цель", "Ответ", None)
        history.compress(1, "Важная цель", None)
        history.add_exchange("Второй вопрос", "Второй ответ", None)
        original = self.path.read_bytes()
        agent = self.agent(1)
        self.create.side_effect = RuntimeError("API error")
        with self.assertRaises(RuntimeError):
            agent.request("Продолжим")
        self.assertEqual(self.path.read_bytes(), original)
        self.assertFalse(agent.history.has_facts)
        self.assertEqual(agent.history.get_messages(), history.get_messages())

    def test_invalid_or_incomplete_updates_leave_memory_and_history_unchanged(self):
        agent = self.agent()
        self.create.side_effect = [facts_response({"goal": "Сохранить"}), completion()]
        agent.request("Цель")
        original = self.path.read_bytes()
        usage = agent.history.get_usage()
        for content, finish_reason in (
            ("not JSON", "stop"), ("", "stop"), (" ", "stop"),
            ('{"goal": "Часть"}', "length"), ("{}", "content_filter"),
            ("null", "stop"), ("[]", "stop"), ('{"goal": 10}', "stop"),
            ('{"goal": true}', "stop"), ('{"goal": []}', "stop"),
            ('{"goal": {"nested": "value"}}', "stop"),
            ('{"goal": " "}', "stop"), ('{"": "value"}', "stop"), ('{" ": null}', "stop"),
        ):
            with self.subTest(content=content, finish_reason=finish_reason):
                response = completion()
                response.choices[0].message.content = content
                response.choices[0].finish_reason = finish_reason
                self.create.reset_mock()
                self.create.side_effect = [response]
                with self.assertRaises(ValueError):
                    agent.request("Измени цель")
                self.assertEqual(self.create.call_count, 1)
                self.assertEqual(self.path.read_bytes(), original)
                self.assertEqual(agent.history.facts, {"goal": "Сохранить"})
                self.assertEqual(agent.history.get_usage(), usage)

    def test_main_failure_keeps_successfully_updated_facts_and_usage(self):
        agent = self.agent()
        self.create.side_effect = [facts_response({"language": "русский"}), completion()]
        agent.request("По-русски")
        dialogue = agent.history.get_compression_messages(0)
        self.create.side_effect = [facts_response({"language": "английский"}), RuntimeError("Main error")]
        with self.assertRaises(RuntimeError):
            agent.request("По-английски")
        restored = HistoryManager(self.path)
        self.assertEqual(restored.facts, {"language": "английский"})
        self.assertEqual(restored.get_compression_messages(0), dialogue)
        self.assertEqual(restored.get_usage(), DialogueUsage(300, 90, 390))

    def test_save_failure_does_not_replace_facts_or_usage(self):
        agent = self.agent()
        self.create.side_effect = [facts_response({"goal": "Старая"}), completion()]
        agent.request("Старая цель")
        original = self.path.read_bytes()
        self.create.reset_mock()
        self.create.side_effect = [facts_response({"goal": "Новая"})]
        with patch("history.os.fsync", side_effect=OSError("Disk error")):
            with self.assertRaises(OSError):
                agent.request("Новая цель")
        self.assertEqual(self.create.call_count, 1)
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(agent.history.facts, {"goal": "Старая"})
        self.assertEqual(agent.history.get_usage(), DialogueUsage(200, 60, 260))
        self.assertEqual(list(self.path.parent.glob("*.tmp")), [])

    def test_empty_facts_remain_initialized_after_restart_and_clear_resets_everything(self):
        agent = self.agent(1)
        self.create.side_effect = [facts_response({"goal": "Цель"}), completion()]
        agent.request("Цель")
        self.create.side_effect = [facts_response({"goal": None}, missing=True), completion()]
        agent.request("Забудь цель")
        restarted = self.agent(1)
        self.assertEqual(restarted.history.facts, {})
        self.assertTrue(restarted.history.has_facts)
        self.create.side_effect = [facts_response({}), completion()]
        result = restarted.request("Привет")
        self.assertFalse(self.update_payload()["initialize"])
        self.assertIn("<facts>\n{}\n</facts>", self.create.call_args.kwargs["messages"][0]["content"])
        self.assertEqual(result.dialogue_usage, DialogueUsage(500, 150, 650, 1))
        restarted.history.clear()
        self.assertEqual(restarted.history.facts, {})
        self.assertFalse(restarted.history.has_facts)
        self.assertEqual(restarted.history.get_messages(), [])
        self.assertEqual(HistoryManager(self.path).get_usage(), DialogueUsage())
        self.assertFalse(HistoryManager(self.path).has_facts)

    def test_meta_prompt_uses_same_facts_at_both_stages_without_synthetic_update(self):
        agent = self.agent(1)
        self.create.side_effect = [facts_response({"language": "русский"}), completion(), completion()]
        _, result = agent.request_with_meta_prompt("По-русски", system="Правила")
        self.assertEqual(self.create.call_count, 3)
        self.assertEqual(self.update_payload(0)["user_message"], "По-русски")
        for call, system in zip(self.create.call_args_list[1:], (META_PROMPT_SYSTEM, "Правила")):
            messages = call.kwargs["messages"]
            self.assertEqual(len(messages), 3)
            self.assertEqual(messages[0], {"role": "system", "content": system})
            self.assertIn('"language": "русский"', messages[1]["content"])
            self.assertEqual(messages[2]["role"], "user")
        self.assertEqual(result.dialogue_usage, DialogueUsage(300, 90, 390))
        self.assertEqual(HistoryManager(self.path).facts, {"language": "русский"})

    def test_invalid_persisted_facts_are_rejected_without_overwriting(self):
        agent = self.agent()
        self.create.side_effect = [facts_response({}), completion()]
        agent.request("Привет")
        saved = json.loads(self.path.read_text(encoding="utf-8"))
        for facts in (None, [], "text", {"": "value"}, {"key": " "}, {"key": None}, {"key": 1}):
            with self.subTest(facts=facts):
                saved["facts"] = facts
                self.path.write_text(json.dumps(saved), encoding="utf-8")
                original = self.path.read_bytes()
                with self.assertRaises(ValueError):
                    self.agent()
                self.assertEqual(self.path.read_bytes(), original)

    def test_facts_property_returns_copy(self):
        agent = self.agent()
        self.create.side_effect = [facts_response({"goal": "Цель"}), completion()]
        agent.request("Цель")
        original = self.path.read_bytes()
        facts = agent.history.facts
        facts["goal"] = "Другая"
        self.assertEqual(agent.history.facts, {"goal": "Цель"})
        self.assertEqual(self.path.read_bytes(), original)

    def test_switching_to_window_removes_facts_and_preserves_usage(self):
        agent = self.agent()
        self.create.side_effect = [facts_response({"goal": "Цель"}), completion()]
        result = agent.request("Цель")
        window = Agent("test", self.path, strategy="window", window_size=1)
        self.assertFalse(window.history.has_facts)
        self.assertEqual(window.history.get_messages(), [{"role": "assistant", "content": "Ответ"}])
        self.assertEqual(HistoryManager(self.path).get_usage(), result.dialogue_usage)
        self.assertNotIn('"facts"', self.path.read_text(encoding="utf-8"))

    def test_agent_and_cli_validation(self):
        for size in (None, 0, -1, True, 1.5):
            with self.subTest(size=size), self.assertRaises(ValueError):
                self.agent(size)
        for options in ({"last_messages": 1}, {"compress_every": 2}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                Agent("test", self.path, strategy="facts", window_size=3, **options)
        self.assertFalse(self.path.exists())
        with patch("sys.argv", ["main.py", "--user", "Запрос", "--strategy", "facts", "--window-size", "3"]):
            args = parse_args()
        self.assertEqual((args.strategy, args.window_size), ("facts", 3))
        for options in (
            [], ["--window-size", "0"], ["--window-size", "-1"],
            ["--window-size", "3", "--last-messages", "1"],
            ["--window-size", "3", "--compress-every", "2"],
        ):
            with self.subTest(options=options), patch(
                "sys.argv", ["main.py", "--user", "Запрос", "--strategy", "facts", *options],
            ):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    parse_args()
                self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
