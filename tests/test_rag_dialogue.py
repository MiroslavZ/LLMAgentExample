import json
import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

from llm_agent.rag_dialogue import empty_memory, format_memory, prepare_turn, validate_memory
from tests.helpers import completion


def change(field="constraints", key="language", value="Python", quote="Python", op="set"):
    return dict(op=op, field=field, key=key, value=value, quote=quote)


class DialogueMemoryTests(unittest.TestCase):
    def setUp(self):
        self.turns = [SimpleNamespace(user="Нужен Python, цель — мини-чат.", status="running")]
        self.memory = empty_memory()
        patcher = patch("llm_agent.rag_dialogue.OpenAI")
        self.openai = patcher.start()
        self.addCleanup(patcher.stop)
        self.create = self.openai.return_value.__enter__.return_value.chat.completions.create

    def prepare(self, delta, query="Самостоятельный вопрос", rewrite=True):
        response = completion()
        response.choices[0].message.content = json.dumps(
            dict(memory_delta=delta, standalone_query=query), ensure_ascii=False,
        )
        self.create.return_value = response
        return self.call(rewrite)

    def call(self, rewrite=True):
        return prepare_turn(self.turns[-1].user, self.memory, self.turns,
                            "test-token", "test-model", rewrite_enabled=rewrite)

    def test_adds_provenance_without_mutating_input_or_double_counting(self):
        result = self.prepare([change(), change("goal", None, "мини-чат", "цель — мини-чат")])
        entry = result["memory"]["constraints"]["language"]
        self.assertEqual(entry, dict(value="Python", quote="Python", turn=1))
        self.assertEqual(self.memory, empty_memory())
        self.assertEqual(result["rewrite"]["status"], "success")
        self.assertEqual(result["diagnostic"]["usage"]["total_tokens"], 130)
        self.assertIsNone(result["rewrite"]["usage"])
        self.assertEqual(result["rewrite"]["elapsed_seconds"], 0)
        self.assertEqual(self.openai.call_args.kwargs["max_retries"], 0)
        self.assertEqual(self.openai.call_args.kwargs["timeout"], 15)
        self.create.assert_called_once()
        validate_memory(result["memory"], self.turns)

    def test_memory_extraction_runs_when_rewrite_disabled(self):
        result = self.prepare([change()], query=None, rewrite=False)
        self.assertIsNone(result["rewrite"])
        self.assertEqual(result["memory"]["constraints"]["language"]["value"], "Python")
        self.create.assert_called_once()

    def test_delta_is_atomic_and_query_is_independent(self):
        result = self.prepare([change(), change(key="fake", quote="не было в вопросе")])
        self.assertEqual(result["memory"], empty_memory())
        self.assertEqual(result["diagnostic"]["status"], "fallback")
        self.assertEqual(result["rewrite"]["status"], "success")

    def test_invalid_query_does_not_discard_valid_memory(self):
        for query in (None, "", " " * 2, "x" * 1501, [], {}):
            with self.subTest(query=str(query)[:30]):
                result = self.prepare([change()], query=query)
                self.assertEqual(result["diagnostic"]["status"], "success")
                self.assertEqual(result["rewrite"]["status"], "fallback")
                self.assertEqual(result["rewrite"]["query"], self.turns[-1].user)

    def test_update_and_explicit_delete(self):
        self.memory = self.prepare([change()])["memory"]
        self.turns.append(SimpleNamespace(user="Теперь Java; отменяю ограничение языка.", status="running"))
        self.memory = self.prepare([change(value="Java", quote="Теперь Java")])["memory"]
        self.assertEqual(self.memory["constraints"]["language"]["turn"], 2)
        invalid = self.prepare([change(value=None, quote="", op="delete")])
        self.assertEqual(invalid["memory"], self.memory)
        result = self.prepare([change(value=None, quote="отменяю ограничение языка", op="delete")])
        self.assertEqual(result["memory"], empty_memory())

    def test_empty_delta_preserves_goal_and_rejects_quotes_from_older_turns(self):
        self.memory = self.prepare([change("goal", None, "мини-чат", "цель — мини-чат")])["memory"]
        self.turns.append(SimpleNamespace(user="А как устроена память?", status="running"))
        self.assertEqual(self.prepare([])["memory"], self.memory)
        self.assertEqual(self.prepare([change()])["memory"], self.memory)

    def test_failure_preserves_state_and_hides_error(self):
        self.memory = self.prepare([change()])["memory"]
        self.create.side_effect = RuntimeError("SECRET_TOKEN")
        result = self.call()
        self.assertEqual(result["memory"], self.memory)
        self.assertNotIn("SECRET_TOKEN", str(result))
        self.assertEqual(result["rewrite"]["query"], self.turns[-1].user)

    def test_rejects_truncated_and_invalid_json(self):
        for content, finish in (("{}", "length"), ("not json", "stop"),
                                ('{"memory_delta":[],"memory_delta":[]}', "stop")):
            response = completion()
            response.choices[0].message.content = content
            response.choices[0].finish_reason = finish
            self.create.return_value = response
            result = self.call()
            self.assertEqual(result["diagnostic"]["status"], "fallback")
            self.assertEqual(result["memory"], self.memory)

    def test_tail_is_bounded_and_does_not_include_document_quotes(self):
        self.turns = [SimpleNamespace(user=str(i) * 2000, status="completed",
                                     rag_answer=dict(answer="x" * 2000, quotes=["SECRET_QUOTE"]),
                                     answer="SECRET_RENDER") for i in range(5)] + self.turns
        self.prepare([])
        payload = json.loads(self.create.call_args.kwargs["messages"][-1]["content"])
        self.assertEqual(len(payload["recent_turns"]), 3)
        self.assertTrue(payload["recent_turns"][0]["user"].startswith("2"))
        self.assertTrue(all(len(turn["assistant"]) == 1500 for turn in payload["recent_turns"]))
        self.assertNotIn("SECRET", str(payload))

    def test_tail_preserves_clarification_within_assistant_budget(self):
        clarification = "Используем Python или Java?"
        self.turns.insert(0, SimpleNamespace(
            user="Помоги выбрать", status="completed",
            rag_answer=dict(answer="Не знаю. " * 300, clarification=clarification,
                            quotes=["SECRET_QUOTE"]),
        ))
        self.prepare([])
        payload = json.loads(self.create.call_args.kwargs["messages"][-1]["content"])
        assistant = payload["recent_turns"][0]["assistant"]
        self.assertEqual(len(assistant), 1500)
        self.assertTrue(assistant.endswith(clarification))
        self.assertNotIn("SECRET", str(payload))

    def test_saved_memory_rejects_bad_types_provenance_and_limits(self):
        valid = self.prepare([change()])["memory"]
        for field, value in (("turn", True), ("turn", 0), ("turn", 2),
                             ("quote", "Made up"), ("value", "x" * 501)):
            memory = deepcopy(valid)
            memory["constraints"]["language"][field] = value
            with self.subTest(field=field, value=str(value)[:20]), self.assertRaises(ValueError):
                validate_memory(memory, self.turns)
        memory = empty_memory()
        memory["terms"] = {str(i): deepcopy(valid["constraints"]["language"]) for i in range(21)}
        with self.assertRaises(ValueError):
            validate_memory(memory, self.turns)
        memory["terms"] = {str(i): dict(value="x" * 500, quote="Python", turn=1) for i in range(20)}
        memory["constraints"] = deepcopy(memory["terms"])
        with self.assertRaises(ValueError):
            validate_memory(memory, self.turns)

    def test_saved_memory_accepts_dict_turns_and_formats_unicode(self):
        memory = self.prepare([change()])["memory"]
        validate_memory(memory, [dict(user=self.turns[0].user)])
        formatted = format_memory(memory)
        self.assertIn("Память задачи RAG", formatted)
        self.assertEqual(json.loads(formatted.split("\n", 1)[1]), memory)


if __name__ == "__main__":
    unittest.main()
