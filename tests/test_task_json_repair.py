"""Одна попытка исправления JSON без обхода автомата и инвариантов."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from llm_agent.agent import Agent, TASK_RESPONSE_RETRY_SYSTEM
from llm_agent.history import DialogueUsage, HistoryManager
from llm_agent.invariants import Invariant, InvariantSet
from llm_agent.service import ConversationService
from llm_agent.task_state import CONTINUE_TASK, TaskResponseError, TaskStage, TaskState
from tests.helpers import completion


BROKEN = '{"answer":"public string Get() { return "рубль"; }","action":"complete_step"}'
CODE = 'public string Get()\n{\n    return "рубль";\n}'


def response(content, *, finish_reason="stop", missing=False):
    result = completion(missing=missing)
    result.choices[0].message.content = content
    result.choices[0].finish_reason = finish_reason
    return result


def payload(action="complete_step", answer=CODE):
    return json.dumps({"answer": answer, "action": action}, ensure_ascii=False)


def check(status="pass"):
    return response(json.dumps({"checks": [{"id": "stack", "status": status, "reason": "Проверено"}]}))


class TaskJSONRepairTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.path = self.directory / "history.json"
        self.task = TaskState(
            title="Реализовать склонение слова рубль", stage=TaskStage.EXECUTION,
            step=2, plan=("Проектирование", "Алгоритм", "Реализация", "Примеры"),
            results=("Выбран Solver на C#", "Исключения 11–14, затем последняя цифра"),
        )
        history = HistoryManager(self.path)
        history.add_exchange("Старый запрос", "Старый ответ", None, task_state=self.task)
        self.create = self.enterContext(patch("llm_agent.agent.OpenAI")).return_value.chat.completions.create

    def agent(self, **options):
        agent = Agent("test", history_path=self.path, **options)
        self.addCleanup(agent.close)
        return agent

    def assert_unchanged(self):
        history = HistoryManager(self.path)
        self.assertEqual(history.task_state, self.task)
        self.assertNotIn(BROKEN, [message["content"] for message in history.get_messages()])

    def test_repairs_third_step_and_commits_once_with_feedback_and_original_context(self):
        self.create.side_effect = [response(BROKEN), response(payload())]
        result = self.agent().request(CONTINUE_TASK, max_tokens=800, temperature=1, stop_sequences=[";"])
        self.assertEqual(result.content, CODE)
        history = HistoryManager(self.path)
        self.assertEqual(history.task_state.step, 3)
        self.assertEqual(history.task_state.stage, TaskStage.EXECUTION)
        self.assertEqual(history.task_state.results, (*self.task.results, CODE))
        self.assertEqual(len(history.get_messages()), 4)
        self.assertEqual(history.get_usage(), DialogueUsage(200, 60, 260, 1))
        initial, repair = [call.kwargs for call in self.create.call_args_list]
        self.assertEqual(repair["messages"][:-2], initial["messages"])
        self.assertEqual(repair["messages"][-2]["content"], TASK_RESPONSE_RETRY_SYSTEM)
        repair_data = json.loads(repair["messages"][-1]["content"])
        self.assertEqual(repair_data["candidate"], BROKEN)
        self.assertIn("Expecting ',' delimiter", repair_data["validation_error"])
        self.assertNotIn("stop", repair)
        self.assertEqual(repair["temperature"], 0)
        self.assertEqual(repair["max_tokens"], 800)
        self.assertEqual(repair["response_format"], {"type": "json_object"})
        self.assertNotIn(TASK_RESPONSE_RETRY_SYSTEM, self.path.read_text(encoding="utf-8"))

    def test_second_bad_json_stops_with_actionable_error_and_counts_both_responses(self):
        self.create.return_value = response(BROKEN)
        with self.assertRaises(TaskResponseError) as raised:
            self.agent().request(CONTINUE_TASK)
        message = str(raised.exception)
        self.assertIn("Однократное автоматическое исправление JSON также не удалось", message)
        self.assertIn("Причина парсера:", message)
        self.assertIn("Текущий шаг: 3 из 4", message)
        self.assertNotIn("Повторите запрос", message)
        self.assertNotIn(BROKEN, message)
        self.assertEqual(self.create.call_count, 2)
        self.assertEqual(HistoryManager(self.path).get_usage().total_tokens, 260)
        self.assert_unchanged()

    def test_repair_api_error_preserves_state_and_first_response_usage(self):
        self.create.side_effect = [response(BROKEN), RuntimeError("network")]
        with self.assertRaisesRegex(RuntimeError, "network"):
            self.agent().request(CONTINUE_TASK)
        self.assertEqual(HistoryManager(self.path).get_usage().total_tokens, 130)
        self.assert_unchanged()

    def test_protocol_errors_retry_once_but_truncation_does_not(self):
        for candidate in (response(payload("finish")), response('{"answer":[],"action":"complete_step"}'),
                          response(payload(), finish_reason="length")):
            with self.subTest(candidate=candidate):
                self.create.reset_mock()
                self.create.return_value = candidate
                with self.assertRaises(TaskResponseError):
                    self.agent().request(CONTINUE_TASK)
                expected_calls = 1 if candidate.choices[0].finish_reason == "length" else 2
                self.assertEqual(self.create.call_count, expected_calls)
                self.assert_unchanged()

    def test_repaired_answer_cannot_skip_state_or_bypass_invariants(self):
        for repaired in (payload("finish"), payload()):
            with self.subTest(repaired=repaired):
                self.create.reset_mock()
                rules = InvariantSet((Invariant("stack", "Только Python"),))
                self.create.side_effect = [check(), response(BROKEN), response(repaired), check("conflict")]
                if repaired == payload("finish"):
                    with self.assertRaises(TaskResponseError):
                        self.agent(invariants=rules).request(CONTINUE_TASK)
                    self.assertEqual(self.create.call_count, 3)
                else:
                    result = self.agent(invariants=rules).request(CONTINUE_TASK)
                    self.assertTrue(result.refused)
                    self.assertNotIn(CODE, result.content)
                    candidate = json.loads(self.create.call_args.kwargs["messages"][-1]["content"])
                    self.assertEqual(candidate["candidate"], repaired)
                    self.assertEqual(candidate["proposed_task_state"]["step"], 3)
                    self.assertEqual(self.create.call_count, 4)
                self.assert_unchanged()

    def test_repair_missing_usage_and_truncation_do_not_publish_candidate(self):
        self.create.side_effect = [response(BROKEN, missing=True), response(payload(), finish_reason="length", missing=True)]
        with self.assertRaisesRegex(TaskResponseError, "finish_reason=length"):
            self.agent().request(CONTINUE_TASK)
        self.assertEqual(HistoryManager(self.path).get_usage().missing_responses, 3)
        self.assert_unchanged()

    def test_web_saves_one_successful_turn_and_only_verified_result(self):
        service = ConversationService(self.directory / "web", "test", invariants_path=self.directory / "rules.json")
        conversation = service.create()
        history = HistoryManager(self.path)
        conversation.working_context = json.loads(self.path.read_text(encoding="utf-8"))
        service.store.save(conversation)
        self.create.side_effect = [response(BROKEN), response(payload())]
        result = service.send(conversation.id, CONTINUE_TASK)
        self.assertEqual(len(result.turns), 1)
        self.assertEqual(result.turns[-1].status, "completed")
        self.assertEqual(result.turns[-1].answer, CODE)
        self.assertEqual(result.task_state.step, history.task_state.step + 1)
        restored = service.get(conversation.id)
        self.assertEqual(restored.task_state, result.task_state)
        self.assertNotIn(BROKEN, [m["content"] for m in restored.working_context["messages"]])


if __name__ == "__main__":
    unittest.main()
