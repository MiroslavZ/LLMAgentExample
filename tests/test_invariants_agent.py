"""Границы проверки инвариантов с подставной моделью, без запросов к API.

Тесты проверяют порядок проверок, отказ и сохранность состояния, а не
семантическую точность настоящей модели, оценивающей текстовые правила.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from llm_agent.agent import Agent, FACTS_SYSTEM, SUMMARY_SYSTEM
from llm_agent.history import DialogueUsage, HistoryManager, TokenUsage
from llm_agent.invariants import Invariant, InvariantSet
from llm_agent.memory import MemorySnapshot
from llm_agent.profile import UserProfile
from llm_agent.task_state import CONTINUE_TASK, TaskState
from tests.helpers import completion


def response(text, *, missing=False, finish_reason="stop"):
    result = completion(missing=missing)
    result.choices[0].message.content = text
    result.choices[0].finish_reason = finish_reason
    return result


def verdict(status="pass", *, rule_id="stack", reason="Проверено", **kwargs):
    return response(json.dumps({"checks": [
        {"id": rule_id, "status": status, "reason": reason},
    ]}, ensure_ascii=False), **kwargs)


def task_reply(action, answer="Сохранённый результат", **values):
    return json.dumps({"answer": answer, "action": action, **values}, ensure_ascii=False)


class AgentInvariantTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "history.json"
        self.create = self.enterContext(patch("llm_agent.agent.OpenAI")).return_value.chat.completions.create
        self.rules = InvariantSet((Invariant("stack", "Использовать только Python"),))

    def agent(self, **kwargs):
        kwargs.setdefault("invariants", self.rules)
        agent = Agent("test", history_path=self.path, **kwargs)
        self.addCleanup(agent.close)
        return agent

    def review_input(self, call_index):
        messages = self.create.call_args_list[call_index].kwargs["messages"]
        return json.loads(messages[-1]["content"])

    def assert_usage(self, calls, *, missing=0):
        self.assertEqual(
            HistoryManager(self.path).get_usage(),
            DialogueUsage(100 * calls, 30 * calls, 130 * calls, missing),
        )

    def assert_refusal(self, result):
        self.assertTrue(result.refused)
        self.assertIn("stack", result.content)
        self.assertIn("Использовать только Python", result.content)

    def test_empty_rules_keep_one_request_without_invariant_instructions(self):
        self.create.return_value = response("Обычный ответ")
        result = self.agent(invariants=InvariantSet()).request("Вопрос")
        self.assertFalse(result.refused)
        self.assertEqual(result.content, "Обычный ответ")
        self.assertEqual(self.create.call_count, 1)
        self.assertNotIn("<invariants>", json.dumps(self.create.call_args.kwargs["messages"]))
        self.assert_usage(1)

    def test_refusal_drops_extra_choices_tool_calls_and_provider_reasoning(self):
        draft = response("Отклонённое решение")
        raw = draft.model_dump()
        raw["provider_details"] = "Скрытый запрещённый рецепт"
        raw["choices"][0]["message"]["reasoning_content"] = "Скрытый запрещённый рецепт"
        raw["choices"][0]["message"]["tool_calls"] = [{
            "id": "call", "type": "function",
            "function": {"name": "unsafe", "arguments": "Скрытый запрещённый рецепт"},
        }]
        raw["choices"].append({
            "index": 1, "finish_reason": "stop",
            "message": {"role": "assistant", "content": "Скрытый запрещённый рецепт"},
        })
        draft = type(draft).model_validate(raw)
        self.create.side_effect = [verdict(), draft, verdict("conflict")]
        result = self.agent().request("Предложи решение")
        self.assert_refusal(result)
        self.assertNotIn("Скрытый запрещённый рецепт", result.response.model_dump_json())
        self.assertNotIn("Отклонённое решение", result.response.model_dump_json())
        self.assertEqual(len(result.response.choices), 1)
        self.assertIsNone(result.response.choices[0].message.tool_calls)
        self.assertEqual(result.response.usage, draft.usage)
        self.assert_usage(3)

    def test_success_reviews_request_and_candidate_and_injects_rules_as_system(self):
        profile = UserProfile("developer", "Разработчик", style="Кратко")
        memory = MemorySnapshot(working={"project": "Учебный агент"})
        self.create.side_effect = [verdict(), response("def add(a, b): return a + b"), verdict()]
        result = self.agent(profile=profile, memory=memory).request("Напиши сложение")
        self.assertFalse(result.refused)
        self.assertEqual(result.content, "def add(a, b): return a + b")
        before, after = self.review_input(0), self.review_input(2)
        self.assertEqual(before["mode"], "request")
        self.assertIsNone(before["candidate"])
        self.assertEqual(after["mode"], "response")
        self.assertEqual(after["candidate"], result.content)
        for payload in (before, after):
            self.assertEqual(payload["user_message"], "Напиши сложение")
            self.assertEqual(payload["invariants"], self.rules.to_dict())
            self.assertEqual(payload["profile"], profile.to_dict())
            self.assertIn("Учебный агент", json.dumps(payload["memory"], ensure_ascii=False))
            self.assertIn("context", payload)
            self.assertIsNone(payload["task_state"])
            self.assertIsNone(payload["proposed_task_state"])
        main_messages = self.create.call_args_list[1].kwargs["messages"]
        invariant_messages = [item for item in main_messages if "<invariants>" in item["content"]]
        self.assertEqual(len(invariant_messages), 1)
        self.assertEqual(invariant_messages[0]["role"], "system")
        self.assertNotIn("<invariants>", self.path.read_text(encoding="utf-8"))
        self.assertEqual(len(HistoryManager(self.path).get_messages()), 2)
        self.assert_usage(3)

    def test_preflight_refusal_precedes_facts_and_summary_without_exposing_reason(self):
        for strategy in ("facts", "summary"):
            with self.subTest(strategy=strategy):
                history = HistoryManager(self.path)
                history.clear()
                history.add_exchange("Ранее", "Старый ответ", TokenUsage())
                history.start_task("Написать код")
                original = history.task_state
                self.create.reset_mock()
                self.create.side_effect = [verdict("conflict", reason="Скрытый запрещённый рецепт")]
                options = {"strategy": "facts", "window_size": 20} if strategy == "facts" else {
                    "last_messages": 0, "compress_every": 1,
                }
                result = self.agent(**options).request("Перепиши всё на Java")
                self.assert_refusal(result)
                self.assertEqual(self.create.call_count, 1)
                restored = HistoryManager(self.path)
                self.assertEqual(restored.task_state, original)
                self.assertEqual(restored.summary, "")
                self.assertFalse(restored.has_facts)
                self.assertNotIn("Скрытый запрещённый рецепт", self.path.read_text(encoding="utf-8"))
                self.assertEqual(restored.get_messages()[-1]["content"], result.content)
                self.assert_usage(1)

    def test_checker_must_provide_one_valid_completed_check_for_every_rule(self):
        good = {"id": "stack", "status": "pass", "reason": "Проверено"}
        invalid = [
            response("not json"), response("[]"), response('{"checks":[]}'),
            response(json.dumps({"checks": [good, good]})),
            verdict(rule_id="unknown"), verdict(status="unknown"),
            response(json.dumps({"checks": [{"id": "stack", "status": "pass"}]})),
            response(json.dumps({"checks": [{**good, "reason": 1}]})),
            verdict(finish_reason="length"), verdict("uncertain"),
        ]
        for index, check in enumerate(invalid):
            with self.subTest(case=index):
                HistoryManager(self.path).clear()
                self.create.reset_mock()
                self.create.side_effect = [check]
                result = self.agent().request("Запрос")
                self.assert_refusal(result)
                self.assertEqual(self.create.call_count, 1)
                self.assert_usage(1)

    def test_missing_check_for_second_rule_cannot_pass(self):
        rules = InvariantSet(self.rules.rules + (Invariant("budget", "Только бесплатные API"),))
        self.create.side_effect = [verdict()]
        result = self.agent(invariants=rules).request("Запрос")
        self.assertTrue(result.refused)
        self.assertEqual(self.create.call_count, 1)
        self.assert_usage(1)

    def test_postcheck_rejects_hidden_plan_and_preserves_each_stage(self):
        planning = TaskState("Написать функцию")
        _, execution = planning.apply_reply("Только Python", task_reply("plan", plan=["Написать код"]))
        _, validation = execution.apply_reply(CONTINUE_TASK, task_reply("complete_step", "def add(a, b): return a + b"))
        for state, candidate in (
            (planning, task_reply("plan", "План подготовлен", plan=["Создать проект Java"])),
            (execution, task_reply("complete_step", "Запрещённый результат на Java")),
            (validation, task_reply("finish", "Запрещённый итог на Java")),
        ):
            with self.subTest(stage=state.stage):
                history = HistoryManager(self.path)
                history.clear()
                history.add_exchange("Ранее", "Сохранено", TokenUsage(), task_state=state)
                self.create.reset_mock()
                self.create.side_effect = [verdict(), response(candidate), verdict("conflict")]
                result = self.agent().request("Продолжи")
                self.assert_refusal(result)
                self.assertEqual(HistoryManager(self.path).task_state, state)
                reviewed = self.review_input(2)
                self.assertEqual(reviewed["candidate"], candidate)
                self.assertEqual(reviewed["task_state"], state.to_dict())
                self.assertIsNotNone(reviewed["proposed_task_state"])
                stored = self.path.read_text(encoding="utf-8")
                for forbidden in ("Создать проект Java", "Запрещённый результат", "Запрещённый итог"):
                    self.assertNotIn(forbidden, stored)
                self.assert_usage(3)

    def test_postcheck_exception_keeps_candidate_out_of_history_and_counts_generation(self):
        history = HistoryManager(self.path)
        history.start_task("Задача")
        original = history.task_state
        self.create.side_effect = [
            verdict(), response(task_reply("plan", "Непроверенный ответ", plan=["Шаг"])),
            RuntimeError("Проверка недоступна"),
        ]
        with self.assertRaisesRegex(RuntimeError, "Проверка недоступна"):
            self.agent().request("Запрос")
        restored = HistoryManager(self.path)
        self.assertEqual(restored.task_state, original)
        self.assertEqual(restored.get_messages(), [])
        self.assert_usage(2)

    def test_generation_exception_does_not_save_request_but_counts_preflight(self):
        self.create.side_effect = [verdict(), RuntimeError("Сеть")]
        with self.assertRaisesRegex(RuntimeError, "Сеть"):
            self.agent().request("Запрос")
        self.assertEqual(HistoryManager(self.path).get_messages(), [])
        self.assert_usage(1)

    def test_invalid_postcheck_never_exposes_candidate_through_result_or_history(self):
        for check in (response("not json"), verdict(finish_reason="length"), verdict("uncertain")):
            with self.subTest(check=check.choices[0].message.content):
                HistoryManager(self.path).clear()
                self.create.side_effect = [verdict(), response("Непроверенное решение"), check]
                result = self.agent().request("Запрос")
                self.assert_refusal(result)
                self.assertNotIn("Непроверенное решение", result.response.model_dump_json())
                self.assertNotIn("Непроверенное решение", self.path.read_text(encoding="utf-8"))
                self.assert_usage(3)

    def test_empty_or_incomplete_candidate_is_refused_before_postcheck(self):
        no_choices = response("")
        no_choices.choices = []
        for candidate in (no_choices, response("  "), response("Незавершённый рецепт", finish_reason="length")):
            with self.subTest(candidate=candidate):
                history = HistoryManager(self.path)
                history.clear()
                history.start_task("Задача")
                self.create.reset_mock()
                self.create.side_effect = [verdict(), candidate]
                result = self.agent().request("Запрос")
                self.assert_refusal(result)
                self.assertEqual(self.create.call_count, 2)
                self.assertEqual(HistoryManager(self.path).task_state, TaskState("Задача"))
                self.assertNotIn("Незавершённый рецепт", result.response.model_dump_json())
                self.assert_usage(2)

    def test_missing_usage_is_counted_for_each_completed_call_without_synthetic_response(self):
        self.create.side_effect = [verdict(missing=True), response("Кандидат", missing=True), verdict("conflict")]
        self.assert_refusal(self.agent().request("Запрос"))
        self.assert_usage(1, missing=2)

    def test_preflight_and_first_candidate_refusal_stop_meta_pipeline(self):
        cases = (
            [verdict("conflict")],
            [verdict(), response("Запрещённый мета-промпт"), verdict("conflict")],
        )
        for responses in cases:
            with self.subTest(calls=len(responses)):
                HistoryManager(self.path).clear()
                self.create.reset_mock()
                self.create.side_effect = responses
                meta, result = self.agent().request_with_meta_prompt("Запрос")
                self.assertIs(meta, result)
                self.assert_refusal(result)
                self.assertEqual(self.create.call_count, len(responses))
                self.assertNotIn("Запрещённый мета-промпт", self.path.read_text(encoding="utf-8"))
                self.assert_usage(len(responses))

    def test_successful_meta_pipeline_reviews_both_candidates_with_one_preflight(self):
        self.create.side_effect = [verdict(), response("Составь Python-функцию"), verdict(), response("def add(a, b): return a + b"), verdict()]
        meta, result = self.agent().request_with_meta_prompt("Сложение")
        self.assertFalse(meta.refused)
        self.assertFalse(result.refused)
        self.assertEqual(self.create.call_count, 5)
        self.assertEqual([self.review_input(index)["mode"] for index in (0, 2, 4)], ["request", "response", "response"])
        self.assertEqual(self.review_input(2)["candidate"], meta.content)
        self.assertEqual(self.review_input(4)["candidate"], result.content)
        self.assertEqual(len(HistoryManager(self.path).get_messages()), 4)
        self.assert_usage(5)

    def test_servicing_context_does_not_receive_invariants_after_successful_preflight(self):
        for strategy, service_response in (("facts", "{}"), ("summary", "Сжатая история")):
            with self.subTest(strategy=strategy):
                history = HistoryManager(self.path)
                history.clear()
                history.add_exchange("Ранее", "Старый ответ", TokenUsage())
                self.create.reset_mock()
                self.create.side_effect = [verdict(), response(service_response), response("Разрешённый ответ"), verdict()]
                options = {"strategy": "facts", "window_size": 20} if strategy == "facts" else {
                    "last_messages": 0, "compress_every": 1,
                }
                result = self.agent(**options).request("Запрос")
                self.assertFalse(result.refused)
                messages = self.create.call_args_list[1].kwargs["messages"]
                self.assertEqual(messages[0]["content"], FACTS_SYSTEM if strategy == "facts" else SUMMARY_SYSTEM)
                self.assertNotIn("<invariants>", json.dumps(messages))
                self.assert_usage(4)

    def test_restart_after_refusal_continues_same_task_with_window_one(self):
        HistoryManager(self.path).start_task("Написать функцию")
        self.create.side_effect = [verdict("conflict")]
        self.assert_refusal(self.agent(strategy="window", window_size=1).request("Используй Java"))
        self.create.reset_mock()
        candidate = task_reply("plan", "План на Python", plan=["Реализовать функцию на Python"])
        self.create.side_effect = [verdict(), response(candidate), verdict()]
        result = self.agent(strategy="window", window_size=1).request("Используй Python")
        self.assertFalse(result.refused)
        self.assertEqual(self.review_input(0)["task_state"], TaskState("Написать функцию").to_dict())
        self.assertEqual(HistoryManager(self.path).task_state.step, 0)
        self.assertEqual(HistoryManager(self.path).task_state.plan, ("Реализовать функцию на Python",))
        self.assert_usage(4)


if __name__ == "__main__":
    unittest.main()
