"""Одна повторная генерация при нарушении протокола уже принятой задачи.

Локальная фабрика воспроизводит противоречие между execution в состоянии
и старым упоминанием planning в истории, без пользовательских файлов и API.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from llm_agent.agent import Agent
from llm_agent.history import DialogueUsage, HistoryManager, TokenUsage
from llm_agent.invariants import Invariant, InvariantSet
from llm_agent.task_state import (
    CONTINUE_TASK, STAGE_ACTIONS, TaskResponseError, TaskStage, TaskState,
)
from tests.helpers import completion


def reply_text(content):
    result = completion()
    result.choices[0].message.content = content
    return result


def task_reply(action, answer="Результат", **fields):
    return reply_text(json.dumps({"answer": answer, "action": action, **fields}, ensure_ascii=False))


def verdict(status="pass"):
    return reply_text(json.dumps({"checks": [{
        "id": "stack", "status": status, "reason": "Результат проверки",
    }]}, ensure_ascii=False))


class TaskProtocolRetryTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "history.json"
        self.create = self.enterContext(patch("llm_agent.agent.OpenAI")).return_value.chat.completions.create
        self.initial = TaskState(
            "Закончить метод расстояния Левенштейна",
            stage=TaskStage.EXECUTION,
            plan=(
                "Зафиксировать C# и полную таблицу динамического программирования",
                "Заполнить базовые случаи",
                "Реализовать переход динамического программирования",
                "Вернуть итоговое расстояние",
                "Проверить результат чтением кода",
            ),
            notes=("Уточнение: этап planning; язык пока не указан, предложи план.",),
        )
        self.reset_history()

    def reset_history(self):
        history = HistoryManager(self.path)
        history.clear()
        history.add_exchange(
            "public static int LevenshteinDistance(string first, string second)",
            "Сейчас выполняю текущий шаг — предлагаю план.",
            TokenUsage(), task_state=self.initial,
        )
        self.original_messages = history.get_messages()

    def agent(self, **options):
        agent = Agent("test", history_path=self.path, **options)
        self.addCleanup(agent.close)
        return agent

    def rejected_plan(self):
        return task_reply("plan", "Непринятое предложение", plan=["Подменённый план"])

    def retry_payload(self, index=1):
        return json.loads(self.create.call_args_list[index].kwargs["messages"][-1]["content"])

    def assert_unchanged(self, calls):
        restored = HistoryManager(self.path)
        self.assertEqual(restored.task_state, self.initial)
        self.assertEqual(restored.get_messages(), self.original_messages)
        self.assertEqual(restored.get_usage(), DialogueUsage(100 * calls, 30 * calls, 130 * calls))

    def test_repeated_plan_is_regenerated_for_current_step_without_replacing_plan(self):
        rejected = self.rejected_plan()
        completed = "Подтверждены C# и полная DP-таблица."
        self.create.side_effect = [rejected, task_reply("complete_step", completed)]
        result = self.agent().request(CONTINUE_TASK)

        self.assertEqual(self.create.call_count, 2)
        self.assertEqual(result.content, completed)
        restored = HistoryManager(self.path)
        self.assertEqual(restored.task_state.stage, TaskStage.EXECUTION)
        self.assertEqual(restored.task_state.step, 1)
        self.assertEqual(restored.task_state.plan, self.initial.plan)
        self.assertEqual(restored.task_state.results, (completed,))
        self.assertEqual(restored.task_state.notes, self.initial.notes)
        self.assertEqual(len(restored.get_messages()), len(self.original_messages) + 2)
        self.assertEqual(restored.get_usage(), DialogueUsage(200, 60, 260))
        payload = self.retry_payload()
        self.assertEqual(payload["candidate"], rejected.choices[0].message.content)
        self.assertIn("plan", payload["validation_error"])
        state = payload["current_state"]
        for field, value in self.initial.to_dict().items():
            self.assertEqual(state[field], value)
        self.assertEqual(state["current_step"], self.initial.current_step)
        self.assertEqual(state["allowed_actions"], list(STAGE_ACTIONS[TaskStage.EXECUTION]))
        self.assertNotIn("Подменённый план", self.path.read_text(encoding="utf-8"))
        self.assertNotIn("Непринятое предложение", result.response.model_dump_json())

    def test_retry_preserves_latest_user_clarification_and_saves_it_once(self):
        self.create.side_effect = [self.rejected_plan(), task_reply("complete_step", "Выбран C# и полная DP-таблица.")]
        self.agent().request("Язык C#")

        messages = self.create.call_args_list[1].kwargs["messages"]
        self.assertIn({"role": "user", "content": "Язык C#"}, messages)
        self.assertEqual(messages[-1]["role"], "user")
        self.assertEqual(self.retry_payload()["current_state"]["step"], 0)
        restored = HistoryManager(self.path)
        self.assertEqual(restored.task_state.notes, self.initial.notes + ("Пользователь: Язык C#",))
        self.assertEqual(sum(item["content"] == "Язык C#" for item in restored.get_messages()), 1)

    def test_second_plan_failure_stops_after_two_generations_without_publishing_proposal(self):
        self.create.side_effect = [self.rejected_plan(), self.rejected_plan()]
        with self.assertRaises(TaskResponseError) as raised:
            self.agent().request(CONTINUE_TASK)
        self.assertEqual(self.create.call_count, 2)
        self.assertIn("plan", str(raised.exception))
        self.assertNotIn("Непринятое предложение", str(raised.exception))
        self.assertNotIn("Подменённый план", self.path.read_text(encoding="utf-8"))
        self.assert_unchanged(2)

    def test_json_and_action_errors_share_one_retry_budget_in_both_orders(self):
        invalid_json = reply_text('{"answer": "Незавершённая строка')
        for responses in (
            [invalid_json, self.rejected_plan()],
            [self.rejected_plan(), invalid_json],
        ):
            with self.subTest(first=responses[0].choices[0].message.content):
                self.reset_history()
                self.create.reset_mock()
                self.create.side_effect = responses
                with self.assertRaises(TaskResponseError):
                    self.agent().request(CONTINUE_TASK)
                self.assertEqual(self.create.call_count, 2)
                self.assert_unchanged(2)

    def test_schema_error_can_retry_but_does_not_grant_another_attempt(self):
        malformed = reply_text('{"answer":42,"action":"complete_step"}')
        for success, final in (
            (True, task_reply("complete_step", "Подтверждены C# и DP-таблица.")),
            (False, self.rejected_plan()),
        ):
            with self.subTest(success=success):
                self.reset_history()
                self.create.reset_mock()
                self.create.side_effect = [malformed, final]
                if success:
                    self.agent().request(CONTINUE_TASK)
                    self.assertEqual(HistoryManager(self.path).task_state.step, 1)
                else:
                    with self.assertRaises(TaskResponseError):
                        self.agent().request(CONTINUE_TASK)
                    self.assert_unchanged(2)
                self.assertEqual(self.create.call_count, 2)

    def test_invariant_postcheck_blocks_unsafe_corrected_response_and_its_state(self):
        rules = InvariantSet((Invariant("stack", "Использовать только C#"),))
        unsafe = task_reply("complete_step", "Запрещённое решение: print('Python')")
        self.create.side_effect = [verdict(), self.rejected_plan(), unsafe, verdict("conflict")]
        result = self.agent(invariants=rules).request("Язык C#")

        self.assertEqual(self.create.call_count, 4)
        self.assertTrue(result.refused)
        check_messages = self.create.call_args_list[3].kwargs["messages"]
        checked = json.loads(check_messages[-1]["content"])
        self.assertEqual(checked["candidate"], unsafe.choices[0].message.content)
        self.assertEqual(checked["task_state"], self.initial.to_dict())
        self.assertEqual(checked["proposed_task_state"]["step"], 1)
        self.assertEqual(self.retry_payload(2)["current_state"]["step"], 0)
        restored = HistoryManager(self.path)
        self.assertEqual(restored.task_state, self.initial)
        self.assertEqual(restored.get_usage(), DialogueUsage(400, 120, 520))
        self.assertEqual(restored.get_messages()[-1]["content"], result.content)
        self.assertNotIn("Запрещённое решение", self.path.read_text(encoding="utf-8"))
        self.assertNotIn("Запрещённое решение", result.response.model_dump_json())
        self.assertNotIn("Подменённый план", self.path.read_text(encoding="utf-8"))

    def test_successful_retry_with_invariants_has_exactly_four_calls(self):
        rules = InvariantSet((Invariant("stack", "Использовать только C#"),))
        self.create.side_effect = [
            verdict(), self.rejected_plan(),
            task_reply("complete_step", "Выбран C# и полная DP-таблица."), verdict(),
        ]
        result = self.agent(invariants=rules).request(CONTINUE_TASK)
        self.assertFalse(result.refused)
        self.assertEqual(self.create.call_count, 4)
        self.assertEqual(HistoryManager(self.path).task_state.step, 1)
        self.assertEqual(HistoryManager(self.path).get_usage(), DialogueUsage(400, 120, 520))
        for index, mode in ((0, "request"), (3, "response")):
            messages = self.create.call_args_list[index].kwargs["messages"]
            self.assertEqual(json.loads(messages[-1]["content"])["mode"], mode)

    def test_explicit_replan_keeps_its_meaning_and_does_not_install_rejected_plan(self):
        for with_retry in (False, True):
            with self.subTest(with_retry=with_retry):
                self.reset_history()
                self.create.reset_mock()
                responses = [self.rejected_plan()] if with_retry else []
                responses.append(task_reply("replan", "Нужно пересмотреть требования к памяти."))
                self.create.side_effect = responses
                self.agent().request(CONTINUE_TASK)
                self.assertEqual(self.create.call_count, len(responses))
                state = HistoryManager(self.path).task_state
                self.assertEqual(state.stage, TaskStage.PLANNING)
                self.assertEqual(state.plan, ())
                self.assertEqual(state.results, ())
                self.assertEqual(state.step, 0)
                self.assertIn(self.initial.plan[0], "\n".join(state.notes))
                self.assertNotIn("Подменённый план", "\n".join(state.notes))

    def test_retry_api_error_preserves_original_state_and_first_generation_usage(self):
        self.create.side_effect = [self.rejected_plan(), RuntimeError("retry unavailable")]
        with self.assertRaisesRegex(RuntimeError, "retry unavailable"):
            self.agent().request("Язык C#")
        self.assertEqual(self.create.call_count, 2)
        self.assert_unchanged(1)


if __name__ == "__main__":
    unittest.main()
