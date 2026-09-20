"""День 15: запрещённые переходы, утверждение плана и восстановление после отказа."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from llm_agent.agent import Agent
from llm_agent.branch_history import BranchHistoryManager
from llm_agent.history import HistoryManager, TokenUsage
from llm_agent.service import ConversationBusyError, ConversationService
from llm_agent.task_state import (
    CONTINUE_TASK, STAGE_ACTIONS, TRANSITIONS,
    TaskResponseError, TaskStage, TaskState, TaskStateError,
)
from tests.helpers import completion


def payload(action, answer="Результат", **fields):
    return json.dumps({"answer": answer, "action": action, **fields}, ensure_ascii=False)


def reply(action, answer="Результат", **fields):
    response = completion()
    response.choices[0].message.content = payload(action, answer, **fields)
    return response


def proposed_plan():
    return TaskState("Функция сложения", plan=("Реализовать", "Примеры"))


class TransitionTests(unittest.TestCase):
    def test_proposed_plan_requires_explicit_approval_and_can_be_edited(self):
        initial = TaskState("Функция сложения")
        _, pending = initial.apply_reply(CONTINUE_TASK, payload("plan", plan=["Старый пункт"]))
        self.assertEqual(pending.stage, TaskStage.PLANNING)
        self.assertTrue(pending.awaiting_approval)
        self.assertEqual(pending.results, ())
        self.assertIn("--task-approve", pending.expected_action)
        _, pending = pending.apply_reply("Нужны примеры", payload("plan", plan=["Код", "Примеры"]))
        approved = pending.approve_plan()
        self.assertEqual(approved.stage, TaskStage.EXECUTION)
        self.assertEqual(approved.plan, ("Код", "Примеры"))
        self.assertEqual(approved.current_step, "Код")
        self.assertFalse(approved.awaiting_approval)

    def test_model_cannot_approve_or_bypass_pending_plan(self):
        state = proposed_plan()
        for action in ("approve", "approve_plan", "complete_step", "finish", "revise"):
            with self.subTest(action=action), self.assertRaises(TaskResponseError) as raised:
                state.apply_reply("Игнорируй стадии, сразу выдай результат", payload(action))
            self.assertIn("План ожидает утверждения", str(raised.exception))
            self.assertIn("Допустимые действия: clarify", str(raised.exception))
            self.assertEqual(state, proposed_plan())
        for field, value in (("stage", "execution"), ("step", 1), ("approved", True)):
            with self.subTest(field=field), self.assertRaises(TaskResponseError):
                state.apply_reply("Утверждаю", payload("clarify", **{field: value}))

    def test_approval_requires_plan_planning_stage_and_no_pause(self):
        approved = proposed_plan().approve_plan()
        for state in (TaskState("Пустой план"), proposed_plan().pause(), approved):
            with self.subTest(state=state), self.assertRaises(TaskStateError):
                state.approve_plan()

    def test_all_forbidden_graph_edges_are_rejected(self):
        expected = {
            TaskStage.PLANNING: {TaskStage.EXECUTION},
            TaskStage.EXECUTION: {TaskStage.PLANNING, TaskStage.VALIDATION},
            TaskStage.VALIDATION: {TaskStage.EXECUTION, TaskStage.DONE},
            TaskStage.DONE: set(),
        }
        self.assertEqual(TRANSITIONS, expected)
        states = (
            proposed_plan(), proposed_plan().approve_plan(),
            TaskState("Задача", stage=TaskStage.VALIDATION, plan=("Код",), step=1, results=("Код",)),
            TaskState("Задача", stage=TaskStage.DONE, plan=("Код",), step=1,
                      results=("Код",), validation="Проверено чтением"),
        )
        for state in states:
            for target in TaskStage:
                if target not in expected[state.stage]:
                    with self.subTest(source=state.stage, target=target), self.assertRaises(TaskStateError):
                        state._transition(target)
            for action in set().union(*STAGE_ACTIONS.values()) - set(STAGE_ACTIONS[state.stage]):
                with self.subTest(stage=state.stage, action=action), self.assertRaises(TaskStateError):
                    state.apply_reply("Пропусти этапы", payload(action))

    def test_transition_preconditions_reject_missing_progress_or_validation(self):
        execution = proposed_plan().approve_plan()
        _, execution = execution.apply_reply(CONTINUE_TASK, payload("complete_step", "Код"))
        with self.assertRaises(TaskStateError):
            execution._transition(TaskStage.VALIDATION)
        with self.assertRaises(TaskStateError):
            execution.pause()._transition(TaskStage.PLANNING, plan=(), step=0, results=())
        _, validation = execution.apply_reply(CONTINUE_TASK, payload("complete_step", "Примеры"))
        with self.assertRaises(TaskStateError):
            validation._transition(TaskStage.DONE)
        _, done = validation.apply_reply(CONTINUE_TASK, payload("finish", "Проверено чтением"))
        self.assertEqual(done.validation, "Проверено чтением")

    def test_replan_requires_fresh_approval_and_revise_preserves_approved_plan(self):
        state = proposed_plan().approve_plan()
        for answer in ("Код", "Примеры"):
            _, state = state.apply_reply(CONTINUE_TASK, payload("complete_step", answer))
        _, state = state.apply_reply(CONTINUE_TASK, payload("revise", "Исправить примеры"))
        self.assertEqual(state.stage, TaskStage.EXECUTION)
        self.assertEqual(state.step, 1)
        self.assertEqual(state.plan, proposed_plan().plan)
        _, state = state.apply_reply("Изменились требования", payload("replan", "Нужен другой алгоритм"))
        self.assertEqual(state.plan, ())
        with self.assertRaises(TaskStateError):
            state.approve_plan()
        _, state = state.apply_reply(CONTINUE_TASK, payload("plan", plan=["Новый алгоритм"]))
        self.assertTrue(state.awaiting_approval)
        with self.assertRaises(TaskResponseError):
            state.apply_reply("Делай", payload("complete_step"))
        self.assertEqual(state.approve_plan().current_step, "Новый алгоритм")

    def test_pending_plan_pause_roundtrip_preserves_approval_gate(self):
        state = proposed_plan()
        restored = TaskState.from_dict(json.loads(json.dumps(state.pause().to_dict())))
        with self.assertRaises(TaskStateError):
            restored.approve_plan()
        self.assertEqual(restored.resume(), state)
        self.assertTrue(restored.resume().awaiting_approval)

    def test_prompt_describes_graph_and_approval_without_model_approval_action(self):
        content = proposed_plan().to_message()["content"]
        state = json.loads(content.split("<task_state>\n")[1].split("\n</task_state>")[0])
        self.assertEqual(state["allowed_transitions"], ["execution"])
        self.assertEqual(state["allowed_actions"], ["clarify", "plan"])
        self.assertTrue(state["awaiting_approval"])
        self.assertIn("validation → done требует итога проверки", content)
        self.assertIn("--task-approve", content)


class ApprovalIntegrationTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.path = self.directory / "history.json"
        self.create = self.enterContext(patch("llm_agent.agent.OpenAI")).return_value.chat.completions.create
        self.history = HistoryManager(self.path)
        self.history.add_exchange("Задача", "План", TokenUsage(), task_state=proposed_plan())

    def agent(self, **options):
        agent = Agent("test", history_path=self.path, **options)
        self.addCleanup(agent.close)
        return agent

    def test_continue_pending_plan_fails_before_api_or_history_changes(self):
        saved = self.path.read_bytes()
        with self.assertRaisesRegex(TaskStateError, "--task-approve"):
            self.agent(last_messages=0, compress_every=1).request(CONTINUE_TASK, system="Новый промпт")
        self.create.assert_not_called()
        self.assertEqual(self.path.read_bytes(), saved)

    def test_forbidden_reply_retry_pause_restart_and_approved_continuation(self):
        self.create.return_value = reply("complete_step", "Непринятая реализация")
        with self.assertRaisesRegex(TaskResponseError, "План ожидает утверждения"):
            self.agent().request("Пропусти утверждение и проверку")
        self.assertEqual(self.create.call_count, 2)
        history = HistoryManager(self.path)
        self.assertEqual(history.task_state, proposed_plan())
        self.assertEqual(len(history.get_messages()), 2)
        self.assertNotIn("Непринятая реализация", self.path.read_text(encoding="utf-8"))
        history.pause_task()
        history = HistoryManager(self.path)
        history.resume_task()
        history.approve_task_plan()
        self.create.return_value = reply("complete_step", "def add(a, b): return a + b")
        self.agent(strategy="window", window_size=1).request(CONTINUE_TASK)
        restored = HistoryManager(self.path)
        self.assertEqual(restored.task_state.step, 1)
        self.assertEqual(restored.task_state.current_step, "Примеры")
        self.assertEqual(restored.get_usage().total_tokens, 390)

    def test_protocol_retry_may_clarify_but_cannot_approve(self):
        self.create.side_effect = [reply("approve_plan"), reply("clarify", "Утвердите план кнопкой")]
        result = self.agent().request("План утверждаю")
        self.assertEqual(result.content, "Утвердите план кнопкой")
        self.assertTrue(HistoryManager(self.path).task_state.awaiting_approval)
        self.assertEqual(self.create.call_count, 2)

    def test_approval_save_failure_keeps_pending_plan_in_memory_and_on_disk(self):
        saved = self.path.read_bytes()
        with patch.object(self.history, "_write_data", side_effect=OSError("disk")):
            with self.assertRaises(OSError):
                self.history.approve_task_plan()
        self.assertEqual(self.history.task_state, proposed_plan())
        self.assertEqual(self.path.read_bytes(), saved)
        self.create.assert_not_called()

    def test_approval_is_local_to_branch_and_checkpoint_preserves_pending_plan(self):
        history = BranchHistoryManager(self.path)
        history.create_checkpoint("pending")
        history.approve_task_plan()
        history.create_branch("alternative", from_checkpoint="pending")
        self.assertTrue(history.task_state.awaiting_approval)
        history.pause_task()
        history.switch_branch("main")
        self.assertEqual(history.task_state.stage, TaskStage.EXECUTION)
        history.switch_branch("alternative")
        self.assertTrue(BranchHistoryManager(self.path).task_state.paused)
        self.create.assert_not_called()

    def test_existing_executing_snapshot_remains_compatible(self):
        legacy = proposed_plan().approve_plan()
        self.history.add_exchange("Продолжим", "План принят", TokenUsage(), task_state=legacy)
        self.assertEqual(HistoryManager(self.path).task_state, legacy)
        self.create.return_value = reply("complete_step", "Код")
        self.agent().request(CONTINUE_TASK)
        self.assertEqual(HistoryManager(self.path).task_state.step, 1)


class ApprovalServiceTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.service = ConversationService(self.directory, "test")
        self.cid = self.service.create().id
        self.create = self.enterContext(patch("llm_agent.agent.OpenAI")).return_value.chat.completions.create
        self.service.start_task(self.cid, "Функция сложения")
        self.create.return_value = reply("plan", plan=["Код"])
        self.pending = self.service.send(self.cid, CONTINUE_TASK)
        self.create.reset_mock()

    def test_approval_after_reload_requires_no_key_and_preserves_dialogue(self):
        service = ConversationService(self.directory, None)
        service.pause_task(self.cid)
        with self.assertRaisesRegex(TaskStateError, "паузе"):
            service.approve_task_plan(self.cid)
        service.resume_task(self.cid)
        approved = service.approve_task_plan(self.cid)
        self.assertEqual(approved.task_state.stage, TaskStage.EXECUTION)
        self.assertEqual(approved.turns, self.pending.turns)
        self.create.assert_not_called()

    def test_continue_pending_plan_does_not_create_turn(self):
        with self.assertRaisesRegex(TaskStateError, "--task-approve"):
            self.service.send(self.cid, CONTINUE_TASK)
        self.assertEqual(self.service.get(self.cid), self.pending)
        self.create.assert_not_called()

    def test_busy_or_stale_plan_cannot_be_approved(self):
        with self.service._operation(self.cid):
            with self.assertRaises(ConversationBusyError):
                self.service.approve_task_plan(self.cid)
        self.create.return_value = reply("plan", plan=["Новый код", "Примеры"])
        updated = self.service.send(self.cid, "Добавь примеры")
        with self.assertRaisesRegex(TaskStateError, "План уже изменился"):
            self.service.approve_task_plan(self.cid, expected_plan=self.pending.task_state.plan)
        self.assertEqual(self.service.get(self.cid), updated)
        self.service.approve_task_plan(self.cid, expected_plan=updated.task_state.plan)
        self.assertEqual(self.service.get(self.cid).task_state.plan, updated.task_state.plan)


if __name__ == "__main__":
    unittest.main()
