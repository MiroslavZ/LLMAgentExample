import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rich.console import Console

from llm_agent.cli import main, parse_args, run
from llm_agent.service import ConversationService
from llm_agent.history import HistoryManager
from llm_agent.task_state import CONTINUE_TASK, TaskStage, TaskState
from tests.helpers import completion


class TaskCliTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.data_dir = self.directory / "conversations"
        self.service = ConversationService(self.data_dir, token=None)
        self.conversation = self.service.create()
        self.history = self.service.store.path(self.conversation.id)
        self.memory_db = self.data_dir / "memory.sqlite3"
        self.profiles_db = self.data_dir / "profiles.sqlite3"
        client = patch("llm_agent.agent.OpenAI")
        self.client = client.start()
        self.addCleanup(client.stop)
        self.create = self.client.return_value.chat.completions.create
        env = patch("llm_agent.cli.load_env")
        self.load_env = env.start()
        self.addCleanup(env.stop)

    def run_cli(self, *options, entrypoint=main):
        output = io.StringIO()
        arguments = [
            "main.py", "--history", str(self.history),
            "--data-dir", str(self.data_dir),
            "--invariants-file", str(self.directory / "invariants.json"), *options,
        ]
        with patch("sys.argv", arguments), patch(
            "llm_agent.cli.console", Console(file=output, width=240, color_system=None),
        ):
            entrypoint()
        return output.getvalue()

    def reply(self, action, answer="Ответ", **data):
        response = completion()
        response.choices[0].message.content = json.dumps(
            {"action": action, "answer": answer, **data}, ensure_ascii=False,
        )
        self.create.return_value = response

    def seed_task(self, state):
        fixture = self.directory / "fixture.json"
        history = HistoryManager(fixture)
        history.clear()
        if state is not None:
            history.add_exchange("Цель", "Ответ", None, task_state=state)
        conversation = self.service.get(self.conversation.id)
        conversation.working_context = json.loads(fixture.read_text(encoding="utf-8"))
        self.service.store.save(conversation)

    def state(self):
        return self.service.get(self.conversation.id).task_state

    def test_show_absent_task_does_not_load_environment(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertIsNone(json.loads(self.run_cli("--task-show")))
            self.assertIsNone(json.loads(self.run_cli("--task-show")))
        self.assertEqual(self.service.get(self.conversation.id).turns, [])
        self.load_env.assert_not_called()
        self.client.assert_not_called()

    def test_local_start_pause_resume_survive_restart_without_api(self):
        title = "Подготовить [доклад] " + "с подробным описанием " * 20
        with patch.dict("os.environ", {}, clear=True):
            self.run_cli("--task-start", title)
            initial = self.state()
            self.assertEqual(initial.stage, TaskStage.PLANNING)
            self.assertEqual(initial.title, title)
            self.assertEqual(json.loads(self.run_cli("--task-show")), initial.to_dict())
            self.run_cli("--task-pause")
            self.assertEqual(self.state(), initial.pause())
            self.run_cli("--task-resume")
            self.assertEqual(self.state(), initial)
        self.load_env.assert_not_called()
        self.client.assert_not_called()

    def test_start_with_user_executes_one_step_and_displays_expected_action(self):
        self.reply("plan", "План готов", plan=["Собрать материал", "Подготовить текст"])
        with patch.dict("os.environ", {"API_KEY": "test"}):
            output = self.run_cli("--task-start", "Доклад", "--user", "Для начинающих")
        self.create.assert_called_once()
        state = self.state()
        self.assertEqual(state.stage, TaskStage.PLANNING)
        self.assertTrue(state.awaiting_approval)
        self.assertEqual(state.step, 0)
        self.assertIn("План готов", output)
        self.assertIn(state.current_step, output)
        self.assertIn(state.expected_action, output)
        self.assertNotIn('"action":', output)

    def test_pause_at_each_stage_and_continue_without_repeating_task(self):
        self.run_cli("--task-start", "Подготовить объяснение генераторов")
        replies = (
            ("plan", "План готов", {"plan": ["Объяснить генераторы"]}),
            ("complete_step", "Генераторы возвращают значения по одному", {}),
            ("finish", "Объяснение проверено по плану", {}),
        )
        expected_stages = (
            TaskStage.PLANNING, TaskStage.EXECUTION, TaskStage.VALIDATION, TaskStage.DONE,
        )
        with patch.dict("os.environ", {"API_KEY": "test"}):
            for index, stage in enumerate(expected_stages):
                with self.subTest(stage=stage):
                    before = self.state()
                    self.assertEqual(before.stage, stage)
                    calls = self.create.call_count
                    self.run_cli("--task-pause")
                    self.assertEqual(self.state(), before.pause())
                    self.run_cli("--task-resume")
                    self.assertEqual(self.state(), before)
                    self.assertEqual(self.create.call_count, calls)
                    if index < len(replies):
                        action, answer, data = replies[index]
                        self.reply(action, answer, **data)
                        self.run_cli("--task-continue")
                        self.assertEqual(self.create.call_count, calls + 1)
                        sent = self.create.call_args.kwargs["messages"]
                        self.assertEqual(sent[-1]["content"], CONTINUE_TASK)
                        self.assertIn(before.title, json.dumps(sent, ensure_ascii=False))
                        if action == "plan":
                            proposed = self.state()
                            self.assertTrue(proposed.awaiting_approval)
                            self.run_cli("--task-pause")
                            self.assertEqual(self.state(), proposed.pause())
                            self.run_cli("--task-resume")
                            self.assertEqual(self.state(), proposed)
                            self.run_cli("--task-approve")
                            self.assertEqual(self.create.call_count, calls + 1)
        self.assertEqual(self.state().results, (replies[1][1],))
        self.assertEqual(self.state().notes, ())

    def test_resume_with_user_uses_preserved_task_step(self):
        self.run_cli("--task-start", "Доклад")
        self.run_cli("--task-pause")
        self.reply("plan", "План с учётом уточнения", plan=["Подготовить текст"])
        with patch.dict("os.environ", {"API_KEY": "test"}):
            self.run_cli("--task-resume", "--user", "Не больше пяти минут")
        self.assertEqual(self.state().stage, TaskStage.PLANNING)
        self.assertTrue(self.state().awaiting_approval)
        self.assertFalse(self.state().paused)
        self.assertIn("Пользователь: Не больше пяти минут", self.state().notes)
        self.create.assert_called_once()

    def test_continue_missing_or_paused_task_fails_before_loading_dependencies(self):
        for paused in (False, True):
            with self.subTest(paused=paused):
                if paused:
                    self.run_cli("--task-start", "Доклад")
                    self.run_cli("--task-pause")
                before = self.history.read_bytes() if self.history.exists() else None
                with self.assertRaises(SystemExit) as error:
                    self.run_cli("--task-continue", entrypoint=run)
                self.assertIn("--task-resume" if paused else "--task-start", str(error.exception))
                after = self.history.read_bytes() if self.history.exists() else None
                self.assertEqual(after, before)
        self.load_env.assert_not_called()
        self.client.assert_not_called()

    def test_continue_finished_task_fails_without_extra_request(self):
        self.run_cli("--task-start", "Доклад")
        with patch.dict("os.environ", {"API_KEY": "test"}):
            for action, data in (
                ("plan", {"plan": ["Подготовить текст"]}),
                ("complete_step", {}), ("finish", {}),
            ):
                self.reply(action, **data)
                self.run_cli("--task-continue")
                if action == "plan":
                    self.run_cli("--task-approve")
        self.create.reset_mock()
        self.load_env.reset_mock()
        with self.assertRaisesRegex(SystemExit, "завершена"):
            self.run_cli("--task-continue", entrypoint=run)
        self.create.assert_not_called()
        self.load_env.assert_not_called()

    def test_plan_approval_is_local_and_survives_restart(self):
        proposed = TaskState("Доклад", plan=("Подготовить текст",))
        self.seed_task(proposed)
        with patch.dict("os.environ", {}, clear=True):
            self.run_cli("--task-approve")
        self.assertEqual(self.state(), proposed.approve_plan())
        self.load_env.assert_not_called()
        self.client.assert_not_called()

    def test_continue_requires_approval_without_sending_request(self):
        proposed = TaskState("Доклад", plan=("Подготовить текст",))
        self.seed_task(proposed)
        before = self.history.read_bytes()
        with self.assertRaisesRegex(SystemExit, "--task-approve"):
            self.run_cli("--task-continue", entrypoint=run)
        self.assertEqual(self.history.read_bytes(), before)
        self.load_env.assert_not_called()
        self.client.assert_not_called()

    def test_approve_with_user_runs_first_step_after_explicit_approval(self):
        proposed = TaskState("Доклад", plan=("Подготовить текст",))
        self.seed_task(proposed)
        self.reply("complete_step", "Готовый текст")
        with patch.dict("os.environ", {"API_KEY": "test"}):
            self.run_cli("--task-approve", "--user", "Используй короткие предложения")
        self.assertEqual(self.state().stage, TaskStage.VALIDATION)
        self.assertEqual(self.state().results, ("Готовый текст",))
        self.create.assert_called_once()
        prompt = json.dumps(self.create.call_args.kwargs["messages"], ensure_ascii=False)
        self.assertIn('\\"stage\\": \\"execution\\"', prompt)

    def test_approve_missing_plan_paused_or_approved_task_preserves_state(self):
        proposed = TaskState("Доклад", plan=("Подготовить текст",))
        for state in (None, TaskState("Доклад"), proposed.pause(), proposed.approve_plan()):
            with self.subTest(state=state):
                self.seed_task(state)
                before = self.history.read_bytes()
                with self.assertRaises(SystemExit):
                    self.run_cli("--task-approve", entrypoint=run)
                self.assertEqual(self.history.read_bytes(), before)
        self.load_env.assert_not_called()
        self.client.assert_not_called()

    def test_show_remains_json_when_combined_with_other_local_edits(self):
        self.run_cli("--task-start", "Доклад")
        output = self.run_cli(
            "--task-show", "--memory-set", "working", "audience", "Начинающие",
            "--profile-clear",
        )
        self.assertEqual(json.loads(output), self.state().to_dict())
        self.client.assert_not_called()

    def test_invalid_model_reply_preserves_task(self):
        self.run_cli("--task-start", "Доклад")
        before = self.state()
        self.reply("finish", "Слишком раннее завершение")
        with patch.dict("os.environ", {"API_KEY": "test"}), self.assertRaises(SystemExit) as raised:
            self.run_cli(
                "--task-continue",
                entrypoint=run,
            )
        self.assertIn('Действие "finish" (завершить задачу) недопустимо', str(raised.exception))
        self.assertIn("Текущий этап: planning", str(raised.exception))
        restored = self.service.get(self.conversation.id)
        self.assertEqual(restored.task_state, before)
        self.assertEqual(restored.turns[-1].status, "error")
        messages, _, _, usage = HistoryManager._decode_data(restored.working_context)
        self.assertEqual(messages, [])
        self.assertEqual(usage.total_tokens, 260)

    def test_start_without_selector_creates_shared_conversation(self):
        output = io.StringIO()
        with patch.dict("os.environ", {}, clear=True), patch("sys.argv", [
            "main.py", "--data-dir", str(self.data_dir), "--task-start", "Новая задача",
        ]), patch("llm_agent.cli.console", Console(file=output)):
            main()
        conversations = self.service.list_conversations()
        self.assertEqual(len(conversations), 2)
        created = next(item for item in conversations if item.id != self.conversation.id)
        self.assertEqual(created.task_state.title, "Новая задача")
        self.load_env.assert_not_called()
        self.client.assert_not_called()

    def test_local_task_operations_require_conversation(self):
        for option in ("--task-show", "--task-pause", "--task-resume", "--task-approve", "--task-continue"):
            with self.subTest(option=option), patch("sys.argv", [
                "main.py", "--data-dir", str(self.data_dir), option,
            ]), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    parse_args()
                self.assertEqual(error.exception.code, 2)

    def test_invalid_combinations_fail_during_argument_parsing(self):
        for options in (
            ["--task-start", " "],
            ["--task-start", "Доклад", "--task-pause"],
            ["--task-pause", "--user", "Запрос"],
            ["--task-show", "--user", "Запрос"],
            ["--task-continue", "--user", "Запрос"],
            ["--task-resume", "--task-continue"],
            ["--task-approve", "--task-continue"],
            ["--task-approve", "--task-pause"],
            ["--task-show", "--memory-show"],
            ["--task-show", "--profile-show"],
            ["--task-show", "--profile-list"],
            ["--strategy", "branch", "--task-show", "--list-branches"],
            ["--task-start", "Доклад", "--system", "Правила"],
        ):
            with self.subTest(options=options), patch("sys.argv", ["main.py", *options]):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    parse_args()
                self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
