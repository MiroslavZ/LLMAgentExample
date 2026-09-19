import contextlib
import io
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from rich.console import Console

from llm_agent.agent import Agent, META_PROMPT_SYSTEM
from llm_agent.branch_history import BranchHistoryManager
from llm_agent.history import DialogueUsage, HistoryManager, TokenUsage
from llm_agent.cli import main, parse_args
from tests.helpers import completion


class BranchTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "llm_agent.history.json"
        client = patch("llm_agent.agent.OpenAI")
        self.client = client.start()
        self.addCleanup(client.stop)
        self.create = self.client.return_value.chat.completions.create
        self.create.return_value = completion()

    def agent(self, branch=None):
        return Agent("test", self.path, strategy="branch", branch=branch)

    def run_cli(self, *options):
        output = io.StringIO()
        with patch("sys.argv", [
            "main.py", "--history", str(self.path),
            "--memory-db", str(self.path.with_suffix(".sqlite3")),
            "--invariants-file", str(self.path.with_name("invariants.json")), *options,
        ]), patch(
            "llm_agent.cli.console", Console(file=output, width=180, color_system=None),
        ):
            main()
        return output.getvalue()

    def test_two_branches_from_same_checkpoint_are_isolated_after_restart(self):
        agent = self.agent()
        self.assertEqual(agent.history.active_branch, "main")
        agent.request("Общая задача", system="Общие правила")
        shared = agent.history.get_messages()
        agent.history.create_checkpoint("start")
        agent.request("Только main")
        main_messages = agent.history.get_messages()

        agent.history.create_branch("a", from_checkpoint="start")
        result_a = agent.request("Решение A")
        self.assertEqual(self.create.call_args.kwargs["messages"], shared + [
            {"role": "user", "content": "Решение A"},
        ])
        messages_a = agent.history.get_messages()
        self.assertEqual(result_a.dialogue_usage, DialogueUsage(200, 60, 260))

        restarted = self.agent()
        self.assertEqual(restarted.history.active_branch, "a")
        restarted.history.create_branch("b", from_checkpoint="start")
        self.assertEqual(restarted.history.get_messages(), shared)
        self.create.return_value = completion(150, 40)
        result_b = restarted.request("Решение B")
        self.assertEqual(self.create.call_args.kwargs["messages"], shared + [
            {"role": "user", "content": "Решение B"},
        ])
        self.assertEqual(result_b.dialogue_usage, DialogueUsage(250, 70, 320))

        switched = self.agent("a")
        self.assertEqual(switched.history.get_messages(), messages_a)
        self.assertEqual(switched.history.get_usage(), result_a.dialogue_usage)
        self.assertEqual(self.agent().history.active_branch, "a")
        switched.request("Продолжение A")
        self.assertEqual(self.create.call_args.kwargs["messages"], messages_a + [
            {"role": "user", "content": "Продолжение A"},
        ])
        switched.history.switch_branch("main")
        self.assertEqual(switched.history.get_messages(), main_messages)
        self.assertEqual(switched.history.list_branches(), ["main", "a", "b"])
        self.assertEqual(switched.history.list_checkpoints(), ["start"])
        for call in self.create.call_args_list:
            self.assertTrue(all(set(m) == {"role", "content"} for m in call.kwargs["messages"]))

    def test_clear_and_system_prompts_are_local_and_checkpoint_is_immutable(self):
        history = BranchHistoryManager(self.path)
        history.set_system_prompt("Общие правила")
        history.add_exchange("Общее", "Ответ", None)
        history.create_checkpoint("start")
        original = history.get_messages()
        history.create_branch("a", from_checkpoint="start")
        history.clear()
        history.set_system_prompt("Правила A")
        history.add_exchange("Ветка A", "Ответ A", TokenUsage(10, 5, 15))
        history.create_checkpoint("nested")
        history.create_branch("nested-a", from_checkpoint="nested")
        self.assertEqual(history.get_system_prompt(), "Правила A")
        self.assertEqual(history.get_usage(), DialogueUsage(10, 5, 15))
        history.create_branch("b", from_checkpoint="start")
        self.assertEqual(history.get_messages(), original)
        self.assertEqual(history.get_usage(), DialogueUsage(missing_responses=1))
        history.switch_branch("main")
        self.assertEqual(history.get_messages(), original)
        history.switch_branch("a")
        self.assertEqual(history.get_system_prompt(), "Правила A")
        history.clear()
        self.assertEqual(history.get_usage(), DialogueUsage())
        self.assertEqual(history.get_messages(), [])
        self.assertEqual(history.list_checkpoints(), ["start", "nested"])
        history.switch_branch("b")
        self.assertEqual(history.get_messages(), original)

    def test_legacy_messages_summary_facts_and_usage_survive_migration(self):
        for compressed, with_facts in ((False, False), (True, False), (True, True)):
            with self.subTest(compressed=compressed, facts=with_facts):
                self.path.unlink(missing_ok=True)
                legacy = HistoryManager(self.path)
                legacy.set_system_prompt("Правила")
                legacy.add_exchange("Первый", "Ответ", None)
                if compressed:
                    legacy.compress(2, "Старая цель", TokenUsage(10, 5, 15))
                if with_facts:
                    legacy.update_facts({"goal": "Цель"}, TokenUsage(20, 10, 30))
                legacy.add_exchange("Второй", "Ответ", TokenUsage(100, 30, 130))
                original = self.path.read_bytes()
                history = BranchHistoryManager(self.path)
                self.assertEqual(self.path.read_bytes(), original)
                history.create_checkpoint("import")
                history.create_branch("copy", from_checkpoint="import")
                restored = BranchHistoryManager(self.path)
                self.assertEqual(restored.get_messages(), legacy.get_messages())
                self.assertEqual(restored.get_usage(), legacy.get_usage())
                self.assertEqual(restored.facts, legacy.facts)
                self.assertEqual(restored.has_facts, legacy.has_facts)
                self.assertEqual(restored.summary, legacy.summary)
                restored.update_facts({"goal": "Новая цель"}, None)
                restored.create_branch("another", from_checkpoint="import")
                self.assertEqual(restored.facts, legacy.facts)

    def test_meta_prompt_uses_and_updates_only_active_branch(self):
        agent = self.agent()
        agent.request("Общее", system="Правила")
        agent.history.create_checkpoint("start")
        agent.history.create_branch("a", from_checkpoint="start")
        agent.request("Только A")
        agent.history.create_branch("b", from_checkpoint="start")
        self.create.reset_mock()
        _, result = agent.request_with_meta_prompt("Задача B")
        first, second = self.create.call_args_list
        self.assertEqual(first.kwargs["messages"][0]["content"], META_PROMPT_SYSTEM)
        self.assertEqual(second.kwargs["messages"][0]["content"], "Правила")
        self.assertEqual([m["content"] for m in second.kwargs["messages"][1:]], [
            "Общее", "Ответ", "Задача B", "Ответ", "Ответ",
        ])
        self.assertEqual(result.dialogue_usage, DialogueUsage(300, 90, 390))
        agent.history.switch_branch("main")
        self.assertEqual(agent.history.get_usage(), DialogueUsage(100, 30, 130))
        agent.history.switch_branch("a")
        self.assertEqual(agent.history.get_messages()[-2]["content"], "Только A")

    def test_api_failure_does_not_change_branch_or_checkpoint(self):
        agent = self.agent()
        agent.request("Общее")
        agent.history.create_checkpoint("start")
        agent.history.create_branch("a", from_checkpoint="start")
        original = self.path.read_bytes()
        self.create.side_effect = RuntimeError("API error")
        with self.assertRaises(RuntimeError):
            agent.request("Не сохранится")
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(agent.history.active_branch, "a")
        self.assertEqual(agent.history.get_usage(), DialogueUsage(100, 30, 130))

    def test_failed_writes_leave_file_and_all_in_memory_state_unchanged(self):
        history = BranchHistoryManager(self.path)
        history.add_exchange("Общее", "Ответ", TokenUsage(10, 5, 15))
        history.create_checkpoint("start")
        history.create_branch("a", from_checkpoint="start")
        original = self.path.read_bytes()
        messages = history.get_messages()
        for operation in (
            lambda: history.create_checkpoint("new"),
            lambda: history.create_branch("b", from_checkpoint="start"),
            lambda: history.switch_branch("main"),
            lambda: history.add_exchange("Новое", "Ответ", TokenUsage(20, 10, 30)),
            history.clear,
        ):
            for target in ("llm_agent.history.os.fsync", "llm_agent.history.Path.replace"):
                with self.subTest(operation=operation, target=target), patch(target, side_effect=OSError("Disk error")):
                    with self.assertRaises(OSError):
                        operation()
                self.assertEqual(self.path.read_bytes(), original)
                self.assertEqual(history.active_branch, "a")
                self.assertEqual(history.list_branches(), ["main", "a"])
                self.assertEqual(history.list_checkpoints(), ["start"])
                self.assertEqual(history.get_messages(), messages)
                self.assertEqual(history.get_usage(), DialogueUsage(10, 5, 15))
                self.assertEqual(list(self.path.parent.glob("*.tmp")), [])

    def test_invalid_names_missing_targets_and_duplicates_do_not_write(self):
        history = BranchHistoryManager(self.path)
        history.create_checkpoint("start")
        original = self.path.read_bytes()
        operations = [
            lambda: history.create_checkpoint("start"),
            lambda: history.create_branch("main", from_checkpoint="start"),
            lambda: history.create_branch("a", from_checkpoint="missing"),
            lambda: history.switch_branch("missing"),
            lambda: self.agent("missing"),
        ]
        for name in ("", " ", " padded ", None, 1, []):
            operations.extend([
                lambda name=name: history.create_checkpoint(name),
                lambda name=name: history.create_branch(name, from_checkpoint="start"),
                lambda name=name: history.switch_branch(name),
                lambda name=name: history.create_branch("a", from_checkpoint=name),
            ])
        for operation in operations:
            with self.subTest(operation=operation), self.assertRaises(ValueError):
                operation()
            self.assertEqual(self.path.read_bytes(), original)
            self.assertEqual(history.active_branch, "main")
            self.assertEqual(history.list_branches(), ["main"])

    def test_corrupt_inactive_branches_and_checkpoints_are_rejected(self):
        history = BranchHistoryManager(self.path)
        history.create_checkpoint("start")
        history.create_branch("a", from_checkpoint="start")
        saved = json.loads(self.path.read_text(encoding="utf-8"))
        malformed = [
            {**saved, "version": True}, {**saved, "version": 2},
            {**saved, "active_branch": "missing"}, {**saved, "active_branch": []},
            {**saved, "branches": {}}, {**saved, "checkpoints": []},
            {**saved, "extra": 1},
        ]
        for key, name in (("branches", "main"), ("checkpoints", "start")):
            for snapshot in (
                None, [{"role": "invalid", "content": "text"}],
                [{"role": "assistant", "content": "text", "usage": {}}],
                {"messages": [], "summary": " ", "archived_usage": {}},
            ):
                data = deepcopy(saved)
                data[key][name] = snapshot
                malformed.append(data)
            data = deepcopy(saved)
            data[key][" "] = []
            malformed.append(data)
        for data in malformed:
            with self.subTest(data=data):
                self.path.write_text(json.dumps(data), encoding="utf-8")
                original = self.path.read_bytes()
                with self.assertRaises(ValueError):
                    BranchHistoryManager(self.path)
                self.assertEqual(self.path.read_bytes(), original)

    def test_other_strategies_cannot_overwrite_branch_store(self):
        BranchHistoryManager(self.path).create_checkpoint("start")
        original = self.path.read_bytes()
        for options in ({}, {"strategy": "window", "window_size": 1}, {"strategy": "facts", "window_size": 1}):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, "--strategy branch"):
                Agent("test", self.path, **options)
            self.assertEqual(self.path.read_bytes(), original)

    def test_stale_instance_cannot_overwrite_another_branch(self):
        history = BranchHistoryManager(self.path)
        history.create_checkpoint("start")
        history.create_branch("a", from_checkpoint="start")
        history.create_branch("b", from_checkpoint="start")
        first = BranchHistoryManager(self.path, branch="a")
        second = BranchHistoryManager(self.path, branch="b")
        second.add_message("user", "Сохранить в B")
        original = self.path.read_bytes()
        for operation in (
            lambda: first.add_message("user", "Устаревшая A"),
            lambda: first.create_checkpoint("stale"),
            lambda: first.create_branch("stale", from_checkpoint="start"),
            lambda: first.switch_branch("main"),
            first.clear,
        ):
            with self.subTest(operation=operation), self.assertRaisesRegex(ValueError, "загрузите её заново"):
                operation()
            self.assertEqual(self.path.read_bytes(), original)
            self.assertEqual(first.active_branch, "a")
            self.assertEqual(first.get_messages(), [])
        refreshed = BranchHistoryManager(self.path, branch="a")
        refreshed.add_message("user", "Сохранить в A")
        refreshed.switch_branch("b")
        self.assertEqual(refreshed.get_messages(), [{"role": "user", "content": "Сохранить в B"}])

    def test_returned_messages_facts_and_names_cannot_mutate_snapshots(self):
        history = BranchHistoryManager(self.path)
        history.add_message("user", "Исходное")
        history.update_facts({"goal": "Цель"}, None)
        history.create_checkpoint("start")
        original = self.path.read_bytes()
        history.get_messages()[-1]["content"] = "Подмена"
        history.facts["goal"] = "Подмена"
        history.list_branches().append("fake")
        history.list_checkpoints().clear()
        self.assertEqual(self.path.read_bytes(), original)
        history.create_branch("copy", from_checkpoint="start")
        self.assertEqual(history.get_messages()[-1]["content"], "Исходное")
        self.assertEqual(history.facts, {"goal": "Цель"})

    def test_cli_management_needs_neither_env_nor_api(self):
        with patch("llm_agent.cli.load_env") as load_env:
            self.run_cli("--strategy", "branch", "--checkpoint", "start")
            self.run_cli("--strategy", "branch", "--create-branch", "a", "--from-checkpoint", "start")
            self.run_cli("--strategy", "branch", "--create-branch", "b", "--from-checkpoint", "start")
            self.run_cli("--strategy", "branch", "--branch", "a")
            output = self.run_cli("--strategy", "branch", "--list-branches")
            load_env.assert_not_called()
        self.client.assert_not_called()
        self.assertIn("Активная ветка: a", output)
        self.assertIn("Ветки: main, a, b", output)
        self.assertIn("Checkpoints: start", output)
        self.assertEqual(BranchHistoryManager(self.path).active_branch, "a")

    def test_cli_checkpoint_after_successful_request_and_branch_selection(self):
        with patch("llm_agent.cli.load_env"), patch.dict("os.environ", {"API_KEY": "test"}):
            self.run_cli("--strategy", "branch", "--user", "Общее", "--checkpoint", "start")
            self.run_cli(
                "--strategy", "branch", "--create-branch", "a", "--from-checkpoint", "start",
                "--user", "Только A",
            )
            self.run_cli("--strategy", "branch", "--branch", "main", "--user", "Только main")
            sent = self.create.call_args.kwargs["messages"]
            self.assertEqual([m["content"] for m in sent], ["Общее", "Ответ", "Только main"])
            self.create.reset_mock()
            with self.assertRaises(ValueError):
                self.run_cli("--strategy", "branch", "--user", "Не отправлять", "--checkpoint", "start")
            self.create.assert_not_called()
            self.create.side_effect = RuntimeError("API error")
            with self.assertRaises(RuntimeError):
                self.run_cli("--strategy", "branch", "--user", "Ошибка", "--checkpoint", "failed")
        history = BranchHistoryManager(self.path)
        self.assertEqual(history.list_checkpoints(), ["start"])
        history.create_branch("b", from_checkpoint="start")
        self.assertEqual([m["content"] for m in history.get_messages()], ["Общее", "Ответ"])

    def test_agent_and_cli_validation(self):
        for options in (
            {"strategy": "branch", "window_size": 2},
            {"strategy": "branch", "last_messages": 2},
            {"strategy": "branch", "compress_every": 2},
            {"branch": "main"}, {"strategy": "window", "window_size": 2, "branch": "main"},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                Agent("test", self.path, **options)
        self.assertFalse(self.path.exists())
        with patch("sys.argv", ["main.py", "--strategy", "branch", "--user", "Запрос"]):
            args = parse_args()
        self.assertEqual(args.strategy, "branch")
        self.assertIsNone(args.window_size)
        for options in (
            [], ["--strategy", "branch"], ["--branch", "main"],
            ["--checkpoint", "start"], ["--list-branches"],
            ["--strategy", "branch", "--create-branch", "a"],
            ["--strategy", "branch", "--from-checkpoint", "start"],
            ["--strategy", "branch", "--checkpoint", " "],
            ["--strategy", "branch", "--user", "Запрос", "--window-size", "2"],
            ["--strategy", "branch", "--user", "Запрос", "--last-messages", "2"],
            ["--strategy", "branch", "--user", "Запрос", "--compress-every", "2"],
            ["--strategy", "branch", "--user", "Запрос", "--list-branches"],
            ["--strategy", "branch", "--branch", "main", "--meta-prompt"],
            ["--strategy", "branch", "--branch", "main", "--system", "Правила"],
            ["--strategy", "branch", "--branch", "main", "--create-branch", "a", "--from-checkpoint", "start"],
        ):
            with self.subTest(options=options), patch("sys.argv", ["main.py", *options]):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    parse_args()
                self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
