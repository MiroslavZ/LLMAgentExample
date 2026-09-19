import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rich.console import Console

from llm_agent.branch_history import BranchHistoryManager
from llm_agent.cli import main, parse_args, run
from llm_agent.history import HistoryManager
from llm_agent.memory import MemoryStore, cli_memory_scope
from tests.helpers import completion


class MemoryCliTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "history.json"
        self.database = Path(directory.name) / "memory.sqlite3"
        client = patch("llm_agent.agent.OpenAI")
        self.client = client.start()
        self.addCleanup(client.stop)
        self.create = self.client.return_value.chat.completions.create
        self.create.return_value = completion()
        env = patch("llm_agent.cli.load_env")
        self.load_env = env.start()
        self.addCleanup(env.stop)

    def run_cli(self, *options, history=None):
        output = io.StringIO()
        arguments = [
            "main.py", "--history", str(history or self.path),
            "--memory-db", str(self.database),
            "--invariants-file", str(self.path.with_name("invariants.json")), *options,
        ]
        with patch("sys.argv", arguments), patch(
            "llm_agent.cli.console", Console(file=output, width=240, color_system=None),
        ):
            main()
        return output.getvalue()

    def snapshot(self, *, history=None, branch=None):
        return MemoryStore(self.database).snapshot(cli_memory_scope(history or self.path, branch))

    def test_manage_and_show_all_layers_without_api_or_environment(self):
        HistoryManager(self.path).add_message("user", "Текущий диалог")
        with patch.dict("os.environ", {}, clear=True):
            self.run_cli("--memory-set", "working", "goal", "Подготовить доклад")
            self.run_cli("--memory-set", "long_term", "language", "Русский")
            shown = json.loads(self.run_cli("--memory-show"))
        self.assertEqual(shown["short_term"]["scope"], cli_memory_scope(self.path))
        self.assertEqual(shown["short_term"]["history"], str(self.path.absolute()))
        self.assertEqual(shown["short_term"]["messages"], [
            {"role": "user", "content": "Текущий диалог"},
        ])
        self.assertEqual(shown["working"], {"goal": "Подготовить доклад"})
        self.assertEqual(shown["long_term"], {"language": "Русский"})
        self.load_env.assert_not_called()
        self.client.assert_not_called()

    def test_working_memory_is_local_and_long_term_is_shared_between_histories(self):
        other_history = self.path.with_name("other.json")
        self.run_cli("--memory-set", "working", "goal", "Доклад")
        self.run_cli("--memory-set", "long_term", "database", "SQLite")
        self.run_cli("--memory-set", "working", "goal", "Письмо", history=other_history)
        self.assertEqual(self.snapshot().working, {"goal": "Доклад"})
        other = self.snapshot(history=other_history)
        self.assertEqual(other.working, {"goal": "Письмо"})
        self.assertEqual(other.long_term, {"database": "SQLite"})
        self.assertFalse(self.path.exists())
        self.assertFalse(other_history.exists())

    def test_deletion_and_clear_do_not_affect_other_layers(self):
        history = HistoryManager(self.path)
        history.add_message("user", "Сохранить диалог")
        before = self.path.read_bytes()
        self.run_cli("--memory-set", "working", "goal", "Доклад")
        self.run_cli("--memory-set", "long_term", "language", "Русский")
        self.run_cli("--memory-set", "long_term", "database", "SQLite")
        self.run_cli("--memory-set", "long_term", "temporary", "Удалить")
        self.run_cli("--memory-delete", "long_term", "temporary")
        self.run_cli("--memory-clear-working")
        snapshot = self.snapshot()
        self.assertEqual(snapshot.working, {})
        self.assertEqual(snapshot.long_term, {"language": "Русский", "database": "SQLite"})
        self.assertEqual(self.path.read_bytes(), before)
        self.run_cli("--memory-set", "working", "goal", "Новая задача")
        self.run_cli("--memory-delete", "working", "goal")
        self.assertEqual(self.snapshot().working, {})
        self.client.assert_not_called()

    def test_memory_written_before_request_is_passed_to_model_without_copying_into_history(self):
        with patch.dict("os.environ", {"API_KEY": "test"}):
            self.run_cli(
                "--memory-set", "working", "goal", "Подготовить доклад",
                "--user", "Предложи следующий шаг",
            )
        sent = json.dumps(self.create.call_args.kwargs["messages"], ensure_ascii=False)
        self.assertIn("Подготовить доклад", sent)
        self.assertIn("Предложи следующий шаг", sent)
        self.assertNotIn("Подготовить доклад", self.path.read_text(encoding="utf-8"))
        self.assertEqual(self.snapshot().working, {"goal": "Подготовить доклад"})
        self.load_env.assert_called_once()

    def test_shared_memory_is_loaded_for_next_request_in_another_dialogue(self):
        self.run_cli("--memory-set", "long_term", "style", "Отвечать кратко")
        with patch.dict("os.environ", {"API_KEY": "test"}):
            self.run_cli("--user", "Объясни SQLite", history=self.path.with_name("next.json"))
        sent = json.dumps(self.create.call_args.kwargs["messages"], ensure_ascii=False)
        self.assertIn("Отвечать кратко", sent)

    def test_branch_memory_uses_selected_or_saved_active_branch(self):
        self.run_cli(
            "--strategy", "branch", "--checkpoint", "start",
            "--memory-set", "working", "goal", "Основная задача",
        )
        self.run_cli(
            "--strategy", "branch", "--memory-set", "long_term", "language", "Русский",
        )
        self.run_cli(
            "--strategy", "branch", "--create-branch", "alternative", "--from-checkpoint", "start",
            "--memory-show",
        )
        self.assertEqual(self.snapshot(branch="alternative").working, {})
        self.assertEqual(self.snapshot(branch="alternative").long_term, {"language": "Русский"})
        self.run_cli("--strategy", "branch", "--memory-set", "working", "goal", "Альтернатива")
        self.assertEqual(self.snapshot(branch="alternative").working, {"goal": "Альтернатива"})
        self.run_cli("--strategy", "branch", "--branch", "main", "--memory-clear-working")
        self.assertEqual(self.snapshot(branch="main").working, {})
        self.assertEqual(self.snapshot(branch="alternative").working, {"goal": "Альтернатива"})
        self.assertEqual(BranchHistoryManager(self.path).active_branch, "main")
        self.load_env.assert_not_called()
        self.client.assert_not_called()

    def test_new_branch_write_targets_new_branch_before_request(self):
        self.run_cli("--strategy", "branch", "--checkpoint", "start")
        with patch.dict("os.environ", {"API_KEY": "test"}):
            self.run_cli(
                "--strategy", "branch", "--create-branch", "alternative", "--from-checkpoint", "start",
                "--memory-set", "working", "goal", "Альтернатива", "--user", "Продолжи",
            )
        self.assertEqual(self.snapshot(branch="main").working, {})
        self.assertEqual(self.snapshot(branch="alternative").working, {"goal": "Альтернатива"})
        sent = json.dumps(self.create.call_args.kwargs["messages"], ensure_ascii=False)
        self.assertIn("Альтернатива", sent)

    def test_invalid_memory_options_fail_before_side_effects(self):
        for options in (
            ["--memory-set", "short_term", "key", "value"],
            ["--memory-set", "working", "key", "value", "--memory-kind", "profile"],
            ["--memory-set", "long_term", "key", "value", "--memory-kind", "profile"],
            ["--memory-kind", "profile", "--memory-show"],
            ["--memory-set", "working", " ", "value"],
            ["--memory-set", "working", "key", " "],
            ["--memory-set", "working", "key", "value", "--memory-clear-working"],
            ["--memory-show", "--system", "Правила"],
            ["--memory-show", "--meta-prompt"],
            ["--memory-db", str(self.database)],
        ):
            with self.subTest(options=options), patch("sys.argv", ["main.py", *options]):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    parse_args()
                self.assertEqual(error.exception.code, 2)
        self.assertFalse(self.database.exists())
        self.client.assert_not_called()

    def test_corrupt_database_reports_readable_error_and_preserves_file(self):
        self.database.write_text("Это не SQLite", encoding="utf-8")
        before = self.database.read_bytes()
        with patch("sys.argv", [
            "main.py", "--history", str(self.path), "--memory-db", str(self.database), "--memory-show",
        ]), self.assertRaises(SystemExit) as error:
            run()
        self.assertIn("Не удалось прочесть или сохранить память", str(error.exception))
        self.assertEqual(self.database.read_bytes(), before)
        self.client.assert_not_called()
        self.load_env.assert_not_called()


if __name__ == "__main__":
    unittest.main()
