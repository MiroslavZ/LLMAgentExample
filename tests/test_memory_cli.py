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
from llm_agent.memory import MemoryStore
from tests.helpers import completion


class MemoryCliTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.data_dir = Path(directory.name) / "conversations"
        self.service = ConversationService(self.data_dir, token=None)
        self.conversation = self.service.create()
        self.path = self.service.store.path(self.conversation.id)
        self.database = self.data_dir / "memory.sqlite3"
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
            "--data-dir", str(self.data_dir),
            "--invariants-file", str(self.data_dir.parent / "invariants.json"), *options,
        ]
        with patch("sys.argv", arguments), patch(
            "llm_agent.cli.console", Console(file=output, width=240, color_system=None),
        ):
            main()
        return output.getvalue()

    def seed_message(self, content):
        conversation = self.service.get(self.conversation.id)
        conversation.working_context = [{"role": "user", "content": content}]
        self.service.store.save(conversation)

    def snapshot(self, *, history=None):
        return MemoryStore(self.database).snapshot(Path(history or self.path).stem)

    def test_manage_and_show_all_layers_without_api_or_environment(self):
        self.seed_message("Текущий диалог")
        with patch.dict("os.environ", {}, clear=True):
            self.run_cli("--memory-set", "working", "goal", "Подготовить доклад")
            self.run_cli("--memory-set", "long_term", "language", "Русский")
            shown = json.loads(self.run_cli("--memory-show"))
        self.assertEqual(shown["short_term"]["scope"], self.conversation.id)
        self.assertEqual(shown["short_term"]["history"], str(self.path.absolute()))
        self.assertEqual(shown["short_term"]["messages"], [
            {"role": "user", "content": "Текущий диалог"},
        ])
        self.assertEqual(shown["working"], {"goal": "Подготовить доклад"})
        self.assertEqual(shown["long_term"], {"language": "Русский"})
        self.load_env.assert_not_called()
        self.client.assert_not_called()

    def test_working_memory_is_local_and_long_term_is_shared_between_conversations(self):
        other_history = self.service.store.path(self.service.create().id)
        self.run_cli("--memory-set", "working", "goal", "Доклад")
        self.run_cli("--memory-set", "long_term", "database", "SQLite")
        self.run_cli("--memory-set", "working", "goal", "Письмо", history=other_history)
        self.assertEqual(self.snapshot().working, {"goal": "Доклад"})
        other = self.snapshot(history=other_history)
        self.assertEqual(other.working, {"goal": "Письмо"})
        self.assertEqual(other.long_term, {"database": "SQLite"})
        self.assertTrue(self.path.exists())
        self.assertTrue(other_history.exists())

    def test_deletion_and_clear_do_not_affect_other_layers(self):
        self.seed_message("Сохранить диалог")
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
            self.run_cli("--user", "Объясни SQLite", history=self.service.store.path(self.service.create().id))
        sent = json.dumps(self.create.call_args.kwargs["messages"], ensure_ascii=False)
        self.assertIn("Отвечать кратко", sent)

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
            ["--data-dir", str(self.data_dir)],
        ):
            with self.subTest(options=options), patch("sys.argv", ["main.py", *options]):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    parse_args()
                self.assertEqual(error.exception.code, 2)
        self.client.assert_not_called()

    def test_corrupt_database_reports_readable_error_and_preserves_file(self):
        self.database.write_text("Это не SQLite", encoding="utf-8")
        before = self.database.read_bytes()
        with patch("sys.argv", [
            "main.py", "--history", str(self.path), "--data-dir", str(self.data_dir), "--memory-show",
        ]), self.assertRaises(SystemExit) as error:
            run()
        self.assertIn("Не удалось прочесть или сохранить память", str(error.exception))
        self.assertEqual(self.database.read_bytes(), before)
        self.client.assert_not_called()
        self.load_env.assert_not_called()


if __name__ == "__main__":
    unittest.main()
