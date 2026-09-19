import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rich.console import Console

from llm_agent.agent import Agent
from llm_agent.cli import main, parse_args, run
from llm_agent.memory import cli_memory_scope
from llm_agent.profile import ProfileStore, UserProfile
from tests.helpers import completion


class ProfileCliTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.history = self.directory / "history.json"
        self.memory_db = self.directory / "memory.sqlite3"
        self.profiles_db = self.directory / "profiles.sqlite3"
        self.profile_file = self.directory / "profile.json"
        self.profile = UserProfile(
            id="developer", name="Разработчик", language="Русский",
            style="Кратко, с техническими деталями", format="Сначала пример кода",
            constraints="Только стандартная библиотека Python", context="Изучаю Python",
        )
        client = patch("llm_agent.agent.OpenAI")
        self.client = client.start()
        self.addCleanup(client.stop)
        self.create = self.client.return_value.chat.completions.create
        self.create.return_value = completion()
        env = patch("llm_agent.cli.load_env")
        self.load_env = env.start()
        self.addCleanup(env.stop)

    def run_cli(self, *options, history=None, entrypoint=main):
        output = io.StringIO()
        arguments = [
            "main.py", "--history", str(history or self.history),
            "--memory-db", str(self.memory_db),
            "--invariants-file", str(self.directory / "invariants.json"), *options,
        ]
        with patch("sys.argv", arguments), patch(
            "llm_agent.cli.console", Console(file=output, width=240, color_system=None),
        ):
            entrypoint()
        return output.getvalue()

    def write_profile(self, profile=None):
        self.profile_file.write_text(
            json.dumps((profile or self.profile).to_dict(), ensure_ascii=False), encoding="utf-8",
        )
        return str(self.profile_file)

    def selected(self, *, history=None, branch=None):
        return ProfileStore(self.profiles_db).selected(cli_memory_scope(history or self.history, branch))

    def test_import_list_select_show_and_clear_without_api_environment_or_memory(self):
        with patch.dict("os.environ", {}, clear=True):
            self.run_cli("--profile-import", self.write_profile())
            self.assertIsNone(self.selected())
            listed = json.loads(self.run_cli("--profile-list"))
            self.assertEqual(listed, [self.profile.to_dict()])
            self.run_cli("--profile", self.profile.id)
            self.assertEqual(json.loads(self.run_cli("--profile-show")), self.profile.to_dict())
            self.run_cli("--profile-clear")
            self.assertIsNone(json.loads(self.run_cli("--profile-show")))
        self.assertFalse(self.history.exists())
        self.assertFalse(self.memory_db.exists())
        self.load_env.assert_not_called()
        self.client.assert_not_called()

    def test_selected_profile_survives_restart_and_is_loaded_for_each_request(self):
        self.run_cli("--profile-import", self.write_profile(), "--profile", self.profile.id)
        with patch.dict("os.environ", {"API_KEY": "test"}), patch("llm_agent.cli.Agent", wraps=Agent) as agent:
            self.run_cli("--user", "Объясни генераторы")
            self.run_cli("--user", "Теперь итераторы")
        self.assertEqual([call.kwargs["profile"] for call in agent.call_args_list], [self.profile, self.profile])
        for call in self.create.call_args_list:
            messages = json.dumps(call.kwargs["messages"], ensure_ascii=False)
            self.assertIn(self.profile.style, messages)
            self.assertIn(self.profile.constraints, messages)
        saved_history = self.history.read_text(encoding="utf-8")
        self.assertNotIn(self.profile.style, saved_history)
        self.assertNotIn(self.profile.constraints, saved_history)

    def test_select_with_request_and_clear_take_effect_immediately(self):
        self.run_cli("--profile-import", self.write_profile())
        with patch.dict("os.environ", {"API_KEY": "test"}), patch("llm_agent.cli.Agent", wraps=Agent) as agent:
            self.run_cli("--profile", self.profile.id, "--user", "Объясни Python")
            self.run_cli("--profile-clear", "--user", "Продолжи")
            self.run_cli("--user", "Ещё пример")
        self.assertEqual(
            [call.kwargs["profile"] for call in agent.call_args_list], [self.profile, None, None],
        )
        for call in self.create.call_args_list[1:]:
            messages = json.dumps(call.kwargs["messages"], ensure_ascii=False)
            self.assertNotIn(self.profile.constraints, messages)
        self.assertIsNone(self.selected())

    def test_reimport_updates_profile_used_by_next_request(self):
        self.run_cli("--profile-import", self.write_profile(), "--profile", self.profile.id)
        updated = UserProfile(id=self.profile.id, name="Обновлённый", style="Подробно и простыми словами")
        self.run_cli("--profile-import", self.write_profile(updated))
        with patch.dict("os.environ", {"API_KEY": "test"}), patch("llm_agent.cli.Agent", wraps=Agent) as agent:
            self.run_cli("--user", "Объясни Python")
        self.assertEqual(agent.call_args.kwargs["profile"], updated)
        self.assertEqual(self.selected(), updated)

    def test_selections_are_independent_for_histories(self):
        other_history = self.directory / "other.json"
        self.run_cli("--profile-import", self.write_profile(), "--profile", self.profile.id)
        self.assertIsNone(json.loads(self.run_cli("--profile-show", history=other_history)))
        self.assertEqual(self.selected(), self.profile)
        self.assertIsNone(self.selected(history=other_history))

    def test_selection_follows_saved_active_branch_and_clear_is_local(self):
        self.run_cli(
            "--strategy", "branch", "--checkpoint", "start",
            "--profile-import", self.write_profile(), "--profile", self.profile.id,
        )
        self.run_cli("--strategy", "branch", "--create-branch", "alternative", "--from-checkpoint", "start")
        self.assertIsNone(json.loads(self.run_cli("--strategy", "branch", "--profile-show")))
        other = UserProfile(id="beginner", name="Новичок", style="Подробно и простыми словами")
        self.run_cli("--strategy", "branch", "--profile-import", self.write_profile(other), "--profile", other.id)
        self.assertEqual(self.selected(branch="main"), self.profile)
        self.assertEqual(self.selected(branch="alternative"), other)
        with patch.dict("os.environ", {"API_KEY": "test"}), patch("llm_agent.cli.Agent", wraps=Agent) as agent:
            self.run_cli("--strategy", "branch", "--user", "Объясни Python")
            self.run_cli("--strategy", "branch", "--branch", "main", "--user", "Объясни Python")
        self.assertEqual([call.kwargs["profile"] for call in agent.call_args_list], [other, self.profile])
        self.run_cli("--strategy", "branch", "--profile-clear")
        self.assertIsNone(self.selected(branch="main"))
        self.assertEqual(self.selected(branch="alternative"), other)

    def test_delete_unselects_profile_in_all_histories(self):
        other_history = self.directory / "other.json"
        self.run_cli("--profile-import", self.write_profile(), "--profile", self.profile.id)
        self.run_cli("--profile", self.profile.id, history=other_history)
        self.run_cli("--profile-delete", self.profile.id)
        self.assertEqual(json.loads(self.run_cli("--profile-list")), [])
        self.assertIsNone(self.selected())
        self.assertIsNone(self.selected(history=other_history))

    def test_profiles_database_can_be_overridden(self):
        custom_database = self.directory / "custom.sqlite3"
        self.run_cli(
            "--profiles-db", str(custom_database), "--profile-import", self.write_profile(),
            "--profile", self.profile.id,
        )
        self.assertEqual(ProfileStore(custom_database).selected(cli_memory_scope(self.history)), self.profile)
        self.assertFalse(self.profiles_db.exists())

    def test_invalid_import_does_not_replace_profile_or_contact_api(self):
        self.run_cli("--profile-import", self.write_profile(), "--profile", self.profile.id)
        for invalid in (None, [], {}, {"id": self.profile.id, "name": 123}, {"id": self.profile.id, "name": "Имя", "unknown": "Поле"}):
            with self.subTest(invalid=invalid):
                self.profile_file.write_text(json.dumps(invalid), encoding="utf-8")
                with self.assertRaises(SystemExit):
                    self.run_cli("--profile-import", str(self.profile_file), "--user", "Продолжи", entrypoint=run)
                self.assertEqual(self.selected(), self.profile)
        self.load_env.assert_not_called()
        self.client.assert_not_called()

    def test_corrupt_json_reports_path_and_preserves_saved_profile(self):
        self.run_cli("--profile-import", self.write_profile(), "--profile", self.profile.id)
        for content in (b'{"id":', b'\xff\xfeinvalid'):
            with self.subTest(content=content):
                self.profile_file.write_bytes(content)
                with self.assertRaises(SystemExit) as error:
                    self.run_cli("--profile-import", str(self.profile_file), entrypoint=run)
                self.assertIn("Некорректный JSON профиля", str(error.exception))
                self.assertIn(str(self.profile_file), str(error.exception))
                self.assertEqual(self.selected(), self.profile)
        self.load_env.assert_not_called()
        self.client.assert_not_called()

    def test_unknown_profile_fails_before_request(self):
        with self.assertRaises(SystemExit):
            self.run_cli("--profile", "missing", "--user", "Запрос", entrypoint=run)
        self.assertFalse(self.history.exists())
        self.load_env.assert_not_called()
        self.client.assert_not_called()

    def test_missing_import_reports_path_without_creating_database(self):
        with self.assertRaises(SystemExit) as error:
            self.run_cli("--profile-import", str(self.profile_file), entrypoint=run)
        self.assertIn(str(self.profile_file), str(error.exception))
        self.assertFalse(self.profiles_db.exists())
        self.load_env.assert_not_called()
        self.client.assert_not_called()

    def test_corrupt_database_reports_readable_error_and_preserves_file(self):
        self.profiles_db.write_text("Это не SQLite", encoding="utf-8")
        before = self.profiles_db.read_bytes()
        with self.assertRaises(SystemExit) as error:
            self.run_cli("--profile-list", entrypoint=run)
        self.assertIn("Не удалось прочесть или сохранить профили", str(error.exception))
        self.assertEqual(self.profiles_db.read_bytes(), before)
        self.load_env.assert_not_called()
        self.client.assert_not_called()

    def test_profile_operations_do_not_pollute_memory_json(self):
        output = self.run_cli(
            "--profile-import", self.write_profile(), "--profile", self.profile.id, "--memory-show",
        )
        self.assertEqual(set(json.loads(output)), {"short_term", "working", "long_term"})
        self.run_cli("--strategy", "branch", "--checkpoint", "start")
        output = self.run_cli("--strategy", "branch", "--profile", self.profile.id, "--memory-show")
        self.assertEqual(json.loads(output)["short_term"]["branch"], "main")

    def test_invalid_combinations_fail_during_argument_parsing(self):
        for options in (
            ["--profile", "id", "--profile-clear"],
            ["--profile-import", "profile.json", "--profile-delete", "id"],
            ["--profile-list", "--profile-show"],
            ["--profile", "id", "--profile-delete", "id"],
            ["--memory-show", "--profile-show"],
            ["--memory-show", "--profile-list"],
            ["--profile", " "],
            ["--profile-delete", " "],
            ["--profile-list", "--system", "Правила"],
            ["--profile-show", "--meta-prompt"],
            ["--profiles-db", str(self.profiles_db)],
        ):
            with self.subTest(options=options), patch("sys.argv", ["main.py", *options]):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    parse_args()
                self.assertEqual(error.exception.code, 2)
        self.assertFalse(self.profiles_db.exists())
        self.client.assert_not_called()


if __name__ == "__main__":
    unittest.main()
