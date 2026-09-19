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
from llm_agent.invariants import Invariant, InvariantSet, InvariantStore
from tests.helpers import completion


def reply(content):
    response = completion()
    response.choices[0].message.content = content
    return response


def verdict(status="pass"):
    return reply(json.dumps({"checks": [{"id": "stack", "status": status, "reason": "Проверено"}]}))


class InvariantCliTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.history = self.directory / "history.json"
        self.memory_db = self.directory / "memory.sqlite3"
        self.invariants_file = self.directory / "invariants.json"
        self.source = self.directory / "import.json"
        self.rules = InvariantSet((Invariant("stack", "Использовать только Python"),))
        self.client = self.enterContext(patch("llm_agent.agent.OpenAI"))
        self.create = self.client.return_value.chat.completions.create
        self.load_env = self.enterContext(patch("llm_agent.cli.load_env"))

    def run_cli(self, *options, entrypoint=main):
        output = io.StringIO()
        arguments = [
            "main.py", "--history", str(self.history), "--memory-db", str(self.memory_db),
            "--invariants-file", str(self.invariants_file), *options,
        ]
        with patch("sys.argv", arguments), patch(
            "llm_agent.cli.console", Console(file=output, width=240, color_system=None),
        ):
            entrypoint()
        return output.getvalue()

    def write_import(self, rules=None):
        self.source.write_text(json.dumps((rules or self.rules).to_dict(), ensure_ascii=False), encoding="utf-8")
        return str(self.source)

    def test_import_show_and_empty_configuration_work_without_api_or_dialogue(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(json.loads(self.run_cli("--invariants-show")), InvariantSet().to_dict())
            self.assertFalse(self.invariants_file.exists())
            shown = self.run_cli("--invariants-import", self.write_import(), "--invariants-show")
            self.assertEqual(json.loads(shown), self.rules.to_dict())
            self.assertEqual(InvariantStore(self.invariants_file).load(), self.rules)
            self.assertEqual(json.loads(self.run_cli("--invariants-show")), self.rules.to_dict())
        self.assertFalse(self.history.exists())
        self.assertFalse(self.memory_db.exists())
        self.assertFalse(self.memory_db.with_name("profiles.sqlite3").exists())
        self.load_env.assert_not_called()
        self.client.assert_not_called()

    def test_invalid_or_missing_import_preserves_saved_rules(self):
        InvariantStore(self.invariants_file).save(self.rules)
        original = self.invariants_file.read_bytes()
        invalid = (
            b'{"version":', b'\xffinvalid', b'{"version":1,"rules":[{"id":"bad"}]}',
            b'{"version":1,"rules":[],"rules":[]}',
        )
        for contents in invalid:
            with self.subTest(contents=contents):
                self.source.write_bytes(contents)
                with self.assertRaises(SystemExit):
                    self.run_cli("--invariants-import", str(self.source), entrypoint=run)
                self.assertEqual(self.invariants_file.read_bytes(), original)
        self.source.unlink()
        with self.assertRaises(SystemExit) as error:
            self.run_cli("--invariants-import", str(self.source), entrypoint=run)
        self.assertIn(str(self.source), str(error.exception))
        self.assertEqual(self.invariants_file.read_bytes(), original)
        self.client.assert_not_called()
        self.load_env.assert_not_called()

    def test_corrupt_file_blocks_request_and_can_be_repaired_by_import(self):
        self.invariants_file.write_text("not JSON", encoding="utf-8")
        with self.assertRaises(SystemExit) as error:
            self.run_cli("--user", "Объясни Python", entrypoint=run)
        self.assertIn("инварианты", str(error.exception))
        self.assertFalse(self.history.exists())
        self.assertEqual(self.invariants_file.read_text(encoding="utf-8"), "not JSON")
        self.client.assert_not_called()
        self.load_env.assert_not_called()
        self.run_cli("--invariants-import", self.write_import())
        self.assertEqual(InvariantStore(self.invariants_file).load(), self.rules)

    def test_each_request_reloads_separate_rules(self):
        replacement = InvariantSet((Invariant("stack", "Использовать Python и SQLite"),))
        self.create.side_effect = [verdict(), reply("Ответ"), verdict()] * 2
        with patch.dict("os.environ", {"API_KEY": "test"}), patch("llm_agent.cli.Agent", wraps=Agent) as agent:
            self.run_cli("--invariants-import", self.write_import(), "--user", "Предложи архитектуру")
            InvariantStore(self.invariants_file).save(replacement)
            self.run_cli("--user", "Продолжи")
        self.assertEqual([call.kwargs["invariants"] for call in agent.call_args_list], [self.rules, replacement])
        saved = self.history.read_text(encoding="utf-8")
        self.assertNotIn(self.rules.rules[0].description, saved)
        self.assertNotIn(replacement.rules[0].description, saved)

    def test_meta_conflict_prints_one_refusal_and_skips_prompt_generation(self):
        InvariantStore(self.invariants_file).save(self.rules)
        self.create.return_value = verdict("conflict")
        with patch.dict("os.environ", {"API_KEY": "test"}), patch("llm_agent.cli.print_response") as print_response:
            self.run_cli("--user", "Замени Python на Java", "--meta-prompt")
        self.create.assert_called_once()
        print_response.assert_called_once()
        refusal = print_response.call_args.args[0]
        self.assertTrue(refusal.refused)
        self.assertIn("stack", refusal.content)
        self.assertIn(self.rules.rules[0].description, refusal.content)

    def test_show_rejects_request_or_unrelated_mutations(self):
        for extra in (["--user", "Запрос"], ["--task-show"], ["--memory-show"], ["--profile-clear"]):
            with self.subTest(extra=extra), patch("sys.argv", ["main.py", "--invariants-show", *extra]):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    parse_args()
                self.assertEqual(error.exception.code, 2)

    def test_empty_checker_choices_print_safe_refusal(self):
        InvariantStore(self.invariants_file).save(self.rules)
        response = completion()
        response.choices = []
        self.create.return_value = response
        with patch.dict("os.environ", {"API_KEY": "test"}):
            output = self.run_cli("--user", "Предложи архитектуру")
        self.assertIn("Не могу подтвердить соблюдение инвариантов", output)
        self.assertIn("stack", output)
        self.create.assert_called_once()


if __name__ == "__main__":
    unittest.main()
