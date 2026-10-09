"""Machine-readable requests for an external scheduler."""

import contextlib
import io
import json
import os
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from rich.console import Console

from llm_agent.cli import run
from llm_agent.models import Turn
from llm_agent.service import ConversationService
from llm_agent.tool_events import ToolCallRecord
from tests.helpers import completion, create_selected_conversation


class ScheduledCliTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.data_dir = Path(directory.name) / "conversations"
        self.service = ConversationService(self.data_dir, token="test-only")
        self.conversation = create_selected_conversation(self.service)
        self.enterContext(patch("llm_agent.cli.load_env"))
        self.enterContext(patch.dict(os.environ, {"API_KEY": "test-only"}))
        client = self.enterContext(patch("llm_agent.agent.OpenAI"))
        self.create = client.return_value.chat.completions.create
        self.create.return_value = completion()

    def invoke(self, *arguments):
        output, errors = io.StringIO(), io.StringIO()
        argv = ["main.py", "--data-dir", str(self.data_dir), *arguments]
        code = 0
        with patch("sys.argv", argv), contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors), \
                patch("llm_agent.cli.console", Console(file=output, width=20)):
            try:
                run()
            except SystemExit as error:
                code = error.code
        return code, output.getvalue(), errors.getvalue()

    def batch(self, *arguments):
        return self.invoke("--batch", "--conversation", self.conversation.id,
                           "--user", "Проверь репозитории", *arguments)

    def test_success_is_one_json_line_with_unicode_and_saved_turn(self):
        self.create.return_value.choices[0].message.content = "Сводка [green]коммитов[/green]\n" + "Ю" * 250
        code, output, errors = self.batch()
        self.assertEqual(code, 0)
        self.assertEqual(errors, "")
        self.assertEqual(len(output.splitlines()), 1)
        self.assertIn("Сводка", output)
        turn = self.service.get(self.conversation.id).turns[-1]
        self.assertEqual(json.loads(output), {"conversation_id": self.conversation.id, **asdict(turn)})

    def test_model_failure_returns_persisted_json_and_nonzero(self):
        self.create.side_effect = RuntimeError("private SDK details")
        code, output, _ = self.batch()
        self.assertEqual(code, 1)
        data = json.loads(output)
        self.assertEqual(data["status"], "error")
        self.assertTrue(data["error"])
        self.assertNotIn("private SDK details", output)
        self.assertEqual(data["error"], self.service.get(self.conversation.id).turns[-1].error)

    def test_redirected_batch_output_is_utf8_even_with_legacy_windows_encoding(self):
        self.create.return_value.choices[0].message.content = "Коммит готов 🚀"
        raw = io.BytesIO()
        output = io.TextIOWrapper(raw, encoding="cp1251")
        argv = ["main.py", "--data-dir", str(self.data_dir), "--batch",
                "--conversation", self.conversation.id, "--user", "Сводка"]
        with patch("sys.argv", argv), contextlib.redirect_stdout(output):
            run()
        self.assertEqual(json.loads(raw.getvalue().decode("utf-8"))["answer"], "Коммит готов 🚀")
        output.detach()

    def test_noncompleted_or_tool_error_returns_json_and_nonzero(self):
        cases = [Turn("request", status=status) for status in ("partial", "interrupted", "running", "error")]
        cases.append(Turn("request", status="completed", error="Ошибка сохранения"))
        cases.append(Turn("request", status="completed", answer="Часть данных получена", tool_calls=[
            ToolCallRecord("call", "MCP", "collect", "{}", "{}", True, 0.1),
        ]))
        for turn in cases:
            with self.subTest(status=turn.status, error=turn.error, tools=turn.tool_calls):
                self.conversation.turns = [turn]
                with patch.object(ConversationService, "send", return_value=self.conversation):
                    code, output, _ = self.batch()
                self.assertEqual(code, 1)
                self.assertEqual(json.loads(output)["status"], turn.status)

    def test_batch_requires_explicit_existing_conversation_and_user(self):
        cases = (
            ("--user", "Запрос"),
            ("--new-conversation", "--user", "Запрос"),
            ("--conversation", self.conversation.id),
            ("--conversation", self.conversation.id, "--task-continue"),
            ("--conversation", "a" * 32, "--user", "Запрос"),
        )
        for arguments in cases:
            with self.subTest(arguments=arguments):
                code, output, errors = self.invoke("--batch", *arguments)
                self.assertEqual(code, 2)
                self.assertEqual(output, "")
                self.assertTrue(errors)
        self.create.assert_not_called()
        self.assertEqual(len(self.service.list_conversations()), 1)

    def test_batch_rejects_administration_and_display_options(self):
        cases = (
            ("--memory-set", "working", "key", "value"), ("--memory-delete", "working", "key"),
            ("--memory-clear-working",), ("--memory-show",), ("--profile", "dev"),
            ("--profile-clear",), ("--profile-import", "profile.json"), ("--profile-delete", "dev"),
            ("--profile-list",), ("--profile-show",), ("--invariants-import", "rules.json"),
            ("--invariants-show",), ("--task-start", "Задача"), ("--task-show",),
            ("--task-pause",), ("--task-resume",), ("--task-approve",), ("--task-continue",),
            ("--mcp-url", "https://example.com/mcp"), ("--list-conversations",), ("--show-conversation",),
        )
        for arguments in cases:
            with self.subTest(arguments=arguments):
                code, output, errors = self.batch(*arguments)
                self.assertEqual(code, 2)
                self.assertEqual(output, "")
                self.assertTrue(errors)
        self.create.assert_not_called()

    def test_early_failure_does_not_print_partial_json_or_rich_output(self):
        self.service.select_model(self.conversation.id, None)
        code, output, _ = self.batch()
        self.assertNotEqual(code, 0)
        self.assertEqual(output, "")


if __name__ == "__main__":
    unittest.main()
