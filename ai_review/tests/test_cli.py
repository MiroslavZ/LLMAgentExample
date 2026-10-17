"""Одноразовый CLI: артефакты, режим retrieval и понятные ошибки без API."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from ai_review import ReviewError
from ai_review import __main__ as cli


def make_context():
    return {
        "base_sha": "a" * 40, "merge_base_sha": "a" * 40, "head_sha": "b" * 40,
        "diff": "@@ -1 +1 @@\n-return 1\n+return None\n",
        "files": [{"path": "service.py", "old_path": "service.py", "status": "M",
                   "before": "return 1\n", "after": "return None\n"}],
        "sources": [], "skipped": [], "retrieval": {},
    }


def make_result():
    return {
        "summary": "Нарушен контракт результата.", "model": "test-model", "usage": {},
        "findings": [{"category": "bug", "severity": "high", "path": "service.py",
                      "side": "after", "line": 1, "description": "Возвращается None вместо числа.",
                      "suggestion": "Восстановить числовой результат.", "sources": []}],
    }


class CliTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / "output"
        self.args = ["--repo", str(self.root), "--base", "base", "--head", "head", "--output", str(self.output)]

    def run_cli(self, *extra):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = cli.main(self.args + list(extra))
        return code, stdout.getvalue() + stderr.getvalue()

    @patch.dict(os.environ, {}, clear=True)
    @patch("ai_review.__main__.generate_review")
    @patch("ai_review.__main__.collect_context")
    def test_retrieve_only_saves_context_without_secret_or_model_call(self, collect, generate):
        collect.return_value = make_context()
        code, _ = self.run_cli("--retrieve-only")
        self.assertEqual(code, 0)
        collect.assert_called_once()
        generate.assert_not_called()
        saved = json.loads((self.output / "context.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["diff"], make_context()["diff"])
        self.assertFalse((self.output / "review.md").exists())

    @patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-secret"}, clear=True)
    @patch("ai_review.__main__.generate_review")
    @patch("ai_review.__main__.collect_context")
    def test_findings_are_success_and_produce_review_artifacts(self, collect, generate):
        collect.return_value = make_context()
        generate.return_value = make_result()
        code, logs = self.run_cli()
        self.assertEqual(code, 0)
        generate.assert_called_once()
        self.assertEqual(json.loads((self.output / "context.json").read_text(encoding="utf-8")), make_context())
        self.assertEqual(json.loads((self.output / "result.json").read_text(encoding="utf-8")), make_result())
        markdown = (self.output / "review.md").read_text(encoding="utf-8")
        self.assertIn(make_result()["findings"][0]["description"], markdown)
        self.assertNotIn("test-secret", logs + markdown)

    @patch.dict(os.environ, {}, clear=True)
    @patch("ai_review.__main__.generate_review")
    @patch("ai_review.__main__.collect_context")
    def test_missing_secret_exits_with_helpful_error(self, collect, generate):
        collect.return_value = make_context()
        code, logs = self.run_cli()
        self.assertEqual(code, 1)
        self.assertIn("DEEPSEEK_API_KEY", logs)
        generate.assert_not_called()

    @patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-secret"}, clear=True)
    @patch("ai_review.__main__.generate_review")
    @patch("ai_review.__main__.collect_context")
    def test_provider_failure_is_error_without_success_report(self, collect, generate):
        collect.return_value = make_context()
        generate.side_effect = ReviewError("Сервис временно недоступен.")
        code, logs = self.run_cli()
        self.assertEqual(code, 1)
        self.assertIn("Сервис временно недоступен", logs)
        self.assertNotIn("test-secret", logs)
        self.assertFalse((self.output / "review.md").exists())

    @patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-secret"}, clear=True)
    @patch("ai_review.__main__.generate_review")
    @patch("ai_review.__main__.collect_context")
    def test_failed_rerun_removes_stale_success_artifacts(self, collect, generate):
        self.output.mkdir()
        for name in ("review.md", "result.json", "context.json"):
            (self.output / name).write_text("old successful result", encoding="utf-8")
        collect.return_value = make_context()
        generate.side_effect = ReviewError("Ошибка нового запроса.")
        code, _ = self.run_cli()
        self.assertEqual(code, 1)
        self.assertFalse((self.output / "review.md").exists())
        self.assertFalse((self.output / "result.json").exists())
        self.assertEqual(json.loads((self.output / "context.json").read_text(encoding="utf-8")), make_context())

    @patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-secret", "DEEPSEEK_MODEL": "env-model"}, clear=True)
    @patch("ai_review.__main__.generate_review")
    @patch("ai_review.__main__.collect_context")
    def test_explicit_model_overrides_environment(self, collect, generate):
        collect.return_value = make_context()
        generate.return_value = make_result()
        for extra, expected in [((), "env-model"), (("--model", "cli-model"), "cli-model")]:
            with self.subTest(model=expected):
                generate.reset_mock()
                code, _ = self.run_cli(*extra)
                self.assertEqual(code, 0)
                generate.assert_called_once_with(make_context(), "test-secret", expected)

    @patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-secret"}, clear=True)
    @patch("ai_review.__main__.generate_review")
    @patch("ai_review.__main__.collect_context")
    def test_invalid_git_context_prevents_model_call(self, collect, generate):
        collect.side_effect = ReviewError("Не удалось прочитать Git revision.")
        code, logs = self.run_cli()
        self.assertEqual(code, 1)
        self.assertIn("Git revision", logs)
        generate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
