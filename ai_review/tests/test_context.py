"""Integration checks against real temporary Git objects; no model calls."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai_review import ReviewError
from ai_review.context import collect_context


class ContextTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.repo = Path(self.temporary.name)
        self.git("init", "-b", "main")
        self.git("config", "user.name", "Review Test")
        self.git("config", "user.email", "review@example.invalid")
        self.git("config", "core.autocrlf", "false")
        self.git("config", "core.hooksPath", str(self.repo / "no-hooks"))
        self.git("config", "commit.gpgsign", "false")
        self.write("README.md", "# Sessions\nThe load_session function must raise LookupError for an unknown session.\n")
        self.write("docs/architecture.md", "# Session loading\nload_session returns a session or raises LookupError.\n")
        self.write("llm_agent/session.py", "def load_session(key):\n    raise LookupError(key)\n")
        self.write("llm_agent/consumer.py", "from .session import load_session\n\ndef consume(key):\n    return load_session(key)\n")
        self.base = self.commit("initial")

    def git(self, *args: str, input_data: bytes | None = None) -> str:
        result = subprocess.run(
            ["git", "-c", "commit.gpgsign=false", "-C", str(self.repo), *args], input=input_data,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
        )
        return result.stdout.decode("utf-8").strip()

    def write(self, path: str, content: str | bytes):
        destination = self.repo / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content if isinstance(content, bytes) else content.encode("utf-8"))

    def commit(self, message: str) -> str:
        self.git("add", "-A")
        self.git("commit", "-m", message)
        return self.git("rev-parse", "HEAD")

    def test_merge_base_excludes_unrelated_base_changes_and_ignores_worktree(self):
        self.write("docs/architecture.md", "# Sessions\nload_session contract on main: MAIN_BASE_DOCUMENT.\n")
        self.write("llm_agent/base_only.py", "MAIN_ONLY_CHANGE = True\n")
        base = self.commit("main evolves")
        self.git("checkout", "-b", "feature", self.base)
        self.write("llm_agent/session.py", "def load_session(key):\n    return None  # FEATURE_CHANGE\n")
        head = self.commit("feature")
        self.write("llm_agent/session.py", "DO_NOT_READ_WORKTREE\n")
        self.write("docs/architecture.md", "DO_NOT_READ_WORKTREE\n")
        self.write(".env", "SECRET_WORKTREE_MARKER\n")

        context = collect_context(self.repo, base, head)

        self.assertEqual(context["merge_base_sha"], self.base)
        self.assertEqual(context["base_sha"], base)
        self.assertEqual(context["head_sha"], head)
        self.assertEqual(context["retrieval"]["snapshot_sha"], base)
        self.assertEqual([item["path"] for item in context["files"]], ["llm_agent/session.py"])
        self.assertIn("FEATURE_CHANGE", context["diff"])
        self.assertNotIn("MAIN_ONLY_CHANGE", context["diff"])
        serialized = json.dumps(context)
        self.assertNotIn("DO_NOT_READ_WORKTREE", serialized)
        self.assertNotIn("SECRET_WORKTREE_MARKER", serialized)
        self.assertIn("MAIN_BASE_DOCUMENT", serialized)

    def test_add_delete_and_rename_preserve_file_versions(self):
        self.git("mv", "llm_agent/consumer.py", "llm_agent/renamed.py")
        (self.repo / "llm_agent/session.py").unlink()
        self.write("llm_agent/new.py", "def newly_added():\n    return 42\n")
        head = self.commit("add delete rename")

        context = collect_context(self.repo, self.base, head)
        files = {item["path"]: item for item in context["files"]}

        self.assertEqual(files["llm_agent/renamed.py"]["status"], "renamed")
        self.assertEqual(files["llm_agent/renamed.py"]["old_path"], "llm_agent/consumer.py")
        self.assertEqual(files["llm_agent/renamed.py"]["before"], files["llm_agent/renamed.py"]["after"])
        self.assertIsNone(files["llm_agent/session.py"]["after"])
        self.assertEqual(files["llm_agent/session.py"]["status"], "deleted")
        self.assertIsNone(files["llm_agent/new.py"]["before"])
        self.assertEqual(files["llm_agent/new.py"]["status"], "added")

    def test_retrieves_documentation_and_related_code_from_base(self):
        self.write("llm_agent/session.py", "def load_session(key):\n    return None\n")
        head = self.commit("change contract")

        context = collect_context(self.repo, self.base, head)

        self.assertEqual({source["kind"] for source in context["sources"]}, {"documentation", "code"})
        self.assertTrue(any(source["path"] == "llm_agent/consumer.py" for source in context["sources"]))
        self.assertTrue(any("raise LookupError" in source["text"] for source in context["sources"]))
        self.assertEqual(len({source["id"] for source in context["sources"]}), len(context["sources"]))
        self.assertTrue(all(source["start_line"] >= 1 for source in context["sources"]))
        self.assertEqual(context["retrieval"]["method"], "sqlite_fts5_bm25")

    def test_binary_large_symlink_and_excluded_paths_are_reported(self):
        self.write("llm_agent/binary.py", b"load_session\0binary")
        self.write("llm_agent/huge.py", "x" * 40_001)
        self.write(".env", "SECRET_COMMITTED_MARKER")
        self.write("data/runtime.json", "RUNTIME_MARKER")
        self.write("docs/CHAT.md", "CHAT_MARKER")
        self.write("docs/evaluation/result.json", "EVALUATION_MARKER")
        self.write("llm_agent/session.py", "def load_session(key):\n    return None\n")
        self.git("add", "-A")
        oid = self.git("hash-object", "-w", "--stdin", input_data=b"../../.env")
        self.git("update-index", "--add", "--cacheinfo", f"120000,{oid},llm_agent/link.py")
        self.git("commit", "-m", "unsafe and unsupported inputs")
        head = self.git("rev-parse", "HEAD")

        context = collect_context(self.repo, self.base, head)

        skipped = {item["path"]: item["reason"] for item in context["skipped"]}
        self.assertEqual(set(skipped), {"llm_agent/binary.py", "llm_agent/huge.py", "llm_agent/link.py", ".env", "data/runtime.json", "docs/CHAT.md", "docs/evaluation/result.json"})
        self.assertTrue(all(skipped.values()))
        serialized = json.dumps(context)
        for marker in ("SECRET_COMMITTED_MARKER", "RUNTIME_MARKER", "CHAT_MARKER", "EVALUATION_MARKER"):
            self.assertNotIn(marker, serialized)
        self.assertEqual(len(context["files"]), 1)

    def test_file_limit_and_total_budget_are_explicit(self):
        for number in range(3):
            self.write(f"llm_agent/added{number}.py", "def load_session(key):\n    return None\n")
        head = self.commit("many files")
        with patch("ai_review.context.MAX_FILES", 2):
            limited = collect_context(self.repo, self.base, head)
        self.assertEqual(len(limited["files"]), 2)
        self.assertEqual(len(limited["skipped"]), 1)
        self.assertIn("лимит", limited["skipped"][0]["reason"])

        self.write("llm_agent/large_one.py", "# session load_session\n" + "a = 1\n" * 5_000)
        self.write("llm_agent/large_two.py", "# session load_session\n" + "b = 2\n" * 5_000)
        self.write("llm_agent/large_three.py", "# session load_session\n" + "c = 3\n" * 5_000)
        head = self.commit("budget overflow")
        bounded = collect_context(self.repo, self.base, head)
        self.assertTrue(any("бюджет" in item["reason"] for item in bounded["skipped"]))
        self.assertLessEqual(len(json.dumps(bounded, ensure_ascii=False).encode("utf-8")), 180_000)

    def test_retrieval_failure_does_not_silently_disable_rag(self):
        self.write("llm_agent/session.py", "def load_session(key):\n    return None\n")
        head = self.commit("change")
        fake_connection = unittest.mock.MagicMock()
        fake_connection.execute.side_effect = sqlite3.OperationalError("no such module: fts5")
        with patch("ai_review.context.sqlite3.connect", return_value=fake_connection):
            with self.assertRaisesRegex(ReviewError, "FTS5"):
                collect_context(self.repo, self.base, head)
        fake_connection.close.assert_called_once()

    def test_russian_documentation_cannot_consume_code_budget(self):
        for number in range(4):
            self.write(f"docs/session{number}.md", ("load_session загрузка диалога проверяет доступ пользователя.\n" * 220))
            self.write(f"llm_agent/session{number}.py", "def load_session(key):\n" + ("    # load_session проверяет доступ пользователя к диалогу\n" * 220))
        self.write("docs/CHAT.md", "load_session PRIVATE_TRAINING_MARKER\n")
        self.write(".env", "load_session PRIVATE_ENV_MARKER\n")
        base = self.commit("large trusted corpus")
        self.write("llm_agent/session.py", "def load_session(key):\n    return None\n")
        head = self.commit("change contract")

        context = collect_context(self.repo, base, head)

        for kind in ("documentation", "code"):
            sources = [source for source in context["sources"] if source["kind"] == kind]
            self.assertTrue(sources, kind)
            self.assertLessEqual(len(sources), 4)
            self.assertLessEqual(sum(len(json.dumps(source, ensure_ascii=False).encode("utf-8")) for source in sources), 12_000)
        serialized = json.dumps(context)
        self.assertNotIn("PRIVATE_TRAINING_MARKER", serialized)
        self.assertNotIn("PRIVATE_ENV_MARKER", serialized)


if __name__ == "__main__":
    unittest.main()
