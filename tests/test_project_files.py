"""Реальные файловые операции в изолированном проекте, без LLM и сети."""

import hashlib
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from LocalMCPServer import project_files
from LocalMCPServer.project_files import ProjectFileError, ProjectFiles


class ProjectFilesTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.root = self.base / "project"
        self.root.mkdir()
        self.files = ProjectFiles(self.root)

    def write(self, path, content):
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content.encode("utf-8") if isinstance(content, str) else content)
        return target

    def seed_project(self):
        self.write("src/client.py", "class ApiClient:\n    pass\n")
        self.write("src/app.py", "from client import ApiClient\nclient = ApiClient()\n")
        self.write("README.md", "# Проект\nКлиент ApiClient пока не описан.\n")

    def assert_error(self, code, function, *args, **kwargs):
        with self.assertRaises(ProjectFileError) as caught:
            function(*args, **kwargs)
        self.assertEqual(caught.exception.code, code)
        return caught.exception

    def test_find_usage_read_three_files_create_and_update_documentation(self):
        self.seed_project()
        found = self.files.search_text("ApiClient")
        paths = {match["path"] for match in found["matches"]}
        self.assertEqual(paths, {"README.md", "src/app.py", "src/client.py"})
        self.assertTrue(all(match["line"] >= 1 for match in found["matches"]))
        for path in paths:
            read = self.files.read_file(path)
            self.assertIn("ApiClient", read["content"])
            self.assertEqual(read["sha256"], hashlib.sha256((self.root / path).read_bytes()).hexdigest())
        created = self.files.create_file("docs/API.md", "# API\nApiClient доступен из src/client.py.\n")
        self.assertTrue(created["changed"])
        self.assertIn("+# API", created["diff"])
        self.assertIn("docs/API.md", created["diff"])
        self.assertEqual(created["bytes_written"], (self.root / "docs/API.md").stat().st_size)
        before = self.files.read_file("README.md")
        edited = self.files.edit_file(
            "README.md", "Клиент ApiClient пока не описан.",
            "Клиент ApiClient описан в docs/API.md.", before["sha256"],
        )
        self.assertTrue(edited["changed"])
        self.assertIn("-Клиент ApiClient пока не описан.", edited["diff"])
        self.assertIn("+Клиент ApiClient описан в docs/API.md.", edited["diff"])
        self.assertIn("docs/API.md", self.files.read_file("README.md")["content"])

    def test_refused_edits_and_create_conflict_preserve_bytes(self):
        original = b"same\nsame\n"
        target = self.write("notes.md", original)
        snapshot = self.files.read_file("notes.md")
        self.assert_error("already_exists", self.files.create_file, "notes.md", "replacement")
        self.assert_error("ambiguous_match", self.files.edit_file,
                          "notes.md", "same", "changed", snapshot["sha256"])
        self.assert_error("text_not_found", self.files.edit_file,
                          "notes.md", "missing", "changed", snapshot["sha256"])
        self.assertEqual(target.read_bytes(), original)
        target.write_bytes(b"User edit\n")
        self.assert_error("stale_file", self.files.edit_file,
                          "notes.md", "same", "changed", snapshot["sha256"])
        self.assertEqual(target.read_bytes(), b"User edit\n")

    def test_overlapping_matches_are_ambiguous_and_preserve_file(self):
        target = self.write("notes.md", "aaaa")
        snapshot = self.files.read_file("notes.md")
        self.assert_error("ambiguous_match", self.files.edit_file,
                          "notes.md", "aaa", "changed", snapshot["sha256"])
        self.assertEqual(target.read_bytes(), b"aaaa")

    def test_crlf_and_missing_final_newline_survive_targeted_edit(self):
        target = self.write("notes.md", b"first\r\nold\r\nlast")
        read = self.files.read_file("notes.md")
        self.assertEqual(read["content"], "first\r\nold\r\nlast")
        self.assertEqual(read["numbered_content"], "1: first\r\n2: old\r\n3: last")
        result = self.files.edit_file("notes.md", "old", "new", read["sha256"])
        self.assertEqual(target.read_bytes(), b"first\r\nnew\r\nlast")
        self.assertEqual(result["sha256"], hashlib.sha256(target.read_bytes()).hexdigest())
        self.assertIn("No newline at end of file", result["diff"])

    def test_noop_edit_has_no_diff_and_keeps_bytes(self):
        target = self.write("notes.md", "Привет\r\n")
        read = self.files.read_file("notes.md")
        result = self.files.edit_file("notes.md", "Привет", "Привет", read["sha256"])
        self.assertFalse(result["changed"])
        self.assertEqual(result["diff"], "")
        self.assertEqual(target.read_bytes(), "Привет\r\n".encode())

    def test_list_pagination_is_stable_and_filter_applies_to_nested_files(self):
        for name in ("src/c.py", "src/a.py", "src/nested/b.py", "notes.md"):
            self.write(name, "content\n")
        first = self.files.list_files(path="src", pattern="*.py", limit=2)
        second = self.files.list_files(path="src", pattern="*.py", limit=2, offset=first["next_offset"])
        self.assertTrue(first["truncated"])
        self.assertFalse(second["truncated"])
        self.assertIsNone(second["next_offset"])
        actual = first["files"] + second["files"]
        self.assertEqual(actual, sorted(["src/a.py", "src/c.py", "src/nested/b.py"]))
        self.assertEqual(len(actual), len(set(actual)))

    def test_search_pagination_preserves_all_matches_and_line_numbers(self):
        self.write("a.py", "needle\nother\nneedle\n")
        self.write("nested/b.py", "NEEDLE\nneedle\n")
        self.write("ignored.md", "needle\n")
        first = self.files.search_text("needle", pattern="*.py", limit=2)
        second = self.files.search_text("needle", pattern="*.py", limit=2, offset=first["next_offset"])
        found = [(item["path"], item["line"]) for item in first["matches"] + second["matches"]]
        self.assertEqual(found, [("a.py", 1), ("a.py", 3), ("nested/b.py", 2)])
        self.assertTrue(first["truncated"])
        self.assertFalse(second["truncated"])
        self.assertIsNone(second["next_offset"])
        insensitive = self.files.search_text("needle", pattern="*.py", case_sensitive=False)
        self.assertEqual(len(insensitive["matches"]), 4)

    def test_search_reports_incomplete_when_a_file_becomes_unreadable(self):
        self.write("a.py", "needle\n")
        self.write("b.py", "needle\n")
        read = self.files._read
        for error_code in ("io_error", "not_found"):
            def unavailable(path):
                if path.name == "b.py":
                    raise ProjectFileError(error_code, "Файл недоступен во время чтения")
                return read(path)

            with self.subTest(error_code=error_code), patch.object(self.files, "_read", side_effect=unavailable):
                result = self.files.search_text("needle")
                self.assertEqual([match["path"] for match in result["matches"]], ["a.py"])
                self.assertFalse(result["scan_complete"])
                self.assertTrue(result["truncated"])
                self.assertIsNone(result["total_matches"])
                self.assertEqual(result["skipped_files"], {error_code: 1})

    def test_read_range_and_continuation_reconstruct_file(self):
        text = "".join(f"line {number:04d} " + "x" * 80 + "\n" for number in range(200))
        self.write("large.md", text)
        result = self.files.read_file("large.md", start_line=2, end_line=3)
        self.assertEqual(result["content"], "".join(text.splitlines(keepends=True)[1:3]))
        self.assertEqual((result["start_line"], result["end_line"], result["total_lines"]), (2, 3, 200))
        self.assertEqual(result["numbered_content"], "".join(
            f"{number}: {line}" for number, line in enumerate(text.splitlines(keepends=True)[1:3], 2)
        ))
        chunks, start = [], 1
        while start is not None:
            result = self.files.read_file("large.md", start_line=start)
            chunks.append(result["content"])
            next_start = result["next_start_line"]
            if next_start is not None:
                self.assertTrue(result["truncated"])
                self.assertGreater(next_start, start)
            start = next_start
        self.assertEqual("".join(chunks), text)

    def test_oversized_and_binary_files_are_not_read_or_searchable(self):
        self.write("oversized.txt", b"needle" + b"x" * project_files.MAX_FILE_BYTES)
        self.write("binary.dat", b"needle\x00binary")
        self.write("valid.txt", "needle\n")
        self.assert_error("too_large", self.files.read_file, "oversized.txt")
        self.assert_error("not_text", self.files.read_file, "binary.dat")
        self.assertEqual([item["path"] for item in self.files.search_text("needle")["matches"]], ["valid.txt"])
        self.assert_error("too_large", self.files.create_file,
                          "new.txt", "x" * (project_files.MAX_FILE_BYTES + 1))
        self.assertFalse((self.root / "new.txt").exists())

    def test_numbered_reads_stay_bounded_without_losing_blank_or_escaped_lines(self):
        for text in ("\n" * 4000, ('"' * 100 + "\n") * 100):
            with self.subTest(blank=text.startswith("\n")):
                self.write("notes.md", text)
                start, chunks = 1, []
                while start is not None:
                    read = self.files.read_file("notes.md", start_line=start)
                    self.assertLess(len(json.dumps(read, ensure_ascii=False)), 20_000)
                    self.assertTrue(read["numbered_content"].startswith(f"{start}: "))
                    self.assertEqual(read["numbered_content"], "".join(
                        f"{number}: {line}" for number, line in enumerate(read["content"].splitlines(keepends=True), start)
                    ))
                    chunks.append(read["content"])
                    start = read["next_start_line"]
                self.assertEqual("".join(chunks), text)

    def test_long_line_returns_explicit_error_instead_of_losing_continuation(self):
        self.write("long.txt", "x" * (project_files.MAX_READ_CHARS + 1))
        self.assert_error("line_too_long", self.files.read_file, "long.txt")

    def test_diff_truncation_does_not_truncate_saved_content(self):
        text = "".join(f"строка {number:04d}\n" for number in range(1500))
        result = self.files.create_file("long.md", text)
        self.assertTrue(result["diff_truncated"])
        self.assertEqual((self.root / "long.md").read_bytes(), text.encode())
        self.assertEqual(result["bytes_written"], len(text.encode()))

    def test_forbidden_files_are_blocked_directly_and_excluded_from_discovery(self):
        forbidden = (".env", ".env.local", ".git/config", ".venv/lib.py",
                     "__pycache__/private.py", ".aws/credentials", "private.pem")
        for path in forbidden:
            target = self.write(path, "secret-needle")
            with self.subTest(path=path):
                self.assert_error("path_forbidden", self.files.read_file, path)
                self.assert_error("path_forbidden", self.files.create_file, path, "new")
                self.assert_error("path_forbidden", self.files.edit_file,
                                  path, "secret-needle", "new", hashlib.sha256(target.read_bytes()).hexdigest())
                self.assertEqual(target.read_bytes(), b"secret-needle")
        self.write("visible.py", "secret-needle")
        self.assertEqual(self.files.list_files()["files"], ["visible.py"])
        self.assertEqual([item["path"] for item in self.files.search_text("secret-needle")["matches"]], ["visible.py"])

    def test_paths_outside_root_cannot_be_read_created_or_edited(self):
        outside = self.base / "outside.txt"
        outside.write_text("outside", encoding="utf-8")
        digest = hashlib.sha256(outside.read_bytes()).hexdigest()
        for path in ("../outside.txt", "..\\outside.txt", str(outside)):
            with self.subTest(path=path):
                for function, args in (
                    (self.files.read_file, (path,)),
                    (self.files.create_file, (path, "changed")),
                    (self.files.edit_file, (path, "outside", "changed", digest)),
                ):
                    with self.assertRaises(ProjectFileError):
                        function(*args)
        self.assertEqual(outside.read_text(encoding="utf-8"), "outside")

    def test_invalid_arguments_are_rejected_before_mutation(self):
        self.write("notes.md", "unchanged")
        calls = (
            (self.files.list_files, {"limit": 0}),
            (self.files.list_files, {"offset": -1}),
            (self.files.search_text, {"query": ""}),
            (self.files.read_file, {"path": "notes.md", "start_line": 0}),
            (self.files.read_file, {"path": "notes.md", "start_line": 3, "end_line": 1}),
        )
        for function, arguments in calls:
            with self.subTest(function=function.__name__, arguments=arguments):
                self.assert_error("invalid_argument", function, **arguments)
        self.assertEqual((self.root / "notes.md").read_text(), "unchanged")

    def test_symlink_file_cannot_escape_direct_read_or_search(self):
        outside = self.base / "outside.txt"
        outside.write_text("outside-secret", encoding="utf-8")
        link = self.root / "alias.txt"
        try:
            link.symlink_to(outside)
        except OSError as error:
            self.skipTest(f"Создание symlink недоступно: {error}")
        self.addCleanup(link.unlink, missing_ok=True)
        with self.assertRaises(ProjectFileError):
            self.files.read_file("alias.txt")
        with self.assertRaises(ProjectFileError):
            self.files.create_file("alias.txt", "changed")
        self.assertEqual(self.files.search_text("outside-secret")["matches"], [])
        self.assertNotIn("alias.txt", self.files.list_files()["files"])
        self.assertEqual(outside.read_text(), "outside-secret")

    @unittest.skipUnless(os.name == "nt", "Windows junction")
    def test_windows_junction_cannot_escape_root_in_search_or_write(self):
        outside = self.base / "outside"
        outside.mkdir()
        (outside / "secret.txt").write_text("junction-secret", encoding="utf-8")
        junction = self.root / "linked"
        command = "$ErrorActionPreference='Stop'; New-Item -ItemType Junction -Path $env:PROJECT_TEST_LINK -Target $env:PROJECT_TEST_TARGET | Out-Null"
        result = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
            env={**os.environ, "PROJECT_TEST_LINK": str(junction), "PROJECT_TEST_TARGET": str(outside)},
            capture_output=True, timeout=15,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if result.returncode:
            self.skipTest("Создание junction недоступно в окружении")
        self.addCleanup(os.rmdir, junction)
        with self.assertRaises(ProjectFileError):
            self.files.read_file("linked/secret.txt")
        with self.assertRaises(ProjectFileError):
            self.files.create_file("linked/new.txt", "changed")
        self.assertEqual(self.files.search_text("junction-secret")["matches"], [])
        self.assertFalse((outside / "new.txt").exists())


if __name__ == "__main__":
    unittest.main()
