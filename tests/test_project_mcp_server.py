"""Настоящий локальный MCP HTTP и клиент проекта, без внешнего API."""

import asyncio
import hashlib
import json
import os
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import unittest
from contextlib import contextmanager
from pathlib import Path

import uvicorn

from LocalMCPServer.server import GitContextError, create_server, read_git_branch
from llm_agent.mcp_client import call_tool, get_tools
from llm_agent.mcp_config import MCPServer


@contextmanager
def running_project_server(root: Path):
    """Поднимает сервер на свободном loopback-порту и обязательно завершает поток."""
    mcp = create_server(root)
    app = mcp.streamable_http_app(json_response=True, stateless_http=True)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(
            app, host="127.0.0.1", port=port, log_level="error", access_log=False,
            log_config=None, timeout_graceful_shutdown=2,
        ))
        failures = []

        def serve():
            try:
                server.run(sockets=[listener])
            except BaseException as error:
                failures.append(error)

        worker = threading.Thread(target=serve, name="project-mcp-test", daemon=True)
        worker.start()
        try:
            deadline = time.monotonic() + 10
            while not server.started:
                if failures or not worker.is_alive():
                    raise RuntimeError("Тестовый MCP-сервер завершился при запуске") from (failures[0] if failures else None)
                if time.monotonic() >= deadline:
                    raise TimeoutError("Тестовый MCP-сервер не запустился")
                time.sleep(0.01)
            yield MCPServer("project-test", "Project test", f"http://127.0.0.1:{port}/mcp", enabled=True)
        finally:
            server.should_exit = True
            worker.join(timeout=5)
            if worker.is_alive():
                server.force_exit = True
                worker.join(timeout=2)
            if worker.is_alive():
                raise RuntimeError("Поток тестового MCP-сервера не завершился")
        if failures:
            raise RuntimeError("Ошибка потока тестового MCP-сервера") from failures[0]


class ProjectMCPServerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_discovery_exposes_six_tools_with_root_fixed_by_server(self):
        async def scenario(config):
            discovery = await get_tools(config)
            tools = {tool.name: tool for tool in discovery.tools}
            self.assertEqual(set(tools), {
                "git_branch", "list_files", "search_text", "read_file", "create_file", "edit_file",
            })
            for tool in tools.values():
                self.assertFalse(tool.input_schema.get("additionalProperties", True))
                self.assertNotIn("project_root", tool.input_schema["properties"])
            self.assertIn("expected_sha256", tools["edit_file"].input_schema["required"])
            self.assertTrue(tools["read_file"].annotations.read_only_hint)
            self.assertFalse(tools["create_file"].annotations.read_only_hint)
        with running_project_server(self.root) as config:
            asyncio.run(scenario(config))

    def test_client_can_search_read_create_and_edit_files_without_git(self):
        (self.root / "src").mkdir()
        (self.root / "src/a.py").write_text("class Component:\n    pass\n", encoding="utf-8")
        (self.root / "src/b.py").write_text("from a import Component\n", encoding="utf-8")
        (self.root / "README.md").write_text("# Component\nОписание устарело.\n", encoding="utf-8")

        async def successful(config, name, arguments):
            result = await call_tool(config, name, arguments)
            self.assertFalse(result.is_error, result.content)
            self.assertIsInstance(result.structured_content, dict)
            text = "".join(item.text for item in result.content if item.type == "text")
            self.assertEqual(json.loads(text), result.structured_content)
            return result.structured_content

        async def scenario(config):
            listed = await successful(config, "list_files", {})
            self.assertEqual(set(listed["files"]), {"src/a.py", "src/b.py", "README.md"})
            found = await successful(config, "search_text", {"query": "Component"})
            self.assertEqual({item["path"] for item in found["matches"]}, set(listed["files"]))
            snapshots = {}
            for path in listed["files"]:
                snapshots[path] = await successful(config, "read_file", {"path": path})
            created = await successful(config, "create_file", {
                "path": "docs/component.md", "content": "# Component\nРеализация: src/a.py.\n",
            })
            self.assertTrue(created["changed"])
            self.assertIn("+# Component", created["diff"])
            edited = await successful(config, "edit_file", {
                "path": "README.md", "old_text": "Описание устарело.",
                "new_text": "Описание: docs/component.md.",
                "expected_sha256": snapshots["README.md"]["sha256"],
            })
            self.assertTrue(edited["changed"])
            self.assertEqual(edited["sha256"], hashlib.sha256((self.root / "README.md").read_bytes()).hexdigest())
            git = await call_tool(config, "git_branch", {})
            self.assertTrue(git.is_error)
            self.assertIn("error", git.structured_content)
        with running_project_server(self.root) as config:
            asyncio.run(scenario(config))
        self.assertIn("docs/component.md", (self.root / "README.md").read_text(encoding="utf-8"))
        self.assertTrue((self.root / "docs/component.md").is_file())

    def test_invalid_schema_arguments_fail_before_any_write(self):
        async def scenario(config):
            invalid = (
                ("create_file", {"path": "new.md", "content": "oops", "project_root": str(self.root)}),
                ("create_file", {"path": "new.md"}),
                ("create_file", {"path": "new.md", "content": 42}),
                ("edit_file", {"path": "new.md", "old_text": "a", "new_text": "b"}),
                ("list_files", {"limit": True}),
                ("git_branch", {"path": "."}),
            )
            for name, arguments in invalid:
                with self.subTest(tool=name, arguments=arguments):
                    result = await call_tool(config, name, arguments)
                    self.assertTrue(result.is_error)
                    self.assertEqual(result.structured_content["code"], "invalid_arguments")
            self.assertFalse((self.root / "new.md").exists())
        with running_project_server(self.root) as config:
            asyncio.run(scenario(config))

    def test_tool_errors_survive_http_and_client_remains_usable(self):
        original = self.root / "notes.md"
        original.write_text("original", encoding="utf-8")

        async def scenario(config):
            for name, arguments, code in (
                ("read_file", {"path": "missing.txt"}, "not_found"),
                ("read_file", {"path": ".env"}, "path_forbidden"),
                ("create_file", {"path": "notes.md", "content": "overwrite"}, "already_exists"),
            ):
                with self.subTest(tool=name, path=arguments["path"]):
                    result = await call_tool(config, name, arguments)
                    self.assertTrue(result.is_error)
                    self.assertEqual(result.structured_content["code"], code)
            unknown = await call_tool(config, "missing_tool", {})
            self.assertTrue(unknown.is_error)
            read = await call_tool(config, "read_file", {"path": "notes.md"})
            self.assertFalse(read.is_error)
            self.assertEqual(read.structured_content["content"], "original")
        with running_project_server(self.root) as config:
            asyncio.run(scenario(config))
        self.assertEqual(original.read_text(), "original")


@unittest.skipUnless(shutil.which("git"), "Git недоступен")
class ProjectGitContextTests(unittest.TestCase):
    def test_unborn_branch_committed_branch_and_detached_head(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def git(*arguments):
                return subprocess.run(
                    ["git", "-c", f"core.hooksPath={root / 'no-hooks'}", "-C", str(root), *arguments],
                    env={key: value for key, value in os.environ.items() if not key.upper().startswith("GIT_")},
                    capture_output=True, text=True,
                    encoding="utf-8", check=True, timeout=10,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                ).stdout.strip()

            git("init", "--initial-branch=demo")
            unborn = read_git_branch(root)
            self.assertEqual(unborn["branch"], "demo")
            self.assertIsNone(unborn["commit"])
            self.assertFalse(unborn["detached"])
            (root / "README.md").write_text("test\n", encoding="utf-8")
            git("add", "README.md")
            git("-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                "-c", "commit.gpgsign=false", "commit", "--no-verify", "-m", "fixture")
            commit = git("rev-parse", "HEAD")
            attached = read_git_branch(root)
            self.assertEqual(attached["commit"], commit)
            self.assertEqual(attached["branch"], "demo")
            git("checkout", "--detach", commit)
            detached = read_git_branch(root)
            self.assertTrue(detached["detached"])
            self.assertIsNone(detached["branch"])
            self.assertEqual(detached["commit"], commit)
            (root / "subdir").mkdir()
            with self.assertRaises(GitContextError):
                read_git_branch(root / "subdir")


if __name__ == "__main__":
    unittest.main()
