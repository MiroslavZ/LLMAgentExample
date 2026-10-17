"""Git-контекст и настоящий Streamable HTTP round-trip без LLM и токенов."""

import asyncio
import json
import os
import shutil
import socket
import subprocess
import tempfile
import unittest
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import patch

import anyio
import uvicorn

from LocalMCPServer.server import DEFAULT_PROJECT_ROOT, GitContextError, create_server, read_git_branch
from llm_agent.mcp_client import call_tool, get_tools
from llm_agent.mcp_config import MCPServer


class RepositoryFixture:
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()

    def git(self, *arguments):
        environment = {key: value for key, value in os.environ.items() if not key.upper().startswith("GIT_")}
        return subprocess.run(
            ["git", "-C", str(self.root), *arguments], capture_output=True,
            encoding="utf-8", check=True, env=environment,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        ).stdout.strip()

    def initialize(self):
        self.git("init", "--quiet", "--initial-branch=main")

    def commit(self):
        self.git(
            "-c", "user.name=MCP Test", "-c", "user.email=mcp@example.test",
            "-c", "commit.gpgsign=false", "-c", "core.hooksPath=disabled-hooks",
            "commit", "--quiet", "--allow-empty", "-m", "initial",
        )
        return self.git("rev-parse", "HEAD")


@unittest.skipUnless(shutil.which("git"), "Для теста требуется Git")
class GitContextTests(RepositoryFixture, unittest.TestCase):
    def test_default_project_is_anchored_to_server_file(self):
        self.assertEqual(DEFAULT_PROJECT_ROOT, Path(__file__).resolve().parent.parent)

    def test_unborn_switch_and_detached_head(self):
        self.initialize()
        self.assertEqual(read_git_branch(self.root), {
            "project_root": str(self.root), "branch": "main", "commit": None, "detached": False,
        })
        commit = self.commit()
        self.git("switch", "--quiet", "-c", "feature/mcp")
        self.assertEqual(read_git_branch(self.root), {
            "project_root": str(self.root), "branch": "feature/mcp", "commit": commit, "detached": False,
        })
        self.git("checkout", "--quiet", "--detach", commit)
        self.assertEqual(read_git_branch(self.root), {
            "project_root": str(self.root), "branch": None, "commit": commit, "detached": True,
        })

    def test_non_repository_and_nested_directory_are_reported(self):
        with self.assertRaisesRegex(GitContextError, "не является корнем"):
            read_git_branch(self.root)
        self.initialize()
        nested = self.root / "ordinary-directory"
        nested.mkdir()
        with self.assertRaisesRegex(GitContextError, "не является корнем"):
            read_git_branch(nested)

    def test_git_environment_cannot_redirect_project(self):
        self.initialize()
        with patch.dict(os.environ, {"GIT_DIR": str(self.root / "missing"), "GIT_WORK_TREE": "missing"}):
            self.assertEqual(read_git_branch(self.root)["branch"], "main")

    def test_git_errors_hide_command_output_and_use_timeout_without_shell(self):
        for error, expected in [
            (subprocess.TimeoutExpired("private-command", 5, stderr="private-output"), "не ответил"),
            (FileNotFoundError("private-path"), "Git не найден"),
            (OSError("private-error"), "Не удалось запустить"),
        ]:
            with self.subTest(error=type(error).__name__):
                with patch("LocalMCPServer.server.subprocess.run", side_effect=error) as run:
                    with self.assertRaisesRegex(GitContextError, expected) as captured:
                        read_git_branch(self.root)
                self.assertNotIn("private", str(captured.exception))
                self.assertFalse(run.call_args.kwargs["shell"])
                self.assertEqual(run.call_args.kwargs["timeout"], 5)
                self.assertEqual(run.call_args.args[0][:3], ["git", "-C", str(self.root)])


@asynccontextmanager
async def serve(project_root):
    app = create_server(project_root).streamable_http_app(json_response=True, stateless_http=True)
    server = uvicorn.Server(uvicorn.Config(app, http="h11", log_level="error", log_config=None, lifespan="on"))
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.setblocking(False)
        port = listener.getsockname()[1]
        serving = asyncio.create_task(server.serve(sockets=[listener]))
        try:
            with anyio.fail_after(10):
                while not server.started:
                    if serving.done():
                        await serving
                        raise RuntimeError("MCP-сервер завершился до запуска")
                    await asyncio.sleep(0.01)
            yield MCPServer("project", "Project context", f"http://127.0.0.1:{port}/mcp")
        finally:
            server.should_exit = True
            with anyio.fail_after(10):
                await serving


@unittest.skipUnless(shutil.which("git"), "Для теста требуется Git")
class ProjectMCPProtocolTests(RepositoryFixture, unittest.IsolatedAsyncioTestCase):
    async def test_real_client_discovers_tool_and_reads_live_git_changes(self):
        self.initialize()
        async with serve(self.root) as server:
            discovery = await get_tools(server, timeout=10)
            self.assertEqual([tool.name for tool in discovery.tools], ["git_branch"])
            tool = discovery.tools[0]
            self.assertEqual(tool.input_schema["properties"], {})
            self.assertFalse(tool.input_schema["additionalProperties"])
            self.assertTrue(tool.annotations.read_only_hint)

            result = await call_tool(server, "git_branch", {}, timeout=10)
            self.assertFalse(result.is_error)
            self.assertEqual(result.structured_content, {
                "project_root": str(self.root), "branch": "main", "commit": None, "detached": False,
            })
            self.assertEqual(json.loads(result.content[0].text), result.structured_content)

            commit = self.commit()
            self.git("switch", "--quiet", "-c", "feature/live-context")
            result = await call_tool(server, "git_branch", {}, timeout=10)
            self.assertEqual(result.structured_content["branch"], "feature/live-context")
            self.assertEqual(result.structured_content["commit"], commit)

            self.git("checkout", "--quiet", "--detach", commit)
            result = await call_tool(server, "git_branch", {}, timeout=10)
            self.assertEqual(result.structured_content, {
                "project_root": str(self.root), "branch": None, "commit": commit, "detached": True,
            })

    async def test_tool_rejects_arguments_unknown_tools_and_non_git_project(self):
        async with serve(self.root) as server:
            for name, arguments, expected in [
                ("git_branch", {"project_root": "another-project"}, "не принимает аргументов"),
                ("git_branch", {"command": "git checkout other"}, "не принимает аргументов"),
                ("git_checkout", {}, "Неизвестный инструмент"),
                ("git_branch", {}, "не является корнем"),
            ]:
                with self.subTest(name=name, arguments=arguments):
                    result = await call_tool(server, name, arguments, timeout=10)
                    self.assertTrue(result.is_error)
                    self.assertIsNone(result.structured_content)
                    self.assertIn(expected, result.content[0].text)


if __name__ == "__main__":
    unittest.main()
