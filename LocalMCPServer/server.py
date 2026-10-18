"""MCP-инструменты файлов и Git фиксированного проекта (Streamable HTTP)."""

import argparse
import json
import os
import subprocess
from pathlib import Path

import anyio
import uvicorn
from jsonschema import Draft202012Validator, ValidationError
from mcp import types
from mcp.server import Server, ServerRequestContext

# Поддерживаем прежний запуск python LocalMCPServer/server.py и запуск через -m.
if __package__:
    from .file_tools import file_tools
    from .project_files import ProjectFileError, ProjectFiles
else:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from LocalMCPServer.file_tools import file_tools
    from LocalMCPServer.project_files import ProjectFileError, ProjectFiles


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parent.parent
GIT_TIMEOUT = 5


class GitContextError(RuntimeError):
    """Безопасная диагностическая ошибка без stderr Git и окружения процесса."""


def _git(project_root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    # Переменные Git не должны перенаправлять запрос к другому репозиторию.
    environment = {key: value for key, value in os.environ.items() if not key.upper().startswith("GIT_")}
    environment.update(GIT_OPTIONAL_LOCKS="0", GIT_TERMINAL_PROMPT="0")
    try:
        return subprocess.run(
            ["git", "-C", str(project_root), *arguments],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            encoding="utf-8", errors="replace", timeout=GIT_TIMEOUT, check=False,
            shell=False, env=environment,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except subprocess.TimeoutExpired:
        raise GitContextError("Git не ответил за отведённое время.") from None
    except FileNotFoundError:
        raise GitContextError("Git не найден. Установите Git и добавьте его в PATH.") from None
    except OSError:
        raise GitContextError("Не удалось запустить Git для проекта.") from None


def read_git_branch(project_root: Path) -> dict[str, str | bool | None]:
    """Читает актуальный HEAD; новый репозиторий и detached HEAD допустимы."""
    root = project_root.expanduser().resolve()
    top_level = _git(root, "rev-parse", "--show-toplevel")
    if top_level.returncode != 0 or Path(top_level.stdout.strip()).resolve() != root:
        raise GitContextError("Настроенная папка не является корнем доступного Git-репозитория.")

    branch_result = _git(root, "symbolic-ref", "--quiet", "--short", "HEAD")
    if branch_result.returncode not in (0, 1):
        raise GitContextError("Не удалось прочитать текущую Git-ветку.")
    branch = branch_result.stdout.strip() if branch_result.returncode == 0 else None

    commit_result = _git(root, "rev-parse", "--verify", "--quiet", "HEAD")
    if commit_result.returncode not in (0, 1) or (commit_result.returncode == 1 and branch is None):
        raise GitContextError("Не удалось прочитать текущий Git-коммит.")
    commit = commit_result.stdout.strip() if commit_result.returncode == 0 else None
    return {"project_root": str(root), "branch": branch, "commit": commit, "detached": branch is None}


def create_server(project_root: str | Path = DEFAULT_PROJECT_ROOT) -> Server:
    """Закрепляет корень при запуске; файловым операциям Git не требуется."""
    root = Path(project_root).expanduser().resolve()
    workspace = ProjectFiles(root)
    git_tool = types.Tool(
        name="git_branch",
        description="Справка о текущей Git-ветке и коммите проекта. Вызывай, когда нужна ветка или коммит. "
                    "Файловые операции не требуют Git и используют уже настроенный корень.",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        output_schema={
            "type": "object",
            "properties": {
                "project_root": {"type": "string"},
                "branch": {"type": ["string", "null"]},
                "commit": {"type": ["string", "null"]},
                "detached": {"type": "boolean"},
            },
            "required": ["project_root", "branch", "commit", "detached"],
            "additionalProperties": False,
        },
        annotations=types.ToolAnnotations(
            read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=False,
        ),
    )
    definitions = [git_tool, *file_tools()]
    handlers = {tool.name: getattr(workspace, tool.name) for tool in definitions[1:]}
    handlers["git_branch"] = lambda: read_git_branch(root)
    validators = {tool.name: Draft202012Validator(tool.input_schema) for tool in definitions}

    async def list_tools(
        context: ServerRequestContext, params: types.PaginatedRequestParams | None,
    ) -> types.ListToolsResult:
        return types.ListToolsResult(tools=definitions)

    async def call_tool(
        context: ServerRequestContext, params: types.CallToolRequestParams,
    ) -> types.CallToolResult:
        code = ""
        try:
            if params.name not in handlers:
                raise ProjectFileError("unknown_tool", "Неизвестный инструмент. Получите каталог tools/list.")
            arguments = params.arguments if params.arguments is not None else {}
            validators[params.name].validate(arguments)
            result = await anyio.to_thread.run_sync(lambda: handlers[params.name](**arguments))
        except ValidationError:
            code, message = "invalid_arguments", "Аргументы инструмента не соответствуют JSON Schema."
        except ProjectFileError as error:
            code, message = error.code, str(error)
        except GitContextError as error:
            code, message = "git_error", str(error)
        except OSError:
            code, message = "io_error", "Операция с файлами проекта не выполнена: ошибка доступа или ввода-вывода."
        if code:
            result = {"code": code, "error": message}
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=json.dumps(result, ensure_ascii=False))],
            structured_content=result, is_error=bool(code),
        )

    return Server("project-context", version="2.0.0", on_list_tools=list_tools, on_call_tool=call_tool)


def main() -> int:
    parser = argparse.ArgumentParser(description="Локальный MCP-сервер для чтения, поиска и изменения файлов проекта")
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT,
                        help="Корень проекта; по умолчанию родитель папки LocalMCPServer")
    parser.add_argument("--port", type=int, default=8001, help="Локальный TCP-порт (по умолчанию 8001)")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("Порт должен быть числом от 1 до 65535")
    try:
        server = create_server(args.project_root)
    except (ProjectFileError, OSError) as error:
        parser.error(str(error))
    uvicorn.run(
        server.streamable_http_app(json_response=True, stateless_http=True),
        host="127.0.0.1", port=args.port, log_level="info", access_log=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
