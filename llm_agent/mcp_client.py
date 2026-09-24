"""Каталог и вызовы MCP-инструментов через Streamable HTTP."""

import argparse
import asyncio
import json
import math
import os
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import anyio
import httpx2
from dotenv import load_dotenv
from mcp import ClientSession, Tool, types
from mcp.client.streamable_http import streamable_http_client

from .mcp_config import GITHUB_MCP_URL, GITHUB_TOKEN_ENV, MCPServer


class MCPConnectionError(RuntimeError):
    """Понятная пользователю ошибка без тела ответа и секретов."""


@dataclass(frozen=True)
class MCPDiscovery:
    server_name: str
    server_version: str
    protocol_version: str
    tools: tuple[Tool, ...]


def _causes(error: BaseException) -> Iterator[BaseException]:
    yield error
    if isinstance(error, BaseExceptionGroup):
        for nested in error.exceptions:
            yield from _causes(nested)


@asynccontextmanager
async def _session(
    server: MCPServer, *, timeout: float,
) -> AsyncIterator[tuple[ClientSession, types.InitializeResult]]:
    """Общие авторизация, initialize, таймаут и безопасные ошибки SDK 2.x."""
    server.validate()
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Таймаут MCP должен быть положительным конечным числом")
    headers = {}
    if server.token_env:
        token = os.getenv(server.token_env, "").strip()
        if not token:
            raise MCPConnectionError(
                f"Не задана переменная {server.token_env}. Добавьте токен в окружение или .env "
                "и перезапустите приложение."
            )
        if any(ord(char) < 33 or ord(char) > 126 for char in token):
            raise MCPConnectionError("Токен содержит недопустимые символы. Проверьте переменную окружения.")
        headers["Authorization"] = f"Bearer {token}"

    failed_status: int | None = None

    async def record_status(response: httpx2.Response) -> None:
        nonlocal failed_status
        if response.request.method == "POST" and response.status_code >= 400:
            failed_status = response.status_code

    try:
        with anyio.fail_after(timeout):
            async with httpx2.AsyncClient(
                headers=headers, timeout=httpx2.Timeout(timeout, connect=min(timeout, 10)),
                follow_redirects=False, event_hooks={"response": [record_status]},
                # Loopback должен оставаться локальным и при включённом прокси
                # в окружении или настройках Windows. Для VPS сохраняем прокси/TLS.
                mounts={"all://localhost": None, "all://127.0.0.1": None, "all://[::1]": None},
            ) as http_client:
                async with streamable_http_client(server.url, http_client=http_client) as (read, write):
                    async with ClientSession(read, write, read_timeout_seconds=timeout) as session:
                        initialized = await session.initialize()
                        if initialized.capabilities.tools is None:
                            raise MCPConnectionError("Соединение установлено, но сервер не поддерживает инструменты MCP.")
                        yield session, initialized
    except Exception as error:
        # Не показываем str(error) от SDK: ответ сервера может содержать секреты.
        causes = tuple(_causes(error))
        for cause in causes:
            if isinstance(cause, MCPConnectionError):
                raise cause from None
        if failed_status in (401, 403):
            message = "MCP-сервер отклонил авторизацию. Проверьте токен, его права и доступ к серверу."
        elif failed_status == 404:
            message = "MCP endpoint не найден (HTTP 404). Проверьте URL сервера."
        elif failed_status is not None:
            message = f"MCP-сервер вернул HTTP {failed_status}. Проверьте настройки или повторите позже."
        elif any(isinstance(cause, (TimeoutError, httpx2.TimeoutException)) for cause in causes):
            message = "MCP-сервер не ответил за отведённое время. Повторите запрос."
        elif any(isinstance(cause, (httpx2.NetworkError, OSError)) for cause in causes):
            message = "Не удалось подключиться к MCP-серверу. Проверьте адрес и сеть."
        else:
            message = "Не удалось выполнить запрос MCP. Проверьте поддержку Streamable HTTP и настройки сервера."
        raise MCPConnectionError(message) from None


async def get_tools(server: MCPServer, *, timeout: float = 30) -> MCPDiscovery:
    """initialize → tools/list (все страницы) → закрытие; общий таймаут 30 с."""
    async with _session(server, timeout=timeout) as (session, initialized):
        tools: list[Tool] = []
        cursor: str | None = None
        seen_cursors: set[str] = set()
        while True:
            page = await session.list_tools(
                params=types.PaginatedRequestParams(cursor=cursor) if cursor is not None else None,
            )
            tools.extend(page.tools)
            cursor = page.next_cursor
            if cursor is None:
                break
            if cursor in seen_cursors:
                raise MCPConnectionError("Сервер повторил курсор списка инструментов. Попробуйте позже.")
            seen_cursors.add(cursor)
        return MCPDiscovery(
            initialized.server_info.name, initialized.server_info.version,
            initialized.protocol_version, tuple(tools),
        )


async def call_tool(
    server: MCPServer, name: str, arguments: dict, *, timeout: float = 30,
) -> types.CallToolResult:
    """Один вызов в новой сессии без автоматического повтора операции.

    Ошибки инструмента (is_error) возвращаются как данные; ошибки транспорта
    становятся MCPConnectionError. SDK проверяет объявленную outputSchema.
    """
    if not isinstance(name, str) or not name or not isinstance(arguments, dict):
        raise ValueError("Для MCP-вызова требуются имя инструмента и объект аргументов")
    async with _session(server, timeout=timeout) as (session, _):
        return await session.call_tool(name, arguments, read_timeout_seconds=timeout)


def main() -> int:
    parser = argparse.ArgumentParser(description="Подключиться к MCP и вывести доступные инструменты")
    parser.add_argument("--url", default=GITHUB_MCP_URL, help="URL Streamable HTTP endpoint")
    parser.add_argument("--token-env", default=GITHUB_TOKEN_ENV, help="Имя переменной с Bearer-токеном; пустая строка для сервера без авторизации")
    args = parser.parse_args()
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    try:
        result = asyncio.run(get_tools(MCPServer(id="cli", name="MCP", url=args.url, token_env=args.token_env)))
    except (ValueError, MCPConnectionError) as error:
        parser.exit(1, f"Ошибка: {error}\n")
    print(json.dumps({
        "server": result.server_name, "version": result.server_version,
        "protocolVersion": result.protocol_version,
        "tools": [tool.model_dump(mode="json", by_alias=True, exclude_none=True) for tool in result.tools],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
