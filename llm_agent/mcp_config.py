"""Настройки MCP-серверов без секретов и их общее локальное хранилище."""

import ipaddress
import json
import re
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator
from urllib.parse import urlsplit

GITHUB_MCP_URL = "https://api.githubcopilot.com/mcp/"
GITHUB_TOKEN_ENV = "GITHUB_PERSONAL_ACCESS_TOKEN"


class MCPStorageError(RuntimeError):
    """Не удалось прочесть или сохранить настройки MCP."""


def _validate_url(url: str) -> None:
    if (
        not isinstance(url, str)
        or not url
        or any(character.isspace() or ord(character) < 32 or ord(character) == 127 for character in url)
        or any(character in url for character in ("\\", "?", "#"))
    ):
        raise ValueError("Укажите URL MCP без пробелов, query-параметров и фрагмента")
    try:
        parsed = urlsplit(url)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as error:
        raise ValueError("Некорректный адрес или порт MCP-сервера") from error
    if parsed.scheme not in {"https", "http"} or not hostname:
        raise ValueError("URL MCP должен содержать http:// или https:// и имя сервера")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("Не указывайте логин или токен в URL MCP; используйте имя переменной окружения")
    if parsed.netloc.endswith(":") or port is not None and not 1 <= port <= 65535:
        raise ValueError("Порт MCP-сервера должен быть числом от 1 до 65535")
    try:
        if ":" in hostname:
            if "%" in hostname or re.fullmatch(r"\[[^\]]+\](?::[0-9]+)?", parsed.netloc) is None:
                raise ValueError("Некорректный IPv6-адрес")
            ipaddress.IPv6Address(hostname)
        elif re.fullmatch(r"[0-9.]+", hostname):
            ipaddress.IPv4Address(hostname)
        else:
            ascii_hostname = hostname.encode("idna").decode("ascii").rstrip(".")
            if len(ascii_hostname) > 253 or any(
                re.fullmatch(r"[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?", label) is None
                for label in ascii_hostname.split(".")
            ):
                raise ValueError("Некорректное имя сервера")
    except (ValueError, UnicodeError) as error:
        raise ValueError("Некорректное имя или IP-адрес MCP-сервера") from error
    if parsed.scheme == "http" and hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("Для удалённого MCP используйте HTTPS; HTTP разрешён только для localhost")


@dataclass(frozen=True)
class MCPServer:
    id: str
    name: str
    url: str
    token_env: str = ""

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not isinstance(self.id, str) or re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", self.id) is None:
            raise ValueError("ID MCP: от 1 до 64 латинских букв, цифр, дефисов или подчёркиваний")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Укажите название MCP-сервера")
        _validate_url(self.url)
        if not isinstance(self.token_env, str) or (
            self.token_env and re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", self.token_env) is None
        ):
            raise ValueError("Укажите имя переменной окружения с токеном, например GITHUB_PERSONAL_ACCESS_TOKEN")

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: object) -> "MCPServer":
        if (
            not isinstance(data, dict)
            or not {"id", "name", "url"} <= data.keys()
            or data.keys() - {"id", "name", "url", "token_env"}
        ):
            raise ValueError("Настройки MCP должны содержать id, name, url и необязательное поле token_env")
        return cls(**data)


class MCPServerStore:
    """Настройки общие для диалогов; соединение и транзакция живут одну операцию."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().absolute()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(self.path, timeout=1.0)
            with connection:
                connection.execute("BEGIN IMMEDIATE")
                version = connection.execute("PRAGMA user_version").fetchone()[0]
                if version not in (0, 1):
                    raise MCPStorageError("Неподдерживаемая версия хранилища MCP. Файл сохранён без изменений.")
                if version == 0:
                    tables = connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
                    if tables:
                        raise MCPStorageError("Неизвестная структура хранилища MCP. Файл сохранён без изменений.")
                    connection.execute(
                        "CREATE TABLE mcp_servers (id TEXT PRIMARY KEY NOT NULL, data TEXT NOT NULL)"
                    )
                    connection.execute("PRAGMA user_version = 1")
                yield connection
        except (OSError, sqlite3.Error) as error:
            raise MCPStorageError(
                "Не удалось прочесть или сохранить настройки MCP. Проверьте файл mcp.sqlite3 и доступ к каталогу."
            ) from error
        finally:
            if connection is not None:
                connection.close()

    @staticmethod
    def _decode(server_id: str, data: str) -> MCPServer:
        try:
            server = MCPServer.from_dict(json.loads(data))
            if server.id != server_id:
                raise ValueError("ID MCP-сервера не совпадает с записью")
            return server
        except (ValueError, TypeError, RecursionError) as error:
            raise MCPStorageError("Повреждены настройки MCP-сервера. Запись сохранена без изменений.") from error

    def list(self) -> list[MCPServer]:
        with self._connection() as connection:
            servers = [self._decode(*row) for row in connection.execute("SELECT id, data FROM mcp_servers")]
        return sorted(servers, key=lambda server: (server.name.casefold(), server.id))

    def get(self, server_id: str) -> MCPServer:
        with self._connection() as connection:
            row = connection.execute("SELECT id, data FROM mcp_servers WHERE id = ?", (server_id,)).fetchone()
            if row is None:
                raise ValueError("MCP-сервер не найден")
            return self._decode(*row)

    def save(self, server: MCPServer, *, expected: MCPServer | None = None) -> None:
        if not isinstance(server, MCPServer):
            raise ValueError("Требуются настройки MCP-сервера")
        server.validate()
        if expected is not None and not isinstance(expected, MCPServer):
            raise ValueError("Требуется исходная версия настроек MCP-сервера")
        with self._connection() as connection:
            row = connection.execute("SELECT id, data FROM mcp_servers WHERE id = ?", (server.id,)).fetchone()
            current = None if row is None else self._decode(*row)
            if expected is not None and current != expected:
                raise ValueError("MCP-сервер изменён или удалён в другой вкладке. Откройте его заново.")
            connection.execute(
                "INSERT INTO mcp_servers (id, data) VALUES (?, ?) "
                "ON CONFLICT(id) DO UPDATE SET data = excluded.data",
                (server.id, json.dumps(server.to_dict(), ensure_ascii=False)),
            )

    def delete(self, server_id: str) -> None:
        with self._connection() as connection:
            row = connection.execute("SELECT id, data FROM mcp_servers WHERE id = ?", (server_id,)).fetchone()
            if row is None:
                raise ValueError("MCP-сервер не найден")
            self._decode(*row)
            connection.execute("DELETE FROM mcp_servers WHERE id = ?", (server_id,))
