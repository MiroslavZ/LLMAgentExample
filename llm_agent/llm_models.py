"""Каталог моделей OpenAI-совместимых серверов, общий для всех диалогов."""

from __future__ import annotations

import ipaddress
import json
import math
import re
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator
from urllib.parse import urlsplit
from uuid import uuid4

from openai import OpenAI

from .llm_client import connection_options


class ModelConnectionError(RuntimeError):
    """Сервер не вернул корректный список моделей; детали запроса скрыты."""


class ModelStorageError(RuntimeError):
    """Каталог моделей недоступен или повреждён."""


def _normalize_url(value: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or any(character.isspace() or ord(character) < 32 or ord(character) == 127 for character in value)
        or any(character in value for character in ("\\", "?", "#"))
    ):
        raise ValueError("Укажите base URL без пробелов, query-параметров и фрагмента")
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError:
        raise ValueError("Некорректный адрес или порт сервера моделей") from None
    if parsed.scheme not in {"http", "https"} or not hostname:
        raise ValueError("Base URL должен содержать http:// или https:// и имя сервера")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("Укажите токен в отдельном поле, без логина и пароля в base URL")
    if parsed.netloc.endswith(":") or port is not None and not 1 <= port <= 65535:
        raise ValueError("Порт сервера должен быть числом от 1 до 65535")
    try:
        if ":" in hostname:
            if "%" in hostname or re.fullmatch(r"\[[^\]]+\](?::[0-9]+)?", parsed.netloc) is None:
                raise ValueError
            ipaddress.IPv6Address(hostname)
        elif re.fullmatch(r"[0-9.]+", hostname):
            ipaddress.IPv4Address(hostname)
        else:
            ascii_hostname = hostname.encode("idna").decode("ascii").rstrip(".")
            if len(ascii_hostname) > 253 or any(
                re.fullmatch(r"[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?", label) is None
                for label in ascii_hostname.split(".")
            ):
                raise ValueError
    except (ValueError, UnicodeError):
        raise ValueError("Некорректное имя или IP-адрес сервера моделей") from None
    return parsed._replace(netloc=parsed.netloc.lower(), path=parsed.path.rstrip("/")).geturl()


def _validate_token(token: str) -> None:
    if not isinstance(token, str) or any(
        character.isspace() or ord(character) < 32 or ord(character) == 127 for character in token
    ):
        raise ValueError("API-токен должен быть строкой без пробелов и управляющих символов")


@dataclass(frozen=True)
class LLMModel:
    id: str
    name: str
    model_id: str
    base_url: str
    token: str = field(default="", repr=False)
    timeout: float = 180
    json_mode: bool = True
    tools_enabled: bool = True

    def __post_init__(self) -> None:
        self.validate()
        object.__setattr__(self, "base_url", _normalize_url(self.base_url))

    def validate(self) -> None:
        if not isinstance(self.id, str) or re.fullmatch(r"[0-9a-f]{32}", self.id) is None:
            raise ValueError("Внутренний ID модели должен быть UUID в формате hex")
        if not isinstance(self.name, str) or not self.name.strip() or any(
            ord(character) < 32 or ord(character) == 127 for character in self.name
        ):
            raise ValueError("Укажите отображаемое имя модели без управляющих символов")
        if not isinstance(self.model_id, str) or not self.model_id or any(
            character.isspace() or ord(character) < 32 or ord(character) == 127 for character in self.model_id
        ):
            raise ValueError("Укажите ID модели на сервере без пробелов и управляющих символов")
        _normalize_url(self.base_url)
        _validate_token(self.token)
        try:
            valid_timeout = type(self.timeout) in (int, float) and math.isfinite(self.timeout) and self.timeout > 0
        except OverflowError:
            valid_timeout = False
        if not valid_timeout:
            raise ValueError("Таймаут модели должен быть положительным конечным числом секунд")
        if type(self.json_mode) is not bool or type(self.tools_enabled) is not bool:
            raise ValueError("Поддержка JSON и инструментов должна задаваться логическими значениями")


class ModelStore:
    """Пустой при создании каталог; соединение и транзакция живут одну операцию."""

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
                    raise ModelStorageError("Неподдерживаемая версия каталога моделей. Файл сохранён без изменений.")
                if version == 0:
                    if connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall():
                        raise ModelStorageError("Неизвестная структура каталога моделей. Файл сохранён без изменений.")
                    connection.execute(
                        "CREATE TABLE llm_models ("
                        "id TEXT PRIMARY KEY NOT NULL, base_url TEXT NOT NULL, model_id TEXT NOT NULL, "
                        "data TEXT NOT NULL, UNIQUE(base_url, model_id))"
                    )
                    connection.execute("PRAGMA user_version = 1")
                yield connection
        except (OSError, sqlite3.Error):
            raise ModelStorageError(
                "Не удалось прочесть или сохранить каталог моделей. Проверьте файл models.sqlite3 и доступ к каталогу."
            ) from None
        finally:
            if connection is not None:
                connection.close()

    @staticmethod
    def _decode(model_key: str, base_url: str, model_id: str, data: str) -> LLMModel:
        try:
            model = LLMModel(**json.loads(data))
            if (model.id, model.base_url, model.model_id) != (model_key, base_url, model_id):
                raise ValueError
            return model
        except (ValueError, TypeError, RecursionError):
            raise ModelStorageError("Повреждены настройки модели. Запись сохранена без изменений.") from None

    def list(self) -> list[LLMModel]:
        with self._connection() as connection:
            models = [self._decode(*row) for row in connection.execute(
                "SELECT id, base_url, model_id, data FROM llm_models"
            )]
        return sorted(models, key=lambda model: (model.name.casefold(), model.id))

    def get(self, model_id: str) -> LLMModel:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT id, base_url, model_id, data FROM llm_models WHERE id = ?", (model_id,)
            ).fetchone()
            if row is None:
                raise KeyError("Модель не найдена в каталоге")
            return self._decode(*row)

    def save(self, model: LLMModel, *, expected: LLMModel | None = None) -> None:
        if not isinstance(model, LLMModel):
            raise ValueError("Требуются настройки модели")
        model.validate()
        if expected is not None and not isinstance(expected, LLMModel):
            raise ValueError("Требуется исходная версия настроек модели")
        with self._connection() as connection:
            row = connection.execute(
                "SELECT id, base_url, model_id, data FROM llm_models WHERE id = ?", (model.id,)
            ).fetchone()
            current = None if row is None else self._decode(*row)
            if expected is not None and current != expected:
                raise ValueError("Модель изменена или удалена в другой вкладке. Откройте её заново.")
            duplicate = connection.execute(
                "SELECT id FROM llm_models WHERE base_url = ? AND model_id = ? AND id != ?",
                (model.base_url, model.model_id, model.id),
            ).fetchone()
            if duplicate is not None:
                raise ValueError("Модель с таким ID уже добавлена для этого сервера")
            connection.execute(
                "INSERT INTO llm_models (id, base_url, model_id, data) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET "
                "base_url = excluded.base_url, model_id = excluded.model_id, data = excluded.data",
                (model.id, model.base_url, model.model_id, json.dumps(asdict(model), ensure_ascii=False)),
            )

    def delete(self, model_id: str, *, expected: LLMModel | None = None) -> None:
        if expected is not None and not isinstance(expected, LLMModel):
            raise ValueError("Требуется исходная версия настроек модели")
        with self._connection() as connection:
            row = connection.execute(
                "SELECT id, base_url, model_id, data FROM llm_models WHERE id = ?", (model_id,)
            ).fetchone()
            if row is None:
                raise KeyError("Модель не найдена в каталоге")
            current = self._decode(*row)
            if expected is not None and current != expected:
                raise ValueError("Модель изменена в другой вкладке. Откройте её заново.")
            connection.execute("DELETE FROM llm_models WHERE id = ?", (model_id,))

    def resolve(self, reference: str) -> LLMModel:
        models = self.list()
        for model in models:
            if model.id == reference:
                return model
        matches = [model for model in models if model.model_id == reference]
        if len(matches) > 1:
            raise ValueError("ID модели неоднозначен: используйте внутренний ID записи каталога")
        if not matches:
            raise KeyError("Модель не найдена в каталоге")
        return matches[0]

    def import_models(self, base_url: str, token: str, model_ids: list[str]) -> list[LLMModel]:
        base_url = _normalize_url(base_url)
        _validate_token(token)
        if not isinstance(model_ids, list):
            raise ValueError("Передайте список ID моделей для импорта")
        candidates = [LLMModel(uuid4().hex, model_id, model_id, base_url, token) for model_id in model_ids]
        imported: dict[str, LLMModel] = {}
        with self._connection() as connection:
            for model in candidates:
                if model.model_id in imported:
                    continue
                row = connection.execute(
                    "SELECT id, base_url, model_id, data FROM llm_models WHERE base_url = ? AND model_id = ?",
                    (base_url, model.model_id),
                ).fetchone()
                if row is not None:
                    imported[model.model_id] = self._decode(*row)
                    continue
                connection.execute(
                    "INSERT INTO llm_models (id, base_url, model_id, data) VALUES (?, ?, ?, ?)",
                    (model.id, model.base_url, model.model_id, json.dumps(asdict(model), ensure_ascii=False)),
                )
                imported[model.model_id] = model
        return list(imported.values())


def discover_models(base_url: str, token: str = "") -> list[str]:
    """Получить ID со всех страниц SDK, не сохраняя модели или токен в каталоге."""
    base_url = _normalize_url(base_url)
    _validate_token(token)
    try:
        with OpenAI(**connection_options(base_url, token), timeout=20, max_retries=0) as client:
            models: set[str] = set()
            # Итератор страницы OpenAI сам получает все последующие страницы.
            for model in client.models.list():
                model_id = model.id
                if not isinstance(model_id, str) or not model_id or any(
                    character.isspace() or ord(character) < 32 or ord(character) == 127 for character in model_id
                ):
                    raise ValueError
                models.add(model_id)
            return sorted(models)
    except Exception as error:
        status = getattr(error, "status_code", None)
        if status in (401, 403):
            message = "Сервер моделей отклонил авторизацию. Проверьте API-токен и права доступа."
        elif status == 404:
            message = "Список моделей недоступен по этому адресу. Проверьте base URL и путь /v1."
        else:
            message = "Не удалось получить список моделей. Проверьте base URL, доступность сервера и API-токен."
        raise ModelConnectionError(message) from None
