"""Чтение проверенных демонстрационных данных без изменения хранилища."""

import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

DEFAULT_DATA_PATH = Path(__file__).with_name("demo_data.json")


class SupportDataError(ValueError):
    """Контекст поддержки недоступен или запрос некорректен."""


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Environment(_Record):
    os: str
    interface: str
    provider: str
    model: str


class Settings(_Record):
    rag_enabled: bool
    index_status: Literal["ready", "missing"]
    mcp_enabled: bool


class User(_Record):
    id: str
    name: str
    summary: str
    environment: Environment
    settings: Settings
    observed_at: str


class Message(_Record):
    role: Literal["user", "assistant", "support"]
    content: str


class Ticket(_Record):
    id: str
    user_id: str
    subject: str
    status: Literal["open", "waiting_user", "resolved"]
    error: str
    attempts: list[str]
    messages: list[Message]


class _Database(_Record):
    demo: Literal[True]
    users: list[User] = Field(min_length=1)
    tickets: list[Ticket]


def _validate_id(value: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", value) is None:
        raise SupportDataError("Некорректный идентификатор пользователя или тикета.")


class SupportStore:
    """Путь задаёт сервер; каждый запрос читает новый снимок JSON целиком."""

    def __init__(self, path: Path = DEFAULT_DATA_PATH) -> None:
        self.path = path

    def _read(self) -> _Database:
        try:
            data = _Database.model_validate(json.loads(self.path.read_text(encoding="utf-8")))
            user_ids = {user.id for user in data.users}
            ticket_ids = {ticket.id for ticket in data.tickets}
            if len(user_ids) != len(data.users) or len(ticket_ids) != len(data.tickets):
                raise ValueError("Duplicate identifiers")
            for identifier in user_ids | ticket_ids:
                _validate_id(identifier)
            if any(ticket.user_id not in user_ids for ticket in data.tickets):
                raise ValueError("Unknown ticket owner")
            return data
        except (OSError, UnicodeError, ValueError) as error:
            raise SupportDataError("Не удалось прочитать демонстрационные данные поддержки.") from error

    @staticmethod
    def _user(data: _Database, user_id: str) -> User:
        _validate_id(user_id)
        for user in data.users:
            if user.id == user_id:
                return user
        raise SupportDataError("Пользователь не найден.")

    def list_users(self) -> dict:
        data = self._read()
        return {"users": [user.model_dump(include={"id", "name", "summary"}) for user in data.users]}

    def list_user_tickets(self, user_id: str) -> dict:
        data = self._read()
        self._user(data, user_id)
        return {"tickets": [
            ticket.model_dump(include={"id", "user_id", "subject", "status"})
            for ticket in data.tickets if ticket.user_id == user_id
        ]}

    def get_support_context(self, user_id: str, ticket_id: str | None = None) -> dict:
        data = self._read()
        user = self._user(data, user_id)
        ticket = None
        if ticket_id is not None:
            _validate_id(ticket_id)
            ticket = next((item for item in data.tickets
                           if item.id == ticket_id and item.user_id == user_id), None)
            if ticket is None:
                raise SupportDataError("Тикет не найден или недоступен выбранному пользователю.")
        return {"user": user.model_dump(), "ticket": ticket.model_dump() if ticket else None, "demo": True}
