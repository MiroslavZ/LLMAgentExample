"""Последние пять завершённых обменов; изоляция по браузеру, клиенту и тикету."""

import json
import sqlite3
from contextlib import closing
from pathlib import Path

from .config import HISTORY_MESSAGES
from .errors import SupportError


class HistoryStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as db, db:
            db.execute("""CREATE TABLE IF NOT EXISTS support_turns (
                id INTEGER PRIMARY KEY, session_id TEXT NOT NULL, user_id TEXT NOT NULL,
                ticket_id TEXT NOT NULL, payload TEXT NOT NULL
            )""")
            db.execute("CREATE INDEX IF NOT EXISTS support_scope ON support_turns(session_id, user_id, ticket_id, id)")

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=5)

    @staticmethod
    def _scope(session_id: str, user_id: str, ticket_id: str | None) -> tuple:
        return session_id, user_id, ticket_id or ""

    def get(self, session_id: str, user_id: str, ticket_id: str | None) -> list[dict]:
        try:
            with closing(self._connect()) as db:
                rows = db.execute(
                    "SELECT payload FROM support_turns WHERE session_id=? AND user_id=? AND ticket_id=? ORDER BY id DESC LIMIT ?",
                    (*self._scope(session_id, user_id, ticket_id), HISTORY_MESSAGES // 2),
                ).fetchall()
            return [json.loads(row[0]) for row in reversed(rows)]
        except (sqlite3.Error, ValueError) as error:
            raise SupportError("Не удалось прочитать историю поддержки.") from error

    def append(self, session_id: str, user_id: str, ticket_id: str | None, turn: dict) -> None:
        scope = self._scope(session_id, user_id, ticket_id)
        try:
            with closing(self._connect()) as db, db:
                db.execute("INSERT INTO support_turns(session_id,user_id,ticket_id,payload) VALUES (?,?,?,?)",
                           (*scope, json.dumps(turn, ensure_ascii=False)))
                db.execute("""DELETE FROM support_turns WHERE session_id=? AND user_id=? AND ticket_id=?
                    AND id NOT IN (SELECT id FROM support_turns WHERE session_id=? AND user_id=? AND ticket_id=?
                                   ORDER BY id DESC LIMIT ?)""", (*scope, *scope, HISTORY_MESSAGES // 2))
        except sqlite3.Error as error:
            raise SupportError("Ответ получен, но не удалось сохранить историю поддержки.") from error

    def clear(self, session_id: str, user_id: str, ticket_id: str | None) -> None:
        try:
            with closing(self._connect()) as db, db:
                db.execute("DELETE FROM support_turns WHERE session_id=? AND user_id=? AND ticket_id=?",
                           self._scope(session_id, user_id, ticket_id))
        except sqlite3.Error as error:
            raise SupportError("Не удалось очистить историю поддержки.") from error
