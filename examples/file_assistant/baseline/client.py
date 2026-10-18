"""Учебный клиент: хранит настройки и возвращает данные без HTTP-запросов."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ApiClient:
    """Настройки будущего транспорта; timeout измеряется в секундах."""

    timeout: int = 30
    retries: int = 3

    def get_record(self, record_id: int) -> dict[str, int | str]:
        """Возвращает локальную запись; сеть и повторные запросы не используются."""
        if record_id < 1:
            raise ValueError("record_id должен быть положительным")
        return {"id": record_id, "status": "ready"}
