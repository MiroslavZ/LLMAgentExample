"""Сервис получения статуса записи."""

from client import ApiClient


def load_status(record_id: int, client: ApiClient | None = None) -> str:
    """Использует переданный клиент или создаёт клиент с настройками по умолчанию."""
    active_client = client if client is not None else ApiClient()
    record = active_client.get_record(record_id)
    return str(record["status"])
