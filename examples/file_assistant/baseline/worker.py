"""Обработка задания с клиентом, переданным в сервис."""

from client import ApiClient
from service import load_status


def run_job(record_id: int, client: ApiClient) -> str:
    """Формирует результат задания из статуса записи."""
    return f"Задание {record_id}: {load_status(record_id, client)}"


if __name__ == "__main__":
    print(run_job(1, ApiClient()))
