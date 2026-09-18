import json
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from .history import Message

SUPPORTED_STRATEGIES = ("window", "facts", "branch")
TMessage = TypeVar("TMessage", bound="Message")


class WindowStrategy:
    """Оставляет системные инструкции и последние N сообщений диалога."""

    def __init__(self, size: int) -> None:
        if type(size) is not int or size <= 0:
            raise ValueError("window_size должен быть положительным целым числом")
        self.size = size

    def apply(self, messages: list[TMessage]) -> list[TMessage]:
        system = [message for message in messages if message["role"] == "system"]
        dialogue = [message for message in messages if message["role"] != "system"]
        return system + dialogue[-self.size:]


class FactsStrategy(WindowStrategy):
    """Дополняет окно локальными фактами диалога из пар ключ-значение."""

    @staticmethod
    def validate(facts: object) -> None:
        if not isinstance(facts, dict) or any(
            not isinstance(key, str) or not key.strip()
            or not isinstance(value, str) or not value.strip()
            for key, value in facts.items()
        ):
            raise ValueError("facts должен быть объектом с непустыми строковыми ключами и значениями")

    @staticmethod
    def update(previous: dict[str, str], content: str) -> dict[str, str]:
        """Применить JSON изменений: пропущенные ключи сохраняются, null удаляет ключ."""
        try:
            updates = json.loads(content)
        except json.JSONDecodeError as error:
            raise ValueError("Модель вернула некорректный JSON обновления facts") from error
        if not isinstance(updates, dict):
            raise ValueError("Обновление facts должно быть JSON-объектом")
        FactsStrategy.validate({key: value for key, value in updates.items() if value is not None})
        if any(not key.strip() for key in updates):
            raise ValueError("Ключи facts должны быть непустыми строками")
        facts = previous.copy()
        for key, value in updates.items():
            if value is None:
                facts.pop(key, None)
            else:
                facts[key] = value
        return facts
