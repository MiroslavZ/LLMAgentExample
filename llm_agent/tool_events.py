"""Сохраняемый результат вызова инструмента, не зависящий от MCP и веб-интерфейса."""

import math
from dataclasses import dataclass, fields


MAX_ATTACHMENT_CHARS = 200_000


@dataclass(frozen=True)
class ToolAttachment:
    """Полученный от инструмента TXT; URI — метаданные, а не путь для чтения."""

    filename: str
    text: str
    uri: str

    @classmethod
    def from_dict(cls, data: object) -> "ToolAttachment":
        if (not isinstance(data, dict) or set(data) != {"filename", "text", "uri"}
                or any(not isinstance(value, str) for value in data.values())):
            raise ValueError("Некорректное вложение инструмента")
        name = data["filename"]
        if (not name.lower().endswith(".txt") or len(name) > 128
                or any(char in name for char in '/\\:*?"<>|')
                or any(ord(char) < 32 for char in name)
                or len(data["text"]) > MAX_ATTACHMENT_CHARS):
            raise ValueError("Некорректное имя или размер TXT-вложения")
        try:
            name.encode("utf-8")
            data["text"].encode("utf-8")
        except UnicodeError:
            raise ValueError("Вложение должно содержать корректный UTF-8 текст") from None
        return cls(**data)


@dataclass(frozen=True)
class ToolCallRecord:
    call_id: str
    server_name: str
    tool_name: str
    arguments: str
    result: str
    is_error: bool
    elapsed_seconds: float
    attachments: tuple[ToolAttachment, ...] = ()

    @classmethod
    def from_dict(cls, data: object) -> "ToolCallRecord":
        required = {field.name for field in fields(cls)} - {"attachments"}
        if (not isinstance(data, dict) or not required <= set(data)
                or set(data) - required - {"attachments"}):
            raise ValueError("Некорректный формат вызова инструмента")
        if any(not isinstance(data[name], str) for name in (
            "call_id", "server_name", "tool_name", "arguments", "result",
        )) or type(data["is_error"]) is not bool:
            raise ValueError("Некорректные данные вызова инструмента")
        elapsed = data["elapsed_seconds"]
        try:
            valid_elapsed = (
                type(elapsed) in (int, float) and elapsed >= 0 and math.isfinite(elapsed)
            )
        except OverflowError:
            valid_elapsed = False
        if not valid_elapsed:
            raise ValueError("Время вызова инструмента должно быть конечным неотрицательным числом")
        attachments = data.get("attachments", [])
        if not isinstance(attachments, (list, tuple)):
            raise ValueError("Некорректный список вложений инструмента")
        return cls(**{**data, "attachments": tuple(ToolAttachment.from_dict(item) for item in attachments)})
