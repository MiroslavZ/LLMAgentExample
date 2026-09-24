"""Сохраняемый результат вызова инструмента, не зависящий от MCP и веб-интерфейса."""

import math
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class ToolCallRecord:
    call_id: str
    server_name: str
    tool_name: str
    arguments: str
    result: str
    is_error: bool
    elapsed_seconds: float

    @classmethod
    def from_dict(cls, data: object) -> "ToolCallRecord":
        if not isinstance(data, dict) or set(data) != {field.name for field in fields(cls)}:
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
        return cls(**data)
