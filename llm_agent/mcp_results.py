"""Передача полных результатов MCP между инструментами одного запроса."""

import copy
import json
import re
from collections.abc import Collection


MAX_TRANSFER_CHARS = 1_000_000
RESULT_REFERENCE_SCHEMA = {
    "type": "object",
    "properties": {
        "$mcp_result": {"type": "string", "minLength": 1,
                        "description": "Скопируй $mcp_result из result_ref предыдущего ответа: это ID вызова call_…, не имя функции mcp_…"},
        "pointer": {"type": "string", "default": "",
                    "description": "JSON Pointer в полном результате: /data или /data/summary"},
    },
    "required": ["$mcp_result"],
    "additionalProperties": False,
}


def with_result_references(schema: dict) -> dict:
    """Расширить только схему для модели; MCP по-прежнему получает исходные типы."""
    schema = copy.deepcopy(schema)
    for name, parameter in schema.get("properties", {}).items():
        schema["properties"][name] = {"anyOf": [parameter, copy.deepcopy(RESULT_REFERENCE_SCHEMA)]}
        if isinstance(parameter, dict) and "description" in parameter:
            schema["properties"][name]["description"] = parameter["description"]
    return schema


class MCPResultStore:
    """Кэш на один пользовательский запрос; ошибки никогда не являются источником."""

    def __init__(self) -> None:
        self._results: dict[str, dict] = {}
        self._calls_by_tool: dict[str, list[str]] = {}
        self._call_ids: set[str] = set()

    def remember(self, call_id: str, payload: dict, *, tool_name: str) -> bool:
        # Учитываем также ошибки: после неудачного повторного вызова имя
        # инструмента не должно незаметно указывать на старый успешный результат.
        self._call_ids.add(call_id)
        self._calls_by_tool.setdefault(tool_name, []).append(call_id)
        if not payload["is_error"] and len(json.dumps(payload, ensure_ascii=False)) <= MAX_TRANSFER_CHARS:
            self._results[call_id] = payload
            return True
        return False

    def reference(self, call_id: str) -> dict:
        payload = self._results[call_id]
        pointer = "/data" if "data" in payload else "/content"
        if pointer == "/content" and payload["content"] and "text" in payload["content"][0]:
            pointer += "/0/text"
        return {"$mcp_result": call_id, "pointer": pointer}

    def available_results(self, *, pending_call_ids: Collection[str] = ()) -> list[dict]:
        return [
            {"tool_name": tool_name, "result_ref": self.reference(call_id)}
            for tool_name, call_ids in self._calls_by_tool.items() for call_id in call_ids
            if call_id in self._results and call_id not in pending_call_ids
        ]

    def resolve(self, arguments: dict, *, pending_call_ids: Collection[str] = ()) -> dict:
        # Только значения верхнего уровня. Вложенный JSON и сами полученные
        # данные не интерпретируются повторно как ссылки/инструкции.
        return {name: self._resolve_value(value, pending_call_ids) for name, value in arguments.items()}

    def _resolve_value(self, value: object, pending_call_ids: Collection[str]) -> object:
        if not isinstance(value, dict) or "$mcp_result" not in value:
            return value
        call_id, pointer = value.get("$mcp_result"), value.get("pointer", "")
        if (set(value) - {"$mcp_result", "pointer"} or not isinstance(call_id, str)
                or not call_id or not isinstance(pointer, str)):
            raise ValueError("Некорректная ссылка $mcp_result: нужны tool_call_id и строка pointer.")
        if call_id not in self._call_ids and call_id in self._calls_by_tool:
            candidates = self._calls_by_tool[call_id]
            if len(candidates) != 1:
                raise ValueError("Имя инструмента в $mcp_result неоднозначно: было несколько вызовов. "
                                 "Скопируйте конкретный ID из result_ref нужного результата или available_results.")
            call_id = candidates[0]
        if call_id in pending_call_ids:
            raise ValueError("Зависимый вызов должен быть в следующем раунде после получения результата источника.")
        if call_id not in self._results:
            raise ValueError("Источник $mcp_result недоступен: нужен успешный предыдущий вызов текущего запроса "
                             "с результатом не более 1 000 000 символов. Скопируйте ID из result_ref "
                             "нужного результата или available_results; не выдумывайте идентификатор.")
        if pointer and (not pointer.startswith("/") or re.search(r"~(?:[^01]|$)", pointer)):
            raise ValueError("pointer должен быть JSON Pointer: пустая строка, /data или /data/summary.")
        selected = self._results[call_id]
        for segment in pointer.split("/")[1:] if pointer else ():
            key = segment.replace("~1", "/").replace("~0", "~")
            if isinstance(selected, dict) and key in selected:
                selected = selected[key]
            elif isinstance(selected, list) and re.fullmatch(r"0|[1-9][0-9]*", key):
                # Сравнение строк не позволяет гигантскому индексу обойти лимит int.
                last = str(len(selected) - 1)
                if len(key) > len(last) or (len(key) == len(last) and key > last) or not selected:
                    raise ValueError("JSON Pointer указывает за пределы массива результата.")
                selected = selected[int(key)]
            else:
                raise ValueError("JSON Pointer не найден в результате MCP-вызова.")
        return copy.deepcopy(selected)
