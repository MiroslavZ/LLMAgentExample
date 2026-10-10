"""Выбор формата JSON без повторов при посторонних ошибках API."""

import re
from collections.abc import Callable

from openai import APIStatusError
from openai.types.chat import ChatCompletion

from .response_schemas import GENERIC_OBJECT_SCHEMA, schema_format


class StructuredOutputError(RuntimeError):
    """Безопасная для интерфейса ошибка совместимости формата ответа."""


def _unsupported_format(error: APIStatusError) -> bool:
    """Распознать отказ именно в типе формата, а не в содержимом схемы.

    Не используем str(error): SDK может включить туда тело запроса. Не считаем
    ошибки схемы, tools, контекста или авторизации поводом повторять запрос.
    """
    if error.status_code not in (400, 422):
        return False
    body = error.body
    if isinstance(body, dict):
        body = body.get("error", body)
    if isinstance(body, dict):
        message = body.get("message", "")
        parameter = body.get("param", "")
    else:
        message, parameter = body, ""
    if not isinstance(message, str):
        return False
    message = message.lower()
    if re.search(r"\b(schema|grammar)\s+(validation|compilation)|invalid\s+(json\s+)?schema", message):
        return False
    if re.search(r"\b(tools?|tool_choice|streaming|context|tokens|memory)\b", message):
        return False
    identifies_type = parameter == "response_format.type" or re.search(
        r"response_format[.'\"\s]+type\b", message,
    )
    rejects_type = re.search(
        r"not supported|does not support|doesn't support|unsupported|unavailable|"
        r"must be|supported (?:values|types)|not (?:a )?valid", message,
    )
    rejects_parameter = re.search(
        r"(?:response_format.{0,12}(?:is not supported|is unsupported)|"
        r"(?:does not support|unsupported parameter).{0,12}response_format)", message,
    )
    return bool(identifies_type and rejects_type or rejects_parameter)


class StructuredOutput:
    """Политика одного подключения; кеш в памяти разделён по API ID модели.

    Внешний каталог отвечает за сохранение результата и проверку актуальности
    снимка подключения. Только принятый сервером формат попадает в кеш.
    """

    def __init__(
        self, mode: str = "auto", *, model_id: str | None = None,
        detected: str | None = None, on_detected: Callable[[str], None] | None = None,
    ) -> None:
        if mode not in ("auto", "json_schema", "json_object", "off"):
            raise ValueError("Неизвестный режим структурированного ответа")
        if detected not in (None, "json_schema", "json_object"):
            raise ValueError("Неизвестный обнаруженный формат ответа")
        self.mode = mode
        self._model_id = model_id
        self._detected = {model_id: detected} if model_id is not None and detected else {}
        self._on_detected = on_detected

    def complete(self, create: Callable[..., ChatCompletion], **request) -> ChatCompletion:
        response_format = request.get("response_format", {})
        if self.mode == "off" or response_format.get("type", "text") == "text":
            request.pop("response_format", None)
            return create(**request)

        model_id = request["model"]
        if self.mode == "auto":
            preferred = self._detected.get(model_id, "json_schema")
            formats = [preferred, "json_object" if preferred == "json_schema" else "json_schema"]
        else:
            formats = [self.mode]
        for output_format in formats:
            if output_format == "json_object":
                wire_format = {"type": "json_object"}
            else:
                # Произвольный JSON также требует схемы: пустой json_schema
                # без тела не является валидным параметром API.
                wire_format = response_format if response_format.get("json_schema") else schema_format(
                    "json_response", GENERIC_OBJECT_SCHEMA, strict=False,
                )
            try:
                response = create(**{**request, "response_format": wire_format})
            except APIStatusError as error:
                if not _unsupported_format(error):
                    raise
                if self.mode != "auto":
                    raise StructuredOutputError(
                        f"Сервер не поддерживает выбранный формат {self.mode}. "
                        "Выберите «Авто» или другой формат в менеджере моделей."
                    ) from error
                continue
            if self.mode == "auto" and self._detected.get(model_id) != output_format:
                self._detected[model_id] = output_format
                if self._on_detected is not None and model_id == self._model_id:
                    self._on_detected(output_format)
            return response
        raise StructuredOutputError(
            "Сервер не поддерживает JSON Schema и JSON Object. "
            "Выберите другую модель или явно отключите структурированный формат "
            "в менеджере моделей; тогда соблюдение JSON зависит от модели."
        )
