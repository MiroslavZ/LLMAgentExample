import time
from dataclasses import dataclass

from openai import OpenAI
from openai.types.chat import ChatCompletion

BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"
META_PROMPT_SYSTEM = (
    "Составь оптимальный промпт для решения следующей задачи. "
    "Верни только текст промпта без пояснений и комментариев."
)
RESPONSE_FORMATS = {
    "text": {"type": "text"},
    "object": {"type": "json_object"},
    "schema": {"type": "json_schema"},
}


@dataclass(frozen=True)
class RequestResult:
    response: ChatCompletion
    elapsed: float

    @property
    def content(self) -> str:
        return self.response.choices[0].message.content or ""


class Agent:
    def __init__(self, token: str) -> None:
        self._client = OpenAI(api_key=token, base_url=BASE_URL)

    def request(
        self,
        user: str,
        *,
        model: str = DEFAULT_MODEL,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        stop_sequences: list[str] | None = None,
        response_format: str = "text",
    ) -> RequestResult:
        """Выполнить запрос с параметрами, действующими только для этого вызова."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        request = {
            "model": model,
            "messages": messages,
            "response_format": RESPONSE_FORMATS[response_format],
        }
        if max_tokens is not None:
            request["max_tokens"] = max_tokens
        if temperature is not None:
            request["temperature"] = temperature
        if stop_sequences:
            request["stop"] = stop_sequences

        started = time.perf_counter()
        response = self._client.chat.completions.create(**request)
        return RequestResult(response, time.perf_counter() - started)

    def request_with_meta_prompt(
        self,
        user: str,
        *,
        model: str = DEFAULT_MODEL,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        stop_sequences: list[str] | None = None,
        response_format: str = "text",
    ) -> tuple[RequestResult, RequestResult]:
        """Сгенерировать промпт и выполнить его; вернуть результаты обоих этапов."""
        options = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stop_sequences": stop_sequences,
        }
        meta_result = self.request(
            user,
            system=META_PROMPT_SYSTEM,
            response_format="text",
            **options,
        )
        result = self.request(
            meta_result.content,
            system=system,
            response_format=response_format,
            **options,
        )
        return meta_result, result
