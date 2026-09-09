import time
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI
from openai.types.chat import ChatCompletion

from history import DEFAULT_HISTORY_PATH, HistoryManager, Message

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
    def __init__(self, token: str, history_path: str | Path = DEFAULT_HISTORY_PATH) -> None:
        self.history = HistoryManager(history_path)
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
        """Выполнить запрос, установив системный промпт, если его ещё нет."""
        self.history.set_system_prompt(system)
        return self._request(
            user,
            messages=self.history.get_messages(),
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            stop_sequences=stop_sequences,
            response_format=response_format,
        )

    def _request(
        self,
        user: str,
        *,
        messages: list[Message],
        model: str,
        max_tokens: int | None,
        temperature: float | None,
        stop_sequences: list[str] | None,
        response_format: str,
    ) -> RequestResult:
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
        result = RequestResult(response, time.perf_counter() - started)
        self.history.add_messages([
            {"role": "user", "content": user},
            {"role": "assistant", "content": result.content},
        ])
        return result

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
        self.history.set_system_prompt(system)
        options = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stop_sequences": stop_sequences,
        }
        meta_messages: list[Message] = [{"role": "system", "content": META_PROMPT_SYSTEM}]
        meta_messages.extend(
            message for message in self.history.get_messages() if message["role"] != "system"
        )
        meta_result = self._request(
            user,
            messages=meta_messages,
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
