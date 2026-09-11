import json
import os
import tempfile
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, TypedDict

DEFAULT_HISTORY_PATH = Path("history.json")


class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


class StoredMessage(Message, total=False):
    usage: dict[str, int]


@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True)
class DialogueUsage(TokenUsage):
    missing_responses: int = 0


class HistoryManager:
    """Хранит сообщения диалога и сохраняет изменения в JSON."""

    def __init__(self, path: str | Path = DEFAULT_HISTORY_PATH) -> None:
        self.path = Path(path).expanduser().absolute()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            data = []
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ValueError(f"Некорректный JSON истории: {self.path}") from error
        self._validate(data)
        self._messages: list[StoredMessage] = data

    @staticmethod
    def _validate(messages: object) -> None:
        if not isinstance(messages, list) or any(
            not isinstance(message, dict)
            or not {"role", "content"} <= set(message)
            or set(message) - {"role", "content", "usage"}
            or message["role"] not in ("system", "user", "assistant")
            or not isinstance(message["content"], str)
            for message in messages
        ):
            raise ValueError(
                "История должна быть списком сообщений с role (system/user/assistant) и строкой content"
            )
        if any(message["role"] == "system" for message in messages[1:]):
            raise ValueError("Системный промпт допускается только один раз в начале истории")
        for message in messages:
            if "usage" not in message:
                continue
            usage = message["usage"]
            if (
                message["role"] != "assistant"
                or not isinstance(usage, dict)
                or set(usage) != {"prompt_tokens", "completion_tokens", "total_tokens"}
                or any(type(value) is not int or value < 0 for value in usage.values())
                or usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]
            ):
                raise ValueError("Некорректная статистика токенов в истории")

    def get_system_prompt(self) -> str | None:
        if self._messages and self._messages[0]["role"] == "system":
            return self._messages[0]["content"]
        return None

    def set_system_prompt(self, content: str | None) -> None:
        """Сохранить первый непустой системный промпт в начале диалога."""
        if content and self.get_system_prompt() is None:
            messages = [{"role": "system", "content": content}] + self._messages
            self._validate(messages)
            self._save(messages)

    def get_messages(self) -> list[Message]:
        """Вернуть копию истории, защищённую от изменений вызывающим кодом."""
        return [{"role": message["role"], "content": message["content"]} for message in self._messages]

    def get_usage(self) -> DialogueUsage:
        """Сумма расходов API; ответы без статистики учитываются отдельно."""
        totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        missing = 0
        for message in self._messages:
            if message["role"] != "assistant":
                continue
            if "usage" not in message:
                missing += 1
                continue
            for key in totals:
                totals[key] += message["usage"][key]
        return DialogueUsage(**totals, missing_responses=missing)

    def add_exchange(self, user: str, assistant: str, usage: TokenUsage | None) -> None:
        """Атомарно сохранить запрос, ответ и его расход токенов."""
        reply: StoredMessage = {"role": "assistant", "content": assistant}
        if usage is not None:
            reply["usage"] = asdict(usage)
        self.add_messages([{"role": "user", "content": user}, reply])

    def add_message(self, role: Literal["user", "assistant"], content: str) -> None:
        self.add_messages([{"role": role, "content": content}])

    def add_messages(self, messages: list[StoredMessage]) -> None:
        """Добавить и сохранить сообщения одной операцией."""
        self._validate(messages)
        updated = self._messages + deepcopy(messages)
        self._validate(updated)
        self._save(updated)

    def clear(self) -> None:
        """Очистить историю в памяти и на диске."""
        self._save([])

    def _save(self, messages: list[StoredMessage]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.path.parent,
                prefix=f".{self.path.name}.", suffix=".tmp", delete=False,
            ) as output:
                temporary_path = Path(output.name)
                json.dump(messages, output, ensure_ascii=False, indent=2)
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            temporary_path.replace(self.path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        self._messages = messages
