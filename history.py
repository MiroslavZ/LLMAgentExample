import json
import os
import tempfile
from pathlib import Path
from typing import Literal, TypedDict

DEFAULT_HISTORY_PATH = Path("history.json")


class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


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
        self._messages: list[Message] = data

    @staticmethod
    def _validate(messages: object) -> None:
        if not isinstance(messages, list) or any(
            not isinstance(message, dict)
            or set(message) != {"role", "content"}
            or message["role"] not in ("system", "user", "assistant")
            or not isinstance(message["content"], str)
            for message in messages
        ):
            raise ValueError(
                "История должна быть списком сообщений с role (system/user/assistant) и строкой content"
            )
        if any(message["role"] == "system" for message in messages[1:]):
            raise ValueError("Системный промпт допускается только один раз в начале истории")

    def get_system_prompt(self) -> str | None:
        if self._messages and self._messages[0]["role"] == "system":
            return self._messages[0]["content"]
        return None

    def set_system_prompt(self, content: str | None) -> None:
        """Сохранить первый непустой системный промпт в начале диалога."""
        if content and self.get_system_prompt() is None:
            messages = [{"role": "system", "content": content}] + self.get_messages()
            self._validate(messages)
            self._save(messages)

    def get_messages(self) -> list[Message]:
        """Вернуть копию истории, защищённую от изменений вызывающим кодом."""
        return [message.copy() for message in self._messages]

    def add_message(self, role: Literal["user", "assistant"], content: str) -> None:
        self.add_messages([{"role": role, "content": content}])

    def add_messages(self, messages: list[Message]) -> None:
        """Добавить и сохранить сообщения одной операцией."""
        self._validate(messages)
        updated = self._messages + [message.copy() for message in messages]
        self._validate(updated)
        self._save(updated)

    def clear(self) -> None:
        """Очистить историю в памяти и на диске."""
        self._save([])

    def _save(self, messages: list[Message]) -> None:
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
