"""Общие модели диалога; не зависят от интерфейса и клиента модели."""

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

from .history import HistoryManager
from .task_state import TaskState
from .tool_events import ToolCallRecord


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class ContextSettings:
    strategy: str = "full"
    window_size: int = 10
    last_messages: int = 6
    compress_every: int = 10
    rag_enabled: bool = False

    def validate(self) -> None:
        if type(self.rag_enabled) is not bool:
            raise ValueError("Режим RAG должен быть логическим значением")
        if self.strategy not in ("full", "window", "facts", "summary"):
            raise ValueError("Выберите поддерживаемую стратегию контекста")
        if type(self.window_size) is not int or self.window_size <= 0:
            raise ValueError("Размер окна должен быть положительным целым числом")
        if type(self.last_messages) is not int or self.last_messages < 0:
            raise ValueError("Число сохраняемых сообщений должно быть целым и не меньше нуля")
        if type(self.compress_every) is not int or self.compress_every <= 0:
            raise ValueError("Порог сжатия должен быть положительным целым числом")


@dataclass(frozen=True)
class RequestOptions:
    meta_prompt: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
    model: str | None = None
    stop_sequences: list[str] | None = None
    response_format: str = "text"

    def validate(self) -> None:
        if type(self.meta_prompt) is not bool:
            raise ValueError("Режим мета-промпта должен быть логическим значением")
        if self.temperature is not None and (
            type(self.temperature) not in (int, float)
            or not 0 <= self.temperature <= 2
            or not math.isfinite(self.temperature)
        ):
            raise ValueError("Температура должна быть числом от 0 до 2")
        if self.max_tokens is not None and (
            type(self.max_tokens) is not int or self.max_tokens <= 0
        ):
            raise ValueError("Лимит токенов должен быть положительным целым числом")
        if self.model is not None and (
            not isinstance(self.model, str) or not self.model.strip()
        ):
            raise ValueError("Название модели должно быть непустой строкой")
        if self.stop_sequences is not None and (
            not isinstance(self.stop_sequences, list)
            or any(not isinstance(value, str) or not value for value in self.stop_sequences)
        ):
            raise ValueError("Стоп-последовательности должны быть списком непустых строк")
        if self.response_format not in ("text", "object", "schema"):
            raise ValueError("Выберите поддерживаемый формат ответа")


TurnStatus = Literal["running", "completed", "error", "partial", "interrupted"]


@dataclass
class Turn:
    user: str
    options: RequestOptions = field(default_factory=RequestOptions)
    answer: str | None = None
    meta_prompt: str | None = None
    status: TurnStatus = "running"
    error: str | None = None
    created_at: str = field(default_factory=utc_now)
    elapsed_seconds: float | None = None
    memory_updated: bool = False
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    rag_enabled: bool = False
    rag_context: dict | None = None


@dataclass
class Conversation:
    id: str
    title: str = "Новый диалог"
    system_prompt: str = ""
    started: bool = False
    settings: ContextSettings = field(default_factory=ContextSettings)
    turns: list[Turn] = field(default_factory=list)
    # Историческое имя поля JSON: это краткосрочный контекст стратегии,
    # а не явная рабочая память задачи (она хранится в MemoryStore).
    working_context: object = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    @property
    def task_state(self) -> TaskState | None:
        return HistoryManager.task_from_data(self.working_context)

    @property
    def busy(self) -> bool:
        return any(turn.status == "running" for turn in self.turns)
