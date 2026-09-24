"""Атомарное хранение полной ленты и рабочего контекста в одном JSON."""

import json
import math
import os
import re
import tempfile
from contextlib import contextmanager, suppress
from dataclasses import asdict, fields
from datetime import datetime
from pathlib import Path
from typing import Iterator

from .history import HistoryManager
from .models import ContextSettings, Conversation, RequestOptions, Turn
from .tool_events import ToolCallRecord


class ConversationStorageError(RuntimeError):
    """Не удалось прочесть или сохранить состояние диалога."""


class ConversationBusyError(RuntimeError):
    """Другая операция уже изменяет этот диалог."""


class ConversationStore:
    VERSION = 1

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir).expanduser().absolute()
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise ConversationStorageError("Не удалось открыть каталог диалогов") from error

    def path(self, conversation_id: str) -> Path:
        if not isinstance(conversation_id, str) or re.fullmatch(r"[0-9a-f]{32}", conversation_id) is None:
            raise ValueError("Некорректный идентификатор диалога")
        return self.data_dir / f"{conversation_id}.json"

    def ids(self) -> list[str]:
        try:
            return [path.stem for path in self.data_dir.glob("*.json")]
        except OSError as error:
            raise ConversationStorageError("Не удалось прочесть список диалогов") from error

    @contextmanager
    def lock(self, conversation_id: str) -> Iterator[None]:
        """Неблокирующая блокировка совместима с несколькими процессами сервера.

        Файлы .lock остаются после использования: удаление inode сделало бы
        блокировку ненадёжной при одновременном открытии другим процессом.
        """
        lock_path = self.path(conversation_id).with_suffix(".lock")
        try:
            lock_file = lock_path.open("a+b")
        except OSError as error:
            raise ConversationStorageError("Не удалось открыть блокировку диалога") from error
        with lock_file:
            lock_file.seek(0, os.SEEK_END)
            if lock_file.tell() == 0:
                lock_file.write(b"\0")
                lock_file.flush()
            lock_file.seek(0)
            try:
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                raise ConversationBusyError("В этом диалоге уже выполняется операция") from error
            try:
                yield
            finally:
                lock_file.seek(0)
                if os.name == "nt":
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def load(self, conversation_id: str) -> Conversation:
        path = self.path(conversation_id)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return self._decode(data, conversation_id)
        except FileNotFoundError:
            raise KeyError("Диалог не найден") from None
        except (OSError, UnicodeDecodeError, ValueError, TypeError, KeyError,
                AttributeError, OverflowError, RecursionError) as error:
            raise ConversationStorageError(
                f"Не удалось прочесть диалог {path.name}. Файл сохранён без изменений."
            ) from error

    @classmethod
    def _decode(cls, data: object, conversation_id: str) -> Conversation:
        if (
            not isinstance(data, dict)
            or type(data.get("version")) is not int
            or data["version"] != cls.VERSION
            or set(data) != {"version", *(field.name for field in fields(Conversation))}
        ):
            raise ValueError("Неизвестный формат диалога")
        values = data.copy()
        del values["version"]
        values["settings"] = ContextSettings(**values["settings"])
        values["settings"].validate()
        turns = []
        turn_fields = {field.name for field in fields(Turn)}
        for item in values["turns"]:
            if not isinstance(item, dict) or set(item) not in (turn_fields, turn_fields - {"tool_calls"}):
                raise ValueError("Некорректный формат хода диалога")
            turn_values = item.copy()
            turn_values["options"] = RequestOptions(**turn_values["options"])
            turn_values["options"].validate()
            tool_calls = turn_values.get("tool_calls", [])
            if not isinstance(tool_calls, list):
                raise ValueError("Вызовы инструментов должны быть списком")
            turn_values["tool_calls"] = [ToolCallRecord.from_dict(record) for record in tool_calls]
            turn = Turn(**turn_values)
            if (
                not isinstance(turn.user, str) or not turn.user.strip()
                or turn.status not in ("running", "completed", "error", "partial", "interrupted")
                or any(value is not None and not isinstance(value, str)
                       for value in (turn.answer, turn.meta_prompt, turn.error))
                or not isinstance(turn.created_at, str)
                or type(turn.memory_updated) is not bool
                or (turn.elapsed_seconds is not None and (
                    type(turn.elapsed_seconds) not in (int, float)
                    or not math.isfinite(turn.elapsed_seconds) or turn.elapsed_seconds < 0
                ))
                or (turn.status == "completed" and turn.answer is None)
            ):
                raise ValueError("Некорректный ход диалога")
            datetime.fromisoformat(turn.created_at)
            turns.append(turn)
        values["turns"] = turns
        conversation = Conversation(**values)
        if (
            conversation.id != conversation_id
            or any(not isinstance(value, str) for value in (
                conversation.title, conversation.system_prompt,
                conversation.created_at, conversation.updated_at,
            ))
            or type(conversation.started) is not bool
            or (bool(turns) and not conversation.started)
            or any(turn.status == "running" for turn in turns[:-1])
        ):
            raise ValueError("Некорректные данные диалога")
        datetime.fromisoformat(conversation.created_at)
        datetime.fromisoformat(conversation.updated_at)
        HistoryManager._decode_data(conversation.working_context)
        return conversation

    def save(self, conversation: Conversation) -> None:
        path = self.path(conversation.id)
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.data_dir,
                prefix=f".{conversation.id}.", suffix=".tmp", delete=False,
            ) as output:
                temporary_path = Path(output.name)
                json.dump({"version": self.VERSION, **asdict(conversation)},
                          output, ensure_ascii=False, indent=2, allow_nan=False)
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            temporary_path.replace(path)
        except (OSError, ValueError, TypeError) as error:
            raise ConversationStorageError("Не удалось сохранить диалог. Проверьте доступ к каталогу данных.") from error
        finally:
            if temporary_path is not None:
                with suppress(OSError):
                    temporary_path.unlink(missing_ok=True)

    def delete(self, conversation_id: str) -> None:
        try:
            self.path(conversation_id).unlink()
        except FileNotFoundError:
            raise KeyError("Диалог не найден") from None
        except OSError as error:
            raise ConversationStorageError("Не удалось удалить диалог") from error
