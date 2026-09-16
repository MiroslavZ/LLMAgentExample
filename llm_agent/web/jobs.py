"""Фоновые запросы живут на сервере независимо от открытой страницы."""

import asyncio
import logging

from nicegui import background_tasks, run

from ..models import ContextSettings, RequestOptions
from ..service import ConversationBusyError, ConversationService

logger = logging.getLogger(__name__)


class RequestRunner:
    def __init__(self, service: ConversationService) -> None:
        self.service = service
        self._tasks: dict[str, asyncio.Task] = {}
        self.errors: dict[str, str] = {}

    def is_running(self, conversation_id: str) -> bool:
        return conversation_id in self._tasks

    def submit(
        self, conversation_id: str, user: str, *,
        system_prompt: str, options: RequestOptions,
        settings: ContextSettings | None = None,
        expected_settings: ContextSettings | None = None,
    ) -> None:
        if self.is_running(conversation_id):
            raise ValueError("В этом диалоге уже выполняется запрос.")
        self.errors.pop(conversation_id, None)
        self._tasks[conversation_id] = background_tasks.create(
            self._send(conversation_id, user, system_prompt, options, settings, expected_settings),
            name=f"agent-request-{conversation_id}",
        )

    async def _send(
        self, conversation_id: str, user: str, system_prompt: str, options: RequestOptions,
        settings: ContextSettings | None, expected_settings: ContextSettings | None,
    ) -> None:
        try:
            await run.io_bound(
                self.service.send, conversation_id, user,
                system_prompt=system_prompt, options=options,
                settings=settings, expected_settings=expected_settings,
            )
        except (ValueError, ConversationBusyError) as error:
            self.errors[conversation_id] = str(error)
        except Exception as error:
            # Ответы SDK и содержимое запроса не попадают ни в UI, ни в журнал.
            logger.error("Request could not finish: %s", type(error).__name__)
            self.errors[conversation_id] = "Не удалось завершить запрос. Проверьте доступ к хранилищу диалогов."
        finally:
            self._tasks.pop(conversation_id, None)
