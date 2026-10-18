"""MCP → документация → DeepSeek; без настроек универсального агента."""

import asyncio
from contextlib import contextmanager
from datetime import datetime, timezone

from llm_agent.models import RAGSettings
from llm_agent.rag import RAGError, Retriever

from .config import Settings
from .crm import SupportCRM
from .errors import SupportError
from .generation import DeepSeekResponder
from .storage import HistoryStore


def search_query(question: str, history: list[dict], context: dict) -> str:
    ticket = context.get("ticket") or {}
    parts = [question[:300], str(ticket.get("subject", ""))[:130], str(ticket.get("error", ""))[:100]]
    if history:
        parts.append(history[-1]["question"][:120])
    return " ".join(part.strip() for part in parts if part.strip())


class SupportService:
    def __init__(self, settings: Settings, *, crm=None, retriever=None, responder=None, store=None) -> None:
        self.crm = crm if crm is not None else SupportCRM(settings.mcp_server())
        self.retriever = retriever if retriever is not None else Retriever(settings.index_path, cache=settings.cache)
        self.responder = responder if responder is not None else DeepSeekResponder(settings)
        self.store = store if store is not None else HistoryStore(settings.database_path)
        self._busy: set[tuple] = set()

    @contextmanager
    def operation(self, session_id: str, user_id: str, ticket_id: str | None):
        # Все публичные операции вызываются в одном event loop FastAPI.
        scope = session_id, user_id, ticket_id
        if scope in self._busy:
            raise SupportError("Ответ для этого обращения уже готовится. Дождитесь его завершения.", 409)
        self._busy.add(scope)
        try:
            yield
        finally:
            self._busy.discard(scope)

    async def history(self, session_id: str, user_id: str, ticket_id: str | None) -> list[dict]:
        await self.crm.context(user_id, ticket_id)
        return self.store.get(session_id, user_id, ticket_id)

    async def clear(self, session_id: str, user_id: str, ticket_id: str | None) -> None:
        with self.operation(session_id, user_id, ticket_id):
            await self.crm.context(user_id, ticket_id)
            self.store.clear(session_id, user_id, ticket_id)

    async def reply(self, session_id: str, user_id: str, ticket_id: str | None, question: str) -> dict:
        question = question.strip()
        if not question or len(question) > 4000:
            raise SupportError("Введите вопрос длиной от 1 до 4000 символов.", 422)
        with self.operation(session_id, user_id, ticket_id):
            context = await self.crm.context(user_id, ticket_id)
            history = self.store.get(session_id, user_id, ticket_id)
            query = search_query(question, history, context)
            warnings = []
            try:
                result = await asyncio.to_thread(
                    self.retriever.retrieve, query,
                    RAGSettings(top_k_before=8, top_k_after=5, filter_enabled=True, similarity_threshold=0.35),
                )
                sources = [{"id": f"D{index}", "source": chunk["source"], "section": chunk["section"], "text": chunk["text"]}
                           for index, chunk in enumerate(result["chunks"], 1)]
                if not sources:
                    warnings.append("В базе знаний поддержки не найдена подходящая инструкция. Ответ может использовать только данные обращения и уточнения.")
            except RAGError:
                sources = []
                warnings.append("База знаний сервиса поддержки недоступна. Доступны только данные пользователя и обращения; администратору нужно проверить индекс поддержки.")
            answer, used_sources = await asyncio.to_thread(
                self.responder.answer, question, history, context, sources, warnings,
            )
            turn = dict(question=question, answer=answer, sources=used_sources, context=context,
                        warnings=warnings, created_at=datetime.now(timezone.utc).isoformat())
            self.store.append(session_id, user_id, ticket_id, turn)
            return turn
