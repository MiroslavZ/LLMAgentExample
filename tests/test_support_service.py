"""Проверки контекста, изоляции и отказов без облачных запросов."""

import asyncio
import tempfile
import unittest
from pathlib import Path

from llm_agent.rag import RAGError
from support_agent.config import Settings
from support_agent.errors import SupportError
from support_agent.generation import make_messages, parse_answer
from support_agent.service import SupportService


class FakeCRM:
    def __init__(self, events=None):
        self.events = events if events is not None else []

    async def list_users(self):
        return [{"id": "u1", "name": "Анна", "summary": "Демо"},
                {"id": "u2", "name": "Иван", "summary": "Демо"}]

    async def list_tickets(self, user_id):
        return [{"id": "t1", "user_id": user_id, "subject": "Документы", "status": "open"}]

    async def context(self, user_id, ticket_id):
        self.events.append("mcp")
        if ticket_id == "foreign":
            raise SupportError("Обращение недоступно", 404)
        return {"demo": True, "user": {"id": user_id, "settings": {"rag_enabled": user_id == "u2"}},
                "ticket": None if ticket_id is None else {
                    "id": ticket_id, "user_id": user_id, "subject": "Документы", "error": "INDEX_MISSING", "status": "open"}}


class FakeRetriever:
    def __init__(self, events=None, *, empty=False, broken=False):
        self.events = events if events is not None else []
        self.empty, self.broken = empty, broken
        self.queries = []

    def retrieve(self, query, settings):
        self.events.append("rag")
        self.queries.append(query)
        if self.broken:
            raise RAGError("Повреждён индекс")
        return {"chunks": [] if self.empty else [{"source": "docs/support/FAQ.md", "section": "RAG", "text": "Включите RAG."}]}


class FakeResponder:
    def __init__(self, events=None):
        self.events = events if events is not None else []
        self.calls = []

    def answer(self, question, history, context, sources, warnings):
        self.events.append("llm")
        self.calls.append((question, history, context, sources, warnings))
        return "RAG включён" if context["user"]["settings"]["rag_enabled"] else "Включите RAG", sources


class SupportServiceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.events = []
        self.settings = Settings(database_path=Path(self.tmp.name) / "history.sqlite3")
        self.crm = FakeCRM(self.events)
        self.retriever = FakeRetriever(self.events)
        self.responder = FakeResponder(self.events)
        self.service = SupportService(self.settings, crm=self.crm, retriever=self.retriever, responder=self.responder)

    def tearDown(self):
        self.tmp.cleanup()

    async def test_context_precedes_search_and_changes_answer(self):
        first = await self.service.reply("browser", "u1", "t1", "Почему не работают документы?")
        second = await self.service.reply("browser", "u2", "t2", "Почему не работают документы?")
        self.assertEqual(self.events, ["mcp", "rag", "llm"] * 2)
        self.assertIn("INDEX_MISSING", self.retriever.queries[0])
        self.assertNotEqual(first["answer"], second["answer"])
        self.assertEqual(self.responder.calls[1][1], [])

    async def test_no_documents_still_allows_ticket_facts(self):
        self.retriever.empty = True
        turn = await self.service.reply("browser", "u1", "t1", "Какой статус обращения?")
        self.assertEqual(turn["sources"], [])
        self.assertTrue(turn["warnings"])
        self.assertEqual(self.responder.calls[0][2]["ticket"]["status"], "open")

    async def test_missing_index_is_explicit_and_does_not_impersonate_client_problem(self):
        self.retriever.broken = True
        turn = await self.service.reply("browser", "u1", None, "Помоги")
        self.assertIn("сервиса поддержки", turn["warnings"][0])
        self.assertIsNone(turn["context"]["ticket"])

    async def test_denied_ticket_never_reaches_model_or_history(self):
        with self.assertRaises(SupportError):
            await self.service.reply("browser", "u1", "foreign", "Вопрос")
        self.assertEqual(self.events, ["mcp"])
        self.assertEqual(self.service.store.get("browser", "u1", "foreign"), [])

    async def test_history_is_five_pairs_scoped_and_survives_restart(self):
        for number in range(7):
            await self.service.reply("browser", "u1", "t1", f"Вопрос {number}")
        self.assertEqual(len(self.responder.calls[-1][1]), 5)
        turns = self.service.store.get("browser", "u1", "t1")
        self.assertEqual([turn["question"] for turn in turns], [f"Вопрос {number}" for number in range(2, 7)])
        for scope in [("other", "u1", "t1"), ("browser", "u2", "t1"), ("browser", "u1", None)]:
            self.assertEqual(self.service.store.get(*scope), [])
        restarted = SupportService(self.settings, crm=self.crm, retriever=self.retriever, responder=self.responder)
        self.assertEqual(restarted.store.get("browser", "u1", "t1"), turns)
        await restarted.clear("browser", "u1", "t1")
        self.assertEqual(restarted.store.get("browser", "u1", "t1"), [])

    async def test_concurrent_send_and_clear_are_rejected_for_busy_scope(self):
        started, release = asyncio.Event(), asyncio.Event()
        original = self.crm.context

        async def delayed(user_id, ticket_id):
            started.set()
            await release.wait()
            return await original(user_id, ticket_id)

        self.crm.context = delayed
        task = asyncio.create_task(self.service.reply("browser", "u1", "t1", "Первый"))
        await started.wait()
        try:
            with self.assertRaises(SupportError) as send_error:
                await self.service.reply("browser", "u1", "t1", "Второй")
            self.assertEqual(send_error.exception.status_code, 409)
            with self.assertRaises(SupportError):
                await self.service.clear("browser", "u1", "t1")
        finally:
            release.set()
            await task

    async def test_generation_error_leaves_history_unchanged(self):
        def failed(*args):
            raise SupportError("DeepSeek недоступен")
        self.responder.answer = failed
        with self.assertRaises(SupportError):
            await self.service.reply("browser", "u1", "t1", "Вопрос")
        self.assertEqual(self.service.store.get("browser", "u1", "t1"), [])
        self.assertEqual(self.service._busy, set())


class SupportAnswerTests(unittest.TestCase):
    def test_answer_only_uses_provided_source_ids(self):
        sources = [{"id": "D1", "text": "Существующий документ"}]
        self.assertEqual(parse_answer('{"answer":"Ответ","source_ids":["D1"]}', sources), ("Ответ", sources))
        for payload in ['{"answer":"Ответ","source_ids":["D2"]}',
                        '{"answer":"Ответ","source_ids":["D1","D1"]}',
                        '{"answer":" ","source_ids":[]}',
                        '{"answer":"Ответ","source_ids":[],"secret":"test"}']:
            with self.assertRaises(SupportError):
                parse_answer(payload, sources)

    def test_history_window_and_current_evidence_are_separate(self):
        history = [{"question": f"Q{i}", "answer": f"A{i}"} for i in range(8)]
        messages = make_messages("Теперь", history, {"user": {"id": "u1"}}, [], [])
        self.assertEqual(len(messages), 13)  # system + 10 прошлых + evidence + вопрос
        self.assertEqual(messages[1]["content"], "Q3")
        self.assertEqual(messages[-1]["content"], "Теперь")
        self.assertIn("mcp_context", messages[-2]["content"])

    def test_api_key_not_in_settings_representation(self):
        self.assertNotIn("private-token", repr(Settings(api_key="private-token")))
