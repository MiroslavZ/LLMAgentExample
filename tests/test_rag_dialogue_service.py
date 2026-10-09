"""Сохранение памяти вне окна, отказов и жизненного цикла процесса."""

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock, patch

from llm_agent.models import ContextSettings
from llm_agent.rag import RAGError
from llm_agent.rag_dialogue import empty_memory
from llm_agent.service import ConversationService
from tests.helpers import create_selected_conversation
from llm_agent.storage import ConversationStorageError
from tests.test_rag import grounded_completion, retrieved


class DialogueServiceTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.service = ConversationService(self.directory, "test-token")
        self.retriever = Mock()
        self.retriever.retrieve.return_value = retrieved()
        self.service.retriever = self.retriever
        self.conversation = create_selected_conversation(self.service)
        self.prepare = self.enterContext(patch("llm_agent.service.prepare_turn"))
        self.prepare.side_effect = self.prepared
        self.client = self.enterContext(patch("llm_agent.agent.OpenAI"))
        self.complete = self.client.return_value.chat.completions.create
        self.complete.return_value = grounded_completion()

    @staticmethod
    def prepared(question, memory, turns, token, model, *, rewrite_enabled, **kwargs):
        memory = deepcopy(memory)
        if len(turns) == 1:
            memory["goal"] = {"value": question, "quote": question, "turn": 1}
        diagnostic = dict(status="success", reason=None, elapsed_seconds=0.1, usage=None)
        rewrite = dict(diagnostic, query=question) if rewrite_enabled else None
        return dict(memory=memory, rewrite=rewrite, diagnostic=diagnostic)

    def send(self, question, **kwargs):
        return self.service.send(self.conversation.id, question, **kwargs)

    def test_memory_survives_window_restart_and_isolation(self):
        for number in range(13):
            if number == 7:
                self.service = ConversationService(self.directory, "test-token")
                self.service.retriever = self.retriever
            result = self.send("Подготовить памятку" if number == 0 else f"Уточнение {number}",
                               settings=ContextSettings(strategy="window", window_size=1, rag_enabled=True))
            self.assertEqual(result.turns[-1].status, "completed")
            self.assertEqual(result.dialogue_task_memory["goal"]["value"], "Подготовить памятку")
            messages = self.complete.call_args.kwargs["messages"]
            self.assertTrue(any("Подготовить памятку" in item["content"] for item in messages))
            self.assertTrue(result.turns[-1].rag_answer["sources"])
            self.assertEqual(len(result.turns), number + 1)
        self.assertEqual(self.retriever.retrieve.call_count, 13)
        self.assertEqual(self.service.store.load(result.id), result)
        self.assertEqual(create_selected_conversation(self.service).dialogue_task_memory, empty_memory())

    def test_memory_saved_before_failed_search(self):
        self.retriever.retrieve.side_effect = RAGError("Индекс недоступен")
        result = self.send("Подготовить памятку", settings=ContextSettings(rag_enabled=True))
        self.assertEqual(result.turns[-1].status, "error")
        self.assertEqual(self.service.store.load(result.id).dialogue_task_memory["goal"]["turn"], 1)
        self.client.assert_not_called()

    def test_unknown_updates_memory_and_disabled_mode_preserves_it(self):
        self.retriever.retrieve.return_value = dict(index="test.sqlite3", chunks=[])
        result = self.send("Подготовить памятку", settings=ContextSettings(rag_enabled=True))
        self.assertEqual(result.turns[-1].rag_answer["status"], "unknown")
        self.assertIn("Источники", result.turns[-1].answer)
        state = deepcopy(result.dialogue_task_memory)
        self.prepare.reset_mock()
        result = self.send("Продолжаем", settings=ContextSettings(rag_enabled=False))
        self.prepare.assert_not_called()
        self.assertEqual(result.dialogue_task_memory, state)

    def test_old_snapshot_and_forged_provenance(self):
        result = self.send("Подготовить памятку", settings=ContextSettings(rag_enabled=True))
        path = self.service.store.path(result.id)
        data = json.loads(path.read_text(encoding="utf-8"))
        forged = deepcopy(data)
        forged["dialogue_task_memory"]["goal"]["quote"] = "Несуществующая реплика"
        path.write_text(json.dumps(forged), encoding="utf-8")
        with self.assertRaises(ConversationStorageError):
            self.service.store.load(result.id)
        del data["dialogue_task_memory"]
        for turn in data["turns"]:
            del turn["rag_preparation"]
        path.write_text(json.dumps(data), encoding="utf-8")
        self.assertEqual(self.service.store.load(result.id).dialogue_task_memory, empty_memory())

    def test_legacy_unknown_is_normalized_without_accepting_tampering(self):
        self.retriever.retrieve.return_value = dict(index="test.sqlite3", chunks=[])
        result = self.send("Вопрос", settings=ContextSettings(rag_enabled=True))
        path = self.service.store.path(result.id)
        data = json.loads(path.read_text(encoding="utf-8"))
        answer = data["turns"][0]["rag_answer"]
        data["turns"][0]["answer"] = answer["answer"] + "\n\n" + answer["clarification"]
        path.write_text(json.dumps(data), encoding="utf-8")
        self.assertIn("Источники", self.service.store.load(result.id).turns[0].answer)
        data["turns"][0]["answer"] = "Подменённый ответ"
        path.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaises(ConversationStorageError):
            self.service.store.load(result.id)
