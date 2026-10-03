import hashlib
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock, patch

from llm_agent.history import HistoryManager
from llm_agent.indexing.chunking import Chunk, Document, corpus_hash
from llm_agent.indexing.store import save_index
from llm_agent.models import ContextSettings, RequestOptions
from llm_agent.rag import RAGError, RAG_INSTRUCTION, Retriever
from llm_agent.service import ConversationService
from tests.helpers import completion


def retrieved(text="Уникальное содержимое базы"):
    return {"index": "test.sqlite3", "chunks": [{
        "chunk_id": "chunk-1", "source": "notes.md", "section": "Правила",
        "text": text, "score": 0.9,
    }]}


class ConversationRAGTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        patcher = patch("llm_agent.agent.OpenAI")
        self.openai = patcher.start()
        self.addCleanup(patcher.stop)
        self.complete = self.openai.return_value.chat.completions.create
        self.complete.return_value = completion()
        self.service = ConversationService(self.directory, "test-token")
        self.service.retriever = Mock()
        self.service.retriever.retrieve.return_value = retrieved()
        self.conversation = self.service.create()

    def send(self, question="Вопрос", *, enabled=True):
        return self.service.send(
            self.conversation.id, question, settings=ContextSettings(rag_enabled=enabled),
        )

    def test_disabled_mode_never_retrieves(self):
        result = self.send(enabled=False)
        self.service.retriever.retrieve.assert_not_called()
        self.assertEqual(result.turns[-1].status, "completed")
        self.assertFalse(result.turns[-1].rag_enabled)
        self.assertIsNone(result.turns[-1].rag_context)
        self.assertEqual(self.complete.call_args.kwargs["messages"], [
            {"role": "user", "content": "Вопрос"},
        ])

    def test_context_reaches_llm_and_persists_but_stays_out_of_history(self):
        saved_at_request = []

        def answer(**kwargs):
            saved_at_request.append(self.service.store.load(self.conversation.id))
            return completion()

        self.complete.side_effect = answer
        result = self.send()
        self.assertEqual(result.turns[-1].status, "completed")
        self.service.retriever.retrieve.assert_called_once_with("Вопрос")
        messages = self.complete.call_args.kwargs["messages"]
        self.assertIn({"role": "system", "content": RAG_INSTRUCTION}, messages)
        self.assertEqual(messages[-1], {"role": "user", "content": "Вопрос"})
        self.assertEqual(messages[-2]["role"], "user")
        self.assertEqual(json.loads(messages[-2]["content"].split("\n", 1)[1]), retrieved()["chunks"])
        self.assertEqual(saved_at_request[0].turns[-1].rag_context, retrieved())
        history = HistoryManager._decode_data(result.working_context)[0]
        self.assertEqual([item["content"] for item in history], ["Вопрос", "Ответ"])
        restored = ConversationService(self.directory, "test-token").get(result.id)
        self.assertEqual(restored, result)
        result.turns[-1].rag_context["chunks"].clear()
        self.assertEqual(self.service.get(result.id).turns[-1].rag_context, retrieved())

    def test_retrieval_error_is_saved_without_llm_or_history_change(self):
        original = deepcopy(self.conversation.working_context)
        self.service.retriever.retrieve.side_effect = RAGError("RAG: индекс отсутствует")
        result = self.send()
        self.openai.assert_not_called()
        self.assertEqual(result.turns[-1].status, "error")
        self.assertIn("индекс отсутствует", result.turns[-1].error)
        self.assertEqual(result.working_context, original)
        self.assertFalse(result.busy)
        self.assertEqual(ConversationService(self.directory, "test-token").get(result.id), result)

    def test_each_question_uses_fresh_context_and_disable_removes_it(self):
        self.service.retriever.retrieve.side_effect = [retrieved("FIRST_ONLY"), retrieved("SECOND_ONLY")]
        self.send("Первый")
        second = self.send("Второй")
        messages = json.dumps(self.complete.call_args.kwargs["messages"])
        self.assertNotIn("FIRST_ONLY", messages)
        self.assertIn("SECOND_ONLY", messages)
        self.assertEqual(second.turns[0].rag_context, retrieved("FIRST_ONLY"))
        third = self.send("Третий", enabled=False)
        self.assertEqual(self.service.retriever.retrieve.call_count, 2)
        messages = json.dumps(self.complete.call_args.kwargs["messages"])
        self.assertNotIn("SECOND_ONLY", messages)
        self.assertIsNone(third.turns[-1].rag_context)

    def test_legacy_json_defaults_to_disabled_rag(self):
        result = self.send(enabled=False)
        path = self.service.store.path(result.id)
        data = json.loads(path.read_text(encoding="utf-8"))
        data["settings"].pop("rag_enabled")
        for turn in data["turns"]:
            turn.pop("rag_enabled")
            turn.pop("rag_context")
        path.write_text(json.dumps(data), encoding="utf-8")
        restored = ConversationService(self.directory, "test-token").get(result.id)
        self.assertFalse(restored.settings.rag_enabled)
        self.assertFalse(restored.turns[0].rag_enabled)
        self.assertIsNone(restored.turns[0].rag_context)

    def test_small_window_keeps_context_for_current_question(self):
        self.service.send(
            self.conversation.id, "Первый",
            settings=ContextSettings(strategy="window", window_size=1, rag_enabled=True),
        )
        result = self.service.send(self.conversation.id, "Второй")
        messages = self.complete.call_args.kwargs["messages"]
        self.assertEqual(messages[-1], {"role": "user", "content": "Второй"})
        self.assertEqual(json.loads(messages[-2]["content"].split("\n", 1)[1]), retrieved()["chunks"])
        self.assertNotIn("Первый", [message["content"] for message in messages])
        history = HistoryManager._decode_data(result.working_context)[0]
        self.assertNotIn(retrieved()["chunks"][0]["text"], json.dumps(history, ensure_ascii=False))
        self.assertEqual(result.turns[-1].status, "completed")

    def test_meta_prompt_reuses_original_retrieval_for_both_stages(self):
        result = self.service.send(
            self.conversation.id, "Исходный вопрос", settings=ContextSettings(rag_enabled=True),
            options=RequestOptions(meta_prompt=True),
        )
        self.service.retriever.retrieve.assert_called_once_with("Исходный вопрос")
        self.assertEqual(self.complete.call_count, 2)
        for call in self.complete.call_args_list:
            messages = call.kwargs["messages"]
            self.assertEqual(json.loads(messages[-2]["content"].split("\n", 1)[1]), retrieved()["chunks"])
            self.assertIn({"role": "system", "content": RAG_INSTRUCTION}, messages)
        self.assertEqual(result.turns[-1].status, "completed")
        history = HistoryManager._decode_data(result.working_context)[0]
        self.assertNotIn(retrieved()["chunks"][0]["text"], json.dumps(history, ensure_ascii=False))

    def test_llm_failure_preserves_retrieved_context_on_disk(self):
        self.complete.side_effect = RuntimeError("secret-provider-detail")
        result = self.send()
        self.assertEqual(result.turns[-1].status, "error")
        self.assertEqual(result.turns[-1].rag_context, retrieved())
        self.assertNotIn("secret-provider-detail", result.turns[-1].error)
        restored = ConversationService(self.directory, "test-token").get(result.id)
        self.assertEqual(restored.turns[-1].rag_context, retrieved())


class RetrieverTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "index.sqlite3"
        text = "alpha beta"
        self.documents = [Document("notes.txt", "Notes", text, hashlib.sha256(text.encode()).hexdigest())]
        self.chunks = [
            Chunk("first", "notes.txt", "Notes", "", [], "fixed", "alpha ", 0, 6, 1),
            Chunk("second", "notes.txt", "Notes", "", [], "fixed", "beta", 6, 10, 1),
        ]
        self.metadata = {
            "schema_version": 1, "strategy": "fixed", "size": 5, "chunk_count": 2,
            "corpus_hash": corpus_hash(self.documents),
            "embedding": {"model": "test-model", "revision": "test-revision", "dimension": 2},
        }
        self.save()
        patcher = patch("llm_agent.rag.Embedder")
        self.embedder_class = patcher.start()
        self.addCleanup(patcher.stop)
        self.embedder = self.embedder_class.return_value
        self.embedder.metadata = deepcopy(self.metadata["embedding"])
        self.embedder.encode.return_value = [[0.0, 1.0]]
        self.retriever = Retriever(self.path, cache="test-cache")

    def save(self, vectors=None):
        save_index(self.path, self.documents, self.chunks,
                   vectors or [[1.0, 0.0], [0.0, 1.0]], self.metadata)

    def test_search_ranks_real_index_and_reuses_embedder(self):
        first = self.retriever.retrieve("Первый вопрос")
        self.retriever.retrieve("Второй вопрос")
        self.assertEqual(first["index"], str(self.path))
        self.assertEqual([chunk["chunk_id"] for chunk in first["chunks"]], ["second", "first"])
        self.assertEqual(first["chunks"][0]["source"], "notes.txt")
        self.assertEqual(first["chunks"][0]["text"], "beta")
        self.assertEqual(first["chunks"][0]["score"], 1.0)
        self.embedder_class.assert_called_once_with("test-model", "test-revision", "test-cache", True)
        self.embedder.encode.assert_called_with(["Второй вопрос"], query=True)

    def test_atomic_rebuild_is_seen_without_reloading_unchanged_model(self):
        self.retriever.retrieve("Вопрос")
        self.save(vectors=[[0.0, 1.0], [1.0, 0.0]])
        result = self.retriever.retrieve("Вопрос")
        self.assertEqual(result["chunks"][0]["chunk_id"], "first")
        self.embedder_class.assert_called_once()

    def test_incompatible_model_is_rejected_before_inference(self):
        self.embedder.metadata["revision"] = "wrong-revision"
        with self.assertRaisesRegex(RAGError, "не совпадают"):
            self.retriever.retrieve("Вопрос")
        self.embedder.encode.assert_not_called()

    def test_changed_embedding_settings_reload_model(self):
        self.retriever.retrieve("Вопрос")
        self.metadata["embedding"]["revision"] = "new-revision"
        self.save()
        replacement = Mock(metadata=deepcopy(self.metadata["embedding"]))
        replacement.encode.return_value = [[1.0, 0.0]]
        self.embedder_class.return_value = replacement
        result = self.retriever.retrieve("Вопрос")
        self.assertEqual(self.embedder_class.call_count, 2)
        self.assertEqual(result["chunks"][0]["chunk_id"], "first")

    def test_missing_index_does_not_load_model_or_create_file(self):
        absent = self.path.with_name("missing.sqlite3")
        with self.assertRaisesRegex(RAGError, "индекс отсутствует"):
            Retriever(absent).retrieve("Вопрос")
        self.assertFalse(absent.exists())
        self.embedder_class.assert_not_called()

    def test_corrupt_index_returns_safe_error_without_loading_model(self):
        self.path.write_bytes(b"secret-invalid-sqlite")
        with self.assertRaisesRegex(RAGError, "индекс отсутствует или повреждён") as error:
            self.retriever.retrieve("Вопрос")
        self.assertNotIn("secret-invalid-sqlite", str(error.exception))
        self.embedder_class.assert_not_called()

    def test_encode_value_error_is_safe_and_does_not_poison_next_query(self):
        self.embedder.encode.side_effect = [ValueError("secret-model-detail"), [[0.0, 1.0]]]
        with self.assertRaisesRegex(RAGError, "сократите слишком длинный вопрос") as error:
            self.retriever.retrieve("Длинный вопрос " * 100)
        self.assertNotIn("secret-model-detail", str(error.exception))
        result = self.retriever.retrieve("Короткий")
        self.assertEqual(result["chunks"][0]["chunk_id"], "second")


if __name__ == "__main__":
    unittest.main()
