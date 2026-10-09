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
from llm_agent.models import ContextSettings, RAGSettings, RequestOptions
from llm_agent.rag import RAGError, RAG_INSTRUCTION, Retriever, rewrite_query, select_candidates
from llm_agent.rag_validation import validate_rag_context
from llm_agent.storage import ConversationStorageError
from llm_agent.service import ConversationService
from tests.helpers import completion, patch_rag_preparation, rag_preparation, create_selected_conversation, register_model


def retrieved(text="Уникальное содержимое базы"):
    return {"index": "test.sqlite3", "chunks": [{
        "chunk_id": "chunk-1", "source": "notes.md", "section": "Правила",
        "text": text, "score": 0.9,
    }]}


def grounded_completion(quote="содержимое"):
    response = completion()
    response.choices[0].message.content = json.dumps({
        "status": "answered", "answer": "Ответ", "sources": ["chunk-1"],
        "quotes": [{"chunk_id": "chunk-1", "text": quote}], "clarification": None,
    }, ensure_ascii=False)
    return response


class ConversationRAGTests(unittest.TestCase):
    def setUp(self):
        patch_rag_preparation(self)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        patcher = patch("llm_agent.agent.OpenAI")
        self.openai = patcher.start()
        self.addCleanup(patcher.stop)
        self.complete = self.openai.return_value.chat.completions.create
        self.complete.return_value = grounded_completion()
        self.service = ConversationService(self.directory, "test-token")
        self.service.retriever = Mock()
        self.service.retriever.retrieve.return_value = retrieved()
        self.conversation = create_selected_conversation(self.service)

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
            return grounded_completion()

        self.complete.side_effect = answer
        result = self.send()
        self.assertEqual(result.turns[-1].status, "completed")
        self.service.retriever.retrieve.assert_called_once_with("Вопрос", RAGSettings(), rewrite=None)
        messages = self.complete.call_args.kwargs["messages"]
        self.assertIn({"role": "system", "content": RAG_INSTRUCTION}, messages)
        self.assertEqual(messages[-1], {"role": "user", "content": "Вопрос"})
        self.assertEqual(messages[-2]["role"], "user")
        self.assertEqual(json.loads(messages[-2]["content"].split("\n", 1)[1]), retrieved()["chunks"])
        self.assertEqual(saved_at_request[0].turns[-1].rag_context, retrieved())
        history = HistoryManager._decode_data(result.working_context)[0]
        self.assertEqual([item["content"] for item in history], ["Вопрос", result.turns[-1].answer])
        self.assertEqual(result.turns[-1].rag_answer["sources"][0]["source"], "notes.md")
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
        self.complete.side_effect = [grounded_completion("FIRST_ONLY"), grounded_completion("SECOND_ONLY"), completion()]
        self.send("Первый")
        second = self.send("Второй")
        messages = self.complete.call_args.kwargs["messages"][-2]["content"]
        self.assertNotIn("FIRST_ONLY", messages)
        self.assertIn("SECOND_ONLY", messages)
        self.assertEqual(second.turns[0].rag_context, retrieved("FIRST_ONLY"))
        third = self.send("Третий", enabled=False)
        self.assertEqual(self.service.retriever.retrieve.call_count, 2)
        messages = self.complete.call_args.kwargs["messages"]
        self.assertFalse(any(item["content"].startswith("Справочные фрагменты базы") for item in messages))
        self.assertIsNone(third.turns[-1].rag_context)

    def test_legacy_json_defaults_to_disabled_rag(self):
        result = self.send(enabled=False)
        path = self.service.store.path(result.id)
        data = json.loads(path.read_text(encoding="utf-8"))
        data["settings"].pop("rag_enabled")
        for key in list(data["settings"]):
            if key.startswith("rag_"):
                del data["settings"][key]
        for turn in data["turns"]:
            turn.pop("rag_enabled")
            turn.pop("rag_context")
            turn.pop("rag_settings")
            turn.pop("rag_answer")
        path.write_text(json.dumps(data), encoding="utf-8")
        restored = ConversationService(self.directory, "test-token").get(result.id)
        self.assertFalse(restored.settings.rag_enabled)
        self.assertFalse(restored.turns[0].rag_enabled)
        self.assertIsNone(restored.turns[0].rag_context)
        self.assertFalse(restored.settings.rag_rewrite_enabled)
        self.assertFalse(restored.settings.rag_filter_enabled)

    def test_day22_rag_enabled_snapshot_remains_readable(self):
        result = self.send()
        path = self.service.store.path(result.id)
        data = json.loads(path.read_text(encoding="utf-8"))
        for key in list(data["settings"]):
            if key.startswith("rag_") and key != "rag_enabled":
                del data["settings"][key]
        data["turns"][0].pop("rag_settings")
        data["turns"][0].pop("rag_answer")
        path.write_text(json.dumps(data), encoding="utf-8")
        restored = self.service.store.load(result.id)
        self.assertTrue(restored.settings.rag_enabled)
        self.assertEqual(restored.turns[0].rag_context, retrieved())
        self.assertIsNone(restored.turns[0].rag_settings)
        self.assertEqual(restored.settings.rag_options(), RAGSettings())

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
        self.complete.side_effect = [completion(), grounded_completion()]
        result = self.service.send(
            self.conversation.id, "Исходный вопрос", settings=ContextSettings(rag_enabled=True),
            options=RequestOptions(meta_prompt=True),
        )
        self.service.retriever.retrieve.assert_called_once_with("Исходный вопрос", RAGSettings(), rewrite=None)
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

    @patch("llm_agent.service.prepare_turn")
    def test_disabled_rag_skips_rewrite_even_if_option_is_enabled(self, rewrite):
        result = self.service.send(self.conversation.id, "Вопрос", settings=ContextSettings(
            rag_enabled=False, rag_rewrite_enabled=True, rag_filter_enabled=True,
        ))
        rewrite.assert_not_called()
        self.service.retriever.retrieve.assert_not_called()
        self.assertIsNone(result.turns[-1].rag_settings)

    @patch("llm_agent.service.prepare_turn")
    def test_rewrite_is_separate_and_original_question_reaches_generator(self, rewrite):
        model = register_model(self.service, token="test-token", model_id="chosen-model")
        self.service.select_model(self.conversation.id, model.id)
        rewrite_result = dict(query="Поисковая формулировка", status="success", reason=None,
                              elapsed_seconds=0.1, usage=None)
        rewrite.side_effect = lambda *args, **kwargs: dict(
            rag_preparation(*args, **kwargs), rewrite=rewrite_result,
        )
        result = self.service.send(self.conversation.id, "Исходный вопрос", settings=ContextSettings(
            rag_enabled=True, rag_rewrite_enabled=True,
        ))
        rewrite.assert_called_once()
        self.assertEqual(rewrite.call_args.args[0], "Исходный вопрос")
        self.assertEqual(rewrite.call_args.args[3:], ("test-token", "chosen-model"))
        self.assertEqual(rewrite.call_args.kwargs, {
            "rewrite_enabled": True, "base_url": model.base_url, "timeout": model.timeout,
        })
        self.service.retriever.retrieve.assert_called_once_with(
            "Исходный вопрос", RAGSettings(rewrite_enabled=True), rewrite=rewrite_result,
        )
        self.assertEqual(self.complete.call_args.kwargs["messages"][-1]["content"], "Исходный вопрос")
        self.assertEqual(result.turns[-1].user, "Исходный вопрос")
        self.assertTrue(result.turns[-1].rag_settings.rewrite_enabled)

    def test_settings_snapshot_survives_later_setting_changes_and_retrieval_error(self):
        self.service.retriever.retrieve.side_effect = RAGError("RAG: индекс отсутствует")
        self.service.send(self.conversation.id, "Вопрос", settings=ContextSettings(
            rag_enabled=True, rag_filter_enabled=True, rag_similarity_threshold=0.8,
        ))
        changed = self.service.update_settings(self.conversation.id, ContextSettings())
        self.assertEqual(changed.turns[-1].rag_settings.similarity_threshold, 0.8)
        self.assertTrue(changed.turns[-1].rag_settings.filter_enabled)
        self.assertFalse(changed.settings.rag_enabled)

    def test_threshold_gate_applies_without_filter_and_before_meta_or_facts(self):
        context = retrieved()
        context["chunks"][0]["score"] = 0.34
        self.service.retriever.retrieve.return_value = context
        result = self.service.send(self.conversation.id, "Вопрос", settings=ContextSettings(
            rag_enabled=True, rag_filter_enabled=False, strategy="facts",
        ), options=RequestOptions(meta_prompt=True))
        self.complete.assert_not_called()
        turn = result.turns[-1]
        self.assertEqual(turn.status, "completed")
        self.assertEqual(turn.rag_answer["status"], "unknown")
        self.assertIsNone(turn.meta_prompt)
        self.assertEqual(self.service.store.load(result.id), result)
        self.assertEqual(HistoryManager._decode_data(result.working_context)[3].total_tokens, 0)

    def test_threshold_equality_allows_generation_and_json_result(self):
        context = retrieved()
        context["chunks"][0]["score"] = 0.35
        self.service.retriever.retrieve.return_value = context
        result = self.service.send(self.conversation.id, "Вопрос", settings=ContextSettings(
            rag_enabled=True,
        ), options=RequestOptions(response_format="object"))
        self.complete.assert_called_once()
        self.assertEqual(json.loads(result.turns[-1].answer), result.turns[-1].rag_answer)
        self.assertEqual(self.service.store.load(result.id), result)

    def test_bad_draft_is_not_published_or_saved_and_usage_is_retained(self):
        self.complete.return_value = completion()
        callbacks = []
        result = self.service.send(self.conversation.id, "Вопрос", settings=ContextSettings(
            rag_enabled=True,
        ), on_response=callbacks.append)
        self.assertEqual(self.complete.call_count, 2)
        self.assertEqual(result.turns[-1].status, "error")
        self.assertIsNone(result.turns[-1].answer)
        self.assertIsNone(result.turns[-1].rag_answer)
        self.assertEqual(callbacks, [])
        history = HistoryManager._decode_data(result.working_context)
        self.assertEqual(history[0], [])
        self.assertEqual(history[3].total_tokens, 260)

    def test_saved_evidence_and_rendered_text_are_checked_on_load(self):
        result = self.send()
        path = self.service.store.path(result.id)
        original = json.loads(path.read_text(encoding="utf-8"))
        for field in ("source", "quote", "answer"):
            with self.subTest(field=field):
                data = deepcopy(original)
                turn = data["turns"][0]
                if field == "source":
                    turn["rag_answer"]["sources"][0]["source"] = "invented.md"
                elif field == "quote":
                    turn["rag_answer"]["quotes"][0]["text"] = "invented quote"
                else:
                    turn["answer"] = "Другой ответ"
                path.write_text(json.dumps(data), encoding="utf-8")
                with self.assertRaises(ConversationStorageError):
                    self.service.store.load(result.id)


class RetrieverTests(unittest.TestCase):
    def setUp(self):
        patch_rag_preparation(self)
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

    def test_sentence_transformer_settings_are_restored_from_index(self):
        self.metadata["embedding"].update(
            backend="sentence_transformers", query_prefix="query instruction: ", passage_prefix="",
        )
        self.save()
        self.embedder.metadata = deepcopy(self.metadata["embedding"])
        self.retriever.retrieve("Вопрос")
        self.embedder_class.assert_called_once_with(
            "test-model", "test-revision", "test-cache", True,
            backend="sentence_transformers", query_prefix="query instruction: ", passage_prefix="",
        )

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

    def test_filter_threshold_boundary_counts_and_top_k(self):
        result = self.retriever.retrieve("Вопрос", RAGSettings(
            filter_enabled=True, similarity_threshold=1.0, top_k_before=2, top_k_after=1,
        ))
        self.assertEqual([chunk["chunk_id"] for chunk in result["chunks"]], ["second"])
        self.assertEqual(result["counts"], dict(found=2, passed=1, selected=1))
        self.assertEqual([c["selection_reason"] for c in result["candidates"]], ["selected", "below_threshold"])
        validate_rag_context(result)
        baseline = self.retriever.retrieve("Вопрос", RAGSettings(top_k_before=2, top_k_after=1))
        self.assertEqual(baseline["counts"], dict(found=2, passed=2, selected=1))
        self.assertEqual(baseline["candidates"][1]["selection_reason"], "top_k")
        validate_rag_context(baseline)

    def test_rewrite_query_is_embedded_and_too_many_tokens_fall_back(self):
        self.embedder.limit = 5
        self.embedder.tokenizer.encode.return_value = [1, 2]
        rewrite = dict(query="Переписанный", status="success", reason=None, elapsed_seconds=0.1, usage=None)
        settings = RAGSettings(rewrite_enabled=True)
        result = self.retriever.retrieve("Исходный", settings, rewrite=rewrite)
        self.embedder.encode.assert_called_with(["Переписанный"], query=True)
        self.assertEqual(result["original_query"], "Исходный")
        self.assertEqual(result["search_query"], "Переписанный")
        validate_rag_context(result)
        self.embedder.tokenizer.encode.return_value = list(range(6))
        result = self.retriever.retrieve("Исходный", settings, rewrite=rewrite)
        self.embedder.encode.assert_called_with(["Исходный"], query=True)
        self.assertEqual(result["rewrite"]["status"], "fallback")
        self.assertIn("токенизатора", result["rewrite"]["reason"])
        self.assertEqual(rewrite["status"], "success")
        validate_rag_context(result)

    def test_empty_context_persists_and_rejected_chunks_never_reach_llm(self):
        self.embedder.encode.return_value = [[-1.0, 0.0]]
        service = ConversationService(self.path.parent / "conversations", "test-token")
        service.retriever = self.retriever
        conversation = create_selected_conversation(service)
        with patch("llm_agent.agent.OpenAI") as client:
            complete = client.return_value.chat.completions.create
            complete.return_value = completion()
            result = service.send(conversation.id, "Вопрос", settings=ContextSettings(
                rag_enabled=True, rag_filter_enabled=True, rag_similarity_threshold=0.5,
            ))
        self.assertEqual(result.turns[-1].status, "completed")
        context = result.turns[-1].rag_context
        self.assertEqual(context["chunks"], [])
        self.assertEqual(context["counts"], dict(found=2, passed=0, selected=0))
        self.assertEqual(service.store.load(result.id), result)
        complete.assert_not_called()
        self.assertEqual(result.turns[-1].rag_answer["status"], "unknown")
        self.assertIn("Не знаю", result.turns[-1].answer)
        for text in ("alpha", "beta"):
            self.assertNotIn(text, json.dumps(result.working_context))
        self.assertEqual(context["settings"]["similarity_threshold"], 0.5)
        path = service.store.path(result.id)
        data = json.loads(path.read_text(encoding="utf-8"))
        data["turns"][-1]["rag_context"]["counts"]["selected"] = 1
        path.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaises(ConversationStorageError):
            service.store.load(result.id)

    def test_ties_keep_index_order_and_baseline_retains_previous_top_five(self):
        self.save(vectors=[[0.0, 1.0], [0.0, 1.0]])
        result = self.retriever.retrieve("Вопрос")
        self.assertEqual([c["chunk_id"] for c in result["chunks"]], ["first", "second"])
        self.assertEqual([c["original_rank"] for c in result["candidates"]], [1, 2])

    def test_saved_diagnostics_reject_inconsistent_or_nonfinite_values(self):
        context = self.retriever.retrieve("Вопрос")
        mutations = [
            lambda c: c.update(retrieval_seconds=float("nan")),
            lambda c: c["settings"].update(top_k_after=21),
            lambda c: c["candidates"][0].update(original_rank=True),
            lambda c: c["candidates"][0].update(selection_reason="below_threshold"),
            lambda c: c["rewrite"].update(status="success"),
            lambda c: c.update(search_query="Другой вопрос при выключенном rewrite"),
            lambda c: c["chunks"].clear(),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                invalid = deepcopy(context)
                mutate(invalid)
                with self.assertRaises(ValueError):
                    validate_rag_context(invalid)
        with self.assertRaisesRegex(ValueError, "снимком"):
            validate_rag_context(context, RAGSettings(filter_enabled=True))

    def test_rewritten_filtered_context_survives_generation_failure(self):
        self.embedder.limit = 512
        self.embedder.tokenizer.encode.return_value = [1, 2]
        service = ConversationService(self.path.parent / "conversations", "test-token")
        service.retriever = self.retriever
        conversation = create_selected_conversation(service)
        rewrite = dict(query="Поисковый запрос", status="success", reason=None, elapsed_seconds=0.1,
                       usage=dict(prompt_tokens=20, completion_tokens=5, total_tokens=25))
        prepared = lambda *args, **kwargs: dict(rag_preparation(*args, **kwargs), rewrite=rewrite)
        with patch("llm_agent.service.prepare_turn", side_effect=prepared), patch("llm_agent.agent.OpenAI") as client:
            complete = client.return_value.chat.completions.create
            complete.side_effect = RuntimeError("secret-api-detail")
            result = service.send(conversation.id, "Исходный вопрос", settings=ContextSettings(
                rag_enabled=True, rag_rewrite_enabled=True, rag_filter_enabled=True,
                rag_similarity_threshold=0.5,
            ))
        self.assertEqual(result.turns[-1].status, "error")
        self.assertNotIn("secret-api-detail", result.turns[-1].error)
        self.assertEqual(service.store.load(result.id), result)
        context = result.turns[-1].rag_context
        self.assertEqual(context["rewrite"]["usage"]["total_tokens"], 25)
        self.assertEqual(context["search_query"], "Поисковый запрос")
        self.assertEqual(context["counts"], dict(found=2, passed=1, selected=1))
        messages = complete.call_args.kwargs["messages"]
        self.assertEqual(messages[-1]["content"], "Исходный вопрос")
        fragments = json.loads(messages[-2]["content"].split("\n", 1)[1])
        self.assertEqual([chunk["text"] for chunk in fragments], ["beta"])
        self.assertNotIn("alpha", json.dumps(messages))
        self.embedder.encode.assert_called_once_with(["Поисковый запрос"], query=True)


class RewriteTests(unittest.TestCase):
    def setUp(self):
        patcher = patch("llm_agent.rag.OpenAI")
        self.client_class = patcher.start()
        self.addCleanup(patcher.stop)
        self.complete = self.client_class.return_value.__enter__.return_value.chat.completions.create
        self.response = completion()
        self.response.choices[0].message.content = "  Запрос для поиска  "
        self.complete.return_value = self.response

    def test_short_service_request_has_no_history_or_tools_and_tracks_usage(self):
        result = rewrite_query("Исходный вопрос", "test-token", "model")
        self.assertEqual(result["query"], "Запрос для поиска")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["usage"]["total_tokens"], 130)
        self.assertGreaterEqual(result["elapsed_seconds"], 0)
        request = self.complete.call_args.kwargs
        self.assertEqual(request["model"], "model")
        self.assertEqual(len(request["messages"]), 2)
        self.assertNotIn("tools", request)
        self.assertEqual(request["max_tokens"], 200)
        self.assertEqual(self.client_class.call_args.kwargs["timeout"], 15.0)
        self.assertEqual(self.client_class.call_args.kwargs["max_retries"], 0)

    def test_empty_truncated_oversized_and_failed_rewrites_use_original(self):
        for text, reason in (("  ", "stop"), ("Часть", "length"), ("x" * 1501, "stop")):
            with self.subTest(text=text[:20], reason=reason):
                self.response.choices[0].message.content = text
                self.response.choices[0].finish_reason = reason
                result = rewrite_query("Исходный", "test-token", "model")
                self.assertEqual(result["query"], "Исходный")
                self.assertEqual(result["status"], "fallback")
                self.assertTrue(result["reason"])
                self.assertEqual(result["usage"]["total_tokens"], 130)
        self.complete.side_effect = RuntimeError("secret-provider-detail")
        result = rewrite_query("Исходный", "test-token", "model")
        self.assertEqual(result["query"], "Исходный")
        self.assertNotIn("secret-provider-detail", json.dumps(result))
        self.assertIsNone(result["usage"])


class RAGSettingsTests(unittest.TestCase):
    def test_invalid_settings_are_rejected(self):
        for values in (
            {"top_k_before": 0}, {"top_k_after": 21}, {"top_k_after": True},
            {"top_k_before": 20.5}, {"similarity_threshold": float("nan")},
            {"similarity_threshold": float("inf")}, {"similarity_threshold": 1.01},
            {"similarity_threshold": -1.01}, {"similarity_threshold": False},
            {"rewrite_enabled": 1}, {"filter_enabled": "true"},
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                RAGSettings(**values).validate()

    def test_negative_score_boundary_and_selection_do_not_mutate_candidates(self):
        matches = [dict(retrieved()["chunks"][0], chunk_id=str(i), score=score)
                   for i, score in enumerate([0.2, -0.5, -0.6])]
        before = deepcopy(matches)
        selected, candidates, counts = select_candidates(matches, RAGSettings(
            filter_enabled=True, similarity_threshold=-0.5, top_k_after=2,
        ))
        self.assertEqual([c["score"] for c in selected], [0.2, -0.5])
        self.assertEqual(counts, dict(found=3, passed=2, selected=2))
        self.assertEqual(matches, before)
        self.assertEqual(candidates[-1]["selection_reason"], "below_threshold")


if __name__ == "__main__":
    unittest.main()
