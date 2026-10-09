import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from llm_agent.rag import select_candidates
from llm_agent.rag_evaluation import citation_metrics, dialogue_usage, evaluation_token, evidence_metrics, load_questions, run_evaluation, summarize
from tests.helpers import completion, patch_rag_preparation, rag_preparation


class RAGEvaluationTests(unittest.TestCase):
    @staticmethod
    def grounded_completion():
        response = completion()
        response.choices[0].message.content = json.dumps({
            "status": "answered", "answer": "Нужный отрывок", "sources": ["a"],
            "quotes": [{"chunk_id": "a", "text": "Нужный отрывок"}], "clarification": None,
        }, ensure_ascii=False)
        return response

    def setUp(self):
        patch_rag_preparation(self)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.index = self.directory / "index.sqlite3"
        self.index.write_bytes(b"fixed-index")
        self.questions = self.directory / "questions.json"
        self.output = self.directory / "report.json"
        self.metadata = {"corpus_hash": "corpus", "embedding": {"model": "test"}}
        self.documents = [SimpleNamespace(source="doc.md", text="Нужный\n отрывок")]
        self.question_data = [{
            "question": "Вопрос", "expected": "EXPECTED_MUST_NOT_REACH_LLM",
            "evidence": [{"source": "doc.md", "quote": "Нужный отрывок"}],
        }]
        self.questions.write_text(json.dumps(self.question_data), encoding="utf-8")
        index_patcher = patch("llm_agent.rag_evaluation.load_index", return_value=(self.metadata, self.documents, [], []))
        index_patcher.start()
        self.addCleanup(index_patcher.stop)
        retriever_patcher = patch("llm_agent.rag_evaluation.Retriever")
        self.retriever = retriever_patcher.start().return_value
        self.retriever.retrieve.side_effect = self.retrieve
        self.addCleanup(retriever_patcher.stop)

    def retrieve(self, question, settings, *, rewrite=None):
        chunks = [{"source": "doc.md", "section": "Раздел", "text": "Нужный\nотрывок", "score": 0.4, "chunk_id": "a", "token_count": 3}]
        selected, candidates, counts = select_candidates(chunks, settings)
        return dict(
            chunks=selected, candidates=candidates, counts=counts,
            rewrite=rewrite or {"status": "disabled"},
            retrieval_seconds=0.01, **self.metadata,
        )

    def run_comparison(self, **kwargs):
        return run_evaluation(
            questions_path=self.questions, index_path=self.index, output=self.output,
            threshold=kwargs.pop("threshold", 0.5), modes=["baseline", "filtered"], **kwargs,
        )

    def test_evidence_uses_source_and_normalized_quote(self):
        self.assertEqual(load_questions(self.questions, self.documents), self.question_data)
        evidence = self.question_data[0]["evidence"]
        self.assertTrue(evidence_metrics(evidence, [{"source": "doc.md", "text": "Нужный\n отрывок"}])["all_hit"])
        self.assertFalse(evidence_metrics(evidence, [{"source": "wrong.md", "text": "Нужный отрывок"}])["any_hit"])
        self.assertFalse(evidence_metrics(evidence, [{"source": "doc.md", "text": "Другой отрывок"}])["any_hit"])
        self.assertIsNone(evidence_metrics([], [])["any_hit"])

    def test_invalid_evidence_stops_before_retrieval_or_api(self):
        self.documents[0].text = "Иной снимок документа"
        with patch("llm_agent.rag_evaluation.ConversationService") as service:
            with self.assertRaisesRegex(ValueError, "evidence не соответствует"):
                self.run_comparison(token="test")
            service.assert_not_called()
        self.retriever.retrieve.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_empty_evidence_requires_explicit_out_of_corpus_group(self):
        self.question_data[0]["evidence"] = []
        self.questions.write_text(json.dumps(self.question_data), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "out_of_corpus"):
            load_questions(self.questions, self.documents)
        self.question_data[0]["group"] = "out_of_corpus"
        self.questions.write_text(json.dumps(self.question_data), encoding="utf-8")
        self.assertEqual(len(load_questions(self.questions, self.documents)), 1)

    def test_offline_report_distinguishes_retrieval_from_answer_quality(self):
        report = self.run_comparison(retrieval_only=True)
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["summary"]["baseline"]["selected_hits"], 1)
        self.assertEqual(report["summary"]["filtered"]["selected_hits"], 0)
        self.assertEqual(report["summary"]["filtered"]["candidate_hits"], 1)
        self.assertEqual(report["summary"]["filtered"]["empty_contexts"], 1)
        self.assertEqual(report["summary"]["baseline"]["answers_completed"], 0)
        self.assertIsNone(report["summary"]["baseline"]["manual_score_sum"])
        self.assertTrue(all(row["answer"] is None for row in report["runs"]))
        self.assertEqual(json.loads(self.output.read_text(encoding="utf-8")), report)

    def test_index_change_aborts_without_mixing_snapshots(self):
        def changing_index(question, settings):
            self.index.write_bytes(b"another-index")
            return self.retrieve(question, settings)

        self.retriever.retrieve.side_effect = changing_index
        with self.assertRaisesRegex(ValueError, "Индекс изменился"):
            self.run_comparison(retrieval_only=True)
        report = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "interrupted")
        self.assertEqual(report["runs"], [])

    def test_generation_uses_fresh_dialogues_and_never_sends_expected_answer(self):
        with patch("llm_agent.agent.OpenAI") as openai:
            generate = openai.return_value.chat.completions.create
            generate.return_value = self.grounded_completion()
            report = self.run_comparison(token="test", threshold=0.35)
        self.assertEqual(report["status"], "completed")
        self.assertEqual(len({row["conversation_id"] for row in report["runs"]}), 2)
        for call in generate.call_args_list:
            messages = call.kwargs["messages"]
            self.assertEqual(messages[-1], {"role": "user", "content": "Вопрос"})
            self.assertNotIn("EXPECTED_MUST_NOT_REACH_LLM", json.dumps(messages))
            self.assertFalse(any(message["role"] == "assistant" for message in messages))
        self.assertEqual(len(generate.call_args_list), 2)
        self.assertTrue(all(row["request_usage"] for row in report["runs"]))
        self.assertTrue(all(row["rag_answer"]["status"] == "answered" for row in report["runs"]))
        self.assertFalse(any(self.directory.glob("rag-evaluation-*")))

    def test_local_model_without_token_uses_requested_endpoint(self):
        base_url = "http://127.0.0.1:1234/v1"
        with patch("llm_agent.agent.OpenAI") as openai:
            generate = openai.return_value.chat.completions.create
            generate.return_value = self.grounded_completion()
            report = self.run_comparison(base_url=base_url, model="gemma", threshold=0.35)
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["base_url"], base_url)
        self.assertTrue(openai.call_args_list)
        for call in openai.call_args_list:
            self.assertEqual(call.kwargs["base_url"], base_url)
            self.assertEqual(call.kwargs["api_key"], "local-no-auth")
        self.assertTrue(all(call.kwargs["model"] == "gemma" for call in generate.call_args_list))

    def test_evaluation_token_is_selected_for_the_requested_server_only(self):
        with patch.dict(os.environ, {"API_KEY": "cloud-secret", "LOCAL_LLM_TOKEN": "local-secret"}, clear=True):
            self.assertEqual(evaluation_token("https://api.deepseek.com/v1", None), "cloud-secret")
            self.assertEqual(evaluation_token("http://127.0.0.1:1234/v1", None), "")
            self.assertEqual(evaluation_token("http://127.0.0.1:1234/v1", "LOCAL_LLM_TOKEN"), "local-secret")
            with self.assertRaisesRegex(ValueError, "отсутствует или пуста"):
                evaluation_token("http://127.0.0.1:1234/v1", "MISSING_TOKEN")

    def test_existing_report_is_never_overwritten(self):
        self.output.write_text("previous result", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "уже существует"):
            self.run_comparison(retrieval_only=True)
        self.assertEqual(self.output.read_text(encoding="utf-8"), "previous result")

    def test_rewrite_fallback_is_not_counted_as_successful_improved_mode(self):
        row = {
            "mode": "rewrite_filtered", "status": "completed", "manual_score": 2,
            "evidence": [{"source": "doc.md", "quote": "Отрывок"}],
            "candidate_evidence": {"any_hit": True}, "selected_evidence": {"any_hit": True},
            "rag_context": {"chunks": ["fallback context"], "rewrite": {"status": "fallback"}},
        }
        result = summarize([row])["rewrite_filtered"]
        self.assertEqual(result["answers_completed"], 1)
        self.assertEqual(result["mode_answers_completed"], 0)
        self.assertEqual(result["rewrite_fallbacks"], 1)
        self.assertEqual(result["evaluated_retrievals"], 0)
        self.assertEqual(result["selected_hits"], 0)
        self.assertIsNone(result["manual_score_sum"])
        row["rag_context"]["rewrite"]["status"] = "success"
        result = summarize([row])["rewrite_filtered"]
        self.assertEqual(result["mode_answers_completed"], 1)
        self.assertEqual(result["rewrite_successes"], 1)
        self.assertEqual(result["selected_hits"], 1)
        self.assertEqual(result["manual_score_sum"], 2)

    def test_completed_answer_after_fallback_marks_comparison_incomplete(self):
        with patch("llm_agent.agent.OpenAI") as openai, patch("llm_agent.service.prepare_turn") as rewrite:
            openai.return_value.chat.completions.create.return_value = self.grounded_completion()
            rewrite.side_effect = lambda *args, **kwargs: dict(
                rag_preparation(*args, **kwargs), rewrite={"query": "Вопрос", "status": "fallback"},
            )
            report = run_evaluation(
                questions_path=self.questions, index_path=self.index, output=self.output,
                threshold=0.35, modes=["rewrite_filtered"], token="test",
            )
        self.assertEqual(report["status"], "completed_with_fallback")
        self.assertEqual(report["summary"]["rewrite_filtered"]["answers_completed"], 1)
        self.assertEqual(report["summary"]["rewrite_filtered"]["mode_answers_completed"], 0)

    def test_citations_require_correct_metadata_and_their_own_chunk(self):
        context = {"chunks": [
            {"chunk_id": "a", "source": "doc.md", "section": "Раздел", "text": "Сервер не требуется."},
            {"chunk_id": "b", "source": "other.md", "section": "Другой", "text": "Сервер требуется."},
        ]}
        answer = {"status": "answered", "sources": [{"chunk_id": "a", "source": "doc.md", "section": "Раздел"}],
                  "quotes": [{"chunk_id": "a", "text": "Сервер\nне требуется."}]}
        self.assertTrue(all(citation_metrics(answer, context).values()))
        answer["sources"][0]["source"] = "invented.md"
        self.assertFalse(citation_metrics(answer, context)["sources_match_context"])
        answer["quotes"][0]["text"] = "Сервер требуется."
        self.assertFalse(citation_metrics(answer, context)["quotes_match_chunks"])
        for value in (None, {"status": "unknown"}):
            self.assertTrue(all(metric is None for metric in citation_metrics(value, context).values()))

    def test_citation_success_never_implicitly_counts_as_semantic_review(self):
        report = self.run_comparison(retrieval_only=True)
        row = report["runs"][0]
        row.update(status="completed", rag_answer={"status": "answered"},
                   citation_metrics=dict(sources_present=True, quotes_present=True,
                                         sources_match_context=True, quotes_match_chunks=True))
        result = summarize([row])["baseline"]
        self.assertEqual(result["citation_rates_among_substantive_answers"]["quotes_match_chunks"], 1)
        self.assertEqual(result["semantic_reviews"], 0)
        self.assertIsNone(result["fully_supported_answers"])
        row["semantic_support"] = "unsupported"
        result = summarize([row])["baseline"]
        self.assertEqual(result["semantic_reviews"], 1)
        self.assertEqual(result["fully_supported_answers"], 0)

    def test_usage_combines_archived_repairs_and_final_response(self):
        data = {"messages": [{"role": "assistant", "content": "Финал",
                              "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14}}],
                "summary": "", "archived_usage": {
                    "prompt_tokens": 20, "completion_tokens": 6, "total_tokens": 26, "missing_responses": 0}}
        self.assertEqual(dialogue_usage(data), {
            "prompt_tokens": 30, "completion_tokens": 10, "total_tokens": 40, "missing_responses": 0})
        del data["messages"][0]["usage"]
        self.assertEqual(dialogue_usage(data)["missing_responses"], 1)

    def test_offline_rewrite_is_rejected_instead_of_faking_success(self):
        with self.assertRaisesRegex(ValueError, "Query rewrite требует LLM"):
            run_evaluation(
                questions_path=self.questions, index_path=self.index, output=self.output,
                threshold=0.35, modes=["rewrite"], retrieval_only=True,
            )


if __name__ == "__main__":
    unittest.main()
