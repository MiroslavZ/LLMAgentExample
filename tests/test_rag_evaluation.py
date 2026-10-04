import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from llm_agent.rag import select_candidates
from llm_agent.rag_evaluation import evidence_metrics, load_questions, run_evaluation, summarize
from tests.helpers import completion


class RAGEvaluationTests(unittest.TestCase):
    def setUp(self):
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
        chunks = [{"source": "doc.md", "text": "Нужный\nотрывок", "score": 0.4, "chunk_id": "a", "token_count": 3}]
        selected, candidates, counts = select_candidates(chunks, settings)
        return dict(
            chunks=selected, candidates=candidates, counts=counts,
            rewrite=rewrite or {"status": "disabled"},
            retrieval_seconds=0.01, **self.metadata,
        )

    def run_comparison(self, **kwargs):
        return run_evaluation(
            questions_path=self.questions, index_path=self.index, output=self.output,
            threshold=0.5, modes=["baseline", "filtered"], **kwargs,
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
            generate.return_value = completion()
            report = self.run_comparison(token="test")
        self.assertEqual(report["status"], "completed")
        self.assertEqual(len({row["conversation_id"] for row in report["runs"]}), 2)
        for call in generate.call_args_list:
            messages = call.kwargs["messages"]
            self.assertEqual(messages[-1], {"role": "user", "content": "Вопрос"})
            self.assertNotIn("EXPECTED_MUST_NOT_REACH_LLM", json.dumps(messages))
            self.assertFalse(any(message["role"] == "assistant" for message in messages))
        self.assertEqual(len(generate.call_args_list), 2)
        self.assertTrue(all(row["request_usage"] for row in report["runs"]))
        self.assertFalse(any(self.directory.glob("rag-evaluation-*")))

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
        with patch("llm_agent.agent.OpenAI") as openai, patch("llm_agent.service.rewrite_query") as rewrite:
            openai.return_value.chat.completions.create.return_value = completion()
            rewrite.return_value = {"query": "Вопрос", "status": "fallback"}
            report = run_evaluation(
                questions_path=self.questions, index_path=self.index, output=self.output,
                threshold=0.35, modes=["rewrite_filtered"], token="test",
            )
        self.assertEqual(report["status"], "completed_with_fallback")
        self.assertEqual(report["summary"]["rewrite_filtered"]["answers_completed"], 1)
        self.assertEqual(report["summary"]["rewrite_filtered"]["mode_answers_completed"], 0)

    def test_offline_rewrite_is_rejected_instead_of_faking_success(self):
        with self.assertRaisesRegex(ValueError, "Query rewrite требует LLM"):
            run_evaluation(
                questions_path=self.questions, index_path=self.index, output=self.output,
                threshold=0.35, modes=["rewrite"], retrieval_only=True,
            )


if __name__ == "__main__":
    unittest.main()
