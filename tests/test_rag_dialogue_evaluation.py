import json
from dataclasses import asdict
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from llm_agent.rag import ROOT, select_candidates
from llm_agent.rag_dialogue_evaluation import load_scenarios, run_evaluation
from tests.helpers import completion, rag_preparation


class RAGDialogueEvaluationTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.index = self.directory / "index.sqlite3"
        self.index.write_bytes(b"fixed-index")
        self.output = self.directory / "report.json"
        self.scenarios_path = self.directory / "scenarios.json"
        self.scenarios = [{"id": name, "turns": [
            {"question": f"Цель: инструкция {name}, вопрос {number}",
             "expected_sources": ["doc.md"], "checks": ["SECRET_EXPECTED_CHECK"]}
            for number in range(1, 14)
        ]} for name in ("one", "two")]
        self.save_scenarios()
        self.metadata = {"corpus_hash": "corpus", "embedding": {"model": "test"}}
        self.documents = [SimpleNamespace(source="doc.md", text="Нужный отрывок")]
        self.enterContext(patch("llm_agent.rag_dialogue_evaluation.load_index",
                                return_value=(self.metadata, self.documents, [], [])))
        self.retriever = self.enterContext(patch("llm_agent.rag_dialogue_evaluation.Retriever")).return_value
        self.retriever.retrieve.side_effect = self.retrieve
        self.preparation = self.enterContext(patch("llm_agent.service.prepare_turn", side_effect=self.prepare))
        self.openai = self.enterContext(patch("llm_agent.agent.OpenAI"))
        response = completion()
        response.choices[0].message.content = json.dumps({
            "status": "answered", "answer": "Нужный отрывок", "sources": ["a"],
            "quotes": [{"chunk_id": "a", "text": "Нужный отрывок"}], "clarification": None,
        }, ensure_ascii=False)
        self.generate = self.openai.return_value.chat.completions.create
        self.generate.return_value = response

    def save_scenarios(self):
        self.scenarios_path.write_text(json.dumps(self.scenarios, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def prepare(question, memory, turns, token, model, *, rewrite_enabled):
        result = rag_preparation(question, memory, turns, token, model, rewrite_enabled=rewrite_enabled)
        if not result["memory"]["goal"]:
            result["memory"]["goal"] = {"value": question, "quote": question, "turn": len(turns)}
        return result

    def retrieve(self, question, settings, *, rewrite=None):
        chunks = [{"source": "doc.md", "section": "Раздел", "text": "Нужный отрывок",
                   "score": 0.9, "chunk_id": "a", "token_count": 3}]
        selected, candidates, counts = select_candidates(chunks, settings)
        return dict(index=str(self.index), chunks=selected, candidates=candidates, counts=counts,
                    rewrite={key: value for key, value in rewrite.items() if key != "query"},
                    original_query=question, search_query=rewrite["query"], settings=asdict(settings),
                    retrieval_seconds=0.01, **self.metadata)

    def run_evaluation(self):
        return run_evaluation(scenarios_path=self.scenarios_path, index_path=self.index,
                              output=self.output, token="test")

    def test_real_service_runs_26_sequential_turns_and_reloads_after_seven(self):
        report = self.run_evaluation()
        self.assertEqual(report["status"], "completed")
        self.assertEqual(len(report["runs"]), 26)
        self.assertEqual(self.retriever.retrieve.call_count, 26)
        self.assertEqual(self.generate.call_count, 26)
        self.assertEqual(report["settings"]["window_size"], 2)
        self.assertTrue(report["settings"]["rag_rewrite_enabled"])
        self.assertTrue(report["settings"]["rag_filter_enabled"])
        self.assertEqual(len(report["restarts"]), 2)
        self.assertTrue(all(item["memory_preserved"] and item["turns_preserved"] for item in report["restarts"]))
        for scenario in self.scenarios:
            rows = [row for row in report["runs"] if row["scenario"] == scenario["id"]]
            self.assertEqual(len({row["conversation_id"] for row in rows}), 1)
            self.assertIsNone(rows[0]["memory_before"]["goal"])
            self.assertEqual(rows[0]["memory_after"], rows[-1]["memory_after"])
            self.assertTrue(all(all(row["citation_metrics"].values()) for row in rows))
            self.assertTrue(all(row["semantic_support"] is None and row["goal_retained"] is None for row in rows))
        self.assertNotEqual(report["runs"][0]["conversation_id"], report["runs"][13]["conversation_id"])
        for call in self.generate.call_args_list:
            self.assertNotIn("SECRET_EXPECTED_CHECK", json.dumps(call.kwargs["messages"]))
        self.assertEqual(json.loads(self.output.read_text(encoding="utf-8")), report)
        self.assertTrue(self.output.with_suffix(".data").is_dir())

    def test_unknown_source_rejected_before_network_and_writes(self):
        self.scenarios[0]["turns"][0]["expected_sources"] = ["missing.md"]
        self.save_scenarios()
        with self.assertRaisesRegex(ValueError, "снимке индекса"):
            self.run_evaluation()
        self.generate.assert_not_called()
        self.retriever.retrieve.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_existing_report_and_store_are_preserved(self):
        self.output.write_text("previous", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "уже существует"):
            self.run_evaluation()
        self.assertEqual(self.output.read_text(encoding="utf-8"), "previous")
        self.output.unlink()
        self.output.with_suffix(".data").mkdir()
        with self.assertRaisesRegex(ValueError, "уже существует"):
            self.run_evaluation()

    def test_changed_index_marks_report_interrupted(self):
        def changing_retrieve(*args, **kwargs):
            self.index.write_bytes(b"new-index")
            return self.retrieve(*args, **kwargs)
        self.retriever.retrieve.side_effect = changing_retrieve
        with self.assertRaisesRegex(ValueError, "Индекс изменился"):
            self.run_evaluation()
        report = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "interrupted")
        self.assertEqual(report["runs"], [])

    def test_bundled_scenarios_have_two_13_turn_dialogues(self):
        sources = [SimpleNamespace(source=f"docs/{name}.md") for name in ("MEMORY", "TASK_STATE", "INDEXING", "RAG")]
        scenarios = load_scenarios(ROOT / "docs/evaluation/day25_scenarios.json", sources)
        self.assertEqual([len(scenario["turns"]) for scenario in scenarios], [13, 13])
        self.assertTrue(all(any(not turn["expected_sources"] for turn in scenario["turns"]) for scenario in scenarios))


if __name__ == "__main__":
    unittest.main()
