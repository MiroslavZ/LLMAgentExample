import copy
import hashlib
import tempfile
import unittest
from pathlib import Path

from llm_agent.indexing.chunking import Chunk, Document, corpus_hash
from llm_agent.indexing.pipeline import build, compare
from llm_agent.indexing.store import load_index
from tests.test_indexing_chunking import FakeTokenizer


class FakeEmbedder:
    metadata = {"dimension": 2, "model": "test-model"}
    tokenizer = FakeTokenizer()

    def __init__(self):
        self.calls = []

    def encode(self, texts, query=False, batch_size=16):
        self.calls.append((list(texts), query, batch_size))
        return [[0.0, 1.0] if "beta" in text else [1.0, 0.0] for text in texts]


class PipelineTests(unittest.TestCase):
    def setUp(self):
        text = "alpha beta gamma"
        self.documents = [Document("notes.txt", "Notes", text, hashlib.sha256(text.encode()).hexdigest())]
        chunks = [Chunk("a", "notes.txt", "Notes", "", [], "fixed", "alpha", 0, 5, 5),
                  Chunk("b", "notes.txt", "Notes", "", [], "fixed", "beta", 6, 10, 4),
                  Chunk("g", "notes.txt", "Notes", "", [], "fixed", "gamma", 11, 16, 5)]
        metadata = {"schema_version": 1, "embedding": FakeEmbedder.metadata,
                    "corpus_hash": corpus_hash(self.documents), "strategy": "fixed",
                    "size": 10, "overlap": 0, "chunk_count": 3}
        self.indexes = {"fixed": (metadata, self.documents, chunks, [[1, 0], [.6, .8], [0, 1]])}
        self.questions = [{"question": "alpha?", "evidence": [{"source": "notes.txt", "quote": "alpha"}]},
                          {"question": "beta?", "evidence": [{"source": "notes.txt", "quote": "beta"}]}]

    def test_build_creates_readable_index_with_model_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.sqlite3"
            embedder = FakeEmbedder()
            metadata = build(path, self.documents, embedder, "fixed", size=8, overlap=2, batch_size=3)
            loaded, documents, chunks, vectors = load_index(path)
            self.assertEqual(loaded, metadata)
            self.assertEqual(documents, self.documents)
            self.assertEqual(metadata["embedding"], embedder.metadata)
            self.assertEqual(len(chunks), len(vectors))
            self.assertEqual(embedder.calls[0][1:], (False, 3))

    def test_compare_reports_actual_ranks_hit_rates_and_mrr(self):
        embedder = FakeEmbedder()
        report = compare(self.indexes, self.questions, embedder, top_k=2)
        result = report["strategies"]["fixed"]
        self.assertEqual([row["first_relevant_rank"] for row in result["questions"]], [1, 2])
        self.assertEqual(result["hit_at_1"], .5)
        self.assertEqual(result["hit_at_k"], 1)
        self.assertEqual(result["mrr_at_k"], .75)
        self.assertEqual(embedder.calls[0][1], True)
        self.assertEqual(result["statistics"]["covered_characters"], 14)
        self.assertEqual(result["statistics"]["repeated_characters"], 0)

    def test_compare_counts_missing_top_k_result_as_zero(self):
        result = compare(self.indexes, self.questions, FakeEmbedder(), top_k=1)["strategies"]["fixed"]
        self.assertEqual(result["hit_at_k"], .5)
        self.assertEqual(result["mrr_at_k"], .5)
        self.assertIsNone(result["questions"][1]["first_relevant_rank"])

    def test_rejects_stale_missing_and_empty_evidence_before_embedding(self):
        for evidence in [[], [{"source": "notes.txt", "quote": "outdated"}],
                         [{"source": "missing.txt", "quote": "alpha"}],
                         [{"source": "notes.txt", "quote": "  "}]]:
            embedder = FakeEmbedder()
            with self.subTest(evidence=evidence), self.assertRaisesRegex(ValueError, "эталон"):
                compare(self.indexes, [{"question": "q", "evidence": evidence}], embedder)
            self.assertEqual(embedder.calls, [])

    def test_rejects_incompatible_index_settings_before_embedding(self):
        for key, value in [("corpus_hash", "changed"), ("size", 20), ("overlap", 1),
                           ("embedding", {"dimension": 2, "model": "other"})]:
            indexes = copy.deepcopy(self.indexes)
            other = copy.deepcopy(indexes["fixed"])
            other[0][key] = value
            indexes["structural"] = other
            embedder = FakeEmbedder()
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "Несовместимые"):
                compare(indexes, self.questions, embedder)
            self.assertEqual(embedder.calls, [])

    def test_rejects_wrong_query_embedder(self):
        embedder = FakeEmbedder()
        embedder.metadata = {"dimension": 2, "model": "different"}
        with self.assertRaisesRegex(ValueError, "модели"):
            compare(self.indexes, self.questions, embedder)

    def test_empty_question_set_is_rejected(self):
        with self.assertRaises(ValueError):
            compare(self.indexes, [], FakeEmbedder())


if __name__ == "__main__":
    unittest.main()
