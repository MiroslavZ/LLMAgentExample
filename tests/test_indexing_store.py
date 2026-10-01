import hashlib
import json
import sqlite3
import tempfile
import unittest
from dataclasses import replace
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from llm_agent.indexing.chunking import Chunk, Document, corpus_hash
from llm_agent.indexing.store import load_index, save_index, search, validate_vector


class IndexStoreTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / "index.sqlite3"
        text = "alpha beta"
        self.documents = [Document("notes.txt", "Notes", text, hashlib.sha256(text.encode()).hexdigest())]
        self.chunks = [
            Chunk("first", "notes.txt", "Notes", "", [], "fixed", "alpha ", 0, 6, 1),
            Chunk("second", "notes.txt", "Notes", "", [], "fixed", "beta", 6, 10, 1),
        ]
        self.vectors = [[1.0, 0.0], [0.0, 1.0]]
        self.metadata = {"schema_version": 1, "strategy": "fixed", "size": 5,
                         "chunk_count": 2, "corpus_hash": corpus_hash(self.documents),
                         "embedding": {"dimension": 2}}

    def save(self):
        save_index(self.path, self.documents, self.chunks, self.vectors, self.metadata)

    def modify_chunk(self, position, **changes):
        with closing(sqlite3.connect(self.path)) as db, db:
            value = json.loads(db.execute("SELECT data FROM chunks WHERE position = ?", (position,)).fetchone()[0])
            value.update(changes)
            db.execute("UPDATE chunks SET data = ? WHERE position = ?", (json.dumps(value), position))

    def test_round_trip_retains_metadata_snapshots_and_vectors(self):
        self.save()
        self.assertEqual(load_index(self.path), (self.metadata, self.documents, self.chunks, self.vectors))

    def test_search_orders_by_cosine_and_keeps_source_metadata(self):
        self.save()
        _, _, chunks, vectors = load_index(self.path)
        found = search(chunks, vectors, [0.0, 1.0], top_k=1)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["chunk_id"], "second")
        self.assertEqual(found[0]["source"], "notes.txt")
        self.assertEqual(found[0]["score"], 1.0)
        tied = search(chunks, vectors, [2 ** -0.5, 2 ** -0.5], top_k=9)
        self.assertEqual([hit["chunk_id"] for hit in tied], ["first", "second"])

    def test_rejects_nan_infinite_zero_nonunit_and_wrong_dimension(self):
        for vector in [[float("nan"), 0], [float("inf"), 0], [0, 0], [2, 0], [1]]:
            with self.subTest(vector=vector), self.assertRaises(ValueError):
                validate_vector(vector, 2)
            with self.subTest(query=vector), self.assertRaises(ValueError):
                search(self.chunks, self.vectors, vector)

    def test_invalid_vectors_do_not_replace_existing_index(self):
        self.save()
        original = self.path.read_bytes()
        with self.assertRaises(ValueError):
            save_index(self.path, self.documents, self.chunks, [[1, 0], [float("nan"), 0]], self.metadata)
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(list(self.root.glob("*.tmp")), [])
        self.assertEqual(load_index(self.path)[2], self.chunks)

    def test_invalid_snapshot_does_not_replace_existing_index(self):
        self.save()
        original = self.path.read_bytes()
        bad_chunks = [replace(self.chunks[0], text="different"), self.chunks[1]]
        with self.assertRaises(ValueError):
            save_index(self.path, self.documents, bad_chunks, self.vectors, self.metadata)
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(list(self.root.glob("*.tmp")), [])

    def test_replace_failure_preserves_old_index_and_removes_temporary(self):
        self.save()
        original = self.path.read_bytes()
        with patch("llm_agent.indexing.store.os.replace", side_effect=PermissionError("locked")):
            with self.assertRaises(PermissionError):
                self.save()
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(list(self.root.glob("*.tmp")), [])

    def test_missing_index_is_never_created(self):
        with self.assertRaises(sqlite3.OperationalError):
            load_index(self.path)
        self.assertFalse(self.path.exists())

    def test_rejects_corrupt_offsets_and_text(self):
        for changes in [{"start": -1}, {"end": 100}, {"text": "wrong"}, {"chunk_id": "wrong"}]:
            with self.subTest(changes=changes):
                self.save()
                self.modify_chunk(0, **changes)
                with self.assertRaises(ValueError):
                    load_index(self.path)

    def test_rejects_uncovered_content_even_when_offsets_and_text_match(self):
        self.save()
        self.modify_chunk(1, start=7, text="eta")
        with self.assertRaisesRegex(ValueError, "покрывают"):
            load_index(self.path)

    def test_rejects_vectors_corrupted_after_saving(self):
        self.save()
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("UPDATE chunks SET vector = ? WHERE position = 1", ("[0, 0]",))
        with self.assertRaises(ValueError):
            load_index(self.path)

    def test_rejects_snapshot_changes_even_if_chunk_text_matches(self):
        self.save()
        with closing(sqlite3.connect(self.path)) as db, db:
            data = json.loads(db.execute("SELECT data FROM documents").fetchone()[0])
            data["text"] = "omega beta"
            db.execute("UPDATE documents SET data = ?", (json.dumps(data),))
        self.modify_chunk(0, text="omega ")
        with self.assertRaisesRegex(ValueError, "хеш снимка"):
            load_index(self.path)

    def test_rejects_empty_and_mismatched_save(self):
        for chunks, vectors in [([], []), (self.chunks, [[1, 0]])]:
            with self.subTest(chunks=chunks), self.assertRaises(ValueError):
                save_index(self.path, self.documents, chunks, vectors, self.metadata)
        self.assertFalse(self.path.exists())

    def test_rejects_unknown_schema(self):
        self.save()
        with closing(sqlite3.connect(self.path)) as db, db:
            metadata = dict(self.metadata, schema_version=999)
            db.execute("UPDATE metadata SET data = ?", (json.dumps(metadata),))
        with self.assertRaisesRegex(ValueError, "версия"):
            load_index(self.path)


if __name__ == "__main__":
    unittest.main()
