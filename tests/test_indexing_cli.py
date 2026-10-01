"""CLI errors and model input limits without downloading neural weights."""

import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from llm_agent.indexing.__main__ import main
from llm_agent.indexing.embeddings import Embedder


class IndexingCliTests(unittest.TestCase):
    def test_missing_index_is_error_and_does_not_create_database(self):
        with tempfile.TemporaryDirectory() as folder:
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(["verify", "--output-dir", folder]), 1)
            self.assertFalse(list(Path(folder).iterdir()))

    def test_invalid_build_parameters_do_not_load_model(self):
        with patch("llm_agent.indexing.__main__.Embedder") as model:
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(["build", "--size", "20", "--overlap", "20"]), 1)
            model.assert_not_called()

    def test_stats_does_not_load_model(self):
        with patch("llm_agent.indexing.__main__.Embedder") as model:
            with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stderr(io.StringIO()):
                main(["stats", "--output-dir", folder])
            model.assert_not_called()

    def test_embedding_rejects_full_input_over_limit_before_forward(self):
        class Tokenizer:
            def encode(self, text, add_special_tokens):
                return list(text) + [1, 2]

        # encode imports torch lazily; the input guard itself needs no torch methods.
        model = Embedder.__new__(Embedder)
        model.tokenizer = Tokenizer()
        model.limit = 10
        model.metadata = {"query_prefix": "query: ", "passage_prefix": "passage: "}
        with patch.dict("sys.modules", {"torch": object()}):
            with self.assertRaisesRegex(ValueError, "лимит"):
                model.encode(["abc"], query=True)
            with self.assertRaises(ValueError):
                model.encode(["abc"], batch_size=0)


if __name__ == "__main__":
    unittest.main()
