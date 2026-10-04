"""Контракт адаптера моделей без загрузки весов из сети."""

import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from llm_agent.indexing.embeddings import E5, RUBERT, Embedder, embedding_options


class EmbeddingModelTests(unittest.TestCase):
    def setUp(self):
        # Resolve lazy imports before patching hub helpers: other modules must
        # not retain a mocked cached_file after this test finishes.
        import sentence_transformers  # noqa: F401

        self.cached = self.enterContext(patch("transformers.utils.hub.cached_file"))
        self.cached.return_value = "/cache/snapshots/pinned-revision/config.json"
        self.factory = self.enterContext(patch("sentence_transformers.SentenceTransformer"))
        self.model = self.factory.return_value.eval.return_value
        self.model.tokenizer = Mock(is_fast=True)
        self.model.tokenizer.encode.side_effect = lambda text, add_special_tokens: list(text) + [0, 1]
        self.model.max_seq_length = 512
        self.model.get_embedding_dimension.return_value = 2
        self.model.prompts = {}
        self.model.default_prompt_name = None
        self.model.encode_query.return_value.tolist.return_value = [[0.6, 0.8]]
        self.model.encode_document.return_value.tolist.return_value = [[1.0, 0.0]]

    def test_arbitrary_exported_model_uses_its_own_pooling(self):
        embedder = Embedder("example/retrieval-model", cache="cache", offline=True)
        self.assertEqual(embedder.encode(["text"]), [[1.0, 0.0]])
        self.model.encode_document.assert_called_once_with(
            ["text"], prompt="", batch_size=16, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        )
        self.factory.assert_called_once_with(
            "example/retrieval-model", revision="pinned-revision", cache_folder="cache",
            local_files_only=True, device="cpu", trust_remote_code=False,
        )
        self.assertEqual(embedder.metadata["pooling"], "model_defined")
        self.assertIn("sentence-transformers", embedder.metadata["versions"])

    def test_e5_base_and_large_have_distinct_query_and_passage_prefixes(self):
        for size in ("base", "large"):
            with self.subTest(size=size):
                embedder = Embedder(f"intfloat/multilingual-e5-{size}")
                embedder.encode(["question"], query=True)
                self.assertEqual(self.model.encode_query.call_args.kwargs["prompt"], "query: ")
                embedder.encode(["document"])
                self.assertEqual(self.model.encode_document.call_args.kwargs["prompt"], "passage: ")

    def test_model_prompts_and_limit_are_used_and_persisted(self):
        self.model.prompts = {"query": "Instruct: find evidence\nQuery: ", "passage": "doc: "}
        self.model.max_seq_length = 8192
        embedder = Embedder("example/instruct-embedder")
        self.assertEqual(embedder.limit, 8192)
        self.assertEqual(embedder.metadata["query_prefix"], self.model.prompts["query"])
        self.assertEqual(embedder.metadata["passage_prefix"], "doc: ")
        restored = Embedder("example/instruct-embedder", **embedding_options(embedder.metadata))
        self.assertEqual(restored.metadata, embedder.metadata)

    def test_explicit_prefixes_override_model_defaults_including_empty_string(self):
        self.model.prompts = {"query": "default-query", "document": "default-document"}
        embedder = Embedder("example/custom", query_prefix="custom: ", passage_prefix="")
        embedder.encode(["q"], query=True)
        self.model.encode_query.assert_called_once()
        self.assertEqual(self.model.encode_query.call_args.args, (["q"],))
        self.assertEqual(self.model.encode_query.call_args.kwargs["prompt"], "custom: ")
        self.assertEqual(embedder.metadata["passage_prefix"], "")

    def test_length_guard_includes_prompt_and_prevents_silent_truncation(self):
        self.model.max_seq_length = 10
        embedder = Embedder("example/custom", query_prefix="long prompt: ")
        with self.assertRaisesRegex(ValueError, "лимит"):
            embedder.encode(["q"], query=True)
        self.model.encode_query.assert_not_called()

    def test_rejects_model_without_embedding_config_before_fallback(self):
        self.cached.side_effect = ["/cache/snapshots/rev/config.json", None]
        with self.assertRaisesRegex(ValueError, "modules.json"):
            Embedder("example/base-language-model")
        self.factory.assert_not_called()

    def test_empty_batch_does_not_call_model(self):
        embedder = Embedder("example/custom")
        self.assertEqual(embedder.encode([]), [])
        self.model.encode_document.assert_not_called()

    def test_legacy_models_keep_original_metadata_and_loading_path(self):
        for name in (E5, RUBERT):
            with self.subTest(model=name), patch("transformers.AutoModel.from_pretrained") as loader, \
                    patch("transformers.AutoTokenizer.from_pretrained"):
                loader.return_value.eval.return_value.config = SimpleNamespace(
                    max_position_embeddings=512, hidden_size=768,
                )
                embedder = Embedder(name)
                self.assertNotIn("backend", embedder.metadata)
                self.assertEqual(embedding_options(embedder.metadata), {})
                self.assertEqual(embedder.metadata["pooling"], "attention_mask_mean")
                self.assertEqual(embedder.metadata["query_prefix"], "query: " if name == E5 else "")
        self.factory.assert_not_called()

    def test_rejects_unknown_backend_or_invalid_prefix(self):
        for options in ({"backend": "typo"}, {"query_prefix": 1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                Embedder("example/custom", **options)
        self.factory.assert_not_called()


class LocalSentenceTransformerTests(unittest.TestCase):
    def test_real_export_build_and_reload_without_network(self):
        import torch
        from transformers import BertConfig, BertModel, BertTokenizerFast
        from sentence_transformers import SentenceTransformer
        from sentence_transformers.sentence_transformer.modules import Transformer, Pooling
        from llm_agent.indexing.chunking import load_corpus
        from llm_agent.indexing.pipeline import build
        from llm_agent.indexing.store import load_index
        from llm_agent.rag import Retriever

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base"
            base.mkdir()
            fast = BertTokenizerFast(vocab={
                "[UNK]": 0, "[PAD]": 1, "alpha": 2, "beta": 3,
                "[CLS]": 4, "[SEP]": 5, "[MASK]": 6,
            })
            fast.save_pretrained(base)
            with torch.random.fork_rng():
                torch.manual_seed(1)
                BertModel(BertConfig(vocab_size=7, hidden_size=8, num_hidden_layers=1,
                                     num_attention_heads=2, intermediate_size=16,
                                     max_position_embeddings=32)).save_pretrained(base)
            # CLS deliberately differs from the old mean pooling implementation.
            model = SentenceTransformer(modules=[
                Transformer(str(base), max_seq_length=32, model_kwargs={"local_files_only": True}),
                Pooling(8, pooling_mode="cls"),
            ], device="cpu", prompts={"query": "alpha "})
            exported = root / "exported"
            model.save_pretrained(str(exported))
            embedder = Embedder(str(exported), offline=True)
            expected = model.encode_query(["beta"], normalize_embeddings=True)
            actual = embedder.encode(["beta"], query=True)
            torch.testing.assert_close(torch.tensor(actual), torch.tensor(expected))
            (root / "notes.txt").write_text("alpha beta", encoding="utf-8")
            (root / "corpus.json").write_text('{"files":["notes.txt"]}', encoding="utf-8")
            path = root / "index.sqlite3"
            build(path, load_corpus(root, root / "corpus.json"), embedder, "fixed", size=8, overlap=0)
            stored = load_index(path)[0]["embedding"]
            self.assertEqual(stored, embedder.metadata)
            context = Retriever(path).retrieve("beta")
            self.assertEqual(context["embedding"], stored)
            self.assertEqual(context["chunks"][0]["text"], "alpha beta")


if __name__ == "__main__":
    unittest.main()
