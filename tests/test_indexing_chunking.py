import hashlib
import json
import re
import tempfile
import unittest
from pathlib import Path

from llm_agent.indexing.chunking import Document, chunk_documents, corpus_hash, load_corpus


class FakeTokenizer:
    """One non-whitespace Unicode character per token, with real offsets."""

    def encode(self, text, add_special_tokens=False):
        return [ord(character) for character in text if not character.isspace()]

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        return {"offset_mapping": [(match.start(), match.end()) for match in re.finditer(r"\S", text)]}


def document(text, source="notes.md"):
    return Document(source, "Notes", text, hashlib.sha256(text.encode()).hexdigest())


class ChunkingTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = FakeTokenizer()

    def assert_valid_chunks(self, original, chunks, size):
        covered = set()
        for chunk in chunks:
            self.assertEqual(chunk.text, original[chunk.start:chunk.end])
            self.assertEqual(chunk.token_count, len(self.tokenizer.encode(chunk.text)))
            self.assertLessEqual(chunk.token_count, size)
            covered.update(range(chunk.start, chunk.end))
        self.assertEqual(covered, set(range(len(original))))
        self.assertEqual(len({chunk.chunk_id for chunk in chunks}), len(chunks))

    def test_fixed_unicode_offsets_overlap_and_determinism(self):
        original = "  Привет 🌍!\r\nЭто пример индексации.  "
        chunks = chunk_documents([document(original)], self.tokenizer, "fixed", 9, 3)
        self.assert_valid_chunks(original, chunks, 9)
        self.assertTrue(any(a.end > b.start for a, b in zip(chunks, chunks[1:])))
        self.assertEqual(chunks, chunk_documents([document(original)], self.tokenizer, "fixed", 9, 3))

    def test_zero_overlap_preserves_exact_source(self):
        original = "\n  abc def\n\nжзий клмн   "
        chunks = chunk_documents([document(original)], self.tokenizer, "fixed", 4, 0)
        self.assertEqual("".join(chunk.text for chunk in chunks), original)

    def test_structural_headings_ignore_fences_and_preserve_hierarchy(self):
        original = "Preface\n\n# Parent\n\nFirst.\n\n```python\n# not a heading\n```\n\n## Child ##\nText\n\n# Next\nEnd\n"
        chunks = chunk_documents([document(original)], self.tokenizer, "structural", 100, 4)
        self.assert_valid_chunks(original, chunks, 100)
        self.assertEqual([chunk.section for chunk in chunks], ["", "Parent", "Parent > Child", "Next"])
        self.assertTrue(all(len(chunk.sections) <= 1 for chunk in chunks))
        self.assertIn("# not a heading", chunks[1].text)

    def test_structural_paragraph_packing_and_fallback(self):
        original = "# Intro\n\nabc\n\ndef\n\n" + "я" * 70 + "\n\nlast\n"
        chunks = chunk_documents([document(original)], self.tokenizer, "structural", 12, 3)
        self.assert_valid_chunks(original, chunks, 12)
        self.assertIn("abc\n\ndef", chunks[0].text)
        self.assertTrue(all(chunk.section == "Intro" for chunk in chunks))

    def test_fixed_records_every_intersected_section(self):
        chunks = chunk_documents([document("# A\nx\n# B\ny")], self.tokenizer, "fixed", 30, 0)
        self.assertEqual(chunks[0].sections, ["A", "B"])

    def test_plain_text_does_not_parse_markdown(self):
        chunks = chunk_documents([document("# plain\nhello", "notes.txt")], self.tokenizer, "structural", 20, 0)
        self.assertEqual(chunks[0].sections, [])

    def test_structural_skips_whitespace_only_preface(self):
        chunks = chunk_documents([document(" \n\n# Title\nText")], self.tokenizer, "structural", 30, 0)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].section, "Title")
        self.assertGreater(chunks[0].token_count, 0)

    def test_fixed_tokenizes_full_document_once(self):
        class RecordingTokenizer(FakeTokenizer):
            def __init__(self):
                self.lengths = []

            def __call__(self, text, **kwargs):
                self.lengths.append(len(text))
                return super().__call__(text, **kwargs)

        tokenizer = RecordingTokenizer()
        original = "abcdefghij" * 100
        chunks = chunk_documents([document(original)], tokenizer, "fixed", 50, 5)
        self.assert_valid_chunks(original, chunks, 50)
        self.assertEqual(tokenizer.lengths[0], len(original))
        self.assertTrue(all(length <= 50 for length in tokenizer.lengths[1:]))

    def test_invalid_options(self):
        for strategy, size, overlap in [("unknown", 5, 0), ("fixed", 0, 0), ("fixed", 5, 5), ("fixed", 5, -1)]:
            with self.subTest(strategy=strategy, size=size, overlap=overlap), self.assertRaises(ValueError):
                chunk_documents([], self.tokenizer, strategy, size, overlap)

    def test_retokenization_enforces_actual_limit(self):
        class BoundaryTokenizer(FakeTokenizer):
            def encode(self, text, add_special_tokens=False):
                tokens = super().encode(text, add_special_tokens)
                return tokens + ([999] if text.startswith("b") else [])

        chunks = chunk_documents([document("abcdefghijk")], BoundaryTokenizer(), "fixed", 4, 3)
        self.assertTrue(all(chunk.token_count <= 4 for chunk in chunks))

    def test_duplicate_unicode_offsets_fail_clearly_when_one_character_exceeds_limit(self):
        class ByteTokenizer(FakeTokenizer):
            def encode(self, text, add_special_tokens=False):
                return list(text.encode("utf-8"))

            def __call__(self, text, **kwargs):
                return {"offset_mapping": [(index, index + 1) for index, char in enumerate(text) for _ in char.encode("utf-8")]}

        with self.assertRaisesRegex(ValueError, "single source character"):
            chunk_documents([document("🌍")], ByteTokenizer(), "fixed", 2, 0)


class CorpusTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.manifest = self.root / "manifest.json"

    def manifest_with(self, files):
        self.manifest.write_text(json.dumps({"files": files}), encoding="utf-8")

    def test_explicit_files_normalize_newlines_skip_empty_and_hash_content(self):
        raw = b"# Title\r\n\r\nText\r\n"
        (self.root / "readme.md").write_bytes(raw)
        (self.root / "empty.txt").write_text(" \n")
        (self.root / "unlisted.txt").write_text("private")
        self.manifest_with(["readme.md", "empty.txt"])
        documents = load_corpus(self.root, self.manifest)
        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].title, "Title")
        normalized = raw.decode().replace("\r\n", "\n")
        self.assertEqual(documents[0].text, normalized)
        self.assertEqual(documents[0].sha256, hashlib.sha256(normalized.encode()).hexdigest())

    def test_hash_is_stable_for_utf8_bom_and_line_endings(self):
        path = self.root / "a.txt"
        self.manifest_with(["a.txt"])
        path.write_bytes(b"\xef\xbb\xbfalpha\r\nbeta\r")
        first = load_corpus(self.root, self.manifest)
        path.write_bytes(b"alpha\nbeta\n")
        self.assertEqual(first, load_corpus(self.root, self.manifest))

    def test_rejects_traversal_absolute_duplicate_and_unsupported_files(self):
        (self.root / "a.txt").write_text("a")
        for files in [["../escape.txt"], [str(self.root / "a.txt")], ["a.txt", "./a.txt"], ["a.pdf"], [None]]:
            with self.subTest(files=files), self.assertRaises(ValueError):
                self.manifest_with(files)
                load_corpus(self.root, self.manifest)

    def test_invalid_manifest_and_missing_file(self):
        self.manifest.write_text("[]")
        with self.assertRaises(ValueError):
            load_corpus(self.root, self.manifest)
        self.manifest_with(["missing.txt"])
        with self.assertRaises(FileNotFoundError):
            load_corpus(self.root, self.manifest)

    def test_hash_is_order_independent_and_content_sensitive(self):
        a, b = document("a", "a.txt"), document("b", "b.txt")
        self.assertEqual(corpus_hash([a, b]), corpus_hash([b, a]))
        self.assertNotEqual(corpus_hash([a]), corpus_hash([document("changed", "a.txt")]))


if __name__ == "__main__":
    unittest.main()
