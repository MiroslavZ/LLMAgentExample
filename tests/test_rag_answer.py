import json
import unittest
from copy import deepcopy

from llm_agent.rag_answer import (
    parse_answer, render_answer, unknown_answer, validate_saved_answer,
)


class RAGAnswerTests(unittest.TestCase):
    def setUp(self):
        self.context = {"chunks": [
            {"chunk_id": "one", "source": "guide.md", "section": "Запуск",
             "text": "Первый шаг.\nЗапустите  python main.py. Следующий шаг."},
            {"chunk_id": "two", "source": "config.md", "section": "Настройки",
             "text": "Модель задаётся в настройках."},
        ]}
        self.answer = {
            "status": "answered", "answer": "Запустите python main.py.",
            "sources": ["one"], "quotes": [{"chunk_id": "one", "text": "Запустите python main.py."}],
            "clarification": None,
        }

    def parse(self, answer=None):
        return parse_answer(json.dumps(self.answer if answer is None else answer), self.context)

    def test_enriches_sources_without_mutating_input(self):
        original = deepcopy(self.answer)
        result = self.parse()
        self.assertEqual(result["sources"], [{"chunk_id": "one", "source": "guide.md", "section": "Запуск"}])
        self.assertEqual(self.answer, original)
        validate_saved_answer(result, self.context)

    def test_allows_only_whitespace_normalization(self):
        self.answer["quotes"][0]["text"] = "  Запустите\n\tpython\u00a0main.py. "
        self.parse()
        for text in ("запустите python main.py.", "Запустите python main.py!", "Запустите … main.py."):
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.answer["quotes"][0]["text"] = text
                self.parse()

    def test_rejects_quote_from_another_chunk(self):
        self.answer["quotes"][0]["text"] = "Модель задаётся в настройках."
        with self.assertRaises(ValueError):
            self.parse()

    def test_requires_exact_fields_and_types(self):
        patches = [
            {"extra": True}, {"answer": " "}, {"answer": 42}, {"status": []},
            {"status": "ok"}, {"sources": "one"}, {"sources": [None]},
            {"sources": [["one"]]}, {"sources": []}, {"quotes": []},
            {"quotes": "quote"}, {"clarification": "Уточните"},
            {"sources": [{"chunk_id": "one", "source": "invented"}]},
        ]
        for patch in patches:
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                self.parse(dict(self.answer, **patch))
        for field in self.answer:
            answer = dict(self.answer)
            del answer[field]
            with self.subTest(missing=field), self.assertRaises(ValueError):
                self.parse(answer)

    def test_rejects_invalid_quotes(self):
        for quote in (None, {}, {"chunk_id": "one", "text": " "},
                      {"chunk_id": [], "text": "text"},
                      {"chunk_id": "missing", "text": "text"},
                      {"chunk_id": "one", "text": "Первый шаг.", "extra": 1}):
            with self.subTest(quote=quote), self.assertRaises(ValueError):
                self.parse(dict(self.answer, quotes=[quote]))

    def test_requires_matching_unique_sources(self):
        for sources in (["one", "one"], ["missing"], ["two"], ["one", "two"]):
            with self.subTest(sources=sources), self.assertRaises(ValueError):
                self.parse(dict(self.answer, sources=sources))
        self.answer["quotes"].append({"chunk_id": "two", "text": "Модель задаётся в настройках."})
        with self.assertRaises(ValueError):
            self.parse()
        self.answer["sources"].append("two")
        self.assertEqual(len(self.parse()["sources"]), 2)

    def test_rejects_invalid_json_and_duplicate_fields(self):
        for content in (None, "[]", "null", "{}", "```json\n{}\n```", "text",
                        '{"status":"answered","status":"unknown"}'):
            with self.subTest(content=content), self.assertRaises(ValueError):
                parse_answer(content, self.context)

    def test_unknown_requires_clarification_and_no_evidence(self):
        result = unknown_answer()
        self.assertIn("Не знаю", result["answer"])
        self.assertEqual(parse_answer(json.dumps(result), {"chunks": []}), result)
        validate_saved_answer(result, {"chunks": []})
        for patch in ({"clarification": None}, {"clarification": " "},
                      {"sources": ["one"]}, {"quotes": self.answer["quotes"]}):
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                self.parse(dict(result, **patch))

    def test_saved_sources_and_quotes_are_revalidated(self):
        for field in ("source", "section", "chunk_id"):
            result = self.parse()
            result["sources"][0][field] = "forged"
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_saved_answer(result, self.context)
        result = self.parse()
        result["quotes"][0]["text"] = "Вымышленная цитата"
        with self.assertRaises(ValueError):
            validate_saved_answer(result, self.context)
        with self.assertRaises(ValueError):
            validate_saved_answer(self.answer, self.context)

    def test_render_text_and_json(self):
        result = self.parse()
        text = render_answer(result)
        for expected in (result["answer"], "Источники:", "Цитаты:", "guide.md", "Запуск", "chunk_id: one"):
            self.assertIn(expected, text)
        for response_format in ("object", "schema"):
            self.assertEqual(json.loads(render_answer(result, response_format)), result)
        result = unknown_answer()
        self.assertIn(result["clarification"], render_answer(result))
        with self.assertRaises(ValueError):
            render_answer(result, "invalid")

    def test_rejects_missing_or_duplicate_context(self):
        for context in ({}, {"chunks": None}, {"chunks": [self.context["chunks"][0]] * 2}):
            with self.subTest(context=context), self.assertRaises(ValueError):
                parse_answer(json.dumps(self.answer), context)


if __name__ == "__main__":
    unittest.main()
