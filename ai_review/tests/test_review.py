"""Проверки контракта ответа и взаимодействия с API без сетевых запросов."""

from copy import deepcopy
import json
import unittest
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError, URLError

from ai_review import ReviewError
from ai_review import review


def make_context():
    return {
        "base_sha": "a" * 40,
        "merge_base_sha": "a" * 40,
        "head_sha": "b" * 40,
        "diff": "--- a/src/service.py\n+++ b/src/service.py\n@@ -1,2 +1,2 @@\n def run():\n-    return 1\n+    return None\n",
        "files": [{
            "path": "src/service.py",
            "old_path": "src/service.py",
            "status": "M",
            "before": "def run():\n    return 1\n",
            "after": "def run():\n    return None\n",
        }],
        "sources": [{
            "id": "R1", "path": "docs/contracts.md", "start_line": 1,
            "text": "run всегда возвращает число.", "kind": "documentation",
        }],
        "skipped": [],
        "retrieval": {},
    }


def make_payload():
    return {
        "summary": "Изменение нарушает контракт результата.",
        "findings": [{
            "category": "bug", "severity": "high", "path": "src/service.py",
            "side": "after", "line": 2,
            "description": "Вызывающий код ожидает число и не обрабатывает None.",
            "suggestion": "Сохранить числовое значение результата.", "sources": ["R1"],
        }],
    }


def api_response(payload=None, *, finish_reason="stop", content=None):
    if content is None:
        content = json.dumps(make_payload() if payload is None else payload, ensure_ascii=False)
    envelope = {
        "choices": [{"finish_reason": finish_reason, "message": {"content": content}}],
        "usage": {"prompt_tokens": 123, "completion_tokens": 45},
    }
    response = MagicMock()
    response.__enter__.return_value = response
    response.read.return_value = json.dumps(envelope).encode("utf-8")
    return response


class ValidateReviewTests(unittest.TestCase):
    def test_valid_finding_keeps_location_and_evidence(self):
        result = review.validate_review(make_payload(), make_context())
        self.assertEqual(result["findings"][0]["line"], 2)
        self.assertEqual(result["findings"][0]["sources"], ["R1"])

    def test_rejects_unverifiable_or_invalid_finding(self):
        changes = [
            {"path": "unrelated.py"}, {"sources": ["R99"]},
            {"line": 0}, {"line": 3}, {"line": True}, {"line": "2"},
            {"side": "left"}, {"category": "style"}, {"severity": "critical"},
            {"description": "x" * 2001}, {"suggestion": "x" * 2001},
            {"sources": "R1"}, {"extra": "unexpected"},
        ]
        for change in changes:
            with self.subTest(change=change):
                payload = make_payload()
                payload["findings"][0].update(change)
                with self.assertRaises(ReviewError):
                    review.validate_review(payload, make_context())

    def test_deleted_file_requires_before_side(self):
        context = make_context()
        context["files"][0].update(status="D", after=None)
        payload = make_payload()
        payload["findings"][0]["side"] = "before"
        review.validate_review(payload, context)
        payload["findings"][0]["side"] = "after"
        with self.assertRaises(ReviewError):
            review.validate_review(payload, context)

    def test_missing_before_text_cannot_be_cited_for_new_file(self):
        context = make_context()
        context["files"][0].update(status="A", before=None, old_path=None)
        payload = make_payload()
        payload["findings"][0]["side"] = "before"
        with self.assertRaises(ReviewError):
            review.validate_review(payload, context)

    def test_all_required_finding_fields_are_checked(self):
        for key in make_payload()["findings"][0]:
            with self.subTest(missing=key):
                payload = make_payload()
                del payload["findings"][0][key]
                with self.assertRaises(ReviewError):
                    review.validate_review(payload, make_context())

    def test_payload_limits_and_shape(self):
        invalid = [
            [], {}, {"summary": "ok", "findings": "none"},
            {"summary": "x" * 2001, "findings": []},
            {**make_payload(), "extra": True},
            {"summary": "ok", "findings": [deepcopy(make_payload()["findings"][0]) for _ in range(16)]},
        ]
        for payload in invalid:
            with self.subTest(payload=payload):
                with self.assertRaises(ReviewError):
                    review.validate_review(payload, make_context())

    def test_no_findings_is_valid(self):
        result = review.validate_review({"summary": "Обоснованных замечаний нет.", "findings": []}, make_context())
        self.assertEqual(result["findings"], [])

    def test_line_numbers_follow_git_newlines(self):
        context = make_context()
        context["files"][0]["after"] = "value = 'a\vb'\n"
        with self.assertRaises(ReviewError):
            review.validate_review(make_payload(), context)


class GenerateReviewTests(unittest.TestCase):
    @patch("ai_review.review.urlopen")
    def test_sends_json_context_and_returns_validated_result(self, urlopen):
        urlopen.return_value = api_response()
        result = review.generate_review(make_context(), "test-secret", "chosen-model")
        request = urlopen.call_args.args[0]
        body = json.loads(request.data)
        self.assertEqual(body["model"], "chosen-model")
        self.assertEqual(body["response_format"], {"type": "json_object"})
        self.assertEqual(urlopen.call_args.kwargs["timeout"], 120)
        self.assertEqual(request.get_header("Authorization"), "Bearer test-secret")
        self.assertIn("src/service.py", json.dumps(body, ensure_ascii=False))
        self.assertIn("R1", json.dumps(body, ensure_ascii=False))
        self.assertEqual(result["model"], "chosen-model")
        self.assertEqual(result["usage"]["prompt_tokens"], 123)
        self.assertEqual(result["findings"], make_payload()["findings"])

    @patch("ai_review.review.time.sleep")
    @patch("ai_review.review.urlopen")
    def test_retries_transient_http_and_network_errors(self, urlopen, sleep):
        for failure in [HTTPError("https://api.deepseek.com", 429, "limited", {}, None),
                        HTTPError("https://api.deepseek.com", 503, "busy", {}, None),
                        URLError("temporary network failure")]:
            with self.subTest(error=type(failure).__name__, code=getattr(failure, "code", None)):
                urlopen.reset_mock()
                urlopen.side_effect = [failure, api_response()]
                result = review.generate_review(make_context(), "test-secret")
                self.assertEqual(urlopen.call_count, 2)
                self.assertEqual(result["findings"], make_payload()["findings"])

    @patch("ai_review.review.time.sleep")
    @patch("ai_review.review.urlopen")
    def test_exhausts_three_attempts_without_exposing_transport_details(self, urlopen, sleep):
        urlopen.side_effect = URLError("private detail test-secret")
        with self.assertRaises(ReviewError) as raised:
            review.generate_review(make_context(), "test-secret")
        self.assertEqual(urlopen.call_count, 3)
        self.assertNotIn("test-secret", str(raised.exception))
        self.assertNotIn("private detail", str(raised.exception))

    @patch("ai_review.review.time.sleep")
    @patch("ai_review.review.urlopen")
    def test_authentication_error_is_not_retried_or_echoed(self, urlopen, sleep):
        urlopen.side_effect = HTTPError("https://api.deepseek.com?key=test-secret", 401, "test-secret", {}, None)
        with self.assertRaises(ReviewError) as raised:
            review.generate_review(make_context(), "test-secret")
        urlopen.assert_called_once()
        sleep.assert_not_called()
        self.assertNotIn("test-secret", str(raised.exception))

    @patch("ai_review.review.urlopen")
    def test_rejects_truncated_or_malformed_model_output(self, urlopen):
        for response in [api_response(finish_reason="length"), api_response(content="not json"),
                         api_response(payload={"summary": "ok", "findings": [{"path": "invented.py"}]})]:
            with self.subTest(response=response):
                urlopen.return_value = response
                with self.assertRaises(ReviewError):
                    review.generate_review(make_context(), "test-secret")


class RenderMarkdownTests(unittest.TestCase):
    def test_includes_findings_locations_and_reviewed_revision(self):
        context = make_context()
        result = {**make_payload(), "model": "test-model", "usage": {}}
        markdown = review.render_markdown(result, context)
        for value in ["src/service.py", "R1", context["head_sha"][:7], result["summary"],
                      result["findings"][0]["description"], result["findings"][0]["suggestion"]]:
            self.assertIn(value, markdown)


if __name__ == "__main__":
    unittest.main()
