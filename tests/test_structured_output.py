"""Проверки выбора формата и границы безопасного повтора запросов."""

import unittest
from copy import deepcopy
from unittest.mock import Mock

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError, BadRequestError

from llm_agent.structured_output import StructuredOutput, StructuredOutputError
from tests.helpers import completion


def api_error(body, status=400):
    request = httpx.Request("POST", "https://example.com/v1/chat/completions")
    response = httpx.Response(status, request=request, json=body)
    error_class = BadRequestError if status == 400 else APIStatusError
    return error_class("Ошибка API", response=response, body=body)


def deepseek_unsupported():
    return api_error({
        "message": "This response_format type is unavailable now (request_id:private-request-id)",
        "type": "invalid_request_error", "param": None, "code": "invalid_request_error",
    })


def lm_studio_unsupported():
    return api_error({"error": "'response_format.type' must be 'json_schema' or 'text'"})


class StructuredOutputTests(unittest.TestCase):
    def setUp(self):
        self.request = {
            "model": "deepseek-flash",
            "messages": [{"role": "user", "content": "Ответь JSON."}],
            "temperature": 0.2,
            "response_format": {
                "type": "json_schema", "json_schema": {
                    "name": "answer", "strict": True, "schema": {
                        "type": "object", "additionalProperties": False,
                        "properties": {"answer": {"type": "string"}}, "required": ["answer"],
                    },
                },
            },
        }
        self.response = completion()
        self.create = Mock(return_value=self.response)

    def formats(self):
        return [call.kwargs.get("response_format") for call in self.create.call_args_list]

    def format_types(self):
        return [value["type"] if value else None for value in self.formats()]

    def test_auto_sends_full_schema_and_preserves_request(self):
        original = deepcopy(self.request)
        result = StructuredOutput().complete(self.create, **self.request)
        self.assertIs(result, self.response)
        self.create.assert_called_once_with(**original)
        self.assertEqual(self.request, original)

    def test_auto_falls_back_for_real_deepseek_error_and_caches_success(self):
        on_detected = Mock()
        output = StructuredOutput(model_id=self.request["model"], on_detected=on_detected)
        self.create.side_effect = [deepseek_unsupported(), self.response, self.response]

        self.assertIs(output.complete(self.create, **self.request), self.response)
        self.assertIs(output.complete(self.create, **self.request), self.response)

        self.assertEqual(self.format_types(), ["json_schema", "json_object", "json_object"])
        self.assertEqual(self.formats()[1], {"type": "json_object"})
        on_detected.assert_called_once_with("json_object")

    def test_auto_replaces_stale_cached_format_for_real_lm_studio_error(self):
        on_detected = Mock()
        output = StructuredOutput(
            model_id=self.request["model"], detected="json_object", on_detected=on_detected,
        )
        self.create.side_effect = [lm_studio_unsupported(), self.response, self.response]
        output.complete(self.create, **self.request)
        output.complete(self.create, **self.request)

        self.assertEqual(self.format_types(), ["json_object", "json_schema", "json_schema"])
        self.assertEqual(self.formats()[1], self.request["response_format"])
        on_detected.assert_called_once_with("json_schema")

    def test_cache_is_separate_for_each_model_id(self):
        output = StructuredOutput()
        self.create.side_effect = [deepseek_unsupported(), self.response, self.response, self.response]
        output.complete(self.create, **self.request)
        output.complete(self.create, **{**self.request, "model": "google/gemma-4-12b-qat"})
        output.complete(self.create, **self.request)
        self.assertEqual(self.format_types(), ["json_schema", "json_object", "json_schema", "json_object"])

    def test_success_callback_only_runs_once_for_changed_bound_model(self):
        on_detected = Mock()
        output = StructuredOutput(model_id=self.request["model"], on_detected=on_detected)
        other_request = {**self.request, "model": "other-model"}
        output.complete(self.create, **other_request)
        on_detected.assert_not_called()
        output.complete(self.create, **self.request)
        output.complete(self.create, **self.request)
        output.complete(self.create, **other_request)
        on_detected.assert_called_once_with("json_schema")

    def test_preloaded_format_does_not_notify_again(self):
        on_detected = Mock()
        output = StructuredOutput(
            model_id=self.request["model"], detected="json_object", on_detected=on_detected,
        )
        output.complete(self.create, **self.request)
        self.assertEqual(self.format_types(), ["json_object"])
        on_detected.assert_not_called()

    def test_both_formats_unsupported_raise_safe_error_without_plain_fallback(self):
        on_detected = Mock()
        output = StructuredOutput(model_id=self.request["model"], on_detected=on_detected)
        self.create.side_effect = [deepseek_unsupported(), lm_studio_unsupported(), self.response]
        with self.assertRaises(StructuredOutputError) as caught:
            output.complete(self.create, **self.request)
        message = str(caught.exception)
        self.assertIn("JSON Schema", message)
        self.assertIn("JSON Object", message)
        self.assertIn("менеджере моделей", message)
        self.assertNotIn("private-request-id", message)
        self.assertEqual(self.format_types(), ["json_schema", "json_object"])
        on_detected.assert_not_called()

        output.complete(self.create, **self.request)
        self.assertEqual(self.format_types(), ["json_schema", "json_object", "json_schema"])

    def test_manual_modes_send_selected_format_without_detection(self):
        for mode in ("json_schema", "json_object"):
            with self.subTest(mode=mode):
                on_detected = Mock()
                create = Mock(return_value=self.response)
                output = StructuredOutput(mode, model_id=self.request["model"], on_detected=on_detected)
                self.assertIs(output.complete(create, **self.request), self.response)
                self.assertEqual(create.call_args.kwargs["response_format"]["type"], mode)
                create.assert_called_once()
                on_detected.assert_not_called()

    def test_unsupported_manual_modes_raise_without_retry_or_response_details(self):
        for mode in ("json_schema", "json_object"):
            with self.subTest(mode=mode):
                create = Mock(side_effect=deepseek_unsupported())
                with self.assertRaises(StructuredOutputError) as caught:
                    StructuredOutput(mode).complete(create, **self.request)
                self.assertIn(mode, str(caught.exception))
                self.assertNotIn("private-request-id", str(caught.exception))
                create.assert_called_once()

    def test_schema_mode_supplies_schema_for_legacy_json_object_request(self):
        for mode in ("auto", "json_schema"):
            with self.subTest(mode=mode):
                create = Mock(return_value=self.response)
                StructuredOutput(mode).complete(
                    create, **{**self.request, "response_format": {"type": "json_object"}},
                )
                wire_format = create.call_args.kwargs["response_format"]
                self.assertEqual(wire_format["type"], "json_schema")
                self.assertEqual(wire_format["json_schema"]["schema"], {"type": "object"})
                self.assertTrue(wire_format["json_schema"]["name"])

    def test_off_removes_format_and_keeps_existing_detection(self):
        on_detected = Mock()
        output = StructuredOutput(
            "off", model_id=self.request["model"], detected="json_object", on_detected=on_detected,
        )
        self.assertIs(output.complete(self.create, **self.request), self.response)
        self.assertNotIn("response_format", self.create.call_args.kwargs)
        on_detected.assert_not_called()
        output.mode = "auto"
        output.complete(self.create, **self.request)
        self.assertEqual(self.format_types(), [None, "json_object"])
        on_detected.assert_not_called()

    def test_plain_requests_remove_format_without_changing_detection(self):
        for response_format in (None, {"type": "text"}, {}):
            with self.subTest(response_format=response_format):
                on_detected = Mock()
                create = Mock(return_value=self.response)
                output = StructuredOutput(
                    model_id=self.request["model"], detected="json_object", on_detected=on_detected,
                )
                request = {key: value for key, value in self.request.items() if key != "response_format"}
                if response_format is not None:
                    request["response_format"] = response_format
                output.complete(create, **request)
                self.assertNotIn("response_format", create.call_args.kwargs)
                on_detected.assert_not_called()
                output.complete(create, **self.request)
                self.assertEqual(create.call_args.kwargs["response_format"], {"type": "json_object"})
                on_detected.assert_not_called()

    def test_unrelated_api_errors_are_not_retried_or_cached(self):
        cases = [
            (400, {"error": {"message": "Invalid JSON schema for response_format.type"}}),
            (400, {"error": {"message": "response_format.type schema validation is not supported"}}),
            (400, {"error": {"message": "response_format.type is unavailable: maximum context exceeded"}}),
            (400, {"error": {"message": "response_format.type is not supported with tools"}}),
            (400, {"error": {"message": "response_format.type is not supported with tool_choice"}}),
            (400, {"error": {"message": "Unsupported keyword oneOf", "param": "response_format.json_schema.schema"}}),
            (400, {"error": {"message": "Invalid request"}}),
            (400, {"error": None}),
            (400, "Bad request"),
            (401, {"message": "response_format.type is unsupported"}),
            (429, {"message": "response_format.type is unsupported"}),
            (500, {"message": "response_format.type is unsupported"}),
        ]
        for status, body in cases:
            with self.subTest(status=status, body=body):
                on_detected = Mock()
                output = StructuredOutput(model_id=self.request["model"], on_detected=on_detected)
                error = api_error(body, status)
                create = Mock(side_effect=error)
                with self.assertRaises(APIStatusError) as caught:
                    output.complete(create, **self.request)
                self.assertIs(caught.exception, error)
                create.assert_called_once()
                on_detected.assert_not_called()

                create.side_effect = None
                create.return_value = self.response
                output.complete(create, **self.request)
                self.assertEqual(create.call_args.kwargs["response_format"]["type"], "json_schema")

    def test_network_errors_are_not_retried(self):
        request = httpx.Request("POST", "https://example.com/v1/chat/completions")
        for error in (APITimeoutError(request=request), APIConnectionError(request=request)):
            with self.subTest(error=type(error).__name__):
                create = Mock(side_effect=error)
                with self.assertRaises(type(error)) as caught:
                    StructuredOutput().complete(create, **self.request)
                self.assertIs(caught.exception, error)
                create.assert_called_once()

    def test_explicit_type_parameter_and_422_are_supported(self):
        self.create.side_effect = [api_error({
            "error": {"message": "Unsupported value", "param": "response_format.type"},
        }, 422), self.response]
        StructuredOutput().complete(self.create, **self.request)
        self.assertEqual(self.format_types(), ["json_schema", "json_object"])

    def test_invalid_configuration_fails_before_request(self):
        with self.assertRaises(ValueError):
            StructuredOutput("unknown")
        with self.assertRaises(ValueError):
            StructuredOutput(detected="off")


if __name__ == "__main__":
    unittest.main()
