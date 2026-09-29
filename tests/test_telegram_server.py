"""Telegram MCP: контракт, последовательность и честный результат доставки без сети."""

import asyncio
import json
import logging
import os
import unittest
from unittest.mock import AsyncMock, patch

import httpx2 as httpx

from TelegramMCPServerExample import telegram_api
from TelegramMCPServerExample.server import mcp, send_message


class SplitTextTests(unittest.TestCase):
    def test_splits_unicode_without_loss_and_keeps_utf16_limit(self):
        for text in ("я" * 10_000, "🐍" * 4097, ("PyTorch 🐍\nVS Code vulnerability " * 400)):
            with self.subTest(length=len(text)):
                parts = telegram_api.split_text(text)
                self.assertEqual("".join(parts), text)
                self.assertTrue(all(0 < len(part.encode("utf-16-le")) // 2 <= 4096 for part in parts))

    def test_prefers_line_boundary_then_word_boundary(self):
        text = "a" * 3000 + "\n" + "b" * 2000
        self.assertEqual(telegram_api.split_text(text)[0], "a" * 3000 + "\n")
        text = "a" * 3000 + " " + "b" * 2000
        self.assertEqual(telegram_api.split_text(text)[0], "a" * 3000 + " ")

    def test_exact_utf16_boundary(self):
        self.assertEqual(telegram_api.split_text("🐍" * 2048), ["🐍" * 2048])
        self.assertEqual(telegram_api.split_text("🐍" * 2049), ["🐍" * 2048, "🐍"])


class TelegramSettingsTests(unittest.TestCase):
    def test_explicit_environment_overrides_server_dotenv_without_mutating_environment(self):
        with patch.dict(os.environ, {"API_TOKEN": "123:environment"}, clear=True), \
                patch.object(telegram_api, "dotenv_values", return_value={
                    "API_TOKEN": "123:dotenv", "CHAT_ID": "12345",
                }):
            settings = telegram_api.Settings.from_env()
            self.assertEqual(settings.bot_token, "123:environment")
            self.assertEqual(settings.chat_id, "12345")
            self.assertNotIn("CHAT_ID", os.environ)
            self.assertNotIn("environment", repr(settings))

    def test_http_request_logging_redacts_token(self):
        record = logging.LogRecord("httpx2", logging.INFO, __file__, 1,
                                   "HTTP Request: %s %s", (
                                       "POST", "https://api.telegram.org/bot123:private-token/sendMessage",
                                   ), None)
        telegram_api._RedactBotToken().filter(record)
        self.assertNotIn("private-token", record.getMessage())
        self.assertIn("/bot<redacted>/sendMessage", record.getMessage())


class TelegramSendTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.settings = patch.object(telegram_api.Settings, "from_env", return_value=telegram_api.Settings(
            bot_token="12345:unit-test-token", chat_id="67890",
        ))
        self.settings.start()
        self.addCleanup(self.settings.stop)
        self.sleep = patch.object(telegram_api.asyncio, "sleep", new_callable=AsyncMock)
        self.sleep_mock = self.sleep.start()
        self.addCleanup(self.sleep.stop)
        self.requests = []

    def client(self, handler):
        def record(request):
            self.requests.append(request)
            return handler(request)
        return patch.object(telegram_api, "create_client", side_effect=lambda: httpx.AsyncClient(
            transport=httpx.MockTransport(record), trust_env=False,
        ))

    async def test_success_sends_plain_text_only_to_configured_chat(self):
        text = "Отчёт <CVE> & **PyTorch**\nhttps://nvd.nist.gov/vuln/detail/CVE-2025-0001"
        with self.client(lambda request: httpx.Response(200, json={"ok": True, "result": {"message_id": 42}})):
            result = await send_message(text)
        self.assertFalse(result.is_error)
        self.assertEqual(result.structured_content, {
            "status": "sent", "message_ids": [42], "total_parts": 1, "sent_parts": 1,
        })
        self.assertEqual(json.loads(self.requests[0].content), {"chat_id": "67890", "text": text})
        self.assertEqual(str(self.requests[0].url), "https://api.telegram.org/bot12345:unit-test-token/sendMessage")
        self.sleep_mock.assert_not_awaited()

    async def test_long_text_is_sent_in_order_without_loss(self):
        text = "Доклад 🐍\n" * 1800
        with self.client(lambda request: httpx.Response(200, json={
            "ok": True, "result": {"message_id": len(self.requests)},
        })):
            result = await send_message(text)
        self.assertFalse(result.is_error)
        self.assertEqual("".join(json.loads(request.content)["text"] for request in self.requests), text)
        self.assertEqual(result.structured_content["message_ids"], list(range(1, len(self.requests) + 1)))
        self.assertEqual(self.sleep_mock.await_count, len(self.requests) - 1)

    async def test_partial_api_error_preserves_confirmed_messages_and_stops(self):
        def handler(request):
            if len(self.requests) == 1:
                return httpx.Response(200, json={"ok": True, "result": {"message_id": 101}})
            return httpx.Response(429, json={
                "ok": False, "error_code": 429, "description": "untrusted unit-test-token",
                "parameters": {"retry_after": 3},
            })
        with self.client(handler):
            result = await send_message("a" * 9000)
        self.assertTrue(result.is_error)
        payload = result.structured_content
        self.assertEqual(payload["status"], "partial")
        self.assertEqual(payload["message_ids"], [101])
        self.assertEqual((payload["sent_parts"], payload["total_parts"], payload["failed_part"]), (1, 3, 2))
        self.assertEqual(payload["retry_after"], 3)
        self.assertFalse(payload["delivery_uncertain"])
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(json.loads(result.content[0].text), payload)
        self.assertNotIn("unit-test-token", result.model_dump_json())

    async def test_timeout_is_unknown_and_is_never_retried(self):
        def handler(request):
            raise httpx.ReadTimeout(f"token in URL {request.url}", request=request)
        with self.client(handler):
            result = await send_message("Report")
        self.assertTrue(result.is_error)
        self.assertEqual(result.structured_content["status"], "unknown")
        self.assertTrue(result.structured_content["delivery_uncertain"])
        self.assertEqual(result.structured_content["message_ids"], [])
        self.assertEqual(len(self.requests), 1)
        self.assertNotIn("unit-test-token", result.model_dump_json())

    async def test_partial_timeout_is_not_reported_as_complete(self):
        def handler(request):
            if len(self.requests) == 1:
                return httpx.Response(200, json={"ok": True, "result": {"message_id": 7}})
            raise httpx.ConnectError("network failure", request=request)
        with self.client(handler):
            result = await send_message("b" * 5000)
        self.assertTrue(result.is_error)
        self.assertEqual(result.structured_content["status"], "partial")
        self.assertEqual(result.structured_content["message_ids"], [7])
        self.assertTrue(result.structured_content["delivery_uncertain"])
        self.assertEqual(len(self.requests), 2)

    async def test_total_deadline_preserves_ids_when_request_is_interrupted(self):
        async def handler(request):
            if len(self.requests) == 1:
                return httpx.Response(200, json={"ok": True, "result": {"message_id": 7}})
            await asyncio.Event().wait()
        with self.client(handler), patch.object(telegram_api, "SEND_TIMEOUT_SECONDS", 0.05):
            result = await send_message("b" * 9000)
        self.assertTrue(result.is_error)
        self.assertEqual(result.structured_content["message_ids"], [7])
        self.assertEqual(result.structured_content["failed_part"], 2)
        self.assertTrue(result.structured_content["delivery_uncertain"])
        self.assertEqual(len(self.requests), 2)

    async def test_total_deadline_during_pause_does_not_claim_unknown_delivery(self):
        async def long_pause(seconds):
            await asyncio.Event().wait()
        self.sleep_mock.side_effect = long_pause
        with self.client(lambda request: httpx.Response(200, json={"ok": True, "result": {"message_id": 9}})), \
                patch.object(telegram_api, "SEND_TIMEOUT_SECONDS", 0.05):
            result = await send_message("b" * 9000)
        self.assertTrue(result.is_error)
        self.assertEqual(result.structured_content["message_ids"], [9])
        self.assertEqual(result.structured_content["failed_part"], 2)
        self.assertFalse(result.structured_content["delivery_uncertain"])
        self.assertEqual(len(self.requests), 1)

    async def test_api_rejection_with_http_200_is_error(self):
        with self.client(lambda request: httpx.Response(200, json={"ok": False, "error_code": 403})):
            result = await send_message("Report")
        self.assertTrue(result.is_error)
        self.assertEqual(result.structured_content["status"], "failed")
        self.assertFalse(result.structured_content["delivery_uncertain"])

    async def test_invalid_responses_never_confirm_delivery(self):
        fixtures = [
            httpx.Response(502, text="Bad Gateway unit-test-token"),
            httpx.Response(200, text="not JSON"),
            httpx.Response(200, json=[]),
            httpx.Response(200, json={"ok": True, "result": {}}),
            httpx.Response(200, json={"ok": True, "result": {"message_id": True}}),
        ]
        for response in fixtures:
            with self.subTest(response=response), self.client(lambda request: response):
                result = await send_message("Report")
            self.assertTrue(result.is_error)
            self.assertEqual(result.structured_content["message_ids"], [])
            self.assertTrue(result.structured_content["delivery_uncertain"])
            self.assertNotIn("unit-test-token", result.model_dump_json())

    async def test_invalid_text_is_rejected_before_network(self):
        for text in ("", " \n\t", "a" * (telegram_api.MAX_TEXT_LENGTH + 1), "invalid\ud800"):
            with self.subTest(length=len(text)), patch.object(telegram_api, "create_client") as factory:
                result = await send_message(text)
            self.assertTrue(result.is_error)
            self.assertEqual(result.structured_content["sent_parts"], 0)
            factory.assert_not_called()

    async def test_missing_configuration_is_rejected_before_network(self):
        with patch.object(telegram_api.Settings, "from_env", return_value=telegram_api.Settings("", "")), \
                patch.object(telegram_api, "create_client") as factory:
            result = await send_message("Report")
        self.assertTrue(result.is_error)
        self.assertEqual(result.structured_content["status"], "failed")
        factory.assert_not_called()

    async def test_mcp_tool_declares_side_effect_and_returns_protocol_error(self):
        tools = await mcp.list_tools()
        self.assertEqual([tool.name for tool in tools], ["send_message"])
        self.assertFalse(tools[0].annotations.read_only_hint)
        self.assertFalse(tools[0].annotations.idempotent_hint)
        with self.client(lambda request: httpx.Response(401, json={"ok": False, "error_code": 401})):
            result = await mcp.call_tool("send_message", {"text": "Report"})
        self.assertTrue(result.is_error)
        self.assertEqual(result.structured_content["status"], "failed")


if __name__ == "__main__":
    unittest.main()
