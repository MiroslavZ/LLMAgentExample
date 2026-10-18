"""API использует серверную сессию, не принимает настройки модели и чужие тикеты."""

import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from support_agent.config import Settings
from support_agent.service import SupportService
from support_agent.web import COOKIE_NAME, create_app
from tests.test_support_service import FakeCRM, FakeResponder, FakeRetriever


class SupportWebTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.service = SupportService(Settings(database_path=Path(self.tmp.name) / "history.sqlite3"),
                                      crm=FakeCRM(), retriever=FakeRetriever(), responder=FakeResponder())
        self.app = create_app(self.service)
        self.client = TestClient(self.app, raise_server_exceptions=False)

    def tearDown(self):
        self.client.close()
        self.tmp.cleanup()

    def test_ui_and_selection(self):
        page = self.client.get("/")
        self.assertEqual(page.status_code, 200)
        self.assertIn("LLMAgent", page.text)
        self.assertIn("httponly", page.headers["set-cookie"].lower())
        self.assertEqual(len(self.client.get("/api/users").json()["users"]), 2)
        self.assertEqual(self.client.get("/api/tickets", params={"user_id": "u1"}).json()["tickets"][0]["id"], "t1")

    def test_browser_user_and_ticket_histories_do_not_mix(self):
        response = self.client.post("/api/chat", json={"user_id": "u1", "ticket_id": "t1", "message": "Помоги"})
        self.assertEqual(response.status_code, 200, response.text)
        params = {"user_id": "u1", "ticket_id": "t1"}
        self.assertEqual(len(self.client.get("/api/history", params=params).json()["turns"]), 1)
        self.assertEqual(self.client.get("/api/history", params={"user_id": "u1"}).json()["turns"], [])
        with TestClient(self.app) as other:
            self.assertEqual(other.get("/api/history", params=params).json()["turns"], [])
        self.assertEqual(self.client.delete("/api/history", params=params).status_code, 200)
        self.assertEqual(self.client.get("/api/history", params=params).json()["turns"], [])

    def test_unknown_ticket_and_invalid_input_do_not_call_model(self):
        for payload in [{"user_id": "u1", "ticket_id": "foreign", "message": "Вопрос"},
                        {"user_id": "u1", "message": " "},
                        {"user_id": "../users", "message": "Вопрос"},
                        {"user_id": "u1", "message": "Вопрос", "model": "local"},
                        {"user_id": "u1", "message": "x" * 4001}]:
            self.assertIn(self.client.post("/api/chat", json=payload).status_code, (404, 422))
        self.assertEqual(self.service.responder.calls, [])

    def test_cookie_is_generated_and_error_hides_exception(self):
        self.client.cookies.set(COOKIE_NAME, "invalid")
        self.client.get("/")
        cookie = self.client.cookies.get(COOKIE_NAME, domain="testserver.local", path="/")
        self.assertRegex(cookie, r"^[0-9a-f]{32}$")
        def fail(*args):
            raise RuntimeError("private-token-from-provider")
        self.service.responder.answer = fail
        response = self.client.post("/api/chat", json={"user_id": "u1", "message": "Вопрос"})
        self.assertEqual(response.status_code, 500)
        self.assertNotIn("private-token", response.text)
