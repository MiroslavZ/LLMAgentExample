"""Проверки конфигурации MCP и её хранения без сетевых запросов."""

import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

from llm_agent.mcp_config import GITHUB_MCP_URL, GITHUB_TOKEN_ENV, MCPServer, MCPServerStore, MCPStorageError


class MCPServerTests(unittest.TestCase):
    def setUp(self):
        self.server = MCPServer("github", "GitHub", GITHUB_MCP_URL, GITHUB_TOKEN_ENV)

    def test_roundtrip_preserves_config_and_is_immutable(self):
        encoded = json.dumps(self.server.to_dict(), ensure_ascii=False)
        self.assertEqual(MCPServer.from_dict(json.loads(encoded)), self.server)
        detached = self.server.to_dict()
        detached["name"] = "Другой сервер"
        self.assertEqual(self.server.name, "GitHub")
        with self.assertRaises(FrozenInstanceError):
            self.server.name = "Изменён"
        self.assertEqual(MCPServer.from_dict({"id": "local", "name": "Local", "url": "http://localhost:8000/mcp"}).token_env, "")

    def test_accepts_https_and_local_http_urls(self):
        for url in (
            GITHUB_MCP_URL, "https://example.com:443/mcp", "https://[2001:db8::1]/mcp",
            "http://localhost:8000/mcp", "http://127.0.0.1:1/mcp", "http://[::1]:65535/mcp",
            "HTTPS://example.com/mcp", "https://пример.рф/mcp",
        ):
            with self.subTest(url=url):
                self.assertEqual(replace(self.server, url=url).url, url)

    def test_rejects_invalid_or_credential_bearing_urls(self):
        for url in (
            None, 1, "", "example.com/mcp", "ftp://example.com/mcp", "https:///mcp",
            "http://example.com/mcp", "http://192.168.1.1/mcp", "http://127.0.0.2/mcp",
            "https://user:secret@example.com/mcp", "https://secret@example.com/mcp",
            "https://example.com/mcp?token=secret", "https://example.com/mcp#secret",
            "https://example.com/mcp?", "https://example.com/mcp#", "https://example.com:/mcp",
            "https://example.com:65536/mcp", "https://example.com:0/mcp", "https://example.com:port/mcp",
            " https://example.com/mcp", "https://exa mple.com/mcp", "https://example.com/\nmcp",
            "https://example.com\\@localhost/mcp", "https://[::1/mcp", "https://999.0.0.1/mcp",
            "https://example..com/mcp", "https://-example.com/mcp", "https://example_.com/mcp",
            "https://[fe80::1%eth0]/mcp", "https://[::1]example.com/mcp", "https://[::1]extra:443/mcp",
        ):
            with self.subTest(url=url), self.assertRaises(ValueError):
                replace(self.server, url=url)

    def test_rejects_invalid_fields_and_unknown_keys(self):
        for field, values in {
            "id": (None, 1, "", "two words", "../mcp", "русский", "x" * 65),
            "name": (None, 1, "", "  "),
            "token_env": (None, 1, "TOKEN=secret", "Bearer token", "${TOKEN}", "1TOKEN", "ТОКЕН"),
        }.items():
            for value in values:
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    replace(self.server, **{field: value})
        for data in (None, [], {}, {"id": "a", "name": "Имя"}, {**self.server.to_dict(), "token": "secret"}):
            with self.subTest(data=data), self.assertRaises(ValueError):
                MCPServer.from_dict(data)


class MCPServerStoreTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "nested" / "mcp.sqlite3"
        self.store = MCPServerStore(self.path)
        self.github = MCPServer("github", "GitHub", GITHUB_MCP_URL, GITHUB_TOKEN_ENV)
        self.local = MCPServer("local", "Local", "http://localhost:8000/mcp")

    def test_storage_is_lazy_and_config_survives_restart(self):
        self.assertFalse(self.path.parent.exists())
        self.store.save(self.local)
        self.store.save(self.github)
        restarted = MCPServerStore(self.path)
        self.assertEqual(restarted.list(), [self.github, self.local])
        self.assertEqual(restarted.get("github"), self.github)
        changed = replace(self.github, name="GitHub новый")
        restarted.save(changed, expected=self.github)
        self.assertEqual(self.store.get("github"), changed)
        restarted.delete("github")
        self.assertEqual(self.store.list(), [self.local])

    def test_stale_edit_cannot_overwrite_updated_or_deleted_server(self):
        self.store.save(self.github)
        changed = replace(self.github, name="Другая вкладка")
        other = MCPServerStore(self.path)
        other.save(changed, expected=self.github)
        for exists in (True, False):
            with self.subTest(exists=exists):
                if not exists:
                    other.delete("github")
                with self.assertRaisesRegex(ValueError, "другой вкладке"):
                    self.store.save(replace(self.github, name="Устаревшая правка"), expected=self.github)
                self.assertEqual(self.store.list(), [changed] if exists else [])

    def test_invalid_save_does_not_create_storage_and_missing_records_fail(self):
        with self.assertRaises(ValueError):
            self.store.save(self.github.to_dict())
        with self.assertRaises(ValueError):
            self.store.save(self.github, expected={})
        self.assertFalse(self.path.exists())
        for operation in (lambda: self.store.get("missing"), lambda: self.store.delete("missing")):
            with self.assertRaisesRegex(ValueError, "не найден"):
                operation()

    def test_corrupt_database_is_not_overwritten(self):
        self.path.parent.mkdir()
        original = "Повреждённая база MCP".encode("utf-8")
        self.path.write_bytes(original)
        for operation in (self.store.list, lambda: self.store.save(self.github), lambda: self.store.delete("github")):
            with self.assertRaises(MCPStorageError):
                operation()
            self.assertEqual(self.path.read_bytes(), original)

    def test_corrupt_record_cannot_be_read_overwritten_or_deleted(self):
        self.store.save(self.github)
        for data in ("{bad", "null", '{"id":"other","name":"GitHub","url":"https://example.com/mcp"}',
                     json.dumps({**self.github.to_dict(), "token_env": "Bearer secret"})):
            with self.subTest(data=data):
                with closing(sqlite3.connect(self.path)) as connection, connection:
                    connection.execute("UPDATE mcp_servers SET data = ?", (data,))
                original = self.path.read_bytes()
                for operation in (self.store.list, lambda: self.store.get("github"),
                                  lambda: self.store.save(self.github), lambda: self.store.delete("github")):
                    with self.assertRaisesRegex(MCPStorageError, "Повреждены"):
                        operation()
                    self.assertEqual(self.path.read_bytes(), original)

    def test_unknown_database_version_or_missing_schema_is_preserved(self):
        self.store.save(self.github)
        for query in ("PRAGMA user_version = 999", "PRAGMA user_version = 0", "DROP TABLE mcp_servers"):
            with self.subTest(query=query):
                with closing(sqlite3.connect(self.path)) as connection, connection:
                    connection.execute("PRAGMA user_version = 1")
                    connection.execute(query)
                original = self.path.read_bytes()
                with self.assertRaises(MCPStorageError):
                    self.store.list()
                self.assertEqual(self.path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
