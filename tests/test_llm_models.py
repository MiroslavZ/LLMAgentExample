"""Каталог моделей и discovery без внешних серверов и настоящих токенов."""

import json
import sqlite3
import tempfile
import traceback
import unittest
from contextlib import closing
from dataclasses import FrozenInstanceError, asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from uuid import uuid4

import httpx
from openai import AuthenticationError

from llm_agent.llm_models import LLMModel, ModelConnectionError, ModelStorageError, ModelStore, discover_models


class LLMModelTests(unittest.TestCase):
    def setUp(self):
        self.model = LLMModel(uuid4().hex, "Локальная модель", "gemma-4", "http://192.168.1.188:1234/v1/")

    def test_config_is_immutable_and_optional_token_is_hidden(self):
        self.assertEqual(self.model.token, "")
        self.assertEqual(self.model.base_url, "http://192.168.1.188:1234/v1")
        with self.assertRaises(FrozenInstanceError):
            self.model.name = "Изменено"
        self.assertNotIn("private-token", repr(replace(self.model, token="private-token")))
        self.assertNotIn("token=", repr(self.model))

    def test_accepts_local_remote_and_ipv6_servers(self):
        for url in (
            "https://api.deepseek.com/v1", "http://localhost:1234/v1", "http://[::1]:1234/v1",
            "http://[2001:db8::1]/v1", "HTTPS://EXAMPLE.COM/v1", "https://пример.рф/v1",
        ):
            with self.subTest(url=url):
                self.assertTrue(replace(self.model, base_url=url).base_url.startswith(("http://", "https://")))

    def test_rejects_invalid_or_credential_bearing_urls_without_echoing_secrets(self):
        for url in (
            None, 1, "", "example.com/v1", "ftp://example.com/v1", "https:///v1",
            "https://user:private-token@example.com/v1", "https://private-token@example.com/v1",
            "https://example.com/v1?token=private-token", "https://example.com/v1#private-token",
            "https://example.com/v1?", "https://example.com/v1#", "https://example.com:/v1",
            "https://example.com:65536/v1", "https://example.com:0/v1", "https://example.com:bad/v1",
            " https://example.com/v1", "https://exa mple.com/v1", "https://example.com/\nv1",
            "https://example.com\\@localhost/v1", "https://[::1/v1", "https://999.0.0.1/v1",
            "https://example..com/v1", "https://-example.com/v1", "https://example_.com/v1",
            "https://[fe80::1%eth0]/v1", "https://[::1]example.com/v1", "https://[::1]extra:443/v1",
        ):
            with self.subTest(url=url), self.assertRaises(ValueError) as captured:
                replace(self.model, base_url=url)
            self.assertNotIn("private-token", str(captured.exception))

    def test_rejects_invalid_fields(self):
        for field, values in {
            "id": (None, 1, "", "model-name", "f" * 31, "z" * 32, "F" * 32),
            "name": (None, 1, "", "  ", "name\n"),
            "model_id": (None, 1, "", "model id", "model\n", "model\x00"),
            "token": (None, 1, "Bearer token", "secret\n"),
            "timeout": (None, "30", True, 0, -1, float("nan"), float("inf"), 10 ** 1000),
            "json_mode": (None, 0, 1, "true"),
            "tools_enabled": (None, 0, 1, "false"),
        }.items():
            for value in values:
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    replace(self.model, **{field: value})


class ModelStoreTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "nested" / "models.sqlite3"
        self.store = ModelStore(self.path)
        self.model = LLMModel(uuid4().hex, "Gemma", "gemma-4", "http://192.168.1.188:1234/v1", "private-token")

    def test_catalog_is_initially_empty_and_does_not_read_environment(self):
        self.assertFalse(self.path.parent.exists())
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "environment-secret"}):
            self.assertEqual(self.store.list(), [])
        self.assertEqual(ModelStore(self.path).list(), [])

    def test_config_survives_restart_update_and_delete(self):
        self.store.save(self.model)
        self.assertEqual(ModelStore(self.path).get(self.model.id), self.model)
        changed = replace(self.model, name="Новое имя", model_id="gemma-new", timeout=30.5,
                          json_mode=False, tools_enabled=False, token="changed-token")
        self.store.save(changed)
        self.assertEqual(ModelStore(self.path).list(), [changed])
        self.store.delete(self.model.id)
        self.assertEqual(self.store.list(), [])

    def test_separate_catalogs_do_not_share_models(self):
        self.store.save(self.model)
        other = ModelStore(self.path.with_name("other.sqlite3"))
        self.assertEqual(other.list(), [])

    def test_import_deduplicates_but_preserves_existing_settings_and_token(self):
        original = replace(self.model, name="Моё имя", timeout=90, json_mode=False, tools_enabled=False)
        self.store.save(original)
        imported = self.store.import_models(self.model.base_url + "/", "new-token", ["gemma-4", "phi", "phi"])
        self.assertEqual(imported[0], original)
        self.assertEqual(imported[1].model_id, "phi")
        self.assertEqual(imported[1].name, "phi")
        self.assertEqual(imported[1].token, "new-token")
        self.assertEqual(len(imported), 2)
        self.assertEqual(len(self.store.list()), 2)
        self.assertEqual(self.store.import_models(self.model.base_url, "", ["gemma-4"]), [original])

    def test_same_api_id_on_different_endpoints_has_separate_identity(self):
        local = self.store.import_models(self.model.base_url, "", ["shared-model"])[0]
        remote = self.store.import_models("https://example.com/v1", "remote-token", ["shared-model"])[0]
        self.assertNotEqual(local.id, remote.id)
        self.assertEqual(local.token, "")
        self.assertEqual(remote.token, "remote-token")
        self.assertEqual(len(self.store.list()), 2)

    def test_resolve_prefers_internal_id_and_rejects_ambiguous_api_id(self):
        self.store.save(self.model)
        self.assertEqual(self.store.resolve(self.model.model_id), self.model)
        second = replace(self.model, id=uuid4().hex, base_url="https://example.com/v1")
        self.store.save(second)
        with self.assertRaisesRegex(ValueError, "неоднозначен"):
            self.store.resolve(self.model.model_id)
        third = replace(self.model, id=uuid4().hex, model_id=self.model.id)
        self.store.save(third)
        self.assertEqual(self.store.resolve(self.model.id), self.model)

    def test_duplicate_manual_save_cannot_overwrite_other_record(self):
        self.store.save(self.model)
        with self.assertRaisesRegex(ValueError, "уже добавлена"):
            self.store.save(replace(self.model, id=uuid4().hex, token="replacement"))
        self.assertEqual(self.store.list(), [self.model])

    def test_stale_save_and_delete_do_not_overwrite_other_changes(self):
        self.store.save(self.model)
        changed = replace(self.model, name="Другая вкладка")
        self.store.save(changed, expected=self.model)
        with self.assertRaisesRegex(ValueError, "другой вкладке"):
            self.store.save(replace(self.model, name="Устаревшая правка"), expected=self.model)
        with self.assertRaisesRegex(ValueError, "другой вкладке"):
            self.store.delete(self.model.id, expected=self.model)
        self.assertEqual(self.store.get(self.model.id), changed)
        self.store.delete(changed.id, expected=changed)
        with self.assertRaisesRegex(ValueError, "другой вкладке"):
            self.store.save(self.model, expected=self.model)
        self.assertEqual(self.store.list(), [])

    def test_missing_records_fail_without_echoing_reference(self):
        for operation in (self.store.get, self.store.delete, self.store.resolve):
            with self.subTest(operation=operation), self.assertRaises(KeyError) as captured:
                operation("private-token")
            self.assertNotIn("private-token", str(captured.exception))

    def test_invalid_import_and_save_do_not_create_or_partially_update_catalog(self):
        for operation in (
            lambda: self.store.import_models(self.model.base_url, "", ["valid", "invalid id"]),
            lambda: self.store.import_models(self.model.base_url, "", "model-id"),
            lambda: self.store.save(asdict(self.model)),
            lambda: self.store.save(self.model, expected={}),
        ):
            with self.assertRaises(ValueError):
                operation()
            self.assertFalse(self.path.exists())

    def test_corrupt_database_is_preserved_and_errors_are_sanitized(self):
        self.path.parent.mkdir()
        original = b"private-token corrupt database"
        self.path.write_bytes(original)
        for operation in (self.store.list, lambda: self.store.save(self.model), lambda: self.store.delete(self.model.id)):
            with self.assertRaises(ModelStorageError) as captured:
                operation()
            self.assertNotIn("private-token", str(captured.exception))
            self.assertEqual(self.path.read_bytes(), original)

    def test_corrupt_record_cannot_be_read_overwritten_or_deleted(self):
        self.store.save(self.model)
        for data in ("{private-token", "null", "[]", json.dumps({**asdict(self.model), "token": None})):
            with self.subTest(data=data):
                with closing(sqlite3.connect(self.path)) as connection, connection:
                    connection.execute("UPDATE llm_models SET data = ?", (data,))
                original = self.path.read_bytes()
                for operation in (self.store.list, lambda: self.store.get(self.model.id),
                                  lambda: self.store.save(self.model), lambda: self.store.delete(self.model.id),
                                  lambda: self.store.import_models(self.model.base_url, "", ["phi", "gemma-4"])):
                    with self.assertRaises(ModelStorageError):
                        operation()
                    self.assertEqual(self.path.read_bytes(), original)

    def test_unknown_schema_and_versions_are_preserved(self):
        self.store.save(self.model)
        for query in ("PRAGMA user_version = 999", "PRAGMA user_version = 0", "DROP TABLE llm_models"):
            with self.subTest(query=query):
                with closing(sqlite3.connect(self.path)) as connection, connection:
                    connection.execute("PRAGMA user_version = 1")
                    connection.execute(query)
                original = self.path.read_bytes()
                with self.assertRaises(ModelStorageError):
                    self.store.list()
                self.assertEqual(self.path.read_bytes(), original)


class DiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.client = MagicMock()
        self.client.__enter__.return_value = self.client
        self.factory_patch = patch("llm_agent.llm_models.OpenAI", return_value=self.client)
        self.factory = self.factory_patch.start()
        self.addCleanup(self.factory_patch.stop)

    def test_discovery_uses_sdk_and_consumes_all_pages(self):
        visited_pages = []

        def items():
            for index, values in enumerate((("zeta", "alpha"), ("beta", "alpha"))):
                visited_pages.append(index)
                yield from (SimpleNamespace(id=value) for value in values)

        self.client.models.list.return_value = items()
        self.assertEqual(discover_models("https://example.com/v1/", "private-token"), ["alpha", "beta", "zeta"])
        self.assertEqual(visited_pages, [0, 1])
        self.factory.assert_called_once_with(base_url="https://example.com/v1", api_key="private-token",
                                             timeout=20, max_retries=0)
        self.client.__exit__.assert_called_once()

    def test_optional_token_does_not_fall_back_to_environment(self):
        self.client.models.list.return_value = [SimpleNamespace(id="gemma-4")]
        with patch.dict("os.environ", {"OPENAI_API_KEY": "environment-secret"}):
            self.assertEqual(discover_models("http://localhost:1234/v1"), ["gemma-4"])
        self.assertEqual(self.factory.call_args.kwargs["api_key"], "local-no-auth")
        transport = self.factory.call_args.kwargs["http_client"]
        self.addCleanup(transport.close)
        self.assertFalse(transport.trust_env)

    def test_connection_errors_hide_server_details_and_close_client(self):
        self.client.models.list.side_effect = RuntimeError("Authorization: private-token; server echoed private-token")
        with self.assertRaises(ModelConnectionError) as captured:
            discover_models("https://example.com/v1", "private-token")
        formatted = "".join(traceback.format_exception(captured.exception))
        self.assertNotIn("private-token", formatted)
        self.client.__exit__.assert_called_once()

    def test_authentication_error_has_actionable_message_without_response_body(self):
        response = httpx.Response(401, request=httpx.Request("GET", "https://example.com/v1/models"))
        self.client.models.list.side_effect = AuthenticationError(
            "private-token is invalid", response=response, body={"error": "private-token"}
        )
        with self.assertRaisesRegex(ModelConnectionError, "авторизацию") as captured:
            discover_models("https://example.com/v1", "private-token")
        self.assertNotIn("private-token", str(captured.exception))

    def test_client_construction_and_invalid_responses_are_sanitized(self):
        self.factory.side_effect = RuntimeError("private-token construction error")
        with self.assertRaises(ModelConnectionError) as captured:
            discover_models("https://example.com/v1", "private-token")
        self.assertNotIn("private-token", str(captured.exception))
        self.factory.side_effect = None
        for response in ([SimpleNamespace(id=None)], [SimpleNamespace(id="invalid id")], [object()]):
            with self.subTest(response=response):
                self.client.models.list.return_value = response
                with self.assertRaises(ModelConnectionError):
                    discover_models("https://example.com/v1")

    def test_invalid_configuration_never_contacts_server(self):
        for base_url, token in (("https://secret@example.com/v1", ""), ("https://example.com/v1", "bad token")):
            with self.assertRaises(ValueError):
                discover_models(base_url, token)
        self.factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
