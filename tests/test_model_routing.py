"""Модель и сервер меняются вместе для всех этапов диалога."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from llm_agent.llm_models import LLMModel
from llm_agent.models import ContextSettings, RequestOptions
from llm_agent.service import ConversationService
from tests.helpers import completion, rag_preparation


class ModelRoutingTests(unittest.TestCase):
    def setUp(self):
        directory = self.enterContext(tempfile.TemporaryDirectory())
        self.directory = Path(directory)
        self.service = ConversationService(self.directory, token="must-not-be-used")
        self.conversation = self.service.create()
        self.local = LLMModel(uuid4().hex, "Gemma", "gemma", "http://127.0.0.1:1234/v1")
        self.cloud = LLMModel(uuid4().hex, "Cloud", "other", "https://example.com/v1", "cloud-secret")
        self.factory = self.enterContext(patch("llm_agent.agent.OpenAI"))
        self.completions = self.factory.return_value.chat.completions.create
        self.completions.return_value = completion()
        self.enterContext(patch("llm_agent.llm_client.DefaultHttpxClient"))

    def register(self, model):
        self.service.models.save(model)
        return self.service.select_model(self.conversation.id, model.id)

    def test_empty_catalog_even_with_legacy_token_or_environment(self):
        self.assertEqual(self.service.models.list(), [])
        with self.assertRaisesRegex(ValueError, "Выберите модель"):
            self.service.send(self.conversation.id, "Hello")
        self.assertFalse(self.service.get(self.conversation.id).started)
        self.factory.assert_not_called()

    def test_switch_routes_endpoint_token_and_all_meta_stages(self):
        self.register(self.local)
        first = self.service.send(self.conversation.id, "Hello")
        self.assertEqual(self.factory.call_args.kwargs["base_url"], self.local.base_url)
        self.assertEqual(self.factory.call_args.kwargs["api_key"], "local-no-auth")
        self.assertEqual(first.turns[-1].model_name, "Gemma")
        self.register(self.cloud)
        second = self.service.send(self.conversation.id, "Follow up", options=RequestOptions(meta_prompt=True))
        self.assertEqual(self.factory.call_args.kwargs["api_key"], "cloud-secret")
        self.assertEqual(self.factory.call_args.kwargs["base_url"], self.cloud.base_url)
        self.assertEqual([call.kwargs["model"] for call in self.completions.call_args_list], ["gemma", "other", "other"])
        restored = ConversationService(self.directory).get(self.conversation.id)
        self.assertEqual(restored, second)
        self.assertEqual(restored.selected_model_id, self.cloud.id)
        self.assertEqual([turn.model_name for turn in restored.turns], ["Gemma", "Cloud"])
        saved = self.service.store.path(restored.id).read_text(encoding="utf-8")
        self.assertNotIn("cloud-secret", saved)
        self.assertNotIn("must-not-be-used", saved)
        self.assertIn("Hello", self.completions.call_args.kwargs["messages"][0]["content"])

    def test_old_conversations_load_without_implicit_cloud_fallback(self):
        path = self.service.store.path(self.conversation.id)
        data = json.loads(path.read_text(encoding="utf-8"))
        del data["selected_model_id"]
        path.write_text(json.dumps(data), encoding="utf-8")
        restored = self.service.get(self.conversation.id)
        self.assertIsNone(restored.selected_model_id)
        self.assertEqual(restored.turns, [])

    def test_deleted_model_does_not_fall_back_or_change_history(self):
        self.register(self.local)
        original = self.service.send(self.conversation.id, "Hello")
        self.service.models.delete(self.local.id)
        self.service.models.save(self.cloud)
        with self.assertRaisesRegex(ValueError, "удалена"):
            self.service.send(self.conversation.id, "Next")
        self.assertEqual(self.service.get(self.conversation.id), original)
        self.assertEqual(self.completions.call_count, 1)

    def test_edit_during_meta_request_uses_one_connection_snapshot(self):
        self.register(self.cloud)

        def respond(**kwargs):
            self.service.models.save(replace(self.cloud, model_id="edited", base_url="https://new.example/v1", token="new-secret"))
            return completion()

        self.completions.side_effect = respond
        result = self.service.send(self.conversation.id, "Hello", options=RequestOptions(meta_prompt=True))
        self.assertEqual([call.kwargs["model"] for call in self.completions.call_args_list], ["other", "other"])
        self.assertEqual(result.turns[-1].model_base_url, self.cloud.base_url)

    def test_summary_uses_selected_model(self):
        self.register(self.local)
        self.service.update_settings(self.conversation.id, ContextSettings(strategy="summary", last_messages=0, compress_every=2))
        self.service.send(self.conversation.id, "One")
        self.service.send(self.conversation.id, "Two")
        self.assertEqual(self.completions.call_count, 3)
        self.assertTrue(all(call.kwargs["model"] == "gemma" for call in self.completions.call_args_list))

    def test_rag_preparation_uses_selected_endpoint_token_and_timeout(self):
        self.register(self.cloud)
        self.service.update_settings(self.conversation.id, ContextSettings(rag_enabled=True))
        with patch("llm_agent.service.prepare_turn", side_effect=rag_preparation) as prepare, patch.object(
            self.service.retriever, "retrieve", side_effect=RuntimeError("stop after preparation")
        ):
            self.service.send(self.conversation.id, "Question")
        self.assertEqual(prepare.call_args.args[3:5], ("cloud-secret", "other"))
        self.assertEqual(prepare.call_args.kwargs["base_url"], self.cloud.base_url)
        self.assertEqual(prepare.call_args.kwargs["timeout"], self.cloud.timeout)

    def test_no_json_mode_omits_format_even_for_facts(self):
        self.register(replace(self.local, json_mode=False))
        self.service.update_settings(self.conversation.id, ContextSettings(strategy="facts", window_size=2))
        facts = completion()
        facts.choices[0].message.content = '{"name":"User"}'
        self.completions.side_effect = [facts, completion()]
        result = self.service.send(self.conversation.id, "Hello")
        self.assertEqual(result.turns[-1].status, "completed")
        self.assertTrue(all("response_format" not in call.kwargs for call in self.completions.call_args_list))

    def test_explicit_json_format_rejected_when_capability_disabled(self):
        self.register(replace(self.local, json_mode=False))
        with self.assertRaisesRegex(ValueError, "JSON mode"):
            self.service.send(self.conversation.id, "Hello", options=RequestOptions(response_format="object"))
        self.assertFalse(self.service.get(self.conversation.id).started)
        self.factory.assert_not_called()

    def test_deleted_internal_key_is_not_resolved_as_another_api_id(self):
        self.register(self.local)
        self.service.models.delete(self.local.id)
        self.service.models.save(replace(self.cloud, model_id=self.local.id))
        with self.assertRaisesRegex(ValueError, "удалена"):
            self.service.send(self.conversation.id, "Hello")
        self.factory.assert_not_called()

    def test_no_tool_support_does_not_connect_mcp(self):
        from llm_agent.mcp_config import MCPServer
        self.register(replace(self.local, tools_enabled=False))
        self.service.mcp_servers.save(MCPServer("one", "Tools", "http://127.0.0.1:8000/mcp", enabled=True))
        with patch("llm_agent.agent.MCPToolCatalog") as catalog:
            result = self.service.send(self.conversation.id, "Hello")
        self.assertEqual(result.turns[-1].status, "completed")
        catalog.assert_not_called()
        self.assertNotIn("tools", self.completions.call_args.kwargs)


if __name__ == "__main__":
    unittest.main()
