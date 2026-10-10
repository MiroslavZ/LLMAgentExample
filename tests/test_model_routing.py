"""Модель и сервер меняются вместе для всех этапов диалога."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from llm_agent.agent import Agent
from llm_agent.history import HistoryManager
from llm_agent.llm_models import LLMModel, ModelStorageError
from llm_agent.models import ContextSettings, RequestOptions
from llm_agent.response_schemas import RAG_PREPARATION_SCHEMA, schema_format
from llm_agent.service import ConversationService
from tests.helpers import completion, rag_preparation
from tests.test_rag import grounded_completion, retrieved
from tests.test_structured_output import deepseek_unsupported, lm_studio_unsupported


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
        self.register(replace(self.local, structured_output="off"))
        self.service.update_settings(self.conversation.id, ContextSettings(strategy="facts", window_size=2))
        facts = completion()
        facts.choices[0].message.content = '{"name":"User"}'
        self.completions.side_effect = [facts, completion()]
        result = self.service.send(self.conversation.id, "Hello")
        self.assertEqual(result.turns[-1].status, "completed")
        self.assertTrue(all("response_format" not in call.kwargs for call in self.completions.call_args_list))

    def test_explicit_json_format_rejected_when_capability_disabled(self):
        self.register(replace(self.local, structured_output="off"))
        with self.assertRaisesRegex(ValueError, "отключён структурированный вывод"):
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

    def test_detected_format_survives_next_turn_and_service_restart(self):
        self.register(self.cloud)
        answer = completion()
        answer.choices[0].message.content = '{"answer":"Ответ"}'
        self.completions.side_effect = [deepseek_unsupported(), answer, answer, answer]
        options = RequestOptions(response_format="object")

        first = self.service.send(self.conversation.id, "Первый вопрос", options=options)
        self.assertEqual(first.turns[-1].status, "completed")
        self.assertEqual(self.service.models.get_detected_format(self.cloud), "json_object")
        second = self.service.send(self.conversation.id, "Второй вопрос", options=options)
        self.assertEqual(second.turns[-1].status, "completed")

        restarted = ConversationService(self.directory)
        self.assertEqual(restarted.models.get_detected_format(self.cloud), "json_object")
        third = restarted.send(self.conversation.id, "Третий вопрос", options=options)
        self.assertEqual(third.turns[-1].status, "completed")
        self.assertEqual([
            call.kwargs["response_format"]["type"] for call in self.completions.call_args_list
        ], ["json_schema", "json_object", "json_object", "json_object"])

    def test_format_cache_read_failure_does_not_block_structured_request(self):
        self.register(self.cloud)
        answer = completion()
        answer.choices[0].message.content = '{"answer":"Ответ"}'
        self.completions.return_value = answer

        with patch.object(
            self.service.models, "get_detected_format", side_effect=ModelStorageError("Кеш недоступен"),
        ) as read_cache:
            result = self.service.send(
                self.conversation.id, "Верни JSON", options=RequestOptions(response_format="object"),
            )

        read_cache.assert_called_once_with(self.cloud)
        self.completions.assert_called_once()
        self.assertEqual(self.completions.call_args.kwargs["response_format"]["type"], "json_schema")
        self.assertEqual(result.turns[-1].status, "completed")
        self.assertEqual(result.turns[-1].answer, answer.choices[0].message.content)
        self.assertIsNone(result.turns[-1].error)

    def test_format_cache_write_failure_preserves_answer_and_usage(self):
        self.register(self.cloud)
        answer = completion(prompt=120, output=35)
        answer.choices[0].message.content = '{"answer":"Полученный ответ"}'
        self.completions.return_value = answer

        with patch.object(
            self.service.models, "remember_detected_format", side_effect=ModelStorageError("Кеш недоступен"),
        ) as write_cache:
            result = self.service.send(
                self.conversation.id, "Верни JSON", options=RequestOptions(response_format="object"),
            )

        write_cache.assert_called_once_with(self.cloud, "json_schema")
        self.completions.assert_called_once()
        self.assertEqual(result.turns[-1].status, "completed")
        self.assertEqual(result.turns[-1].answer, answer.choices[0].message.content)
        self.assertIsNone(result.turns[-1].error)
        restored = ConversationService(self.directory).get(result.id)
        self.assertEqual(restored, result)
        messages = HistoryManager._decode_data(restored.working_context)[0]
        self.assertEqual(
            messages[-1]["usage"],
            {"prompt_tokens": 120, "completion_tokens": 35, "total_tokens": 155},
        )

    def test_manual_format_selection_reaches_sdk_for_structured_stage(self):
        for mode in ("json_schema", "json_object", "off"):
            with self.subTest(mode=mode):
                model = replace(self.local, structured_output=mode)
                self.register(model)
                facts = completion()
                facts.choices[0].message.content = '{"name":"Пользователь"}'
                self.completions.reset_mock()
                self.completions.side_effect = [facts, completion()]

                result = self.service.send(
                    self.conversation.id, "Меня зовут Пользователь",
                    settings=ContextSettings(strategy="facts", window_size=2),
                )

                self.assertEqual(result.turns[-1].status, "completed")
                requests = self.completions.call_args_list
                self.assertEqual(len(requests), 2)
                if mode == "off":
                    self.assertNotIn("response_format", requests[0].kwargs)
                else:
                    self.assertEqual(requests[0].kwargs["response_format"]["type"], mode)
                if mode == "json_schema":
                    self.assertEqual(
                        requests[0].kwargs["response_format"]["json_schema"]["schema"]["type"],
                        "object",
                    )
                self.assertNotIn("response_format", requests[1].kwargs)
                self.assertIsNone(self.service.models.get_detected_format(model))

    def test_plain_dialogue_does_not_detect_or_send_structured_format(self):
        self.register(self.cloud)
        for question in ("Привет", "Продолжим"):
            result = self.service.send(self.conversation.id, question)
            self.assertEqual(result.turns[-1].status, "completed")
            self.assertIsNone(self.service.models.get_detected_format(self.cloud))
        self.assertEqual(self.completions.call_count, 2)
        self.assertTrue(all(
            "response_format" not in call.kwargs for call in self.completions.call_args_list
        ))

    def test_rag_preparation_and_final_answer_share_policy_and_detected_format(self):
        self.register(self.cloud)
        preparation = completion()
        preparation.choices[0].message.content = '{"memory_delta":[],"standalone_query":null}'
        self.completions.side_effect = [deepseek_unsupported(), preparation, grounded_completion()]

        def prepare_with_model(question, memory, turns, token, model, **kwargs):
            kwargs["output_policy"].complete(
                self.completions, model=model,
                messages=[{"role": "user", "content": question}],
                response_format=schema_format("rag_preparation", RAG_PREPARATION_SCHEMA),
            )
            return rag_preparation(question, memory, turns, token, model, **kwargs)

        with patch("llm_agent.service.prepare_turn", side_effect=prepare_with_model) as prepare, patch.object(
            self.service.retriever, "retrieve", return_value=retrieved(),
        ), patch("llm_agent.service.Agent", wraps=Agent) as agent:
            result = self.service.send(
                self.conversation.id, "Что сказано в базе?", settings=ContextSettings(rag_enabled=True),
            )

        self.assertEqual(result.turns[-1].status, "completed")
        self.assertEqual(result.turns[-1].rag_answer["status"], "answered")
        self.assertIs(agent.call_args.kwargs["output_policy"], prepare.call_args.kwargs["output_policy"])
        self.assertEqual([
            call.kwargs["response_format"]["type"] for call in self.completions.call_args_list
        ], ["json_schema", "json_object", "json_object"])
        self.assertEqual(self.service.models.get_detected_format(self.cloud), "json_object")

    def test_unsupported_formats_produce_safe_persisted_turn_error(self):
        self.register(self.cloud)
        self.completions.side_effect = [deepseek_unsupported(), lm_studio_unsupported()]
        result = self.service.send(
            self.conversation.id, "Верни JSON", options=RequestOptions(response_format="object"),
        )

        turn = result.turns[-1]
        self.assertEqual(turn.status, "error")
        self.assertIn("JSON Schema", turn.error)
        self.assertIn("JSON Object", turn.error)
        self.assertIn("менеджере моделей", turn.error)
        self.assertNotIn("private-request-id", turn.error)
        self.assertNotIn("cloud-secret", turn.error)
        self.assertNotIn("response_format.type", turn.error)
        self.assertEqual(self.completions.call_count, 2)
        self.assertIsNone(self.service.models.get_detected_format(self.cloud))
        self.assertEqual(ConversationService(self.directory).get(result.id).turns[-1].error, turn.error)


if __name__ == "__main__":
    unittest.main()
