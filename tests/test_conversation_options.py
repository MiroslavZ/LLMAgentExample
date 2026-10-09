"""Параметры CLI сохраняются в общей истории и переживают смену интерфейса."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from llm_agent.models import ContextSettings, RequestOptions
from llm_agent.service import ConversationService
from tests.helpers import completion, create_selected_conversation, register_model


class ConversationOptionsTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.service = ConversationService(self.directory, "test-token")
        self.conversation = create_selected_conversation(self.service)
        client = self.enterContext(patch("llm_agent.agent.OpenAI")).return_value
        self.create_completion = client.chat.completions.create
        self.create_completion.return_value = completion()

    def test_options_reach_both_request_modes_and_survive_restart(self):
        register_model(self.service, model_id="custom-model")
        for meta_prompt in (False, True):
            with self.subTest(meta_prompt=meta_prompt):
                options = RequestOptions(
                    meta_prompt=meta_prompt, model="custom-model", stop_sequences=["STOP"],
                    response_format="object", temperature=0.3, max_tokens=500,
                )
                responses = []
                result = self.service.send(
                    self.conversation.id, "Вопрос", options=options,
                    on_response=responses.append,
                )
                self.assertEqual(result.turns[-1].status, "completed")
                kwargs = self.create_completion.call_args.kwargs
                self.assertEqual(kwargs["model"], "custom-model")
                self.assertEqual(kwargs["stop"], ["STOP"])
                self.assertEqual(kwargs["response_format"], {"type": "json_object"})
                self.assertEqual(kwargs["temperature"], 0.3)
                self.assertEqual(kwargs["max_tokens"], 500)
                self.assertEqual(len(responses), 2 if meta_prompt else 1)
                restored = ConversationService(self.directory, None).get(result.id)
                self.assertEqual(restored.turns[-1].options, options)

    def test_old_web_options_load_with_defaults(self):
        self.service.send(self.conversation.id, "Вопрос")
        path = self.service.store.path(self.conversation.id)
        data = json.loads(path.read_text(encoding="utf-8"))
        data["turns"][0]["options"] = {
            "meta_prompt": False, "temperature": 0.4, "max_tokens": 300,
        }
        path.write_text(json.dumps(data), encoding="utf-8")
        restored = self.service.get(self.conversation.id)
        self.assertEqual(restored.turns[0].options, RequestOptions(temperature=0.4, max_tokens=300))

    def test_invalid_options_do_not_start_turn(self):
        for changes in (
            {"model": " "}, {"model": 42}, {"stop_sequences": "stop"},
            {"stop_sequences": [""]}, {"stop_sequences": [False]},
            {"response_format": "unknown"},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.service.send(self.conversation.id, "Вопрос", options=RequestOptions(**changes))
        self.assertFalse(self.service.get(self.conversation.id).started)
        self.create_completion.assert_not_called()

    def test_response_callback_error_does_not_rewrite_completed_turn(self):
        def fail(_result):
            raise RuntimeError("Ошибка вывода")

        with self.assertRaisesRegex(RuntimeError, "Ошибка вывода"):
            self.service.send(self.conversation.id, "Вопрос", on_response=fail)
        restored = self.service.get(self.conversation.id)
        self.assertEqual(restored.turns[-1].status, "completed")
        self.assertIsNone(restored.turns[-1].error)

    def test_compression_callback_error_does_not_rewrite_completed_turn(self):
        self.service.update_settings(self.conversation.id, ContextSettings(
            strategy="summary", last_messages=0, compress_every=2,
        ))
        self.service.send(self.conversation.id, "Первый вопрос")

        def fail(_result):
            raise RuntimeError("Ошибка статистики")

        with self.assertRaisesRegex(RuntimeError, "Ошибка статистики"):
            self.service.send(self.conversation.id, "Второй вопрос", on_compression=fail)
        restored = self.service.get(self.conversation.id)
        self.assertEqual(restored.turns[-1].status, "completed")
        self.assertTrue(restored.turns[-1].memory_updated)


if __name__ == "__main__":
    unittest.main()
