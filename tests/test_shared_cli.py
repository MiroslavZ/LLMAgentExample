"""CLI and the web service resume the same on-disk conversations without a server."""

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rich.console import Console

from llm_agent.cli import main, run
from llm_agent.models import ContextSettings, RequestOptions
from llm_agent.profile import UserProfile
from llm_agent.service import ConversationService
from llm_agent.storage import DEFAULT_DATA_DIR
from tests.helpers import completion, create_selected_conversation, register_model


class SharedConversationTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.data_dir = self.root / "shared conversations"
        self.client = self.enterContext(patch("llm_agent.agent.OpenAI"))
        self.create = self.client.return_value.chat.completions.create
        self.create.return_value = completion()
        self.enterContext(patch("llm_agent.cli.load_env"))
        self.enterContext(patch.dict(os.environ, {"API_KEY": "test-only"}))

    def service(self):
        return ConversationService(self.data_dir, token="test-only")

    def cli(self, *args, data_dir=True):
        output = io.StringIO()
        arguments = ["main.py", *(('--data-dir', str(self.data_dir)) if data_dir else ()), *args]
        with patch("sys.argv", arguments), patch("llm_agent.cli.console", Console(file=output, width=240)):
            run()
        return output.getvalue()

    def test_web_then_cli_by_path_then_web_preserves_context_and_transcript(self):
        web = self.service()
        conversation = create_selected_conversation(web)
        web.send(conversation.id, "Первый вопрос в вебе", system_prompt="Отвечай кратко")
        path = web.store.path(conversation.id)
        del web  # no server/service instance needs to remain alive
        output = self.cli('--conversation', str(path), '--user', 'Второй вопрос в CLI', data_dir=False)
        self.assertIn(str(path), output)
        messages = self.create.call_args.kwargs["messages"]
        self.assertEqual([item["content"] for item in messages], [
            "Отвечай кратко", "Первый вопрос в вебе", "Ответ", "Второй вопрос в CLI",
        ])
        reopened = self.service()
        self.assertEqual([item.id for item in reopened.list_conversations()], [conversation.id])
        self.assertEqual([turn.user for turn in reopened.get(conversation.id).turns], [
            "Первый вопрос в вебе", "Второй вопрос в CLI",
        ])
        reopened.send(conversation.id, "Третий вопрос снова в вебе")
        self.assertEqual(self.create.call_args.kwargs["messages"][-3:], [
            {"role": "user", "content": "Второй вопрос в CLI"},
            {"role": "assistant", "content": "Ответ"},
            {"role": "user", "content": "Третий вопрос снова в вебе"},
        ])

    def test_cli_then_web_then_cli_by_id_and_history_alias(self):
        model = register_model(self.service())
        self.cli('--model', model.id, '--user', 'Начало из CLI')
        web = self.service()
        conversation, = web.list_conversations()
        web.send(conversation.id, "Продолжение в вебе")
        self.cli('--conversation', conversation.id, '--user', 'Снова CLI')
        self.cli('--history', str(web.store.path(conversation.id)), '--user', 'Через alias')
        self.assertEqual([turn.user for turn in web.get(conversation.id).turns], [
            "Начало из CLI", "Продолжение в вебе", "Снова CLI", "Через alias",
        ])
        # Omitting selection explicitly starts another independent conversation.
        self.cli('--model', model.id, '--user', 'Другой диалог')
        self.assertEqual(len(web.list_conversations()), 2)
        self.assertEqual(self.create.call_args.kwargs['messages'], [{"role": "user", "content": "Другой диалог"}])

    def test_window_settings_survive_handoff_and_transcript_remains_complete(self):
        web = self.service()
        conversation = create_selected_conversation(web)
        settings = ContextSettings(strategy="window", window_size=2)
        web.send(conversation.id, "Первая реплика", settings=settings)
        self.cli('--conversation', conversation.id, '--user', 'Вторая реплика')
        current = web.get(conversation.id)
        self.assertEqual(current.settings, settings)
        self.assertEqual(len(current.turns), 2)
        self.assertNotIn("Первая реплика", json.dumps(self.create.call_args.kwargs['messages'], ensure_ascii=False))
        self.assertEqual(current.turns[0].user, "Первая реплика")
        before = web.store.path(conversation.id).read_bytes()
        with self.assertRaisesRegex(SystemExit, "стратег"):
            self.cli('--conversation', conversation.id, '--strategy', 'full', '--user', 'Не отправлять')
        self.assertEqual(web.store.path(conversation.id).read_bytes(), before)

    def test_summary_and_facts_are_used_after_restart(self):
        for strategy, response in (("summary", "Сводка предыдущего диалога"), ("facts", '{"goal": "Общая цель"}')):
            with self.subTest(strategy=strategy):
                web = self.service()
                conversation = create_selected_conversation(web)
                settings = ContextSettings(strategy=strategy, window_size=2, last_messages=0, compress_every=2)
                web.send(conversation.id, "Первый вопрос", settings=settings)
                intermediate = completion()
                intermediate.choices[0].message.content = response
                self.create.side_effect = [intermediate, completion()]
                self.cli('--conversation', conversation.id, '--user', 'Второй вопрос')
                self.create.side_effect = None
                restored = self.service().get(conversation.id)
                self.assertEqual(restored.settings, settings)
                self.assertEqual(len(restored.turns), 2)
                self.assertEqual(restored.turns[-1].status, 'completed')
                sent = json.dumps(self.create.call_args.kwargs['messages'], ensure_ascii=False)
                self.assertIn("Сводка" if strategy == 'summary' else "Общая цель", sent)

    def test_working_memory_and_profile_are_shared_by_conversation_id(self):
        web = self.service()
        conversation = create_selected_conversation(web)
        profile = UserProfile(id="dev", name="Разработчик", language="Русский")
        web.save_profile(conversation.id, profile)
        web.select_profile(conversation.id, profile.id)
        web.remember_memory(conversation.id, "working", "goal", "Объяснить генераторы")
        self.cli('--conversation', conversation.id, '--user', 'Начнём')
        sent = json.dumps(self.create.call_args.kwargs['messages'], ensure_ascii=False)
        self.assertIn('Объяснить генераторы', sent)
        self.assertIn('Русский', sent)
        self.cli('--conversation', conversation.id, '--memory-set', 'working', 'goal', 'Объяснить декораторы')
        self.assertEqual(web.get_memory(conversation.id).working, {"goal": "Объяснить декораторы"})
        self.cli('--conversation', conversation.id, '--profile-clear')
        self.assertIsNone(web.get_profile(conversation.id))

    def test_cli_errors_are_persisted_and_web_can_resume(self):
        self.cli('--new-conversation')
        service = self.service()
        conversation, = service.list_conversations()
        service.select_model(conversation.id, register_model(service).id)
        self.create.side_effect = RuntimeError("private SDK details")
        with self.assertRaises(SystemExit) as error:
            self.cli('--conversation', conversation.id, '--user', 'Неудачный запрос')
        self.assertNotIn('private SDK details', str(error.exception))
        web = self.service()
        self.assertEqual(web.get(conversation.id).turns[-1].status, 'error')
        self.create.side_effect = None
        web.send(conversation.id, 'Повторный запрос')
        self.assertEqual([turn.status for turn in web.get(conversation.id).turns], ['error', 'completed'])

    def test_live_web_request_blocks_cli_changes(self):
        web = self.service()
        conversation = create_selected_conversation(web)
        def reply(**_kwargs):
            with self.assertRaisesRegex(SystemExit, 'операц|запрос'):
                self.cli('--conversation', conversation.id, '--user', 'Конкурирующий запрос')
            with self.assertRaises(SystemExit):
                self.cli('--conversation', conversation.id, '--memory-set', 'working', 'goal', 'Конфликт')
            self.assertEqual(web.get(conversation.id).turns[-1].status, 'running')
            return completion()
        self.create.side_effect = reply
        web.send(conversation.id, 'Запрос веба')
        self.assertEqual(len(web.get(conversation.id).turns), 1)
        self.assertEqual(web.get_memory(conversation.id).working, {})

    def test_list_and_show_without_api_and_no_implicit_creation(self):
        self.assertEqual(json.loads(self.cli('--list-conversations')), [])
        self.cli('--new-conversation')
        listed, = json.loads(self.cli('--list-conversations'))
        self.assertEqual(Path(listed['path']).parent, self.data_dir)
        shown = json.loads(self.cli('--conversation', listed['id'], '--show-conversation'))
        self.assertEqual(shown['id'], listed['id'])
        self.assertEqual(shown['turns'], [])
        self.client.assert_not_called()

    def test_invalid_or_missing_paths_never_create_another_dialogue(self):
        for selector in ('0' * 32, str(self.data_dir / ('0' * 32 + '.json')), 'history.json'):
            with self.subTest(selector=selector), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    self.cli('--conversation', selector, '--user', 'Не создавать')
        self.assertFalse(self.data_dir.exists())
        self.client.assert_not_called()

    def test_default_catalog_does_not_depend_on_working_directory(self):
        from llm_agent.cli import parse_args
        from llm_agent.web.app import DEFAULT_DATA_DIR as WEB_DATA_DIR
        previous = Path.cwd()
        try:
            os.chdir(self.root)
            with patch('sys.argv', ['main.py', '--new-conversation']):
                self.assertEqual(parse_args().data_dir, DEFAULT_DATA_DIR)
        finally:
            os.chdir(previous)
        self.assertEqual(DEFAULT_DATA_DIR, WEB_DATA_DIR)


if __name__ == '__main__':
    unittest.main()
