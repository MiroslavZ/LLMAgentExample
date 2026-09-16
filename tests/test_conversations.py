import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from llm_agent.history import DialogueUsage, HistoryManager
from llm_agent.models import ContextSettings, RequestOptions, Turn
from llm_agent.service import ConversationBusyError, ConversationService, ConversationStorageError
from tests.helpers import completion


def reply(content):
    response = completion()
    response.choices[0].message.content = content
    return response


def context_messages(conversation):
    return HistoryManager._decode_data(conversation.working_context)[0]


class ConversationTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        patcher = patch("llm_agent.agent.OpenAI")
        self.openai = patcher.start()
        self.addCleanup(patcher.stop)
        self.client = self.openai.return_value
        self.create_completion = self.client.chat.completions.create
        self.create_completion.return_value = reply("Ответ")
        self.service = ConversationService(self.directory, "test-secret")
        self.conversation = self.service.create()

    def send(self, user="Вопрос", **kwargs):
        return self.service.send(self.conversation.id, user, **kwargs)

    def test_persistence_request_options_and_client_lifetime(self):
        options = RequestOptions(temperature=0.7, max_tokens=128)
        first = self.send("Первый вопрос", system_prompt="Правила", options=options)
        second = self.send("Второй вопрос")
        restored = ConversationService(self.directory, "new-secret").get(first.id)
        self.assertEqual(restored, second)
        self.assertEqual(restored.system_prompt, "Правила")
        self.assertEqual(restored.turns[0].options, options)
        self.assertTrue(all(turn.status == "completed" for turn in restored.turns))
        self.assertFalse(restored.busy)
        self.assertEqual(self.client.close.call_count, 2)
        self.openai.assert_called_with(
            api_key="test-secret", base_url="https://api.deepseek.com", timeout=60.0, max_retries=0,
        )
        self.assertEqual(self.create_completion.call_args_list[0].kwargs["temperature"], 0.7)
        self.assertEqual(self.create_completion.call_args_list[0].kwargs["max_tokens"], 128)
        saved = self.service.store.path(first.id).read_text(encoding="utf-8")
        self.assertNotIn("test-secret", saved)
        self.assertNotIn("new-secret", saved)

    def test_snapshot_mutation_does_not_change_state(self):
        snapshot = self.send()
        snapshot.turns[0].answer = "Изменено извне"
        snapshot.working_context.clear()
        self.assertEqual(self.service.get(snapshot.id).turns[0].answer, "Ответ")
        listed = self.service.list_conversations()
        listed[0].turns.clear()
        self.assertEqual(len(self.service.get(snapshot.id).turns), 1)

    def test_window_retains_full_transcript_and_usage(self):
        self.service.update_settings(self.conversation.id, ContextSettings(strategy="window", window_size=2))
        for number in range(4):
            snapshot = self.send(f"Вопрос {number}", system_prompt="Правила")
        self.assertEqual([turn.user for turn in snapshot.turns], [f"Вопрос {number}" for number in range(4)])
        messages, _, _, usage = HistoryManager._decode_data(snapshot.working_context)
        self.assertEqual(len(messages), 3)  # Системный промпт и два последних сообщения.
        self.assertEqual(usage, DialogueUsage(300, 90, 390))
        self.assertEqual(self.service.get(snapshot.id), snapshot)

    def test_summary_keeps_full_transcript_and_commits_memory_before_answer(self):
        self.service.update_settings(self.conversation.id, ContextSettings(
            strategy="summary", last_messages=0, compress_every=2,
        ))
        self.send("Первый вопрос")
        self.create_completion.side_effect = [reply("Краткая память"), reply("Следующий ответ")]
        snapshot = self.send("Второй вопрос")
        self.assertEqual([turn.user for turn in snapshot.turns], ["Первый вопрос", "Второй вопрос"])
        self.assertEqual(snapshot.working_context["summary"], "Краткая память")
        self.assertEqual(len(context_messages(snapshot)), 2)
        self.assertTrue(snapshot.turns[-1].memory_updated)
        self.assertEqual(snapshot.turns[-1].status, "completed")

    def test_facts_failure_after_memory_is_partial(self):
        self.service.update_settings(self.conversation.id, ContextSettings(strategy="facts", window_size=2))
        self.create_completion.side_effect = [reply('{"name": "Мирослав"}'), RuntimeError("secret in response")]
        snapshot = self.send("Меня зовут Мирослав")
        self.assertEqual(snapshot.turns[-1].status, "partial")
        self.assertTrue(snapshot.turns[-1].memory_updated)
        self.assertEqual(snapshot.working_context["facts"], {"name": "Мирослав"})
        self.assertIsNone(snapshot.turns[-1].answer)
        self.assertNotIn("secret", snapshot.turns[-1].error)

    def test_meta_second_stage_failure_keeps_successful_first_stage(self):
        self.create_completion.side_effect = [reply("Подготовленный промпт"), RuntimeError("Bearer test-secret")]
        snapshot = self.send("Задача", options=RequestOptions(meta_prompt=True))
        turn = snapshot.turns[-1]
        self.assertEqual(turn.meta_prompt, "Подготовленный промпт")
        self.assertEqual(turn.status, "partial")
        self.assertIsNone(turn.answer)
        self.assertEqual([message["content"] for message in context_messages(snapshot)], ["Задача", "Подготовленный промпт"])
        self.assertNotIn("test-secret", self.service.store.path(snapshot.id).read_text(encoding="utf-8"))
        self.assertEqual(ConversationService(self.directory, "test").get(snapshot.id), snapshot)
        self.client.close.assert_called_once()

    def test_meta_success_has_one_turn_and_two_context_exchanges(self):
        self.create_completion.side_effect = [reply("Промпт"), reply("Итог")]
        snapshot = self.send("Задача", options=RequestOptions(meta_prompt=True))
        self.assertEqual(len(snapshot.turns), 1)
        self.assertEqual(snapshot.turns[0].meta_prompt, "Промпт")
        self.assertEqual(snapshot.turns[0].answer, "Итог")
        self.assertEqual(snapshot.turns[0].status, "completed")
        self.assertEqual(len(context_messages(snapshot)), 4)

    def test_system_and_strategy_locked_after_failed_first_request(self):
        self.create_completion.side_effect = RuntimeError("API failed")
        snapshot = self.send(system_prompt="Правила")
        self.assertTrue(snapshot.started)
        self.assertEqual(snapshot.turns[-1].status, "error")
        with self.assertRaises(ValueError):
            self.service.update_settings(snapshot.id, ContextSettings(strategy="window"))
        with self.assertRaises(ValueError):
            self.send(system_prompt="Другие правила")
        with self.assertRaisesRegex(ValueError, "Тип стратегии фиксируется"):
            self.send(settings=ContextSettings(strategy="window"), expected_settings=snapshot.settings)
        self.assertEqual(len(self.service.get(snapshot.id).turns), 1)

    def test_initial_empty_system_is_also_locked(self):
        self.send()
        with self.assertRaises(ValueError):
            self.send(system_prompt="Новые правила")

    def test_mutable_window_parameter_applies_at_next_request(self):
        self.service.update_settings(self.conversation.id, ContextSettings(strategy="window", window_size=6))
        self.send("Первый")
        before = self.send("Второй")
        updated = self.service.update_settings(before.id, replace(before.settings, window_size=1))
        self.assertEqual(updated.working_context, before.working_context)
        snapshot = self.send("Третий")
        self.assertEqual(len(context_messages(snapshot)), 1)
        self.assertEqual(len(snapshot.turns), 3)
        self.assertEqual(self.create_completion.call_args.kwargs["messages"], [{"role": "user", "content": "Третий"}])

    def test_update_settings_rejects_stale_snapshot_without_overwriting_newer_settings(self):
        original = self.conversation.settings
        latest = self.service.update_settings(
            self.conversation.id, ContextSettings(strategy="window", window_size=5),
            expected_settings=original,
        )
        path = self.service.store.path(latest.id)
        saved = path.read_bytes()
        with self.assertRaisesRegex(ValueError, "Настройки изменились в другой вкладке"):
            self.service.update_settings(
                latest.id, ContextSettings(strategy="summary"), expected_settings=original,
            )
        self.assertEqual(path.read_bytes(), saved)
        updated = self.service.update_settings(
            latest.id, replace(latest.settings, window_size=3), expected_settings=latest.settings,
        )
        self.assertEqual(updated.settings.window_size, 3)

    def test_send_rejects_stale_settings_without_starting_or_calling_api(self):
        latest = self.service.update_settings(
            self.conversation.id, ContextSettings(strategy="window", window_size=5),
        )
        path = self.service.store.path(latest.id)
        saved = path.read_bytes()
        with self.assertRaisesRegex(ValueError, "Настройки изменились в другой вкладке"):
            self.send(settings=ContextSettings(strategy="summary"), expected_settings=self.conversation.settings)
        self.assertEqual(path.read_bytes(), saved)
        self.assertFalse(self.service.get(latest.id).started)
        self.openai.assert_not_called()

    def test_first_send_atomically_saves_settings_with_started_turn(self):
        settings = ContextSettings(strategy="window", window_size=1)
        snapshots = []
        original_save = self.service.store.save

        def capture(snapshot):
            snapshots.append(deepcopy(snapshot))
            original_save(snapshot)

        with patch.object(self.service.store, "save", side_effect=capture):
            snapshot = self.send(
                system_prompt="Правила", settings=settings,
                expected_settings=self.conversation.settings,
            )
        self.assertEqual(snapshots[0].settings, settings)
        self.assertTrue(snapshots[0].started)
        self.assertEqual(snapshots[0].system_prompt, "Правила")
        self.assertEqual(snapshots[0].turns[-1].status, "running")
        self.assertEqual(snapshot.settings, settings)
        self.assertEqual(len(context_messages(snapshot)), 2)
        with self.assertRaisesRegex(ValueError, "Тип стратегии фиксируется"):
            self.send(settings=ContextSettings(strategy="summary"), expected_settings=settings)
        self.assertEqual(self.service.get(snapshot.id), snapshot)

    def test_send_without_edited_settings_uses_latest_server_values(self):
        latest = self.service.update_settings(
            self.conversation.id, ContextSettings(strategy="window", window_size=1),
        )
        snapshot = self.send(expected_settings=self.conversation.settings)
        self.assertEqual(snapshot.settings, latest.settings)
        self.assertEqual(len(context_messages(snapshot)), 1)

    def test_invalid_send_settings_do_not_change_conversation(self):
        for settings in (ContextSettings(strategy="branch"), ContextSettings(window_size=0), "window"):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                self.send(settings=settings)
        self.assertEqual(self.service.get(self.conversation.id), self.conversation)
        self.openai.assert_not_called()

    def test_invalid_request_and_settings_do_not_start_conversation(self):
        for temperature in (float("nan"), float("inf"), -0.1, 2.1, True, "1", 10 ** 500):
            with self.subTest(temperature=temperature), self.assertRaises(ValueError):
                self.send(options=RequestOptions(temperature=temperature))
        for max_tokens in (0, -1, 1.1, True):
            with self.subTest(max_tokens=max_tokens), self.assertRaises(ValueError):
                self.send(options=RequestOptions(max_tokens=max_tokens))
        for user in ("", "  \n", None):
            with self.subTest(user=user), self.assertRaises(ValueError):
                self.send(user)
        for settings in (
            ContextSettings(strategy="branch"), ContextSettings(window_size=0),
            ContextSettings(last_messages=-1), ContextSettings(compress_every=0),
        ):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                self.service.update_settings(self.conversation.id, settings)
        snapshot = self.service.get(self.conversation.id)
        self.assertFalse(snapshot.started)
        self.assertEqual(snapshot.turns, [])
        self.openai.assert_not_called()

    def test_missing_api_key_is_safe_and_persisted(self):
        service = ConversationService(self.directory, None)
        self.assertFalse(service.token_available)
        snapshot = service.send(self.conversation.id, "Вопрос")
        self.assertEqual(snapshot.turns[-1].status, "error")
        self.assertIn("API_KEY", snapshot.turns[-1].error)
        self.assertTrue(snapshot.started)
        self.openai.assert_not_called()

    def test_restart_marks_unfinished_turn_interrupted_and_keeps_meta(self):
        snapshot = self.service.get(self.conversation.id)
        snapshot.started = True
        snapshot.turns.append(Turn(user="Задача", meta_prompt="Готовый промпт"))
        self.service.store.save(snapshot)
        restored = ConversationService(self.directory, "test").get(snapshot.id)
        self.assertEqual(restored.turns[-1].status, "interrupted")
        self.assertEqual(restored.turns[-1].meta_prompt, "Готовый промпт")
        self.assertFalse(restored.busy)

    def test_corrupt_file_is_reported_preserved_and_other_dialogues_available(self):
        broken_id = "a" * 32
        path = self.directory / f"{broken_id}.json"
        content = b"{ invalid json"
        path.write_bytes(content)
        service = ConversationService(self.directory, "test")
        self.assertEqual([item.id for item in service.list_conversations()], [self.conversation.id])
        self.assertEqual(path.read_bytes(), content)
        self.assertEqual(len(service.storage_errors), 1)
        with self.assertRaises(ConversationStorageError):
            service.get(broken_id)

    def test_atomic_save_failure_preserves_previous_file(self):
        self.send()
        snapshot = self.service.get(self.conversation.id)
        path = self.service.store.path(snapshot.id)
        original = path.read_bytes()
        snapshot.turns[-1].answer = "Новая версия"
        with patch("llm_agent.storage.os.fsync", side_effect=OSError("Disk failed")):
            with self.assertRaises(ConversationStorageError):
                self.service.store.save(snapshot)
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(list(self.directory.glob("*.tmp")), [])

    def test_persistent_disk_failure_preserves_answer_in_memory_and_ends_busy_state(self):
        original_save = self.service.store.save

        def failing_save(snapshot):
            if snapshot.turns and snapshot.turns[-1].answer is not None:
                raise ConversationStorageError("Disk failed")
            original_save(snapshot)

        with patch.object(self.service.store, "save", side_effect=failing_save):
            snapshot = self.send()
        self.assertFalse(snapshot.busy)
        self.assertEqual(snapshot.turns[-1].answer, "Ответ")
        self.assertEqual(snapshot.turns[-1].status, "partial")
        self.assertIn("только в памяти", snapshot.turns[-1].error)
        self.assertEqual(self.service.get(snapshot.id), snapshot)
        self.assertEqual(self.service.list_conversations()[0], snapshot)
        self.assertTrue(self.service.storage_errors)
        # Обычная последующая запись восстанавливает сохранность результата.
        saved = self.service.update_settings(snapshot.id, snapshot.settings)
        self.assertEqual(ConversationService(self.directory, "test").get(snapshot.id), saved)
        self.assertEqual(self.service.storage_errors, [])

    def test_context_and_transcript_success_are_in_same_snapshot(self):
        snapshots = []
        original_save = self.service.store.save

        def capture(snapshot):
            snapshots.append(deepcopy(snapshot))
            original_save(snapshot)

        with patch.object(self.service.store, "save", side_effect=capture):
            self.send(system_prompt="Правила")
        for snapshot in snapshots:
            has_response = any(item["role"] == "assistant" for item in context_messages(snapshot))
            self.assertEqual(has_response, snapshot.turns[-1].answer is not None)
        self.assertEqual(snapshots[-1].turns[-1].status, "completed")

    def test_delete_and_identifier_validation(self):
        with self.assertRaises(ValueError):
            self.service.get("../outside")
        self.service.delete(self.conversation.id)
        self.assertEqual(self.service.list_conversations(), [])
        with self.assertRaises(KeyError):
            self.service.get(self.conversation.id)

    def test_concurrent_mutations_rejected_reads_and_other_dialogues_work(self):
        entered = threading.Event()
        release = threading.Event()
        other = self.service.create()

        def respond(**kwargs):
            if kwargs["messages"][-1]["content"] == "Ожидание":
                entered.set()
                if not release.wait(timeout=5):
                    raise TimeoutError("Test worker timed out")
            return reply("Ответ")

        self.create_completion.side_effect = respond
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self.send, "Ожидание")
            try:
                self.assertTrue(entered.wait(timeout=5))
                self.assertTrue(self.service.get(self.conversation.id).busy)
                for operation in (
                    lambda: self.send("Дублирование"),
                    lambda: self.service.update_settings(self.conversation.id, ContextSettings()),
                    lambda: self.service.delete(self.conversation.id),
                ):
                    with self.assertRaises(ConversationBusyError):
                        operation()
                # Второй экземпляр не восстанавливает активный запрос как прерванный.
                another_service = ConversationService(self.directory, "test")
                self.assertTrue(another_service.get(self.conversation.id).busy)
                with self.assertRaises(ConversationBusyError):
                    another_service.send(self.conversation.id, "Дублирование из другого сервера")
                self.assertEqual(self.service.send(other.id, "Независимый запрос").turns[-1].status, "completed")
            finally:
                release.set()
            self.assertEqual(future.result(timeout=5).turns[-1].status, "completed")
        self.assertEqual(len(self.service.get(self.conversation.id).turns), 1)


if __name__ == "__main__":
    unittest.main()
