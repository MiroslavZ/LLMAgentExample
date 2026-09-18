"""Изоляция слоёв памяти и их передача модели без сетевых запросов.

Детерминированная модель в одном тесте демонстрирует передачу контекста,
а не проверяет качество ответов настоящей LLM.
"""

import json
import sqlite3
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, contextmanager, nullcontext
from pathlib import Path
from unittest.mock import patch

from llm_agent.agent import Agent, FACTS_SYSTEM, META_PROMPT_SYSTEM, SUMMARY_SYSTEM
from llm_agent.history import HistoryManager
from llm_agent.memory import (
    MemorySnapshot,
    MemoryStorageError,
    MemoryStore,
    cli_memory_scope,
)
from llm_agent.models import ContextSettings, RequestOptions, Turn
from llm_agent.service import ConversationService
from llm_agent.storage import ConversationBusyError, ConversationStorageError
from tests.helpers import completion


def response_with_text(text):
    response = completion()
    response.choices[0].message.content = text
    return response


def memory_payload(messages, layer):
    start, end = f"<{layer}_memory>\n", f"\n</{layer}_memory>"
    matches = [message["content"] for message in messages if start in message["content"]]
    if len(matches) != 1:
        raise AssertionError(f"Ожидался один блок памяти {layer}, получено: {len(matches)}")
    return json.loads(matches[0].split(start, 1)[1].split(end, 1)[0])


class MemoryStoreTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "memory.sqlite3"
        self.store = MemoryStore(self.path)

    def test_tasks_are_isolated_long_term_is_shared_and_survives_restart(self):
        self.store.remember("первая", "working", "цель", "Создать агента")
        self.store.remember("вторая", "working", "цель", "Подготовить отчёт")
        self.store.remember("первая", "long_term", "язык", "русский")
        self.store.remember("первая", "long_term", "хранилище", "SQLite")

        restarted = MemoryStore(self.path)
        first, second = restarted.snapshot("первая"), restarted.snapshot("вторая")
        self.assertEqual(first.working, {"цель": "Создать агента"})
        self.assertEqual(second.working, {"цель": "Подготовить отчёт"})
        self.assertEqual(first.long_term, second.long_term)
        self.assertEqual(first.long_term, {"язык": "русский", "хранилище": "SQLite"})
        self.assertEqual(restarted.snapshot("новая").working, {})
        with closing(sqlite3.connect(self.path)) as connection, connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM working_memory").fetchone()[0], 2)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM long_term_memory").fetchone()[0], 2)
            self.assertEqual(
                [column[1] for column in connection.execute("PRAGMA table_info(long_term_memory)")],
                ["key", "value"],
            )

    def test_explicit_update_delete_and_clear_do_not_touch_other_targets(self):
        self.store.remember("первая", "working", " цель ", " Черновик ")
        self.store.remember("первая", "working", "цель", "Готовый результат")
        self.store.remember("вторая", "working", "цель", "Другая задача")
        self.store.remember("первая", "long_term", "формат", "Кратко")
        self.store.remember("первая", "long_term", "язык", "русский")
        self.store.remember("вторая", "long_term", "формат", "Подробно")
        self.assertEqual(self.store.snapshot("первая").working, {"цель": "Готовый результат"})
        self.assertEqual(self.store.snapshot("первая").long_term, {"формат": "Подробно", "язык": "русский"})

        self.store.forget("вторая", "long_term", "формат")
        self.store.forget("первая", "working", "несуществующий")
        self.store.clear_working("первая")
        snapshot = self.store.snapshot("первая")
        self.assertEqual(snapshot.working, {})
        self.assertEqual(snapshot.long_term, {"язык": "русский"})
        self.assertEqual(self.store.snapshot("вторая").working, {"цель": "Другая задача"})
        self.store.forget("вторая", "working", "цель")
        self.assertEqual(self.store.snapshot("вторая").working, {})

    def test_invalid_targets_and_values_are_rejected_before_writing(self):
        invalid = [
            ("", "working", "ключ", "значение"),
            ("задача", "short_term", "ключ", "значение"),
            ("задача", "working", " ", "значение"),
            ("задача", "working", "ключ", " "),
            ("задача", "working", None, "значение"),
            ("задача", "working", "ключ", 12),
            ("задача", "long_term", " ", "значение"),
            ("задача", "long_term", "ключ", 12),
        ]
        for scope, layer, key, value in invalid:
            with self.subTest(scope=scope, layer=layer, key=key, value=value):
                with self.assertRaises(ValueError):
                    self.store.remember(scope, layer, key, value)
        self.assertFalse(self.path.exists())
        with self.assertRaises(ValueError):
            self.store.forget("задача", "long_term", " ")
        with self.assertRaises(ValueError):
            self.store.snapshot(" ")
        with self.assertRaises(ValueError):
            self.store.clear_working(" ")

    def test_snapshot_changes_do_not_modify_persisted_data(self):
        self.store.remember("задача", "working", "цель", "Агент")
        self.store.remember("задача", "long_term", "язык", "русский")
        snapshot = self.store.snapshot("задача")
        snapshot.working.clear()
        snapshot.long_term["язык"] = "английский"
        self.assertEqual(self.store.snapshot("задача").working, {"цель": "Агент"})
        self.assertEqual(self.store.snapshot("задача").long_term, {"язык": "русский"})

    def test_concurrent_connections_preserve_independent_updates(self):
        self.store.snapshot("задача")

        def remember(index):
            independent_store = MemoryStore(self.path)
            independent_store.remember("задача", "working", f"ключ-{index}", f"Значение {index}")
            independent_store.remember(
                "задача", "long_term", f"ключ-{index}", f"Знание {index}",
            )

        with ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(remember, range(12)))
        snapshot = self.store.snapshot("задача")
        self.assertEqual(snapshot.working, {f"ключ-{i}": f"Значение {i}" for i in range(12)})
        self.assertEqual(snapshot.long_term, {f"ключ-{i}": f"Знание {i}" for i in range(12)})

    def test_failed_transaction_rolls_back_partial_write(self):
        self.store.remember("задача", "working", "сохранено", "Исходное значение")
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute(
                "CREATE TRIGGER fail_write AFTER INSERT ON working_memory "
                "BEGIN SELECT RAISE(FAIL, 'Тестовый отказ записи'); END"
            )
        with self.assertRaises(MemoryStorageError):
            self.store.remember("задача", "working", "новое", "Не должно сохраниться")
        self.assertEqual(self.store.snapshot("задача").working, {"сохранено": "Исходное значение"})

    def test_corrupted_database_is_reported_and_not_overwritten(self):
        original = "Повреждённый файл вместо SQLite".encode("utf-8")
        self.path.write_bytes(original)
        for operation in (
            lambda: self.store.snapshot("задача"),
            lambda: self.store.remember("задача", "working", "ключ", "значение"),
            lambda: self.store.clear_working("задача"),
        ):
            with self.subTest(operation=operation):
                with self.assertRaises(MemoryStorageError):
                    operation()
                self.assertEqual(self.path.read_bytes(), original)

    def test_unsupported_database_version_is_preserved_without_migration(self):
        self.store.remember("задача", "working", "ключ", "значение")
        for version in (1, 999):
            with self.subTest(version=version):
                with closing(sqlite3.connect(self.path)) as connection, connection:
                    connection.execute(f"PRAGMA user_version = {version}")
                original = self.path.read_bytes()
                with self.assertRaisesRegex(MemoryStorageError, "версия"):
                    self.store.snapshot("задача")
                self.assertEqual(self.path.read_bytes(), original)

    def test_cli_scopes_are_stable_for_paths_and_distinct_for_branches(self):
        history = self.path.parent / "history.json"
        alias = self.path.parent / "несуществующий" / ".." / "history.json"
        self.assertEqual(cli_memory_scope(history), cli_memory_scope(alias))
        scopes = {
            cli_memory_scope(history),
            cli_memory_scope(history, "первая"),
            cli_memory_scope(history, "вторая"),
            cli_memory_scope(self.path.parent / "other.json"),
        }
        self.assertEqual(len(scopes), 4)


class AgentMemoryTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "history.json"
        self.store = MemoryStore(self.path.parent / "memory.sqlite3")
        self.store.remember("задача", "working", "цель", "Реализовать слои памяти")
        self.store.remember("задача", "long_term", "язык", "русский")
        self.snapshot = self.store.snapshot("задача")
        self.create = self.enterContext(patch("llm_agent.agent.OpenAI")).return_value.chat.completions.create
        self.create.return_value = completion()

    def agent(self, **options):
        return Agent("тестовый-ключ", self.path, memory=self.snapshot, **options)

    def assert_explicit_memory(self, messages):
        self.assertEqual(memory_payload(messages, "working"), self.snapshot.working)
        self.assertEqual(memory_payload(messages, "long_term"), {"язык": "русский"})

    def assert_no_memory_blocks(self, value):
        text = json.dumps(value, ensure_ascii=False)
        self.assertNotIn("<working_memory>", text)
        self.assertNotIn("<long_term_memory>", text)

    def test_empty_layers_do_not_add_messages(self):
        self.assertEqual(MemorySnapshot().to_messages(), [])
        agent = Agent("тестовый-ключ", self.path)
        agent.request("Вопрос", system="Правила")
        self.assertEqual(self.create.call_args.kwargs["messages"], [
            {"role": "system", "content": "Правила"},
            {"role": "user", "content": "Вопрос"},
        ])

    def test_window_discards_old_dialogue_but_keeps_explicit_memory(self):
        history = HistoryManager(self.path)
        history.add_exchange("Старое сообщение", "Старый ответ", None)
        agent = self.agent(strategy="window", window_size=1)
        agent.request("Продолжим", system="Правила")
        sent = self.create.call_args.kwargs["messages"]
        self.assert_explicit_memory(sent)
        self.assertEqual(sent[0], {"role": "system", "content": "Правила"})
        self.assertEqual([message for message in sent if message["role"] != "system"], [
            {"role": "user", "content": "Продолжим"},
        ])
        self.assertNotIn("Старое сообщение", self.path.read_text(encoding="utf-8"))
        self.assertEqual(self.store.snapshot("задача"), self.snapshot)
        self.assert_no_memory_blocks(agent.history.get_messages())

    def test_summary_does_not_receive_explicit_layers_and_answer_receives_them(self):
        agent = self.agent(last_messages=0, compress_every=2)
        agent.history.add_exchange("Старый вопрос", "Старый ответ", None)
        self.create.side_effect = [response_with_text("Сжатая история"), completion()]
        agent.request("Продолжим", system="Правила")
        compressed, answered = self.create.call_args_list
        self.assertEqual(compressed.kwargs["messages"][0]["content"], SUMMARY_SYSTEM)
        self.assert_no_memory_blocks(compressed.kwargs["messages"])
        payload = json.loads(compressed.kwargs["messages"][1]["content"])
        self.assertEqual(payload["messages"], [
            {"role": "user", "content": "Старый вопрос"},
            {"role": "assistant", "content": "Старый ответ"},
        ])
        self.assert_explicit_memory(answered.kwargs["messages"])
        self.assertTrue(any("<summary>" in message["content"] for message in answered.kwargs["messages"]))
        self.assertEqual(HistoryManager(self.path).summary, "Сжатая история")
        self.assert_no_memory_blocks(json.loads(self.path.read_text(encoding="utf-8")))
        self.assertEqual(self.store.snapshot("задача"), self.snapshot)

    def test_facts_and_user_request_are_not_automatically_promoted(self):
        agent = self.agent(strategy="facts", window_size=1)
        self.create.side_effect = [
            response_with_text('{"имя": "Мирослав", "язык": "английский"}'), completion(),
        ]
        agent.request("Меня зовут Мирослав. Запомни это навсегда", system="Правила")
        extractor, answered = self.create.call_args_list
        self.assertEqual(extractor.kwargs["messages"][0]["content"], FACTS_SYSTEM)
        self.assert_no_memory_blocks(extractor.kwargs["messages"])
        payload = json.loads(extractor.kwargs["messages"][1]["content"])
        self.assertNotIn("working", payload)
        self.assertNotIn("long_term", payload)
        self.assertEqual(payload["user_message"], "Меня зовут Мирослав. Запомни это навсегда")
        self.assert_explicit_memory(answered.kwargs["messages"])
        self.assertEqual(agent.history.facts, {"имя": "Мирослав", "язык": "английский"})
        self.assertEqual(self.store.snapshot("задача"), self.snapshot)
        self.assert_no_memory_blocks(json.loads(self.path.read_text(encoding="utf-8")))

    def test_both_meta_stages_use_same_detached_snapshot(self):
        agent = self.agent(strategy="facts", window_size=1)
        self.snapshot.working["цель"] = "Изменённая внешняя копия"

        def respond(**options):
            messages = options["messages"]
            if messages[0]["content"] == FACTS_SYSTEM:
                self.assert_no_memory_blocks(messages)
                return response_with_text("{}")
            if messages[0]["content"] == META_PROMPT_SYSTEM:
                self.store.remember("задача", "working", "цель", "Изменение во время запроса")
                return response_with_text("Сформулированный промпт")
            return completion()

        self.create.side_effect = respond
        agent.request_with_meta_prompt("Задача", system="Правила")
        extractor, meta, answer = self.create.call_args_list
        self.assertEqual(extractor.kwargs["messages"][0]["content"], FACTS_SYSTEM)
        for call in (meta, answer):
            self.assertEqual(memory_payload(call.kwargs["messages"], "working"), {"цель": "Реализовать слои памяти"})
            self.assertEqual(memory_payload(call.kwargs["messages"], "long_term"), {"язык": "русский"})
        self.assertEqual(meta.kwargs["messages"][0]["content"], META_PROMPT_SYSTEM)
        self.assertEqual(answer.kwargs["messages"][0]["content"], "Правила")
        self.assert_no_memory_blocks(agent.history.get_messages())

    def test_repeated_requests_do_not_duplicate_snapshots_into_history(self):
        agent = self.agent()
        agent.request("Первый вопрос", system="Правила")
        agent.request("Второй вопрос")
        for call in self.create.call_args_list:
            self.assert_explicit_memory(call.kwargs["messages"])
        saved = HistoryManager(self.path)
        self.assertEqual(len(saved.get_messages()), 5)
        self.assert_no_memory_blocks(saved.get_messages())
        self.assertEqual(self.store.snapshot("задача"), self.snapshot)

    def test_clearing_short_term_history_preserves_other_layers(self):
        agent = self.agent()
        agent.request("Вопрос", system="Правила")
        agent.history.clear()
        restarted = self.agent()
        self.assertEqual(restarted.history.get_messages(), [])
        restarted.request("Новый диалог")
        self.assert_explicit_memory(self.create.call_args.kwargs["messages"])
        self.assertEqual(self.store.snapshot("задача"), self.snapshot)


class ServiceMemoryTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.service = ConversationService(self.directory, token="тестовый-ключ")
        self.conversation = self.service.create()
        self.create = self.enterContext(patch("llm_agent.agent.OpenAI")).return_value.chat.completions.create
        self.create.return_value = completion()

    def remember_both_layers(self):
        self.service.remember_memory(self.conversation.id, "working", "цель", "Создать агента")
        self.service.remember_memory(
            self.conversation.id, "long_term", "язык", "русский",
        )

    def test_new_conversation_reset_delete_and_restart_keep_shared_memory(self):
        self.remember_both_layers()
        other = self.service.create()
        self.service.remember_memory(other.id, "working", "цель", "Другая задача")
        restarted = ConversationService(self.directory, token="тестовый-ключ")
        self.assertEqual(restarted.get_memory(self.conversation.id).working, {"цель": "Создать агента"})
        self.assertEqual(restarted.get_memory(other.id).long_term, {"язык": "русский"})
        restarted.clear_working_memory(self.conversation.id)
        self.assertEqual(restarted.get_memory(self.conversation.id).working, {})
        self.assertEqual(restarted.get_memory(other.id).working, {"цель": "Другая задача"})
        restarted.delete(other.id)
        self.assertEqual(restarted.memory.snapshot(other.id).working, {})
        new = restarted.create()
        self.assertEqual(restarted.get_memory(new.id).working, {})
        self.assertEqual(restarted.get_memory(new.id).long_term, {"язык": "русский"})
        with self.assertRaises(KeyError):
            restarted.get_memory(other.id)

    def test_memory_operations_validate_conversation_and_reject_busy_task(self):
        missing = "a" * 32
        with self.assertRaises(KeyError):
            self.service.remember_memory(missing, "working", "цель", "Недопустимая запись")
        self.assertEqual(self.service.memory.snapshot(missing).working, {})
        self.remember_both_layers()
        busy = self.service.get(self.conversation.id)
        busy.started = True
        busy.turns.append(Turn(user="Запрос в работе"))
        self.service.store.save(busy)
        operations = [
            lambda: self.service.remember_memory(busy.id, "working", "цель", "Другая цель"),
            lambda: self.service.forget_memory(busy.id, "working", "цель"),
            lambda: self.service.clear_working_memory(busy.id),
        ]
        for operation in operations:
            with self.subTest(operation=operation):
                with self.assertRaises(ConversationBusyError):
                    operation()
        self.assertEqual(self.service.get_memory(busy.id).working, {"цель": "Создать агента"})

    def test_service_forget_updates_shared_memory_without_affecting_working(self):
        self.remember_both_layers()
        other = self.service.create()
        self.service.forget_memory(other.id, "long_term", "язык")
        self.assertEqual(self.service.get_memory(self.conversation.id).long_term, {})
        self.assertEqual(self.service.get_memory(self.conversation.id).working, {"цель": "Создать агента"})

    def test_deterministic_fixture_changes_answer_when_memory_is_explicitly_saved(self):
        """Проверка влияния переданных messages; оценка реальной LLM здесь не выполняется."""
        def fixture_model(**options):
            messages = options["messages"]
            if not any("<working_memory>" in message["content"] for message in messages):
                return response_with_text("Уточните цель задачи и язык ответа")
            working = memory_payload(messages, "working")
            long_term = memory_payload(messages, "long_term")
            return response_with_text(f"Цель: {working['цель']}. Язык: {long_term['язык']}.")

        self.create.side_effect = fixture_model
        first = self.service.send(self.conversation.id, "Какой следующий шаг?")
        self.assertEqual(first.turns[-1].answer, "Уточните цель задачи и язык ответа")
        self.remember_both_layers()
        second = self.service.send(self.conversation.id, "Какой следующий шаг?")
        self.assertEqual(second.turns[-1].answer, "Цель: Создать агента. Язык: русский.")
        self.assertEqual(second.turns[-1].status, "completed")
        self.assertNotIn("<working_memory>", json.dumps(second.working_context, ensure_ascii=False))

    def test_send_facts_with_meta_keeps_explicit_layers_separate(self):
        self.remember_both_layers()
        self.create.side_effect = [response_with_text('{"имя": "Мирослав"}'), completion(), completion()]
        result = self.service.send(
            self.conversation.id, "Меня зовут Мирослав", options=RequestOptions(meta_prompt=True),
            settings=ContextSettings(strategy="facts", window_size=1),
        )
        self.assertEqual(result.turns[-1].status, "completed")
        self.assertEqual(result.working_context["facts"], {"имя": "Мирослав"})
        self.assertEqual(self.service.get_memory(result.id).working, {"цель": "Создать агента"})
        self.assertEqual(self.service.get_memory(result.id).long_term, {"язык": "русский"})
        for call in self.create.call_args_list[1:]:
            self.assertEqual(memory_payload(call.kwargs["messages"], "working"), {"цель": "Создать агента"})
        self.assertNotIn("<long_term_memory>", json.dumps(result.working_context, ensure_ascii=False))

    def test_corrupt_memory_returns_visible_error_without_calling_model(self):
        original = "Повреждённая память".encode("utf-8")
        self.service.memory.path.write_bytes(original)
        result = self.service.send(self.conversation.id, "Вопрос")
        self.assertEqual(result.turns[-1].status, "error")
        self.assertIn("память", result.turns[-1].error)
        self.assertIsNone(result.turns[-1].answer)
        self.create.assert_not_called()
        self.assertEqual(self.service.memory.path.read_bytes(), original)
        self.assertEqual(self.service.get(result.id).turns[-1].status, "error")

    def test_delete_preserves_conversation_when_memory_database_is_corrupted(self):
        path = self.service.store.path(self.conversation.id)
        original = path.read_bytes()
        damaged = "Повреждённая память".encode("utf-8")
        self.service.memory.path.write_bytes(damaged)
        with self.assertRaises(MemoryStorageError):
            self.service.delete(self.conversation.id)
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(self.service.memory.path.read_bytes(), damaged)
        self.assertEqual(self.service.get(self.conversation.id), self.conversation)

    def test_delete_rolls_back_working_memory_when_json_deletion_fails(self):
        self.remember_both_layers()
        original = self.service.get_memory(self.conversation.id)
        with patch.object(self.service.store, "delete", side_effect=ConversationStorageError("Отказ удаления")):
            with self.assertRaises(ConversationStorageError):
                self.service.delete(self.conversation.id)
        self.assertEqual(self.service.get_memory(self.conversation.id), original)
        self.assertEqual(self.service.get(self.conversation.id), self.conversation)

    def test_delete_restores_conversation_after_commit_failure_with_in_memory_fallback(self):
        self.remember_both_layers()
        original = self.service.get_memory(self.conversation.id)
        original_connection = self.service.memory._connection

        @contextmanager
        def failing_commit():
            with original_connection() as connection:
                yield connection
                raise sqlite3.OperationalError("Тестовый отказ фиксации транзакции")

        for restore_fails in (False, True):
            with self.subTest(restore_fails=restore_fails):
                save_failure = (
                    patch.object(self.service.store, "save", side_effect=ConversationStorageError("Отказ восстановления"))
                    if restore_fails else nullcontext()
                )
                with patch.object(self.service.memory, "_connection", failing_commit), save_failure:
                    with self.assertRaises(MemoryStorageError):
                        self.service.delete(self.conversation.id)
                self.assertEqual(self.service.get_memory(self.conversation.id), original)
                self.assertEqual(self.service.get(self.conversation.id), self.conversation)
                self.assertEqual(self.service.list_conversations(), [self.conversation])
                self.assertEqual(self.service.store.path(self.conversation.id).exists(), not restore_fails)
                self.assertEqual(bool(self.service.storage_errors), restore_fails)


if __name__ == "__main__":
    unittest.main()
