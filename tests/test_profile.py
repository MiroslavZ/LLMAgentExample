"""Персонализация и изоляция её настроек от памяти без сетевых запросов.

Подставная модель проверяет передачу профиля, а не качество следования
предпочтениям настоящей LLM.
"""

import json
import sqlite3
import tempfile
import unittest
from contextlib import closing, contextmanager
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from unittest.mock import patch

from llm_agent.agent import Agent, FACTS_SYSTEM, META_PROMPT_SYSTEM, SUMMARY_SYSTEM
from llm_agent.history import HistoryManager
from llm_agent.memory import MemorySnapshot, MemoryStorageError
from llm_agent.models import RequestOptions, Turn
from llm_agent.profile import PROFILE_FIELDS, ProfileStorageError, ProfileStore, UserProfile
from llm_agent.service import ConversationService
from llm_agent.storage import ConversationBusyError, ConversationStorageError
from tests.helpers import completion


def response_with_text(text):
    response = completion()
    response.choices[0].message.content = text
    return response


def profile_payload(messages):
    start, end = "<user_profile>\n", "\n</user_profile>"
    matches = [message for message in messages if start in message["content"]]
    if len(matches) != 1:
        raise AssertionError(f"Ожидался один блок профиля, получено: {len(matches)}")
    if matches[0]["role"] != "system":
        raise AssertionError("Профиль должен передаваться системным сообщением")
    return json.loads(matches[0]["content"].split(start, 1)[1].split(end, 1)[0])


class UserProfileTests(unittest.TestCase):
    def test_json_roundtrip_preserves_all_preferences_and_returns_detached_dict(self):
        profile = UserProfile(
            id="student-1", name="Ученик", address="Мирослав", language="русский",
            style="Простыми словами", format="Нумерованные шаги",
            constraints="Без готового решения", context="Изучаю Python\nПервый проект",
        )
        encoded = json.dumps(profile.to_dict(), ensure_ascii=False)
        self.assertEqual(UserProfile.from_dict(json.loads(encoded)), profile)
        self.assertEqual(profile_payload([profile.to_message()]), profile.to_dict())
        data = profile.to_dict()
        data["language"] = "английский"
        self.assertEqual(profile.language, "русский")
        with self.assertRaises(FrozenInstanceError):
            profile.style = "Другой стиль"

    def test_missing_preferences_have_empty_defaults(self):
        profile = UserProfile.from_dict({"id": "basic", "name": "Базовый"})
        for field in PROFILE_FIELDS.keys() - {"name"}:
            with self.subTest(field=field):
                self.assertEqual(getattr(profile, field), "")

    def test_invalid_fields_and_unknown_json_keys_are_rejected(self):
        invalid_ids = [None, 1, "", " ", "русский", "two words", "../profile", "x" * 65]
        invalid_data = [None, [], {}, {"id": "valid"}, {"id": "valid", "name": "  "}]
        invalid_data += [{"id": value, "name": "Профиль"} for value in invalid_ids]
        invalid_data += [
            {"id": "valid", "name": "Профиль", field: value}
            for field in PROFILE_FIELDS for value in (None, 1, [], {})
        ]
        invalid_data.append({"id": "valid", "name": "Профиль", "unknown": "лишнее"})
        for data in invalid_data:
            with self.subTest(data=data), self.assertRaises(ValueError):
                UserProfile.from_dict(data)
        self.assertEqual(UserProfile("A_0-" + "x" * 60, "Имя").id, "A_0-" + "x" * 60)


class ProfileStoreTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "profiles.sqlite3"
        self.store = ProfileStore(self.path)
        self.student = UserProfile("student", "Ученик", style="Объясняй по шагам")
        self.expert = UserProfile("expert", "Эксперт", style="Кратко и технически")

    def save_profiles(self):
        self.store.save(self.student)
        self.store.save(self.expert)

    def test_profiles_and_independent_selections_survive_restart(self):
        self.save_profiles()
        self.store.select("first", self.student.id)
        self.store.select("second", self.expert.id)
        restarted = ProfileStore(self.path)
        self.assertEqual(set(restarted.list()), {self.student, self.expert})
        self.assertEqual(restarted.selected("first"), self.student)
        self.assertEqual(restarted.selected("second"), self.expert)
        self.assertIsNone(restarted.selected("new"))
        restarted.select("first", None)
        self.assertIsNone(self.store.selected("first"))
        self.assertEqual(self.store.selected("second"), self.expert)
        self.assertEqual(self.store.get(self.student.id), self.student)

    def test_shared_profile_update_keeps_old_snapshot_and_delete_detaches_all_scopes(self):
        self.save_profiles()
        for scope in ("first", "second"):
            self.store.select(scope, self.student.id)
        self.store.select("third", self.expert.id)
        snapshot = self.store.selected("first")
        updated = replace(self.student, style="Подробно с примерами")
        ProfileStore(self.path).save(updated, expected=self.student)
        self.assertEqual(snapshot, self.student)
        self.assertEqual(self.store.selected("first"), updated)
        self.assertEqual(self.store.selected("second"), updated)
        self.store.delete(self.student.id)
        self.assertIsNone(self.store.selected("first"))
        self.assertIsNone(self.store.selected("second"))
        self.assertEqual(self.store.selected("third"), self.expert)
        self.assertEqual(self.store.list(), [self.expert])

    def test_stale_edit_does_not_overwrite_updated_or_deleted_profile(self):
        self.store.save(self.student)
        updated = replace(self.student, style="Изменено в другой вкладке")
        other_store = ProfileStore(self.path)
        other_store.save(updated, expected=self.student)
        stale = replace(self.student, language="русский")
        with self.assertRaisesRegex(ValueError, "другой вкладке"):
            self.store.save(stale, expected=self.student)
        self.assertEqual(self.store.get(self.student.id), updated)
        other_store.delete(self.student.id)
        with self.assertRaisesRegex(ValueError, "другой вкладке"):
            self.store.save(stale, expected=self.student)
        self.assertEqual(self.store.list(), [])

    def test_invalid_scope_or_missing_profile_does_not_replace_selection(self):
        for scope in (None, "", " "):
            for operation in (lambda: self.store.select(scope, "student"), lambda: self.store.selected(scope)):
                with self.subTest(scope=scope), self.assertRaises(ValueError):
                    operation()
        with self.assertRaises(ValueError):
            self.store.save({"id": "student", "name": "Ученик"})
        self.assertFalse(self.path.exists())
        self.store.save(self.student)
        self.store.select("first", self.student.id)
        for operation in (
            lambda: self.store.select("first", "missing"),
            lambda: self.store.get("missing"), lambda: self.store.delete("missing"),
        ):
            with self.assertRaises(ValueError):
                operation()
        self.assertEqual(self.store.selected("first"), self.student)

    def test_corrupt_database_is_reported_without_overwriting_it(self):
        original = "Повреждённая база профилей".encode("utf-8")
        self.path.write_bytes(original)
        for operation in (self.store.list, lambda: self.store.selected("first")):
            with self.assertRaises(ProfileStorageError):
                operation()
            self.assertEqual(self.path.read_bytes(), original)

    def test_corrupt_profile_or_dangling_selection_is_not_silently_disabled(self):
        self.store.save(self.student)
        self.store.select("first", self.student.id)
        for data in ("{bad", "null", '{"id":"wrong","name":"Имя"}', '{"id":"student","name":2}'):
            with self.subTest(data=data):
                with closing(sqlite3.connect(self.path)) as connection, connection:
                    connection.execute("UPDATE profiles SET data = ?", (data,))
                original = self.path.read_bytes()
                for operation in (
                    self.store.list, lambda: self.store.get(self.student.id),
                    lambda: self.store.selected("first"), lambda: self.store.select("second", self.student.id),
                ):
                    with self.assertRaises(ProfileStorageError):
                        operation()
                    self.assertEqual(self.path.read_bytes(), original)
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute("DELETE FROM profiles")  # Нарушение FK внешним редактором.
        with self.assertRaises(ProfileStorageError):
            self.store.selected("first")

    def test_unsupported_database_version_is_preserved(self):
        self.store.save(self.student)
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute("PRAGMA user_version = 999")
        original = self.path.read_bytes()
        with self.assertRaisesRegex(ProfileStorageError, "версия"):
            self.store.list()
        self.assertEqual(self.path.read_bytes(), original)


class AgentProfileTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.profile = UserProfile("student", "Ученик", language="русский", constraints="Без спойлеров")
        self.create = self.enterContext(patch("llm_agent.agent.OpenAI")).return_value.chat.completions.create
        self.create.return_value = completion()

    def assert_no_profile(self, value):
        self.assertNotIn("<user_profile>", json.dumps(value, ensure_ascii=False))

    def test_absent_profile_keeps_existing_request_and_invalid_profile_is_rejected(self):
        agent = Agent("тестовый-ключ", self.directory / "plain.json")
        agent.request("Вопрос", system="Базовые правила")
        self.assertEqual(self.create.call_args.kwargs["messages"], [
            {"role": "system", "content": "Базовые правила"},
            {"role": "user", "content": "Вопрос"},
        ])
        with self.assertRaises(ValueError):
            Agent("тестовый-ключ", self.directory / "invalid.json", profile={"id": "student"})

    def test_all_context_strategies_add_profile_once_per_answer_and_keep_history_clean(self):
        strategies = {
            "full": {}, "window": {"strategy": "window", "window_size": 1},
            "facts": {"strategy": "facts", "window_size": 1},
            "summary": {"last_messages": 0, "compress_every": 2},
            "branch": {"strategy": "branch"},
        }

        def respond(**options):
            system = options["messages"][0]["content"]
            if system == FACTS_SYSTEM:
                return response_with_text('{"цель":"Изучить Python"}')
            if system == SUMMARY_SYSTEM:
                return response_with_text("Пользователь изучает Python")
            return completion()

        self.create.side_effect = respond
        for name, options in strategies.items():
            with self.subTest(strategy=name):
                self.create.reset_mock()
                path = self.directory / f"{name}.json"
                memory = MemorySnapshot(working={"цель": "Изучить Python"}, long_term={"редактор": "IDE"})
                agent = Agent("тестовый-ключ", path, profile=self.profile, memory=memory, **options)
                agent.request("Первый вопрос", system="Базовые правила")
                agent.request("Второй вопрос")
                answer_calls = []
                for call in self.create.call_args_list:
                    messages = call.kwargs["messages"]
                    if messages[0]["content"] in (FACTS_SYSTEM, SUMMARY_SYSTEM):
                        self.assert_no_profile(messages)
                        continue
                    answer_calls.append(call)
                    self.assertEqual(profile_payload(messages), self.profile.to_dict())
                    self.assertEqual(messages[0], {"role": "system", "content": "Базовые правила"})
                    self.assertEqual(messages[1], self.profile.to_message())
                    self.assertTrue(any("<working_memory>" in message["content"] for message in messages))
                    self.assertTrue(any("<long_term_memory>" in message["content"] for message in messages))
                self.assertEqual(len(answer_calls), 2)
                self.assert_no_profile(agent.history.get_messages())
                self.assert_no_profile(json.loads(path.read_text(encoding="utf-8")))

    def test_meta_stages_get_profile_but_intermediate_summary_does_not(self):
        path = self.directory / "meta.json"
        agent = Agent("тестовый-ключ", path, profile=self.profile, last_messages=0, compress_every=2)
        self.create.side_effect = [
            response_with_text("Подготовленный промпт"), response_with_text("Краткая история"), completion(),
        ]
        agent.request_with_meta_prompt("Объясни Python", system="Базовые правила", response_format="object")
        meta, compressed, answer = self.create.call_args_list
        for call in (meta, answer):
            self.assertEqual(profile_payload(call.kwargs["messages"]), self.profile.to_dict())
        self.assertEqual(meta.kwargs["messages"][0]["content"], META_PROMPT_SYSTEM)
        self.assertEqual(meta.kwargs["response_format"], {"type": "text"})
        self.assertEqual(answer.kwargs["messages"][0]["content"], "Базовые правила")
        self.assertEqual(answer.kwargs["response_format"], {"type": "json_object"})
        self.assertEqual(compressed.kwargs["messages"][0]["content"], SUMMARY_SYSTEM)
        self.assert_no_profile(compressed.kwargs["messages"])
        self.assert_no_profile(json.loads(path.read_text(encoding="utf-8")))

    def test_clearing_history_preserves_profile_for_next_request(self):
        path = self.directory / "clear.json"
        agent = Agent("тестовый-ключ", path, profile=self.profile)
        agent.request("Первый вопрос")
        agent.history.clear()
        self.assertEqual(HistoryManager(path).get_messages(), [])
        agent.request("Новый вопрос")
        self.assertEqual(profile_payload(self.create.call_args.kwargs["messages"]), self.profile.to_dict())


class ServiceProfileTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.service = ConversationService(self.directory, token="тестовый-ключ")
        self.conversation = self.service.create()
        self.profile = UserProfile("student", "Ученик", language="русский", style="Подробно")
        self.service.save_profile(self.conversation.id, self.profile)
        self.service.select_profile(self.conversation.id, self.profile.id)
        self.create = self.enterContext(patch("llm_agent.agent.OpenAI")).return_value.chat.completions.create
        self.create.return_value = completion()

    def test_service_reloads_profile_on_each_request_and_does_not_mix_profiles(self):
        other = self.service.create()
        expert = UserProfile("expert", "Эксперт", language="English", style="Кратко")
        self.service.save_profile(other.id, expert)
        self.service.select_profile(other.id, expert.id)
        self.service.remember_memory(self.conversation.id, "working", "цель", "Программа")
        initial_memory = self.service.get_memory(self.conversation.id)
        restarted = ConversationService(self.directory, token="тестовый-ключ")
        requests = [
            (self.conversation.id, self.profile), (other.id, expert),
        ]
        for conversation_id, profile in requests:
            result = restarted.send(conversation_id, "Что делать дальше?")
            self.assertEqual(result.turns[-1].status, "completed")
            self.assertEqual(profile_payload(self.create.call_args.kwargs["messages"]), profile.to_dict())
            self.assertNotIn("<user_profile>", json.dumps(result.working_context, ensure_ascii=False))
        changed = replace(self.profile, style="Только краткий список")
        self.service.save_profile(other.id, changed, expected=self.profile)
        restarted.send(self.conversation.id, "Что делать дальше?")
        self.assertEqual(profile_payload(self.create.call_args.kwargs["messages"]), changed.to_dict())
        restarted.select_profile(self.conversation.id, expert.id)
        restarted.send(self.conversation.id, "Что делать дальше?")
        self.assertEqual(profile_payload(self.create.call_args.kwargs["messages"]), expert.to_dict())
        restarted.select_profile(self.conversation.id, None)
        restarted.send(self.conversation.id, "Без персонализации")
        self.assertNotIn("<user_profile>", json.dumps(self.create.call_args.kwargs["messages"]))
        self.assertEqual(self.service.get_memory(self.conversation.id), initial_memory)

    def test_meta_uses_one_snapshot_despite_profile_edit_from_other_conversation(self):
        other = self.service.create()
        self.service.select_profile(other.id, self.profile.id)
        changed = replace(self.profile, language="English")

        def respond(**options):
            if options["messages"][0]["content"] == META_PROMPT_SYSTEM:
                self.service.save_profile(other.id, changed, expected=self.profile)
                return response_with_text("Подготовленный промпт")
            return completion()

        self.create.side_effect = respond
        result = self.service.send(self.conversation.id, "Вопрос", options=RequestOptions(meta_prompt=True))
        self.assertEqual(result.turns[-1].status, "completed")
        self.assertEqual(self.create.call_count, 2)
        for call in self.create.call_args_list:
            self.assertEqual(profile_payload(call.kwargs["messages"]), self.profile.to_dict())
        self.assertEqual(self.service.get_profile(self.conversation.id), changed)
        self.service.send(self.conversation.id, "Следующий вопрос")
        self.assertEqual(profile_payload(self.create.call_args.kwargs["messages"]), changed.to_dict())

    def test_profile_mutations_reject_busy_and_missing_conversations(self):
        busy = self.service.get(self.conversation.id)
        busy.started = True
        busy.turns.append(Turn(user="Запрос в работе"))
        self.service.store.save(busy)
        for conversation_id, error in ((busy.id, ConversationBusyError), ("a" * 32, KeyError)):
            operations = [
                lambda: self.service.select_profile(conversation_id, None),
                lambda: self.service.save_profile(conversation_id, replace(self.profile, style="Изменён")),
                lambda: self.service.delete_profile(conversation_id, self.profile.id),
            ]
            for operation in operations:
                with self.subTest(conversation=conversation_id), self.assertRaises(error):
                    operation()
        self.assertEqual(self.service.get_profile(busy.id), self.profile)
        self.assertEqual(self.service.profiles.list(), [self.profile])
        with self.assertRaises(KeyError):
            self.service.get_profile("a" * 32)

    def test_corrupt_profiles_produce_visible_error_without_model_call(self):
        original = "Повреждённый файл профилей".encode("utf-8")
        self.service.profiles.path.write_bytes(original)
        result = self.service.send(self.conversation.id, "Вопрос")
        self.assertEqual(result.turns[-1].status, "error")
        self.assertIn("профил", result.turns[-1].error)
        self.assertIsNone(result.turns[-1].answer)
        self.assertEqual(self.service.get(result.id).turns[-1].status, "error")
        self.assertEqual(self.service.profiles.path.read_bytes(), original)
        self.create.assert_not_called()

    def test_deleting_conversation_clears_only_its_selection_and_keeps_profiles(self):
        other = self.service.create()
        self.service.select_profile(other.id, self.profile.id)
        self.service.delete(self.conversation.id)
        self.assertIsNone(self.service.profiles.selected(self.conversation.id))
        self.assertEqual(self.service.get_profile(other.id), self.profile)
        self.assertEqual(self.service.profiles.list(), [self.profile])
        self.assertIsNone(self.service.get_profile(self.service.create().id))

    def test_failed_json_deletion_rolls_back_selection_and_memory(self):
        self.service.remember_memory(self.conversation.id, "working", "цель", "Агент")
        original = self.service.get_memory(self.conversation.id)
        with patch.object(self.service.store, "delete", side_effect=ConversationStorageError("Отказ удаления")):
            with self.assertRaises(ConversationStorageError):
                self.service.delete(self.conversation.id)
        self.assertEqual(self.service.get_profile(self.conversation.id), self.profile)
        self.assertEqual(self.service.get_memory(self.conversation.id), original)
        self.assertEqual(self.service.get(self.conversation.id), self.conversation)

    def test_failed_commit_restores_conversation_selection_and_memory(self):
        self.service.remember_memory(self.conversation.id, "working", "цель", "Агент")
        original_memory = self.service.get_memory(self.conversation.id)
        for target, error in ((self.service.profiles, ProfileStorageError), (self.service.memory, MemoryStorageError)):
            original_connection = target._connection

            @contextmanager
            def failing_commit():
                with original_connection() as connection:
                    yield connection
                    raise sqlite3.OperationalError("Отказ фиксации транзакции")

            with self.subTest(target=type(target).__name__):
                with patch.object(target, "_connection", failing_commit):
                    with self.assertRaises(error):
                        self.service.delete(self.conversation.id)
                self.assertEqual(self.service.get(self.conversation.id), self.conversation)
                self.assertEqual(self.service.get_profile(self.conversation.id), self.profile)
                self.assertEqual(self.service.get_memory(self.conversation.id), original_memory)

    def test_corrupt_profiles_prevent_conversation_deletion(self):
        path = self.service.store.path(self.conversation.id)
        original = path.read_bytes()
        damaged = b"broken profile database"
        self.service.profiles.path.write_bytes(damaged)
        with self.assertRaises(ProfileStorageError):
            self.service.delete(self.conversation.id)
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(self.service.profiles.path.read_bytes(), damaged)


if __name__ == "__main__":
    unittest.main()
