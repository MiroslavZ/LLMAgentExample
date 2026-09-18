"""Редактор явной памяти: реальные элементы NiceGUI и локальное хранилище."""

import asyncio
import inspect
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from nicegui import Client, core, ui

from llm_agent.memory import MemoryStorageError
from llm_agent.models import ContextSettings, Turn
from llm_agent.service import ConversationService
from llm_agent.web.jobs import RequestRunner
from llm_agent.web.page import ChatPage


class MemoryPageTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.service = ConversationService(Path(directory.name), token="test-token")
        self.conversation = self.service.create()
        self.runner = RequestRunner(self.service)
        self.enterContext(patch("nicegui.background_tasks.create_or_defer",
                                side_effect=lambda coroutine, **_: coroutine.close()))
        self.enterContext(patch("llm_agent.web.page.ui.timer"))
        self.enterContext(patch("llm_agent.web.page.ui.navigate.history.replace"))
        self.notify = self.enterContext(patch("llm_agent.web.page.ui.notify"))
        self.enterContext(patch("llm_agent.agent.OpenAI",
                                side_effect=AssertionError("API запрещён в тестах UI")))
        self.enterContext(patch.object(core, "loop", asyncio.get_running_loop()))
        self.client = Client(ui.page("/memory-unit-test"))
        self.addCleanup(self.client.delete)

    def build_page(self):
        page = ChatPage(self.service, self.runner, token_available=True,
                        conversation_id=self.conversation.id)
        page.build()
        page.open_memory()
        return page

    @staticmethod
    def labels(container):
        return [element.text for element in container.descendants() if isinstance(element, ui.label)]

    async def click_button(self, button):
        handler = next(listener.handler for listener in button._event_listeners.values()
                       if listener.type == "click")
        with patch("nicegui.elements.button.handle_event") as dispatch:
            handler(None)
        with button.parent_slot:
            result = dispatch.call_args.args[0]()
            if inspect.isawaitable(result):
                await result

    async def test_three_layers_show_local_strategy_context_separately(self):
        self.conversation.started = True
        self.conversation.settings = ContextSettings(strategy="facts")
        self.conversation.turns = [Turn("Хочу изучить Python", answer="Начнём", status="completed")]
        self.conversation.working_context = [
            {"role": "user", "content": "Хочу изучить Python"},
            {"role": "assistant", "content": "Начнём"},
        ]
        self.service.store.save(self.conversation)
        with self.client:
            page = self.build_page()
        labels = self.labels(page.memory_dialog)
        self.assertTrue(any("обычные сообщения не добавляют" in label for label in labels))
        self.assertTrue(any("Факты текущего диалога" in label for label in labels))
        self.assertTrue(any("Хочу изучить Python" in label for label in labels))
        self.assertEqual(set(page.memory_editors), {"working", "long_term"})
        self.assertEqual(self.service.get_memory(self.conversation.id).working, {})

    async def test_working_editor_saves_replaces_and_deletes_through_buttons(self):
        with self.client:
            page = self.build_page()
            editor = page.memory_editors["working"]
            editor.key.set_value("цель")
            editor.value.set_value("Пройти курс")
        save = next(element for element in editor.key.parent_slot.children
                    if isinstance(element, ui.button) and element.text == "Сохранить запись")
        await self.click_button(save)
        self.assertEqual(self.service.get_memory(self.conversation.id).working, {"цель": "Пройти курс"})
        self.assertEqual(editor.key.value, "")
        edit = next(element for element in page.working_memory_records.descendants()
                    if isinstance(element, ui.button) and element.props.get("icon") == "edit")
        await self.click_button(edit)
        self.assertEqual(editor.key.value, "цель")
        with self.client:
            editor.value.set_value("Сделать агента")
        await self.click_button(save)
        self.assertEqual(self.service.get_memory(self.conversation.id).working, {"цель": "Сделать агента"})
        delete = next(element for element in page.working_memory_records.descendants()
                      if isinstance(element, ui.button) and element.props.get("icon") == "delete_outline")
        await self.click_button(delete)
        self.assertEqual(self.service.get_memory(self.conversation.id).working, {})

    async def test_long_term_is_shared_and_working_isolated_across_dialogues(self):
        other = self.service.create()
        with self.client:
            page = self.build_page()
            working = page.memory_editors["working"]
            working.key.set_value("цель")
            working.value.set_value("Первый проект")
            page.save_memory("working")
            long_term = page.memory_editors["long_term"]
            long_term.key.set_value("хранилище")
            long_term.value.set_value("SQLite")
            page.save_memory("long_term")
            page.select(other.id)
            self.assertFalse(page.memory_dialog.value)
            page.open_memory()
        self.assertNotIn("Первый проект", self.labels(page.working_memory_records))
        self.assertIn("SQLite", self.labels(page.long_memory_records))
        edit = next(element for element in page.long_memory_records.descendants()
                    if isinstance(element, ui.button) and element.props.get("icon") == "edit")
        await self.click_button(edit)
        editor = page.memory_editors["long_term"]
        self.assertEqual(editor.key.value, "хранилище")
        self.assertEqual(editor.value.value, "SQLite")
        with self.client:
            editor.value.set_value("PostgreSQL")
            page.save_memory("long_term")
        self.assertEqual(self.service.get_memory(self.conversation.id).long_term, {"хранилище": "PostgreSQL"})
        delete = next(element for element in page.long_memory_records.descendants()
                      if isinstance(element, ui.button) and element.props.get("icon") == "delete_outline")
        await self.click_button(delete)
        memory = self.service.get_memory(self.conversation.id)
        self.assertEqual(memory.working, {"цель": "Первый проект"})
        self.assertEqual(memory.long_term, {})

    async def test_busy_request_disables_edits_and_handlers(self):
        self.service.remember_memory(self.conversation.id, "working", "цель", "Сохранить")
        with self.client:
            page = self.build_page()
            editor = page.memory_editors["working"]
            editor.key.set_value("новая")
            editor.value.set_value("Нельзя сохранить")
            self.runner._tasks[self.conversation.id] = Mock()
            page.refresh()
            self.assertFalse(editor.key.enabled)
            self.assertTrue(all(not element.enabled for element in page.working_memory_records.descendants()
                                if isinstance(element, ui.button)))
            page.save_memory("working")
            page.delete_memory("working", "цель")
            page.clear_working_memory()
            self.assertEqual(self.service.get_memory(self.conversation.id).working, {"цель": "Сохранить"})
            del self.runner._tasks[self.conversation.id]
            page.refresh()
            self.assertTrue(editor.key.enabled)
            self.assertEqual(editor.value.value, "Нельзя сохранить")

    async def test_refresh_reads_other_tab_changes_without_erasing_editor(self):
        with self.client:
            page = self.build_page()
            editor = page.memory_editors["working"]
            editor.key.set_value("черновик")
            editor.value.set_value("Не терять")
            self.service.remember_memory(self.conversation.id, "working", "извне", "Другая вкладка")
            page.refresh()
        self.assertIn("Другая вкладка", self.labels(page.working_memory_records))
        self.assertEqual(editor.key.value, "черновик")
        self.assertEqual(editor.value.value, "Не терять")

    async def test_storage_failure_preserves_editor_and_read_failure_disables_it(self):
        with self.client:
            page = self.build_page()
            editor = page.memory_editors["working"]
            editor.key.set_value("цель")
            editor.value.set_value("Не терять")
            with patch.object(self.service, "remember_memory", side_effect=MemoryStorageError("Секретные детали")):
                page.save_memory("working")
            self.assertEqual(editor.value.value, "Не терять")
            self.notify.assert_called_with("Не удалось сохранить память.", type="negative")
            with patch.object(self.service, "get_memory", side_effect=MemoryStorageError("Секретные детали")):
                page.refresh()
            self.assertFalse(editor.key.enabled)
            self.assertIn("Не удалось прочитать память", page.memory_status.text)
            page.refresh()
            self.assertTrue(editor.key.enabled)
            self.assertEqual(editor.value.value, "Не терять")

    async def test_clear_working_keeps_transcript_and_long_term(self):
        self.service.remember_memory(self.conversation.id, "working", "цель", "Текущая задача")
        self.service.remember_memory(self.conversation.id, "long_term", "язык", "Русский")
        self.conversation.started = True
        self.conversation.turns = [Turn("Привет", answer="Здравствуйте", status="completed")]
        self.service.store.save(self.conversation)
        with self.client:
            page = self.build_page()
            page.clear_working_memory()
        memory = self.service.get_memory(self.conversation.id)
        self.assertEqual(memory.working, {})
        self.assertEqual(memory.long_term, {"язык": "Русский"})
        self.assertEqual(self.service.get(self.conversation.id).turns, self.conversation.turns)


if __name__ == "__main__":
    unittest.main()
