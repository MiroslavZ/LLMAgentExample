"""Профили в NiceGUI: реальные элементы, временное хранилище, без API."""

import asyncio
import inspect
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch
from uuid import uuid4

from nicegui import Client, core, ui

from llm_agent.models import Turn
from llm_agent.profile import PROFILE_FIELDS, ProfileStorageError, UserProfile
from llm_agent.service import ConversationService
from llm_agent.web.jobs import RequestRunner
from llm_agent.web.page import ChatPage


class ProfilePageTests(unittest.IsolatedAsyncioTestCase):
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
        self.notify = self.enterContext(patch("llm_agent.web.profiles.ui.notify"))
        self.enterContext(patch("llm_agent.agent.OpenAI",
                                side_effect=AssertionError("API запрещён в тестах UI")))
        self.enterContext(patch.object(core, "loop", asyncio.get_running_loop()))
        self.client = Client(ui.page("/profile-unit-test"))
        self.addCleanup(self.client.delete)

    def build_page(self):
        page = ChatPage(self.service, self.runner, token_available=True,
                        conversation_id=self.conversation.id)
        page.build()
        page.profile_panel.open()
        return page

    def add_profile(self, name="Новичок", **fields):
        profile = UserProfile(id=uuid4().hex, name=name, **fields)
        self.service.profiles.save(profile)
        return profile

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

    async def test_create_choose_and_clear_profile_through_controls(self):
        with self.client:
            page = self.build_page()
        panel = page.profile_panel
        self.assertIsNone(panel.selector.value)
        self.assertEqual(set(panel.fields), set(PROFILE_FIELDS))
        await self.click_button(panel.new_button)
        with self.client:
            panel.fields["name"].set_value("Новичок")
            panel.fields["language"].set_value("Русский")
            panel.fields["style"].set_value("Объяснять простыми словами")
            panel.fields["format"].set_value("Пошаговый список")
            panel.fields["constraints"].set_value("Без платных инструментов")
        await self.click_button(panel.save_button)
        profile, = self.service.profiles.list()
        self.assertFalse(panel.editor.visible)
        self.assertEqual(profile.style, "Объяснять простыми словами")
        self.assertEqual(panel.selector.options[profile.id], "Новичок")
        with self.client:
            panel.selector.set_value(profile.id)
        self.assertEqual(self.service.get_profile(self.conversation.id), profile)
        self.assertEqual(panel.summary.text, "Новичок")
        self.assertIn("Без платных инструментов", self.labels(panel.details))
        with self.client:
            panel.selector.set_value(None)
        self.assertIsNone(self.service.get_profile(self.conversation.id))
        self.assertEqual(panel.summary.text, "Профиль не выбран")
        self.assertEqual(self.service.profiles.list(), [profile])

    async def test_selection_is_per_dialogue_and_dialog_closes_on_navigation(self):
        other = self.service.create()
        first = self.add_profile()
        second = self.add_profile("Эксперт", style="Кратко и технически")
        self.service.select_profile(self.conversation.id, first.id)
        self.service.select_profile(other.id, second.id)
        with self.client:
            page = self.build_page()
            panel = page.profile_panel
            self.assertEqual(panel.selector.value, first.id)
            page.select(other.id)
            self.assertFalse(panel.dialog.value)
            self.assertEqual(panel.summary.text, "Эксперт")
            panel.open()
            self.assertEqual(panel.selector.value, second.id)
            page.select(self.conversation.id)
        self.assertEqual(panel.summary.text, "Новичок")
        self.assertEqual(self.service.get_profile(other.id), second)

    async def test_edit_common_profile_updates_all_selected_dialogues(self):
        profile = self.add_profile(style="Кратко")
        other = self.service.create()
        self.service.select_profile(self.conversation.id, profile.id)
        self.service.select_profile(other.id, profile.id)
        with self.client:
            page = self.build_page()
        panel = page.profile_panel
        await self.click_button(panel.edit_button)
        self.assertEqual(panel.fields["style"].value, "Кратко")
        with self.client:
            panel.fields["style"].set_value("Подробно, с примерами")
        await self.click_button(panel.save_button)
        self.assertEqual(self.service.get_profile(other.id).style, "Подробно, с примерами")
        self.assertIn("Подробно, с примерами", self.labels(panel.details))

    async def test_refresh_keeps_editor_draft_and_rejects_stale_save(self):
        profile = self.add_profile(style="Кратко")
        self.service.select_profile(self.conversation.id, profile.id)
        with self.client:
            page = self.build_page()
            panel = page.profile_panel
            panel.edit()
            panel.fields["style"].set_value("Мой черновик")
            changed = replace(profile, style="Другая вкладка")
            self.service.profiles.save(changed, expected=profile)
            page.refresh()
            self.assertEqual(panel.fields["style"].value, "Мой черновик")
            self.assertIn("Другая вкладка", self.labels(panel.details))
            self.assertIn("изменён или удалён", panel.status.text)
            panel.save()
        self.assertEqual(self.service.profiles.get(profile.id), changed)
        self.assertTrue(panel.editor.visible)
        self.assertEqual(panel.fields["style"].value, "Мой черновик")
        self.assertEqual(self.notify.call_args.kwargs["type"], "negative")

    async def test_busy_disables_controls_and_mutation_handlers(self):
        profile = self.add_profile(style="Кратко")
        self.service.select_profile(self.conversation.id, profile.id)
        with self.client:
            page = self.build_page()
            panel = page.profile_panel
            panel.edit()
            panel.fields["style"].set_value("Черновик")
            self.runner._tasks[self.conversation.id] = Mock()
            page.refresh()
            for control in (panel.selector, panel.new_button, panel.edit_button,
                            panel.delete_button, panel.save_button, *panel.fields.values()):
                self.assertFalse(control.enabled)
            panel.save()
            panel.select(None)
            panel.confirm_delete()
            self.assertIsNone(panel.confirm_dialog)
            self.assertEqual(self.service.get_profile(self.conversation.id), profile)
            del self.runner._tasks[self.conversation.id]
            page.refresh()
        self.assertTrue(panel.save_button.enabled)
        self.assertEqual(panel.fields["style"].value, "Черновик")

    async def test_storage_failure_preserves_draft_and_read_failure_disables_edits(self):
        with self.client:
            page = self.build_page()
            panel = page.profile_panel
            panel.create()
            panel.fields["name"].set_value("Сохранить черновик")
            with patch.object(self.service, "save_profile", side_effect=ProfileStorageError("Секретные детали")):
                panel.save()
            self.notify.assert_called_with("Не удалось сохранить профиль.", type="negative")
            self.assertTrue(panel.editor.visible)
            self.assertEqual(panel.fields["name"].value, "Сохранить черновик")
            with patch.object(self.service, "get_profile", side_effect=ProfileStorageError("Секретные детали")):
                page.refresh()
            self.assertFalse(panel.save_button.enabled)
            self.assertIn("Не удалось прочитать профили", panel.status.text)
            page.refresh()
        self.assertTrue(panel.save_button.enabled)
        self.assertEqual(panel.fields["name"].value, "Сохранить черновик")

    async def test_delete_confirmation_removes_shared_profile_and_preserves_memory(self):
        profile = self.add_profile()
        other = self.service.create()
        self.service.select_profile(self.conversation.id, profile.id)
        self.service.select_profile(other.id, profile.id)
        self.service.remember_memory(self.conversation.id, "working", "цель", "Изучить Python")
        self.conversation.started = True
        self.conversation.turns = [Turn("Привет", answer="Здравствуйте", status="completed")]
        self.service.store.save(self.conversation)
        with self.client:
            page = self.build_page()
        panel = page.profile_panel
        await self.click_button(panel.delete_button)
        self.assertEqual(self.service.get_profile(other.id), profile)
        confirm = next(element for element in panel.confirm_dialog.descendants()
                       if isinstance(element, ui.button) and element.text == "Удалить")
        await self.click_button(confirm)
        self.assertFalse(panel.confirm_dialog.value)
        self.assertEqual(self.service.profiles.list(), [])
        self.assertIsNone(self.service.get_profile(other.id))
        self.assertIsNone(panel.selector.value)
        self.assertEqual(self.service.get_memory(self.conversation.id).working, {"цель": "Изучить Python"})
        self.assertEqual(self.service.get(self.conversation.id).turns, self.conversation.turns)

    async def test_invalid_profile_keeps_form_for_correction(self):
        with self.client:
            page = self.build_page()
            panel = page.profile_panel
            panel.create()
            panel.fields["style"].set_value("Кратко")
            panel.save()
        self.assertEqual(self.service.profiles.list(), [])
        self.assertTrue(panel.editor.visible)
        self.assertEqual(panel.fields["style"].value, "Кратко")
        self.assertEqual(self.notify.call_args.kwargs["type"], "negative")

    async def test_old_delete_confirmation_cannot_act_in_another_dialogue(self):
        profile = self.add_profile()
        other = self.service.create()
        self.service.select_profile(self.conversation.id, profile.id)
        self.service.select_profile(other.id, profile.id)
        with self.client:
            page = self.build_page()
        panel = page.profile_panel
        await self.click_button(panel.delete_button)
        confirmation = panel.confirm_dialog
        confirm = next(element for element in confirmation.descendants()
                       if isinstance(element, ui.button) and element.text == "Удалить")
        with self.client:
            page.select(other.id)
            panel.open()
        self.assertFalse(confirmation.value)
        await self.click_button(confirm)
        self.assertEqual(self.service.get_profile(other.id), profile)
        self.assertEqual(self.service.profiles.list(), [profile])


if __name__ == "__main__":
    unittest.main()
