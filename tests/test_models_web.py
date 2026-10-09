"""Выбор LLM и менеджер моделей: реальные элементы NiceGUI без сети."""

import asyncio
import inspect
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

from nicegui import Client, core, ui

from llm_agent.llm_models import LLMModel, ModelConnectionError, ModelStorageError
from llm_agent.models import Turn
from llm_agent.service import ConversationService
from llm_agent.web.components import render_turn
from llm_agent.web.jobs import RequestRunner
from llm_agent.web.page import ChatPage


class ModelPageTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        directory = self.enterContext(tempfile.TemporaryDirectory())
        self.service = ConversationService(Path(directory), token="legacy-token")
        self.runner = RequestRunner(self.service)
        self.enterContext(patch("nicegui.background_tasks.create_or_defer",
                                side_effect=lambda coroutine, **_: coroutine.close()))
        self.enterContext(patch("llm_agent.web.page.ui.timer"))
        self.enterContext(patch("llm_agent.web.page.ui.navigate.history.replace"))
        self.notify = self.enterContext(patch("llm_agent.web.models.ui.notify"))
        self.enterContext(patch("llm_agent.agent.OpenAI", side_effect=AssertionError("Сеть запрещена в UI-тестах")))
        self.enterContext(patch.object(core, "loop", asyncio.get_running_loop()))
        self.client = Client(ui.page("/models-unit-test"))
        self.addCleanup(self.client.delete)
        with self.client:
            self.page = ChatPage(self.service, self.runner, token_available=True)
            self.page.build()
        self.panel = self.page.models_panel

    @staticmethod
    def labels(container):
        return [element.text for element in container.descendants() if isinstance(element, ui.label)]

    async def click(self, button):
        handler = next(listener.handler for listener in button._event_listeners.values() if listener.type == "click")
        with patch("nicegui.elements.button.handle_event") as dispatch:
            handler(None)
        with button.parent_slot:
            result = dispatch.call_args.args[0]()
            if inspect.isawaitable(result):
                await result

    def add_model(self, *, model_id="gemma", token="", name=None):
        model = LLMModel(uuid4().hex, name or model_id, model_id, "http://localhost:1234/v1", token=token)
        self.service.models.save(model)
        with self.client:
            self.page.refresh()
        return model

    async def test_empty_catalog_does_not_use_legacy_token_and_opens_manager(self):
        self.assertEqual(self.service.models.list(), [])
        self.assertFalse(self.page.send_button.enabled)
        self.assertFalse(self.page.model_selector.enabled)
        self.assertIsNone(self.page.model_selector.value)
        self.assertTrue(any("Добавьте модель" in text for text in self.labels(self.page.banner)))
        button = next(element for element in self.client.elements.values()
                      if isinstance(element, ui.button) and element.props.get("aria-label") == "Менеджер моделей")
        await self.click(button)
        self.assertTrue(self.panel.dialog.value)
        self.assertFalse(self.panel.edit_button.enabled)
        self.assertFalse(self.panel.import_button.enabled)

    async def test_manual_add_defaults_name_and_explicit_selection_preserves_drafts(self):
        with self.client:
            self.page.user_input.set_value("Черновик сообщения")
            self.page.system_input.set_value("Черновик роли")
            self.page.temperature_input.set_value(0.7)
            self.panel.open()
            self.panel.create()
            self.panel.id_input.set_value("gemma")
            self.panel.url_input.set_value("http://localhost:1234/v1")
            self.panel.save()
            saved, = self.service.models.list()
            self.assertEqual(saved.name, "gemma")
            self.assertEqual(saved.token, "")
            self.assertIsNone(self.page.model_selector.value)
            self.assertFalse(self.page.send_button.enabled)
            self.page.model_selector.set_value(saved.id)
            self.assertTrue(self.page.send_button.enabled)
            self.assertEqual(self.service.get(self.page.conversation_id).selected_model_id, saved.id)
            self.assertEqual(self.page.user_input.value, "Черновик сообщения")
            self.assertEqual(self.page.system_input.value, "Черновик роли")
            self.assertEqual(self.page.temperature_input.value, 0.7)
            self.assertEqual(self.service.get(self.page.conversation_id).system_prompt, "")
            self.page.model_selector.set_value(None)
        self.assertIsNone(self.service.get(self.page.conversation_id).selected_model_id)
        self.assertFalse(self.page.send_button.enabled)

    async def test_model_selection_is_per_conversation_and_busy_prevents_change(self):
        first = self.add_model()
        second = self.add_model(model_id="another")
        other = self.service.create()
        with self.client:
            self.page.model_selector.set_value(first.id)
            first_conversation = self.page.conversation_id
            self.page.select(other.id)
            self.assertIsNone(self.page.model_selector.value)
            self.page.model_selector.set_value(second.id)
            self.page.select(first_conversation)
            self.assertEqual(self.page.model_selector.value, first.id)
            self.runner._tasks[first_conversation] = Mock()
            self.page.refresh()
            self.assertFalse(self.page.model_selector.enabled)
            self.page.model_selector.set_value(second.id)
            self.assertEqual(self.page.model_selector.value, first.id)
            self.assertEqual(self.service.get(first_conversation).selected_model_id, first.id)

    async def test_local_model_without_token_can_submit_and_continue_task(self):
        model = self.add_model()
        with self.client, patch.object(self.runner, "submit") as submit:
            self.page.model_selector.set_value(model.id)
            self.page.user_input.set_value("Привет")
            self.page.send()
            submit.assert_called_once()
            self.service.start_task(self.page.conversation_id, "Напиши программу")
            self.page.refresh()
            self.assertTrue(self.page.task_panel.continue_button.enabled)
            self.page.model_selector.set_value(None)
            self.assertFalse(self.page.task_panel.continue_button.enabled)
            self.page.continue_task()
            self.assertTrue(self.panel.dialog.value)
            self.assertEqual(submit.call_count, 1)

    async def test_edit_never_returns_saved_token_and_empty_field_preserves_secret(self):
        model = self.add_model(token="secret-never-in-browser")
        with self.client:
            self.panel.open()
            self.panel.edit()
            self.assertEqual(self.panel.token_input.value, "")
            self.assertNotIn(model.token, str([element._props for element in self.client.elements.values()]))
            self.panel.name_input.set_value("Моя Gemma")
            self.panel.id_input.set_value("gemma-v2")
            self.panel.url_input.set_value("http://localhost:4321/v1")
            self.panel.timeout_input.set_value(240)
            self.panel.json_input.set_value(False)
            self.panel.tools_input.set_value(False)
            self.panel.save()
            saved = self.service.models.get(model.id)
            self.assertEqual(saved, replace(model, name="Моя Gemma", model_id="gemma-v2",
                                            base_url="http://localhost:4321/v1", timeout=240,
                                            json_mode=False, tools_enabled=False))
            self.panel.edit()
            self.panel.token_input.set_value("replacement-token")
            self.panel.save()
            self.assertEqual(self.service.models.get(model.id).token, "replacement-token")
            self.assertEqual(self.panel.token_input.value, "")
            self.panel.edit()
            self.panel.remove_token.set_value(True)
            self.panel.save()
        self.assertEqual(self.service.models.get(model.id).token, "")

    async def test_stale_editor_does_not_overwrite_newer_settings(self):
        model = self.add_model(token="original-token")
        with self.client:
            self.panel.open()
            self.panel.edit()
            newer = replace(model, name="Имя из другой вкладки", token="new-token")
            self.service.models.save(newer)
            self.page.refresh()
            self.panel.name_input.set_value("Устаревшее имя")
            self.panel.save()
        self.assertEqual(self.service.models.get(model.id), newer)
        self.assertTrue(self.panel.editor.visible)
        self.assertIn("другой вкладке", self.notify.call_args.args[0])

    async def test_delete_requires_confirmation_and_preserves_chat_and_draft(self):
        model = self.add_model()
        with self.client:
            self.page.model_selector.set_value(model.id)
            self.page.user_input.set_value("Важный черновик")
            self.panel.open()
        await self.click(self.panel.delete_button)
        self.assertEqual(self.service.models.get(model.id), model)
        dialog = next(element for element in self.client.elements.values()
                      if isinstance(element, ui.dialog) and element is not self.panel.dialog and element.value)
        confirm = next(element for element in dialog.descendants()
                       if isinstance(element, ui.button) and element.text == "Удалить")
        await self.click(confirm)
        self.assertEqual(self.service.models.list(), [])
        self.assertFalse(self.page.send_button.enabled)
        self.assertIsNone(self.page.model_selector.value)
        self.assertEqual(self.page.user_input.value, "Важный черновик")
        self.assertEqual(len(self.service.list_conversations()), 1)
        self.assertTrue(any("удалена" in text for text in self.labels(self.page.banner)))

    async def test_external_catalog_changes_refresh_selector_and_send_availability(self):
        model = self.add_model()
        with self.client:
            self.page.model_selector.set_value(model.id)
            self.service.models.save(replace(model, name="Новое имя"))
            self.page.refresh()
            self.assertEqual(self.page.model_selector.options[model.id], "Новое имя")
            self.service.models.delete(model.id)
            self.page.refresh()
        self.assertFalse(self.page.send_button.enabled)
        self.assertIsNone(self.page.model_selector.value)

    async def test_discovery_uses_worker_and_imports_only_selected_models_without_duplicates(self):
        with self.client:
            self.panel.open()
            self.panel.server_url.set_value("http://localhost:1234/v1")
        with patch("llm_agent.web.models.run.io_bound", new=AsyncMock(return_value=["gemma", "qwen"])) as discover:
            await self.click(self.panel.discover_button)
        self.assertEqual(discover.await_args.args[1:], ("http://localhost:1234/v1", ""))
        self.assertEqual(self.service.models.list(), [])
        self.assertEqual(self.panel.discovered.value, [])
        with self.client:
            self.panel.discovered.set_value(["gemma"])
            self.panel.import_selected()
            saved, = self.service.models.list()
            self.assertEqual(saved.model_id, "gemma")
            self.assertEqual(saved.name, "gemma")
            self.assertIsNone(self.page.model_selector.value)
            self.service.models.save(replace(saved, name="Переименованная"))
            self.panel.import_selected()
        self.assertEqual(len(self.service.models.list()), 1)
        self.assertEqual(self.service.models.get(saved.id).name, "Переименованная")
        self.assertIn("Добавлено моделей: 0", self.panel.result_status.text)

    async def test_discovery_error_clears_previous_results_and_changed_address_requires_discovery(self):
        with self.client:
            self.panel.open()
            self.panel.server_url.set_value("http://localhost:1234/v1")
        with patch("llm_agent.web.models.run.io_bound", new=AsyncMock(return_value=["gemma"])):
            await self.click(self.panel.discover_button)
        with self.client:
            self.panel.discovered.set_value(["gemma"])
            self.panel.server_url.set_value("http://localhost:4321/v1")
            self.panel.import_selected()
        self.assertEqual(self.service.models.list(), [])
        self.assertFalse(self.panel.import_button.enabled)
        with patch("llm_agent.web.models.run.io_bound", new=AsyncMock(side_effect=ModelConnectionError("Нет соединения"))):
            await self.click(self.panel.discover_button)
        self.assertEqual(self.panel.result_status.text, "Нет соединения")
        self.assertEqual(self.panel.discovered.options, [])
        self.assertTrue(self.panel.discover_button.enabled)

    async def test_pending_discovery_allows_other_work_and_close_cancels_it(self):
        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def wait_for_server(*_):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        with self.client:
            self.panel.open()
            self.panel.server_url.set_value("http://localhost:1234/v1")
            self.panel.server_token.set_value("secret")
        with patch("llm_agent.web.models.run.io_bound", side_effect=wait_for_server):
            with self.client:
                request = asyncio.create_task(self.panel.discover())
            await asyncio.wait_for(started.wait(), 1)
            self.assertFalse(self.panel.discover_button.enabled)
            with self.client:
                self.page.user_input.set_value("Пока идёт поиск")
                self.panel.on_close()
            with self.assertRaises(asyncio.CancelledError):
                await request
        self.assertTrue(cancelled.is_set())
        self.assertEqual(self.panel.server_token.value, "")
        self.assertFalse(self.panel.import_button.enabled)
        self.assertEqual(self.page.user_input.value, "Пока идёт поиск")

    async def test_corrupt_catalog_blocks_send_and_recovers_after_refresh(self):
        model = self.add_model()
        with self.client:
            self.page.model_selector.set_value(model.id)
            self.page.user_input.set_value("Не потерять сообщение")
            with patch.object(self.service.models, "list", side_effect=ModelStorageError("Каталог недоступен")):
                self.page.refresh()
                self.assertFalse(self.page.send_button.enabled)
                self.assertIn("Каталог недоступен", self.labels(self.page.banner))
            self.page.refresh()
        self.assertTrue(self.page.send_button.enabled)
        self.assertEqual(self.page.user_input.value, "Не потерять сообщение")

    async def test_history_uses_model_snapshot_and_legacy_turn_needs_no_model(self):
        turns = [
            Turn(user="Первый", answer="Ответ", status="completed", model_name="Gemma тогда",
                 model_id="gemma", model_base_url="http://localhost:1234/v1"),
            Turn(user="Второй", answer="Ответ", status="completed", model_name="<b>Другая модель</b>",
                 model_id="other", model_base_url="https://example.com/v1"),
            Turn(user="Старая переписка", answer="Ответ", status="completed"),
        ]
        with self.client, ui.column() as transcript:
            for turn in turns:
                render_turn(turn, restore=lambda: None, busy=False)
        labels = [element for element in transcript.descendants()
                  if isinstance(element, ui.label) and "message-model" in element.classes]
        self.assertEqual([element.text for element in labels], ["Gemma тогда", "<b>Другая модель</b>"])
        tooltips = [element.text for element in transcript.descendants() if isinstance(element, ui.tooltip)]
        self.assertIn("gemma · http://localhost:1234/v1", tooltips)
        self.assertIn("other · https://example.com/v1", tooltips)


if __name__ == "__main__":
    unittest.main()
