"""Управление задачей через настоящие элементы NiceGUI без вызовов модели."""

import asyncio
import inspect
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import Mock, patch

from nicegui import Client, core, ui

from llm_agent.history import DialogueUsage
from llm_agent.service import ConversationService, ConversationStorageError
from llm_agent.task_state import CONTINUE_TASK, TaskStage, TaskState
from llm_agent.web.jobs import RequestRunner
from llm_agent.web.page import ChatPage


class TaskPageTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.service = ConversationService(self.directory, token="test-token")
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
        self.client = Client(ui.page("/task-unit-test"))
        self.addCleanup(self.client.delete)

    def build_page(self, *, token_available=True):
        page = ChatPage(self.service, self.runner, token_available=token_available,
                        conversation_id=self.conversation.id)
        page.build()
        return page

    def save_task(self, task):
        conversation = self.service.get(self.conversation.id)
        conversation.working_context = {
            "messages": [], "summary": "", "archived_usage": asdict(DialogueUsage()),
            "task_state": task.to_dict(),
        }
        self.service.store.save(conversation)

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

    async def test_explicit_start_preserves_chat_draft_and_disables_meta_prompt(self):
        with self.client:
            page = self.build_page()
            panel = page.task_panel
            self.assertIsNone(page.snapshot.task_state)
            self.assertTrue(panel.new_task.visible)
            self.assertFalse(panel.progress.visible)
            page.user_input.set_value("Сохранить черновик")
            page.meta_input.set_value(True)
            panel.title_input.set_value("Разобраться в генераторах Python")
        await self.click_button(panel.start_button)
        self.assertEqual(page.snapshot.task_state.title, "Разобраться в генераторах Python")
        self.assertEqual(page.snapshot.task_state.stage, TaskStage.PLANNING)
        self.assertIn("Планирование", panel.stage.text)
        self.assertTrue(panel.progress.visible)
        self.assertFalse(panel.new_task.visible)
        self.assertEqual(page.user_input.value, "Сохранить черновик")
        self.assertFalse(page.meta_input.enabled)
        self.assertFalse(page.meta_input.value)
        self.assertEqual(self.service.get(self.conversation.id).turns, [])

    async def test_pause_and_resume_each_active_stage_preserve_progress_after_reload(self):
        states = (
            TaskState("Задача"),
            TaskState("Задача", stage=TaskStage.EXECUTION, plan=("Первый", "Второй"),
                      step=1, results=("Готов первый",), notes=("Уточнение",)),
            TaskState("Задача", stage=TaskStage.VALIDATION, plan=("Первый",),
                      step=1, results=("Готов первый",)),
        )
        with self.client:
            page = self.build_page()
        for state in states:
            with self.subTest(stage=state.stage):
                self.save_task(state)
                with self.client:
                    page.refresh()
                    page.user_input.set_value("Сохранить при паузе")
                await self.click_button(page.task_panel.pause_button)
                reloaded = ConversationService(self.directory, token="test-token")
                self.assertEqual(reloaded.get(self.conversation.id).task_state, state.pause())
                self.assertTrue(page.task_panel.resume_button.visible)
                self.assertFalse(page.task_panel.pause_button.visible)
                self.assertFalse(page.task_panel.continue_button.enabled)
                self.assertFalse(page.send_button.enabled)
                self.assertFalse(page.meta_input.enabled)
                with self.client, patch.object(self.runner, "submit") as submit:
                    page.send()
                    page.continue_task()
                submit.assert_not_called()
                await self.click_button(page.task_panel.resume_button)
                self.assertEqual(page.snapshot.task_state, state)
                self.assertTrue(page.send_button.enabled)
                self.assertTrue(page.task_panel.continue_button.enabled)
                self.assertEqual(page.user_input.value, "Сохранить при паузе")

    async def test_continue_uses_standard_submission_and_disables_controls_while_busy(self):
        self.service.start_task(self.conversation.id, "Изучить Python")

        def queue(conversation_id, *_args, **_kwargs):
            self.runner._tasks[conversation_id] = Mock()

        with self.client:
            page = self.build_page()
            page.temperature_input.set_value(0.4)
            page.max_tokens_input.set_value(512)
        with patch.object(self.runner, "submit", side_effect=queue) as submit:
            await self.click_button(page.task_panel.continue_button)
        submit.assert_called_once()
        self.assertEqual(submit.call_args.args, (self.conversation.id, CONTINUE_TASK))
        self.assertEqual(submit.call_args.kwargs["options"].temperature, 0.4)
        self.assertEqual(submit.call_args.kwargs["options"].max_tokens, 512)
        self.assertFalse(submit.call_args.kwargs["options"].meta_prompt)
        self.assertEqual(page.user_input.value, "")
        for control in (page.task_panel.start_button, page.task_panel.pause_button,
                        page.task_panel.resume_button, page.task_panel.continue_button,
                        page.task_panel.title_input, page.send_button):
            self.assertFalse(control.enabled)
        before = page.snapshot.task_state
        with self.client:
            page.task_panel.start()
            page.task_panel.pause()
            page.task_panel.resume()
        self.assertEqual(self.service.get(self.conversation.id).task_state, before)

    async def test_continue_does_not_overwrite_unsent_message(self):
        self.service.start_task(self.conversation.id, "Изучить Python")
        with self.client:
            page = self.build_page()
            page.user_input.set_value("Важное уточнение")
        with patch.object(self.runner, "submit") as submit:
            await self.click_button(page.task_panel.continue_button)
        submit.assert_not_called()
        self.assertEqual(page.user_input.value, "Важное уточнение")
        self.assertEqual(self.notify.call_args.kwargs["type"], "warning")

    async def test_progress_refresh_shows_plan_results_and_preserves_message_draft(self):
        self.service.start_task(self.conversation.id, "План урока")
        with self.client:
            page = self.build_page()
            page.user_input.set_value("Мой черновик")
            state = TaskState("План урока", stage=TaskStage.EXECUTION,
                              plan=("Введение", "Примеры"), step=1,
                              results=("Готовое введение",), notes=("Для новичков",))
            self.save_task(state)
            page.refresh()
        panel = page.task_panel
        self.assertIn("Выполнение", panel.stage.text)
        self.assertIn("Примеры", panel.current_step.text)
        self.assertIn(state.expected_action, panel.expected_action.text)
        self.assertIn("✓ Введение", self.labels(panel.details))
        self.assertIn("2. Примеры", self.labels(panel.details))
        self.assertIn("1. Готовое введение", self.labels(panel.details))
        self.assertIn("Для новичков", self.labels(panel.details))
        self.assertEqual(page.user_input.value, "Мой черновик")

    async def test_done_allows_chat_and_new_task_and_shows_validation(self):
        completed = TaskState("Первая", stage=TaskStage.DONE, plan=("Шаг",),
                              step=1, results=("Результат",), validation="Проверено")
        self.save_task(completed)
        with self.client:
            page = self.build_page()
            panel = page.task_panel
            self.assertTrue(panel.new_task.visible)
            self.assertFalse(panel.continue_button.visible)
            self.assertTrue(panel.pause_button.visible)
            self.assertFalse(panel.resume_button.visible)
            self.assertTrue(page.send_button.enabled)
            self.assertTrue(page.meta_input.enabled)
            self.assertIn("Проверено", self.labels(panel.details))
            panel.title_input.set_value("Следующая задача")
        await self.click_button(panel.start_button)
        self.assertEqual(page.snapshot.task_state, TaskState("Следующая задача"))
        self.assertFalse(page.meta_input.enabled)

    async def test_done_task_pause_blocks_chat_until_resume(self):
        completed = TaskState("Готовая", stage=TaskStage.DONE, plan=("Шаг",),
                              step=1, results=("Результат",), validation="Проверено")
        self.save_task(completed)
        with self.client:
            page = self.build_page()
            page.user_input.set_value("Обсудить результат")
        panel = page.task_panel
        await self.click_button(panel.pause_button)
        self.assertEqual(page.snapshot.task_state, completed.pause())
        self.assertTrue(panel.resume_button.visible)
        self.assertFalse(panel.pause_button.visible)
        self.assertFalse(panel.continue_button.visible)
        self.assertFalse(page.send_button.enabled)
        self.assertFalse(page.meta_input.enabled)
        self.assertTrue(panel.start_button.enabled)
        with self.client, patch.object(self.runner, "submit") as submit:
            page.send()
        submit.assert_not_called()
        await self.click_button(panel.resume_button)
        self.assertEqual(page.snapshot.task_state, completed)
        self.assertTrue(page.send_button.enabled)
        self.assertTrue(page.meta_input.enabled)
        self.assertEqual(page.user_input.value, "Обсудить результат")

    async def test_switching_dialogues_preserves_separate_task_title_drafts(self):
        other = self.service.create()
        with self.client:
            page = self.build_page()
            panel = page.task_panel
            panel.title_input.set_value("Первая цель")
            page.select(other.id)
            self.assertEqual(panel.title_input.value, "")
            panel.title_input.set_value("Вторая цель")
            page.select(self.conversation.id)
            self.assertEqual(panel.title_input.value, "Первая цель")
            panel.start()
            page.select(other.id)
            self.assertIsNone(page.snapshot.task_state)
            self.assertTrue(page.meta_input.enabled)
            self.assertEqual(panel.title_input.value, "Вторая цель")
            page.select(self.conversation.id)
        self.assertEqual(panel.title.text, "Первая цель")
        self.assertFalse(page.meta_input.enabled)

    async def test_start_validation_and_storage_errors_keep_title_draft(self):
        with self.client:
            page = self.build_page()
            panel = page.task_panel
            panel.start()
            self.assertIsNone(page.snapshot.task_state)
            panel.title_input.set_value("Сохранить цель")
            with patch.object(self.service, "start_task", side_effect=ConversationStorageError("Детали")):
                panel.start()
        self.assertEqual(panel.title_input.value, "Сохранить цель")
        self.assertTrue(panel.new_task.visible)
        self.notify.assert_called_with("Не удалось сохранить состояние задачи.", type="negative")

    async def test_local_controls_work_without_api_token(self):
        with self.client:
            page = self.build_page(token_available=False)
            panel = page.task_panel
            panel.title_input.set_value("Офлайн-цель")
            panel.start()
            self.assertTrue(panel.pause_button.enabled)
            self.assertFalse(panel.continue_button.enabled)
            panel.pause()
            self.assertTrue(panel.resume_button.enabled)
            panel.resume()
        self.assertFalse(page.send_button.enabled)
        self.assertFalse(page.snapshot.task_state.paused)


if __name__ == "__main__":
    unittest.main()
