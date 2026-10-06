"""Проверки серверных обработчиков NiceGUI без браузера, сервера и API."""

import asyncio
import inspect
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import Mock, patch

from nicegui import Client, core, ui
from nicegui.elements.timer import Timer

from llm_agent.models import ContextSettings, RAGSettings, RequestOptions, Turn
from llm_agent.rag_answer import render_answer, unknown_answer
from llm_agent.service import ConversationService, ConversationStorageError
from llm_agent.tool_events import ToolAttachment, ToolCallRecord
from llm_agent.web.components import render_turn
from llm_agent.web.jobs import RequestRunner
from llm_agent.web.page import ChatPage


class ChatPageTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.service = ConversationService(Path(directory.name), token="test-token")
        self.conversation = self.service.create()
        self.runner = RequestRunner(self.service)

        # Оставляем настоящие Client, дерево элементов и обработчики значений.
        # Убираем только транспорт браузера, навигацию и автоматический опрос.
        self.enterContext(patch("nicegui.background_tasks.create_or_defer",
                                side_effect=lambda coroutine, **_: coroutine.close()))
        self.enterContext(patch("llm_agent.web.page.ui.timer"))
        self.enterContext(patch("llm_agent.web.page.ui.navigate.history.replace"))
        self.notify = self.enterContext(patch("llm_agent.web.page.ui.notify"))
        self.enterContext(patch("llm_agent.agent.OpenAI",
                                side_effect=AssertionError("API запрещён в тестах UI")))
        self.enterContext(patch.object(core, "loop", asyncio.get_running_loop()))
        self.client = Client(ui.page("/web-unit-test"))
        self.addCleanup(self.client.delete)

    def build_page(self, *, token_available=True):
        page = ChatPage(
            self.service, self.runner, token_available=token_available,
            conversation_id=self.conversation.id,
        )
        page.build()
        return page

    def queue_submissions(self):
        """Приостановить запрос до первого сохранённого Turn."""
        def queue(conversation_id, *_args, **_kwargs):
            self.runner._tasks[conversation_id] = Mock()

        return self.enterContext(patch.object(self.runner, "submit", side_effect=queue))

    @staticmethod
    def labels(container):
        return [element.text for element in container.descendants() if isinstance(element, ui.label)]

    async def test_build_has_three_panels_and_empty_first_message_form(self):
        with self.client:
            page = self.build_page(token_available=False)

        app = next(element for element in self.client.content.descendants()
                   if "chat-app" in element.classes)
        self.assertEqual([element.tag for element in app], ["aside", "main", "aside"])
        self.assertIn("С чего начнём?", self.labels(page.transcript))
        self.assertEqual(page.user_input.value, "")
        self.assertEqual(page.system_input.value, "")
        self.assertTrue(page.strategy_input.enabled)
        self.assertFalse(page.send_button.enabled)
        self.assertTrue(any("API_KEY" in label for label in self.labels(page.banner)))
        self.assertEqual(len(self.service.list_conversations()), 1)

    async def test_rag_setting_persists_and_can_change_after_conversation_started(self):
        self.conversation.started = True
        self.conversation.turns.append(Turn(user="Первый вопрос", answer="Ответ", status="completed"))
        self.service.store.save(self.conversation)
        with self.client:
            page = self.build_page()
            self.assertTrue(page.rag_input.enabled)
            self.assertFalse(page.rag_input.value)
            page.rag_input.set_value(True)
            page.save_settings()
            self.assertTrue(self.service.get(self.conversation.id).settings.rag_enabled)
            page.rag_input.set_value(False)
            page.save_settings()
        self.assertFalse(self.service.get(self.conversation.id).settings.rag_enabled)

    async def test_rag_sources_are_collapsed_and_rendered_as_literal_text(self):
        text = '<script>alert("source")</script>\n```html\n<b>Текст</b>\n```'
        turn = Turn(user="Вопрос", rag_enabled=True, rag_context={
            "index": "test-index",
            "chunks": [{"source": "<b>file.md</b>", "section": "Раздел", "text": text,
                        "score": 0.75, "chunk_id": "chunk-1"}],
        })
        with self.client, ui.column() as transcript:
            render_turn(turn, restore=lambda: None, busy=False)
        self.assertIn("С RAG", self.labels(transcript))
        self.assertIn(text, self.labels(transcript))
        self.assertFalse(any(isinstance(element, ui.markdown) for element in transcript.descendants()))
        expansions = [element for element in transcript.descendants() if isinstance(element, ui.expansion)]
        self.assertEqual([element.text for element in expansions], ["Найденные фрагменты RAG · 1", "[1] <b>file.md</b>"])
        self.assertTrue(all(not element.value for element in expansions))

    async def test_rag_controls_react_before_save_and_preserve_disabled_settings(self):
        with self.client:
            page = self.build_page()
            dependent = (page.rag_rewrite_input, page.rag_filter_input,
                         page.rag_top_k_before_input, page.rag_top_k_after_input)
            self.assertTrue(all(not widget.enabled for widget in dependent))
            self.assertFalse(page.rag_threshold_input.enabled)
            page.rag_input.set_value(True)
            self.assertTrue(all(widget.enabled for widget in dependent))
            self.assertTrue(page.rag_threshold_input.enabled)
            page.rag_filter_input.set_value(True)
            self.assertTrue(page.rag_threshold_input.enabled)
            page.rag_rewrite_input.set_value(True)
            page.rag_top_k_before_input.set_value(12)
            page.rag_top_k_after_input.set_value(3)
            page.rag_threshold_input.set_value(0.77)
            page.save_settings()
            expected = ContextSettings(rag_enabled=True, rag_filter_enabled=True, rag_rewrite_enabled=True,
                                       rag_top_k_before=12, rag_top_k_after=3, rag_similarity_threshold=0.77)
            self.assertEqual(self.service.get(self.conversation.id).settings, expected)
            page.rag_input.set_value(False)
            self.assertFalse(page.rag_filter_input.enabled)
            self.assertFalse(page.rag_rewrite_input.enabled)
            self.assertFalse(page.rag_threshold_input.enabled)
            page.save_settings()
        self.assertEqual(self.service.get(self.conversation.id).settings, replace(expected, rag_enabled=False))

    async def test_rag_unknown_always_displays_sources_without_changing_json_format(self):
        result = unknown_answer()
        legacy_text = result["answer"] + "\n\n" + result["clarification"]
        for response_format in ("text", "object", "schema"):
            with self.subTest(response_format=response_format):
                turn = Turn(user="Вопрос", answer=legacy_text, rag_enabled=True,
                            rag_answer=result, options=RequestOptions(response_format=response_format),
                            status="completed")
                with self.client, ui.column() as transcript:
                    render_turn(turn, restore=lambda: None, busy=False)
                answer = next(element for element in transcript.descendants()
                              if isinstance(element, ui.markdown))
                self.assertEqual(answer.content, render_answer(result, response_format))
                if response_format == "text":
                    self.assertIn("Источники: подтверждающие материалы не найдены", answer.content)

    async def test_running_request_disables_all_rag_controls(self):
        self.queue_submissions()
        self.service.update_settings(self.conversation.id, ContextSettings(rag_enabled=True, rag_filter_enabled=True))
        with self.client:
            page = self.build_page()
            page.user_input.set_value("Вопрос")
            page.send()
            self.assertTrue(page.busy)
            self.assertTrue(all(not widget.enabled for widget in (
                page.rag_input, page.rag_rewrite_input, page.rag_filter_input,
                page.rag_top_k_before_input, page.rag_top_k_after_input, page.rag_threshold_input,
            )))

    async def test_invalid_rag_limits_do_not_submit_or_clear_draft(self):
        submit = self.queue_submissions()
        with self.client:
            page = self.build_page()
            page.rag_input.set_value(True)
            page.rag_top_k_before_input.set_value(2)
            page.rag_top_k_after_input.set_value(3)
            page.user_input.set_value("Сохранить вопрос")
            page.send()
        submit.assert_not_called()
        self.assertEqual(page.user_input.value, "Сохранить вопрос")
        self.assertFalse(self.service.get(self.conversation.id).settings.rag_enabled)
        self.assertIn("top-K", self.notify.call_args.args[0])

    async def test_rag_empty_context_displays_diagnostics_and_literal_fallback(self):
        settings = RAGSettings(rewrite_enabled=True, filter_enabled=True, similarity_threshold=0.9)
        context = {
            "index": "test-index", "chunks": [], "settings": asdict(settings),
            "original_query": "<b>Вопрос</b>", "search_query": "<b>Вопрос</b>",
            "rewrite": {"status": "fallback", "reason": "<script>failed</script>",
                        "elapsed_seconds": 0.12, "usage": {"prompt_tokens": 20, "completion_tokens": 2, "total_tokens": 22}},
            "candidates": [{"source": "<b>file.md</b>", "section": "", "text": "Текст", "score": 0.8,
                            "chunk_id": "c1", "original_rank": 1, "selection_reason": "below_threshold"}],
            "counts": {"found": 1, "passed": 0, "selected": 0}, "retrieval_seconds": 0.3,
            "embedding": {"model": "test-model", "revision": "abc"}, "corpus_hash": "corpus-hash",
        }
        with self.client, ui.column() as transcript:
            render_turn(Turn(user="Вопрос", rag_enabled=True, rag_settings=settings, rag_context=context),
                        restore=lambda: None, busy=False)
        labels = self.labels(transcript)
        self.assertIn("RAG + rewrite + фильтр", labels)
        self.assertIn("После отбора релевантных фрагментов не осталось. Для нового RAG-ответа требуется уточнение вопроса.", labels)
        self.assertIn("Исходный вопрос: <b>Вопрос</b>", labels)
        self.assertIn("Причина fallback: <script>failed</script>", labels)
        self.assertIn("Найдено: 1 · Прошло порог: 0 · Передано модели: 0", labels)
        self.assertIn("Токены rewrite: вход 20 → выход 2 (всего 22)", labels)
        table = next(element for element in transcript.descendants() if isinstance(element, ui.table))
        self.assertEqual(table.rows, [{"rank": 1, "source": "<b>file.md</b>", "chunk_id": "c1",
                                       "score": "0.800", "reason": "Ниже порога"}])
        self.assertFalse(any(isinstance(element, ui.markdown) for element in transcript.descendants()))
        self.assertTrue(all(not element.value for element in transcript.descendants() if isinstance(element, ui.expansion)))

    async def test_rag_mode_label_uses_each_turn_snapshot_even_after_error(self):
        turns = [
            Turn(user="Первый", rag_enabled=True, rag_settings=RAGSettings()),
            Turn(user="Второй", rag_enabled=True, rag_settings=RAGSettings(filter_enabled=True),
                 status="error", error="Индекс недоступен"),
            Turn(user="Третий", rag_enabled=False, rag_settings=RAGSettings(rewrite_enabled=True)),
        ]
        with self.client, ui.column() as transcript:
            for turn in turns:
                render_turn(turn, restore=lambda: None, busy=False)
        labels = self.labels(transcript)
        self.assertIn("Базовый RAG", labels)
        self.assertIn("RAG + фильтр", labels)
        self.assertIn("Без RAG", labels)
        self.assertNotIn("RAG + rewrite", labels)

    async def test_cli_command_tracks_selected_conversation_and_uses_quoted_path(self):
        with self.client:
            page = self.build_page()
            other = self.service.create()
            page.select(other.id)
        button = next(element for element in self.client.content.descendants()
                      if isinstance(element, ui.button) and element.props.get('icon') == 'terminal')
        await self.click_button(button)
        dialog = next(element for element in self.client.elements.values()
                      if isinstance(element, ui.dialog) and element.value)
        code = next(element for element in dialog.descendants() if isinstance(element, ui.code))
        self.assertIn(f'--conversation "{self.service.store.path(other.id)}"', code.content)
        self.assertNotIn(self.conversation.id, code.content)
        self.assertIn('--user "Продолжим"', code.content)

    async def click_button(self, button):
        handler = next(listener.handler for listener in button._event_listeners.values()
                       if listener.type == "click")
        # Контекст родительского слота, как при настоящем событии NiceGUI.
        with patch("nicegui.elements.button.handle_event") as dispatch:
            handler(None)
        callback = dispatch.call_args.args[0]
        with button.parent_slot:
            result = callback()
            if inspect.isawaitable(result):
                await result

    async def delete_from_list(self, page, conversation):
        button = next(element for element in page.conversation_list.descendants()
                      if isinstance(element, ui.button)
                      and "delete-conversation" in element.classes
                      and any(isinstance(sibling, ui.button) and sibling.text == conversation.title
                              for sibling in element.parent_slot.children))
        await self.click_button(button)
        dialog = next(element for element in self.client.elements.values()
                      if isinstance(element, ui.dialog) and element.value)
        confirm = next(element for element in dialog.descendants()
                       if isinstance(element, ui.button) and element.text == "Удалить")
        with patch("llm_agent.web.page.ui.timer", Timer):
            await self.click_button(confirm)
        self.assertFalse(dialog.value)
        self.assertNotIn(conversation.id, [item.id for item in self.service.list_conversations()])
        self.assertEqual(page.snapshot.id, page.conversation_id)
        self.assertTrue(page.send_button.enabled)

    async def test_delete_active_dialogue_from_button_context(self):
        other = self.service.create()
        other.title = "Другой диалог"
        self.service.store.save(other)
        with self.client:
            page = self.build_page()
        await self.delete_from_list(page, self.conversation)
        self.assertEqual(page.conversation_id, other.id)

    async def test_delete_last_dialogue_from_button_context(self):
        with self.client:
            page = self.build_page()
        await self.delete_from_list(page, self.conversation)
        self.assertNotEqual(page.conversation_id, self.conversation.id)
        self.assertEqual(len(self.service.list_conversations()), 1)

    async def test_delete_inactive_dialogue_preserves_active_draft(self):
        other = self.service.create()
        other.title = "Другой диалог"
        self.service.store.save(other)
        with self.client:
            page = self.build_page()
            page.user_input.set_value("Сохранить черновик")
        await self.delete_from_list(page, other)
        self.assertEqual(page.conversation_id, self.conversation.id)
        self.assertEqual(page.user_input.value, "Сохранить черновик")

    async def test_select_keeps_each_dialogue_draft_and_request_options(self):
        other = self.service.create()
        with self.client:
            page = self.build_page()
            page.user_input.set_value("Первый черновик")
            page.system_input.set_value("Первая роль")
            page.meta_input.set_value(True)
            page.temperature_input.set_value(0.7)
            page.max_tokens_input.set_value(123)
            page.select(other.id)
            self.assertEqual(page.user_input.value, "")
            self.assertEqual(page.system_input.value, "")
            page.user_input.set_value("Другой черновик")
            page.system_input.set_value("Другая роль")
            page.select(self.conversation.id)
            self.assertEqual(page.user_input.value, "Первый черновик")
            self.assertEqual(page.system_input.value, "Первая роль")
            self.assertTrue(page.meta_input.value)
            self.assertEqual(page.temperature_input.value, 0.7)
            self.assertEqual(page.max_tokens_input.value, 123)
            page.select(other.id)
            self.assertEqual(page.user_input.value, "Другой черновик")
            self.assertEqual(page.system_input.value, "Другая роль")
            self.assertFalse(page.meta_input.value)

    async def test_pre_save_failure_restores_text_after_switching_dialogues(self):
        other = self.service.create()
        submitted = asyncio.Event()
        finish = asyncio.Event()

        async def fail_before_save(*_args, **_kwargs):
            submitted.set()
            await finish.wait()
            raise ConversationStorageError("Тестовая ошибка записи")

        with patch("llm_agent.web.jobs.run.io_bound", side_effect=fail_before_save):
            with self.client:
                page = self.build_page()
                page.user_input.set_value("Важный запрос")
                page.system_input.set_value("Роль")
                page.send()
                task = self.runner._tasks[self.conversation.id]
                self.assertEqual(page.user_input.value, "")
                self.assertTrue(page.busy)
                self.assertFalse(page.send_button.enabled)
                page.select(other.id)
                page.user_input.set_value("Черновик другого диалога")
            await submitted.wait()
            finish.set()
            with self.assertLogs("llm_agent.web.jobs", level="ERROR"):
                await task
            with self.client:
                page.refresh()
                self.assertEqual(page.user_input.value, "Черновик другого диалога")
                page.select(self.conversation.id)
                self.assertEqual(page.user_input.value, "Важный запрос")
                self.assertEqual(page.system_input.value, "Роль")
                self.assertTrue(page.send_button.enabled)
                self.assertFalse(page.busy)
                self.assertEqual(self.service.get(self.conversation.id).turns, [])
                page.select(other.id)
                self.assertEqual(page.user_input.value, "Черновик другого диалога")

    async def test_send_with_stale_unchanged_settings_keeps_other_tab_configuration(self):
        original = ContextSettings(strategy="window", window_size=10)
        self.service.update_settings(self.conversation.id, original)
        submit = self.queue_submissions()
        with self.client:
            page = self.build_page()
            latest = replace(original, window_size=3)
            self.service.update_settings(self.conversation.id, latest)
            page.user_input.set_value("Сообщение из старой вкладки")
            page.send()

        submit.assert_called_once()
        self.assertIsNone(submit.call_args.kwargs["settings"])
        self.assertEqual(self.service.get(self.conversation.id).settings, latest)

    async def test_send_with_edited_settings_passes_expected_snapshot(self):
        original = ContextSettings(strategy="window", window_size=10)
        self.service.update_settings(self.conversation.id, original)
        submit = self.queue_submissions()
        with self.client:
            page = self.build_page()
            page.window_input.set_value(4)
            page.user_input.set_value("Настройки вместе с запросом")
            page.send()

        submit.assert_called_once()
        self.assertEqual(submit.call_args.kwargs["settings"], replace(original, window_size=4))
        self.assertEqual(submit.call_args.kwargs["expected_settings"], original)
        # Сохранение выполняется внутри операции отправки, когда она начнётся.
        self.assertEqual(self.service.get(self.conversation.id).settings, original)

    async def test_storage_warning_appears_and_clears_during_regular_refresh(self):
        other = self.service.create()
        path = self.service.store.path(other.id)
        valid_data = path.read_text(encoding="utf-8")
        with self.client:
            page = self.build_page()
            self.assertEqual(self.labels(page.banner), [])
            path.write_text("{broken", encoding="utf-8")
            page.refresh()
            self.assertTrue(any(other.id in label for label in self.labels(page.banner)))
            path.write_text(valid_data, encoding="utf-8")
            page.refresh()
            self.assertEqual(self.labels(page.banner), [])

    async def test_started_dialogue_locks_strategy_but_allows_parameter_update(self):
        conversation = self.service.get(self.conversation.id)
        conversation.started = True
        conversation.system_prompt = "Зафиксированная роль"
        conversation.settings = ContextSettings(strategy="window", window_size=10)
        conversation.turns = [Turn(user="Начало", answer="Ответ", status="completed")]
        self.service.store.save(conversation)
        with self.client:
            page = self.build_page()
            self.assertIsNone(page.system_input)
            self.assertFalse(page.strategy_input.enabled)
            self.assertTrue(page.window_input.enabled)
            self.assertTrue(page.save_settings_button.enabled)
            page.window_input.set_value(5)
            page.save_settings()
        saved = self.service.get(conversation.id)
        self.assertEqual(saved.settings, replace(conversation.settings, window_size=5))
        self.assertEqual(saved.system_prompt, conversation.system_prompt)
        self.assertEqual(saved.turns, conversation.turns)

    async def test_invalid_request_retains_draft_without_scheduling(self):
        submit = self.queue_submissions()
        with self.client:
            page = self.build_page()
            page.user_input.set_value("Не потерять при валидации")
            page.max_tokens_input.set_value(1.5)
            page.send()

        submit.assert_not_called()
        self.assertEqual(page.user_input.value, "Не потерять при валидации")
        self.assertEqual(page.drafts[self.conversation.id].user, page.user_input.value)
        self.assertFalse(self.service.get(self.conversation.id).started)
        self.notify.assert_called_once()

    async def test_running_turn_displays_collapsed_tool_results_as_literal_text(self):
        result = 'Результат\n```\n<strong>Обычный текст</strong>\n```\nКонец'
        calls = [
            ToolCallRecord("call_1", "GitHub", "get_issue", '{"number": 17}', result, False, 0.2),
            ToolCallRecord("call_2", "GitHub", "get_issue", "invalid JSON", "Ошибка параметров", True, 0.1),
        ]
        with self.client, ui.column() as transcript:
            render_turn(Turn(user="Найди задачу", tool_calls=calls), restore=lambda: None, busy=True)
        expansions = [element for element in transcript.descendants() if isinstance(element, ui.expansion)]
        self.assertEqual([element.text for element in expansions], [
            "1. GitHub · get_issue · Успешно", "2. GitHub · get_issue · Ошибка",
        ])
        self.assertTrue(all(not element.value for element in expansions))
        blocks = [element for element in transcript.descendants() if isinstance(element, ui.code)]
        self.assertEqual([element.content for element in blocks], [
            calls[0].arguments, result, calls[1].arguments, calls[1].result,
        ])
        html = blocks[1].markdown._props["innerHTML"]
        self.assertIn("&lt;strong&gt;", html)
        self.assertNotIn("<strong>", html)
        self.assertIn("Агент готовит ответ…", self.labels(transcript))

    async def test_tool_attachment_downloads_embedded_bytes_without_reading_remote_uri(self):
        attachment = ToolAttachment("сводка.txt", 'Точный текст "🐍"\r\n', "file:///remote/summary.txt")
        record = ToolCallRecord("save", "MCP", "save_to_file", "{}", "{}", False, 0.1, (attachment,))
        with self.client, ui.column() as transcript:
            render_turn(Turn(user="Сохрани", tool_calls=[record]), restore=lambda: None, busy=False)
        button = next(element for element in transcript.descendants()
                      if isinstance(element, ui.button) and element.text == "Скачать сводка.txt")
        handler = next(listener.handler for listener in button._event_listeners.values() if listener.type == "click")
        with patch("nicegui.elements.button.handle_event") as dispatch:
            handler(None)
        with self.client, patch("llm_agent.web.components.ui.download.content") as download:
            dispatch.call_args.args[0](None)
        download.assert_called_once_with(
            attachment.text.encode("utf-8"), filename="сводка.txt", media_type="text/plain; charset=utf-8",
        )


if __name__ == "__main__":
    unittest.main()
