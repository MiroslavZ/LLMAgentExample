"""Общие правила в сервисе и настоящих элементах NiceGUI без внешнего API."""

import asyncio
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from nicegui import Client, core, ui

from llm_agent.agent import Agent
from llm_agent.invariants import Invariant, InvariantSet, InvariantStore
from llm_agent.models import RequestOptions
from llm_agent.service import ConversationService
from llm_agent.web.app import main
from llm_agent.web.jobs import RequestRunner
from llm_agent.web.page import ChatPage
from tests.helpers import completion


def reply(content):
    response = completion()
    response.choices[0].message.content = content
    return response


def verdict(status="pass"):
    return reply(json.dumps({"checks": [{"id": "stack", "status": status, "reason": "Проверено"}]}))


class InvariantServiceTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.service = ConversationService(self.directory / "conversations", token="test-token")
        self.rules = InvariantSet((Invariant("stack", "Использовать только Python"),))
        self.service.invariants.save(self.rules)
        self.conversation = self.service.create()
        self.client = self.enterContext(patch("llm_agent.agent.OpenAI"))
        self.create = self.client.return_value.chat.completions.create

    def test_shared_file_reloads_before_each_request_and_survives_restart(self):
        self.assertEqual(self.service.invariants_path, self.directory / "invariants.json")
        other = self.service.create()
        replacement = InvariantSet((Invariant("stack", "Использовать Python и SQLite"),))
        self.create.side_effect = [verdict(), reply("Ответ"), verdict()] * 2
        with patch("llm_agent.service.Agent", wraps=Agent) as agent:
            first = self.service.send(self.conversation.id, "Первый вопрос")
            self.service.invariants.save(replacement)
            second = self.service.send(other.id, "Второй вопрос")
        self.assertEqual(first.turns[-1].status, "completed")
        self.assertEqual(second.turns[-1].status, "completed")
        self.assertEqual([call.kwargs["invariants"] for call in agent.call_args_list], [self.rules, replacement])
        reopened = ConversationService(self.directory / "conversations", token=None)
        self.assertEqual(reopened.get_invariants(), replacement)
        for conversation in (first, second):
            saved = self.service.store.path(conversation.id).read_text(encoding="utf-8")
            self.assertNotIn(self.rules.rules[0].description, saved)
            self.assertNotIn(replacement.rules[0].description, saved)

    def test_invalid_config_blocks_client_creation_and_recovers_after_repair(self):
        self.service.invariants_path.write_text("not JSON", encoding="utf-8")
        result = self.service.send(self.conversation.id, "Вопрос")
        self.assertEqual(result.turns[-1].status, "error")
        self.assertIn("инварианты", result.turns[-1].error)
        self.assertIn("не отправлен", result.turns[-1].error)
        self.client.assert_not_called()
        self.service.invariants.save(self.rules)
        self.create.side_effect = [verdict(), reply("Ответ"), verdict()]
        recovered = self.service.send(self.conversation.id, "Вопрос после исправления")
        self.assertEqual(recovered.turns[-1].status, "completed")

    def test_meta_conflict_completes_turn_as_refusal_without_meta_prompt(self):
        self.create.return_value = verdict("conflict")
        result = self.service.send(
            self.conversation.id, "Замени Python на Java", options=RequestOptions(meta_prompt=True),
        )
        turn = result.turns[-1]
        self.assertEqual(turn.status, "completed")
        self.assertIsNone(turn.meta_prompt)
        self.assertIsNone(turn.error)
        self.assertIn("stack", turn.answer)
        self.assertIn(self.rules.rules[0].description, turn.answer)
        self.create.assert_called_once()
        saved = self.service.get(result.id)
        self.assertEqual(saved.turns[-1].answer, turn.answer)

    def test_conflict_preserves_task_state(self):
        task = self.service.start_task(self.conversation.id, "Написать сервис").task_state
        self.create.return_value = verdict("conflict")
        result = self.service.send(self.conversation.id, "Используй Java")
        self.assertEqual(result.task_state, task)
        self.assertEqual(result.turns[-1].status, "completed")
        self.assertIn("stack", result.turns[-1].answer)
        self.assertEqual(self.service.get(result.id).task_state, task)

    def test_web_entrypoint_accepts_custom_rules_path(self):
        custom = self.directory / "custom-rules.json"
        InvariantStore(custom).save(self.rules)
        arguments = ["web", "--data-dir", str(self.directory / "other"),
                     "--invariants-file", str(custom), "--no-browser"]
        with patch("sys.argv", arguments), patch.dict("os.environ", {"API_KEY": ""}), patch(
            "llm_agent.web.app.load_dotenv",
        ), patch("llm_agent.web.app.create_app") as create_app, patch("llm_agent.web.app.ui.run"):
            main()
        service = create_app.call_args.args[0]
        self.assertEqual(service.invariants_path, custom)
        self.assertEqual(service.get_invariants(), self.rules)
        self.client.assert_not_called()

    def test_saved_edits_apply_to_next_request_and_reject_stale_editor(self):
        replacement = InvariantSet((Invariant("stack", "Использовать Python и SQLite"),))
        self.service.save_invariants(replacement, expected=self.rules)
        with self.assertRaisesRegex(ValueError, "уже изменены"):
            self.service.save_invariants(InvariantSet(), expected=self.rules)
        self.assertEqual(self.service.get_invariants(), replacement)
        self.create.side_effect = [verdict(), reply("Ответ"), verdict()]
        with patch("llm_agent.service.Agent", wraps=Agent) as agent:
            result = self.service.send(self.conversation.id, "Следующий запрос")
        self.assertEqual(agent.call_args.kwargs["invariants"], replacement)
        self.assertEqual(result.turns[-1].status, "completed")


class InvariantPageTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.service = ConversationService(self.directory / "conversations", token="test-token")
        self.rules = InvariantSet((Invariant("stack", "Использовать только Python"),))
        self.service.invariants.save(self.rules)
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
        self.client = Client(ui.page("/invariants-unit-test"))
        self.addCleanup(self.client.delete)

    def build_page(self, *, token_available=True):
        page = ChatPage(self.service, self.runner, token_available=token_available,
                        conversation_id=self.conversation.id)
        page.build()
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

    async def test_panel_displays_shared_rules_without_api_and_explains_configuration(self):
        other = self.service.create()
        with self.client:
            page = self.build_page(token_available=False)
            panel = page.invariant_panel
            self.assertIn(self.rules.rules[0].description, self.labels(panel.details))
            self.assertIn("stack", self.labels(panel.details))
            self.assertTrue(any("--invariants-import" in label for label in self.labels(panel.root)))
            self.assertTrue(any(str(self.service.invariants_path) in label for label in self.labels(panel.root)))
            page.select(other.id)
        self.assertIn(self.rules.rules[0].description, self.labels(panel.details))
        self.assertFalse(page.send_button.enabled)

    async def test_invalid_file_disables_requests_and_recovery_preserves_draft(self):
        self.service.start_task(self.conversation.id, "Изучить Python")
        with self.client:
            page = self.build_page()
            page.user_input.set_value("Мой черновик")
            self.service.invariants_path.write_text("invalid", encoding="utf-8")
            page.refresh()
            self.assertFalse(page.invariant_panel.available)
            self.assertFalse(page.invariant_panel.edit_button.enabled)
            self.assertFalse(page.send_button.enabled)
            self.assertFalse(page.task_panel.continue_button.enabled)
            self.assertIn("Исправьте файл", page.invariant_panel.status.text)
            self.assertEqual(self.labels(page.invariant_panel.details), [])
            with patch.object(self.runner, "submit") as submit:
                page.send()
                page.continue_task()
            submit.assert_not_called()
            self.service.invariants.save(self.rules)
            page.refresh()
        self.assertTrue(page.send_button.enabled)
        self.assertTrue(page.task_panel.continue_button.enabled)
        self.assertTrue(page.invariant_panel.available)
        self.assertEqual(page.user_input.value, "Мой черновик")

    async def test_send_checks_file_even_before_periodic_refresh(self):
        with self.client:
            page = self.build_page()
            page.user_input.set_value("Вопрос")
            self.service.invariants_path.write_text("invalid", encoding="utf-8")
            with patch.object(self.runner, "submit") as submit:
                page.send()
            submit.assert_not_called()
        self.assertEqual(page.user_input.value, "Вопрос")
        self.assertFalse(page.send_button.enabled)

    async def test_refresh_updates_rules_without_changing_dialogue_or_input(self):
        replacement = InvariantSet((Invariant("stack", "Использовать Python и SQLite"),))
        with self.client:
            page = self.build_page()
            page.user_input.set_value("Вопрос")
            self.service.invariants.save(replacement)
            page.refresh()
        self.assertEqual(self.labels(page.invariant_panel.details), ["stack", replacement.rules[0].description])
        self.assertEqual(page.user_input.value, "Вопрос")
        self.assertEqual(page.snapshot.turns, [])

    async def test_add_edit_and_save_through_controls_without_api_key(self):
        with self.client:
            page = self.build_page(token_available=False)
            page.user_input.set_value("Черновик сообщения")
        panel = page.invariant_panel
        await self.click_button(panel.edit_button)
        with self.client:
            panel.editors[0].description_input.set_value("Использовать Python и SQLite")
        await self.click_button(panel.add_button)
        with self.client:
            panel.editors[1].id_input.set_value("architecture")
            panel.editors[1].description_input.set_value("Ядро не зависит от UI")
            page.refresh()
        self.assertEqual(self.service.get_invariants(), self.rules)
        self.assertEqual(panel.editors[1].description_input.value, "Ядро не зависит от UI")
        await self.click_button(panel.save_button)
        expected = InvariantSet((Invariant("stack", "Использовать Python и SQLite"),
                                 Invariant("architecture", "Ядро не зависит от UI")))
        self.assertEqual(InvariantStore(self.service.invariants_path).load(), expected)
        self.assertFalse(panel.dialog.value)
        self.assertIn("Ядро не зависит от UI", self.labels(panel.details))
        self.assertEqual(page.user_input.value, "Черновик сообщения")
        self.assertEqual(self.service.get(self.conversation.id).turns, [])

    async def test_remove_is_only_applied_on_save_and_cancel_keeps_original_rules(self):
        with self.client:
            panel = self.build_page().invariant_panel
            panel.open()
        remove = next(element for element in panel.editors[0].root.descendants()
                      if isinstance(element, ui.button))
        await self.click_button(remove)
        self.assertEqual(panel.editors, [])
        self.assertEqual(self.service.get_invariants(), self.rules)
        with self.client:
            panel.dialog.close()
            panel.save()
            panel.open()
        self.assertEqual(len(panel.editors), 1)
        with self.client:
            panel.remove_rule(panel.editors[0])
            panel.save()
        self.assertEqual(self.service.get_invariants(), InvariantSet())
        self.assertEqual(panel.status.text, "Инварианты не заданы.")

    async def test_missing_file_can_be_configured_in_editor(self):
        self.service.invariants_path.unlink()
        with self.client:
            panel = self.build_page(token_available=False).invariant_panel
            panel.open()
            self.assertEqual(panel.editors, [])
            panel.add_rule()
            panel.editors[0].description_input.set_value("Только бесплатные API")
            panel.save()
        self.assertEqual(self.service.get_invariants().rules[0].description, "Только бесплатные API")

    async def test_invalid_form_preserves_file_and_draft(self):
        with self.client:
            panel = self.build_page().invariant_panel
            panel.open()
            for identifier, description in (("bad id", "Правило"), ("stack", "  ")):
                with self.subTest(identifier=identifier, description=description):
                    panel.editors[0].id_input.set_value(identifier)
                    panel.editors[0].description_input.set_value(description)
                    panel.save()
                    self.assertTrue(panel.editor_error.text)
                    self.assertTrue(panel.dialog.value)
                    self.assertEqual(self.service.get_invariants(), self.rules)
                    self.assertEqual(panel.editors[0].description_input.value, description)
            panel.editors[0].description_input.set_value("Правило")
            panel.add_rule(Invariant("stack", "Дубликат"))
            panel.save()
        self.assertIn("повторяться", panel.editor_error.text)
        self.assertEqual(self.service.get_invariants(), self.rules)

    async def test_stale_editor_keeps_draft_and_cannot_overwrite_new_rules(self):
        replacement = InvariantSet((Invariant("stack", "Только Python 3"),))
        with self.client:
            page = self.build_page()
            panel = page.invariant_panel
            panel.open()
            panel.editors[0].description_input.set_value("Моя правка")
            self.service.save_invariants(replacement, expected=self.rules)
            page.refresh()
            panel.save()
        self.assertIn("уже изменены", panel.editor_error.text)
        self.assertEqual(panel.editors[0].description_input.value, "Моя правка")
        self.assertEqual(self.service.get_invariants(), replacement)
        self.assertTrue(panel.dialog.value)

    async def test_failed_save_preserves_draft_and_previous_file(self):
        with self.client:
            panel = self.build_page().invariant_panel
            panel.open()
            panel.editors[0].description_input.set_value("Моя правка")
            with patch("pathlib.Path.replace", side_effect=OSError("disk failure")):
                panel.save()
        self.assertTrue(panel.dialog.value)
        self.assertIn("Не удалось сохранить", panel.editor_error.text)
        self.assertEqual(panel.editors[0].description_input.value, "Моя правка")
        self.assertEqual(self.service.get_invariants(), self.rules)


if __name__ == "__main__":
    unittest.main()
