import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from llm_agent.agent import Agent
from llm_agent.branch_history import BranchHistoryManager
from llm_agent.history import HistoryManager
from llm_agent.memory import MemorySnapshot
from llm_agent.models import RequestOptions, Turn
from llm_agent.profile import UserProfile
from llm_agent.service import ConversationBusyError, ConversationService, ConversationStorageError
from llm_agent.task_state import CONTINUE_TASK, TaskResponseError, TaskStage, TaskState, TaskStateError
from tests.helpers import completion


def payload(action, answer="Результат", **values):
    return json.dumps({"action": action, "answer": answer, **values}, ensure_ascii=False)


def reply(action, answer="Результат", **values):
    response = completion()
    response.choices[0].message.content = payload(action, answer, **values)
    return response


def stages():
    planning = TaskState("Написать функцию сложения")
    _, execution = planning.apply_reply("Только Python", payload("plan", plan=["Реализовать", "Привести примеры"]))
    _, second_step = execution.apply_reply(CONTINUE_TASK, payload("complete_step", "def add(a, b): return a + b"))
    _, validation = second_step.apply_reply(CONTINUE_TASK, payload("complete_step", "add(2, 3) == 5"))
    _, done = validation.apply_reply(CONTINUE_TASK, payload("finish", "Код проверен чтением, запуск не выполнялся"))
    return planning, execution, second_step, validation, done


class TaskStateTests(unittest.TestCase):
    def test_lifecycle_keeps_goal_requirements_artifacts_and_validation(self):
        planning, execution, second_step, validation, done = stages()
        self.assertEqual([state.stage for state in (planning, execution, validation, done)], list(TaskStage))
        self.assertEqual(second_step.step, 1)
        self.assertEqual(second_step.current_step, "Привести примеры")
        self.assertEqual(done.results, ("def add(a, b): return a + b", "add(2, 3) == 5"))
        self.assertIn("Только Python", done.notes[0])
        self.assertIn("запуск не выполнялся", done.validation)
        self.assertEqual(TaskState.from_dict(done.to_dict()), done)

    def test_pause_and_resume_every_stage_and_middle_step_without_losing_progress(self):
        for state in stages():
            with self.subTest(stage=state.stage, step=state.step):
                paused = state.pause()
                restored = TaskState.from_dict(json.loads(json.dumps(paused.to_dict())))
                self.assertTrue(restored.paused)
                self.assertEqual(restored.current_step, state.current_step)
                self.assertIn("Снять паузу", restored.expected_action)
                with self.assertRaises(TaskStateError):
                    restored.apply_reply(CONTINUE_TASK, payload("clarify"))
                self.assertEqual(restored.resume(), state)
        with self.assertRaises(TaskStateError):
            stages()[-1].apply_reply(CONTINUE_TASK, payload("finish"))

    def test_clarification_and_rework_keep_context(self):
        _, state = TaskState("API").apply_reply("Нужна бесплатная API", payload("clarify", "Какой протокол?"))
        self.assertEqual(state.stage, TaskStage.PLANNING)
        self.assertEqual(len(state.notes), 2)
        _, state = state.apply_reply("HTTP", payload("plan", plan=["Сделать API"]))
        _, state = state.apply_reply(CONTINUE_TASK, payload("complete_step", "Вариант 1"))
        _, state = state.apply_reply(CONTINUE_TASK, payload("revise", "Не хватает обработки ошибок"))
        self.assertEqual(state.stage, TaskStage.EXECUTION)
        self.assertEqual(state.step, 0)
        self.assertIn("Предыдущий результат: Вариант 1", state.notes)
        _, state = state.apply_reply("Изменились требования", payload("replan", "Нужен новый план"))
        self.assertEqual(state.stage, TaskStage.PLANNING)
        self.assertEqual(state.plan, ())
        self.assertIn("Какой протокол?", " ".join(state.notes))

    def test_invalid_model_proposals_cannot_change_state(self):
        state = TaskState("Задача")
        for content in (
            "Не JSON", "null", "[]", payload("finish"), payload("complete_step"),
            payload("plan"), payload("plan", plan=[]), payload("plan", plan=[""]),
            payload("plan", plan="список"), payload("plan", plan=[3]),
            payload("clarify", stage="done"), payload("clarify", plan=["скрытый план"]),
            payload("clarify", answer=""), payload([], answer="Неверный тип действия"),
        ):
            with self.subTest(content=content), self.assertRaises(TaskStateError):
                state.apply_reply("Перепрыгни этапы", content)
        self.assertEqual(state, TaskState("Задача"))

    def test_corrupt_persisted_state_is_rejected(self):
        for change in (
            {"stage": "unknown"}, {"stage": "done"}, {"step": True}, {"step": -1},
            {"paused": "true"}, {"plan": [1]}, {"notes": "text"}, {"results": ["лишний"]},
        ):
            with self.subTest(change=change), self.assertRaises(TaskStateError):
                TaskState.from_dict({**TaskState("Задача").to_dict(), **change})

    def test_reply_errors_explain_the_exact_format_or_plan_problem(self):
        state = TaskState("Задача")
        cases = (
            ('{"answer":', "Некорректный JSON в строке 1, столбце 11"),
            ("[]", "получено: массив"),
            ("null", "получено: null"),
            ('{"answer":"Ответ"}', "Отсутствуют обязательные поля: action"),
            (payload("clarify", step=3), 'Лишние поля: "step"'),
            (payload("clarify", answer={}), "Поле answer должно быть непустой строкой; получено: объект"),
            (payload("clarify", answer=" "), "Поле answer содержит пустую строку"),
            (payload(["finish"]), "Поле action должно быть непустой строкой; получено: массив"),
            (payload(" "), "Поле action содержит пустую строку"),
            (payload("execute"), 'Неизвестное действие "execute"'),
            (payload("plan"), "Для действия plan отсутствует поле plan"),
            (payload("plan", plan=[]), "План пуст"),
            (payload("plan", plan="Первый шаг"), "Поле plan должно быть массивом непустых строк; получено: строка"),
            (payload("plan", plan=["Первый шаг", " "]), "Пункт плана №2 пуст"),
            (payload("plan", plan=["Первый шаг", 7]), "Пункт плана №2 должен быть непустой строкой; получено: число"),
            (payload("clarify", plan=[]), 'Поле plan передано вместе с действием "clarify"'),
        )
        for content, expected in cases:
            with self.subTest(content=content), self.assertRaises(TaskResponseError) as raised:
                state.apply_reply(CONTINUE_TASK, content)
            message = str(raised.exception)
            self.assertIn(expected, message)
            self.assertIn("Текущий этап: planning", message)
            self.assertIn("Этап и шаг сохранены", message)
            self.assertNotIn("Повторите описание задачи", message)

    def test_forbidden_action_shows_stage_step_and_allowed_actions(self):
        state = stages()[2]
        with self.assertRaises(TaskResponseError) as raised:
            state.apply_reply(CONTINUE_TASK, payload("finish"))
        message = str(raised.exception)
        for expected in (
            'Действие "finish" (завершить задачу) недопустимо', "Текущий этап: execution",
            "Текущий шаг: 2 из 2", "clarify (запросить уточнение)",
            "complete_step (завершить текущий шаг)", "replan (вернуться к планированию)",
        ):
            self.assertIn(expected, message)
        self.assertEqual(state, stages()[2])

    def test_untrusted_names_in_errors_are_bounded_and_do_not_include_answer(self):
        for fields in ({"action": "x" * 1000 + "\n"}, {"action": "clarify", "y" * 1000: 1}):
            with self.subTest(fields=tuple(fields)), self.assertRaises(TaskResponseError) as raised:
                TaskState("Задача").apply_reply(CONTINUE_TASK, json.dumps({"answer": "Содержимое ответа", **fields}))
            self.assertLess(len(str(raised.exception)), 700)
            self.assertNotIn("Содержимое ответа", str(raised.exception))


class TaskIntegrationTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.path = self.directory / "history.json"
        self.client = self.enterContext(patch("llm_agent.agent.OpenAI")).return_value
        self.create = self.client.chat.completions.create

    def agent(self, **kwargs):
        agent = Agent("test", history_path=self.path, **kwargs)
        self.addCleanup(agent.close)
        return agent

    def test_resume_after_restart_with_window_one_has_full_task_context(self):
        history = HistoryManager(self.path)
        history.start_task("Написать функцию сложения")
        self.create.return_value = reply("plan", plan=["Реализовать", "Примеры"])
        self.agent(strategy="window", window_size=1).request("Только Python")
        self.create.return_value = reply("complete_step", "def add(a, b): return a + b")
        self.agent(strategy="window", window_size=1).request(CONTINUE_TASK)
        HistoryManager(self.path).pause_task()
        self.create.reset_mock()
        with self.assertRaisesRegex(TaskStateError, "паузе"):
            self.agent(strategy="window", window_size=1).request(CONTINUE_TASK)
        self.create.assert_not_called()
        HistoryManager(self.path).resume_task()
        self.create.return_value = reply("complete_step", "add(2, 3) == 5")
        result = self.agent(strategy="window", window_size=1).request(CONTINUE_TASK)
        self.assertEqual(result.content, "add(2, 3) == 5")
        self.assertEqual(self.create.call_args.kwargs["response_format"], {"type": "json_object"})
        messages = self.create.call_args.kwargs["messages"]
        prompt = next(message["content"] for message in messages if "<task_state>" in message["content"])
        for value in ("Написать функцию сложения", "Только Python", "def add", '"step": 1', "Примеры"):
            self.assertIn(value, prompt)
        self.assertEqual([message for message in messages if message["role"] != "system"],
                         [{"role": "user", "content": CONTINUE_TASK}])
        restored = HistoryManager(self.path)
        self.assertEqual(restored.task_state.stage, TaskStage.VALIDATION)
        self.assertNotIn("<task_state>", self.path.read_text(encoding="utf-8"))
        self.assertEqual(restored.get_usage().total_tokens, 390)

    def test_task_is_in_main_prompt_with_profile_memory_but_not_summary_or_facts(self):
        for strategy in ("summary", "facts"):
            with self.subTest(strategy=strategy):
                history = HistoryManager(self.path)
                history.clear()
                history.add_exchange("Старый вопрос", "Старый ответ", None)
                history.start_task("Новая задача")
                service_reply = completion()
                service_reply.choices[0].message.content = "{}" if strategy == "facts" else "Краткий контекст"
                self.create.side_effect = [service_reply, reply("plan", plan=["Первый шаг"])]
                self.create.reset_mock()
                options = {"strategy": "facts", "window_size": 1} if strategy == "facts" else {
                    "last_messages": 0, "compress_every": 1,
                }
                agent = self.agent(
                    profile=UserProfile(id="dev", name="Разработчик", style="Кратко"),
                    memory=MemorySnapshot(working={"constraint": "без зависимостей"}), **options,
                )
                agent.request("Начни")
                calls = self.create.call_args_list
                self.assertEqual(len(calls), 2)
                self.assertNotIn("<task_state>", json.dumps(calls[0].kwargs["messages"]))
                main_prompt = json.dumps(calls[1].kwargs["messages"], ensure_ascii=False)
                for value in ("<task_state>", "Кратко", "без зависимостей"):
                    self.assertIn(value, main_prompt)
                self.assertEqual(HistoryManager(self.path).task_state.stage, TaskStage.EXECUTION)

    def test_api_error_invalid_json_or_truncation_does_not_advance_or_save_answer(self):
        HistoryManager(self.path).start_task("Задача")
        invalid = completion()
        truncated = reply("plan", plan=["шаг"])
        truncated.choices[0].finish_reason = "length"
        for error in (RuntimeError("network"), invalid, truncated, reply("finish")):
            with self.subTest(error=type(error).__name__):
                agent = self.agent()
                previous_usage = agent.history.get_usage()
                self.create.side_effect = error if isinstance(error, Exception) else None
                self.create.return_value = error
                with self.assertRaises((RuntimeError, TaskStateError)):
                    agent.request(CONTINUE_TASK)
                self.assertEqual(agent.history.task_state, TaskState("Задача"))
                restored = HistoryManager(self.path)
                self.assertEqual(restored.task_state, TaskState("Задача"))
                self.assertEqual(restored.get_messages(), [])
                expected_tokens = previous_usage.total_tokens + (
                    0 if isinstance(error, Exception) else 130 if error is truncated else 260
                )
                self.assertEqual(restored.get_usage().total_tokens, expected_tokens)

    def test_rejected_response_without_usage_marks_total_incomplete(self):
        HistoryManager(self.path).start_task("Задача")
        self.create.return_value = completion(missing=True)
        with self.assertRaises(TaskStateError):
            self.agent().request(CONTINUE_TASK)
        self.assertEqual(HistoryManager(self.path).get_usage().missing_responses, 2)

    def test_failed_save_does_not_advance_in_memory_or_on_disk(self):
        HistoryManager(self.path).start_task("Задача")
        agent = self.agent()
        self.create.return_value = reply("plan", plan=["Шаг"])
        with patch.object(agent.history, "_write_data", side_effect=OSError("disk")):
            with self.assertRaises(OSError):
                agent.request(CONTINUE_TASK)
        self.assertEqual(agent.history.task_state, TaskState("Задача"))
        self.assertEqual(HistoryManager(self.path).task_state, TaskState("Задача"))
        self.assertEqual(agent.history.get_messages(), [])

    def test_meta_and_custom_format_are_rejected_before_context_or_api_changes(self):
        HistoryManager(self.path).start_task("Задача")
        agent = self.agent(last_messages=0, compress_every=1)
        saved = self.path.read_bytes()
        with self.assertRaises(TaskStateError):
            agent.request_with_meta_prompt("Запрос", system="Правила")
        with self.assertRaises(TaskStateError):
            agent.request("Запрос", response_format="object", system="Правила")
        self.assertEqual(self.path.read_bytes(), saved)
        self.create.assert_not_called()

    def test_legacy_history_branches_checkpoints_and_clear(self):
        self.path.write_text('[{"role":"user","content":"Старый чат"}]', encoding="utf-8")
        history = BranchHistoryManager(self.path)
        self.assertIsNone(history.task_state)
        history.start_task("Задача")
        history.create_checkpoint("planning")
        history.pause_task()
        history.create_branch("alternative", from_checkpoint="planning")
        self.assertFalse(history.task_state.paused)
        history.clear()
        self.assertIsNone(history.task_state)
        history.switch_branch("main")
        self.assertTrue(history.task_state.paused)
        self.assertEqual(history.get_messages()[0]["content"], "Старый чат")
        self.assertTrue(BranchHistoryManager(self.path).task_state.paused)


class TaskServiceTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.service = ConversationService(self.directory, "test")
        self.conversation = self.service.create()
        self.create = self.enterContext(patch("llm_agent.agent.OpenAI")).return_value.chat.completions.create

    def test_specific_rejection_is_saved_and_can_be_retried_without_losing_progress(self):
        cid = self.conversation.id
        initial = self.service.start_task(cid, "Задача")
        for response, reason in (
            (reply("finish"), 'Действие "finish" (завершить задачу) недопустимо'),
            (reply("plan", plan=[]), "План пуст"),
            (reply("plan", plan=["Шаг", None]), "Пункт плана №2"),
        ):
            with self.subTest(reason=reason):
                self.create.return_value = response
                failed = self.service.send(cid, CONTINUE_TASK)
                self.assertEqual(failed.task_state, initial.task_state)
                self.assertEqual(failed.turns[-1].status, "error")
                self.assertIsNone(failed.turns[-1].answer)
                self.assertIn(reason, failed.turns[-1].error)
                restored = ConversationService(self.directory, "test").get(cid)
                self.assertEqual(restored.turns[-1].error, failed.turns[-1].error)
        self.create.return_value = reply("plan", plan=["Первый шаг"])
        completed = self.service.send(cid, CONTINUE_TASK)
        self.assertEqual(completed.task_state.stage, TaskStage.EXECUTION)
        self.assertEqual(completed.task_state.step, 0)
        self.assertEqual(completed.turns[-1].status, "completed")

    def test_incomplete_reply_explains_finish_reason_without_assuming_user_limit(self):
        cid = self.conversation.id
        initial = self.service.start_task(cid, "Задача")
        for finish_reason, expected in (
            ("length", "Лимит может действовать и без настройки в интерфейсе"),
            ("content_filter", "фильтром содержимого"),
            ("tool_calls", "вызов инструмента"),
            ("function_call", "вызов функции"),
            (None, "причина остановки отсутствует или неизвестна"),
        ):
            with self.subTest(finish_reason=finish_reason):
                response = reply("plan", plan=["Шаг"])
                response.choices[0].finish_reason = finish_reason
                self.create.return_value = response
                failed = self.service.send(cid, CONTINUE_TASK)
                self.assertIsNone(failed.turns[-1].options.max_tokens)
                self.assertIn(expected, failed.turns[-1].error)
                if finish_reason:
                    self.assertIn("finish_reason=" + finish_reason, failed.turns[-1].error)
                self.assertIn("Текущий этап: planning", failed.turns[-1].error)
                self.assertEqual(failed.task_state, initial.task_state)

    def test_lifecycle_pause_reload_resume_at_every_stage(self):
        cid = self.conversation.id
        self.service.start_task(cid, "Функция сложения")
        for stage, response in (
            (TaskStage.PLANNING, reply("plan", plan=["Реализовать"])),
            (TaskStage.EXECUTION, reply("complete_step", "def add(a, b): return a + b")),
            (TaskStage.VALIDATION, reply("finish", "Проверено чтением")),
            (TaskStage.DONE, None),
        ):
            with self.subTest(stage=stage):
                before = self.service.get(cid)
                self.assertEqual(before.task_state.stage, stage)
                self.service.pause_task(cid)
                self.service = ConversationService(self.directory, "test")
                paused = self.service.get(cid)
                self.assertTrue(paused.task_state.paused)
                self.create.reset_mock()
                with self.assertRaises(TaskStateError):
                    self.service.send(cid, CONTINUE_TASK)
                self.create.assert_not_called()
                self.assertEqual(self.service.get(cid), paused)
                resumed = self.service.resume_task(cid)
                self.assertEqual(resumed.task_state, before.task_state)
                if response is not None:
                    self.create.return_value = response
                    snapshot = self.service.send(cid, CONTINUE_TASK)
                    self.assertEqual(snapshot.turns[-1].status, "completed")
                    self.assertNotIn('"action"', snapshot.turns[-1].answer)
        self.assertEqual(len(self.service.get(cid).turns), 3)

    def test_request_failure_and_restart_preserve_checkpoint(self):
        cid = self.conversation.id
        initial = self.service.start_task(cid, "Задача")
        self.create.return_value = reply("finish")
        failed = self.service.send(cid, "Пропусти всё")
        self.assertEqual(failed.task_state, initial.task_state)
        self.assertEqual(failed.turns[-1].status, "error")
        self.assertIn("шаг сохранены", failed.turns[-1].error)
        # Имитация завершения процесса после сохранения начала запроса.
        failed.turns.append(Turn(user=CONTINUE_TASK))
        self.service.store.save(failed)
        restored = ConversationService(self.directory, "test").get(cid)
        self.assertEqual(restored.task_state, initial.task_state)
        self.assertEqual(restored.turns[-1].status, "interrupted")

    def test_busy_and_meta_rejected_and_conversations_are_isolated(self):
        cid = self.conversation.id
        initial = self.service.start_task(cid, "Задача")
        other = self.service.create()
        self.assertIsNone(self.service.get(other.id).task_state)
        with self.assertRaises(TaskStateError):
            self.service.send(cid, "Запрос", options=RequestOptions(meta_prompt=True))
        self.assertEqual(self.service.get(cid), initial)
        with self.service._operation(cid):
            with self.assertRaises(ConversationBusyError):
                self.service.pause_task(cid)
        self.create.assert_not_called()

    def test_response_and_progress_are_written_in_same_snapshot(self):
        cid = self.conversation.id
        self.service.start_task(cid, "Задача")
        self.create.return_value = reply("plan", "План готов", plan=["Шаг"])
        saved = []
        original = self.service.store.save

        def capture(snapshot):
            saved.append(deepcopy(snapshot))
            original(snapshot)

        with patch.object(self.service.store, "save", side_effect=capture):
            self.service.send(cid, CONTINUE_TASK)
        for snapshot in saved:
            if snapshot.task_state.stage == TaskStage.EXECUTION:
                self.assertEqual(snapshot.turns[-1].answer, "План готов")
                self.assertEqual(snapshot.turns[-1].status, "completed")
        self.assertEqual(saved[-1].task_state.stage, TaskStage.EXECUTION)

    def test_corrupt_state_is_reported_without_overwriting_file(self):
        cid = self.conversation.id
        self.service.start_task(cid, "Задача")
        path = self.service.store.path(cid)
        data = json.loads(path.read_text(encoding="utf-8"))
        data["working_context"]["task_state"]["stage"] = "done"
        path.write_text(json.dumps(data), encoding="utf-8")
        saved = path.read_bytes()
        with self.assertRaises(ConversationStorageError):
            self.service.get(cid)
        self.assertEqual(path.read_bytes(), saved)


if __name__ == "__main__":
    unittest.main()
