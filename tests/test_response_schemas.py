import json
import unittest
from copy import deepcopy

from jsonschema import Draft202012Validator, ValidationError, validate

from llm_agent.context_strategy import FactsStrategy
from llm_agent.invariants import Invariant, InvariantSet, InvariantVerdict
from llm_agent.rag_answer import parse_answer, unknown_answer
from llm_agent.rag_dialogue import _apply_delta, empty_memory
from llm_agent.response_schemas import (
    FACTS_SCHEMA, GENERIC_OBJECT_SCHEMA, INVARIANT_SCHEMA, RAG_ANSWER_SCHEMA,
    RAG_PREPARATION_SCHEMA, schema_format, task_response_format, task_response_schema,
)
from llm_agent.task_state import TaskStage, TaskState, TaskStateError


class ResponseSchemaTests(unittest.TestCase):
    def setUp(self):
        self.context = {"chunks": [
            {"chunk_id": "guide", "source": "guide.md", "section": "Запуск",
             "text": "Запустите python main.py."},
        ]}
        self.answer = {
            "status": "answered", "answer": "Запустите python main.py.",
            "sources": ["guide"],
            "quotes": [{"chunk_id": "guide", "text": "Запустите python main.py."}],
            "clarification": None,
        }
        self.tasks = [
            TaskState("Подготовить результат"),
            TaskState("Подготовить результат", plan=("Проверить данные",)),
            TaskState("Подготовить результат", stage=TaskStage.EXECUTION,
                      plan=("Проверить данные",)),
            TaskState("Подготовить результат", stage=TaskStage.VALIDATION,
                      plan=("Проверить данные",), step=1, results=("Проверено",)),
        ]

    def test_schemas_are_valid_json_schema(self):
        schemas = [FACTS_SCHEMA, GENERIC_OBJECT_SCHEMA, INVARIANT_SCHEMA,
                   RAG_ANSWER_SCHEMA, RAG_PREPARATION_SCHEMA]
        schemas += [task_response_schema(task, rag=rag)
                    for task in self.tasks for rag in (False, True)]
        for schema in schemas:
            with self.subTest(schema=schema):
                Draft202012Validator.check_schema(schema)

    def test_format_is_complete_and_does_not_share_mutable_schema(self):
        result = schema_format("rag_answer", RAG_ANSWER_SCHEMA)
        self.assertEqual(result["type"], "json_schema")
        self.assertEqual(result["json_schema"]["name"], "rag_answer")
        self.assertTrue(result["json_schema"]["strict"])
        result["json_schema"]["schema"]["properties"].clear()
        self.assertEqual(len(RAG_ANSWER_SCHEMA["properties"]), 5)
        self.assertFalse(schema_format("facts", FACTS_SCHEMA, strict=False)["json_schema"]["strict"])

    def test_rag_schema_accepts_both_parser_statuses(self):
        for answer in (self.answer, unknown_answer()):
            with self.subTest(status=answer["status"]):
                validate(answer, RAG_ANSWER_SCHEMA)
                parse_answer(json.dumps(answer), self.context)

    def test_rag_schema_rejects_missing_extra_and_incorrect_fields(self):
        for field in self.answer:
            candidate = dict(self.answer)
            del candidate[field]
            with self.subTest(missing=field), self.assertRaises(ValidationError):
                validate(candidate, RAG_ANSWER_SCHEMA)
        for patch in ({"extra": True}, {"status": "ok"}, {"answer": 1},
                      {"sources": [{"chunk_id": "guide"}]},
                      {"quotes": [{"chunk_id": "guide", "text": "text", "extra": True}]},
                      {"clarification": False}):
            with self.subTest(patch=patch), self.assertRaises(ValidationError):
                validate(dict(self.answer, **patch), RAG_ANSWER_SCHEMA)

    def test_facts_support_dynamic_keys_deletion_and_no_changes(self):
        for payload in ({}, {"preferences.language": "русский", "deadline": None}):
            with self.subTest(payload=payload):
                validate(payload, FACTS_SCHEMA)
                FactsStrategy.update({}, json.dumps(payload))
        for payload in ({"goal": 42}, {"goal": []}, {"goal": {"nested": True}}):
            with self.subTest(payload=payload), self.assertRaises(ValidationError):
                validate(payload, FACTS_SCHEMA)
        validate({"nested": {"custom": [1, True, None]}}, GENERIC_OBJECT_SCHEMA)

    def test_invariant_schema_accepts_every_verdict(self):
        invariants = InvariantSet((Invariant("stack", "Использовать Python"),))
        for status in ("pass", "conflict", "uncertain"):
            payload = {"checks": [{"id": "stack", "status": status, "reason": "Проверено"}]}
            with self.subTest(status=status):
                validate(payload, INVARIANT_SCHEMA)
                InvariantVerdict.parse(json.dumps(payload), invariants)
        validate({"checks": []}, INVARIANT_SCHEMA)
        for payload in ({"checks": [{"id": "stack", "status": "ok", "reason": "Проверено"}]},
                        {"checks": [{"id": "stack", "status": "pass"}]},
                        {"checks": [], "extra": True}):
            with self.subTest(payload=payload), self.assertRaises(ValidationError):
                validate(payload, INVARIANT_SCHEMA)

    def test_preparation_schema_accepts_existing_memory_operations(self):
        question = "Нужно запустить агента. Использовать только CLI. Отменяю прежнее ограничение."
        turns = [{"user": question}]
        operations = [
            [],
            [{"op": "set", "field": "goal", "key": None,
              "value": "Запустить агента", "quote": "Нужно запустить агента."}],
            [{"op": "delete", "field": "goal", "key": None,
              "value": None, "quote": "Отменяю прежнее ограничение."}],
        ]
        for group in ("clarifications", "constraints", "terms"):
            for op in ("set", "delete"):
                operations.append([{
                    "op": op, "field": group, "key": "interface",
                    "value": "Только CLI" if op == "set" else None,
                    "quote": "Использовать только CLI." if op == "set" else "Отменяю прежнее ограничение.",
                }])
        for delta in operations:
            for query in (None, "Как запустить агента через CLI?"):
                with self.subTest(delta=delta, query=query):
                    validate({"memory_delta": delta, "standalone_query": query}, RAG_PREPARATION_SCHEMA)
                    _apply_delta(empty_memory(), delta, question, turns)

    def test_task_schemas_accept_each_stage_action_and_optional_plan(self):
        for task in self.tasks:
            schema = task_response_schema(task)
            for action in schema["properties"]["action"]["enum"]:
                payload = {"answer": "Результат или уточнение", "action": action}
                if action == "plan":
                    payload["plan"] = ["Проверить данные"]
                with self.subTest(stage=task.stage, action=action):
                    validate(payload, schema)
                    task.apply_reply("Продолжить", json.dumps(payload))
                    structured = task_response_format(task)["json_schema"]
                    self.assertEqual(structured["strict"], task.stage != TaskStage.PLANNING)
            for patch in ({"stage": task.stage.value}, {"action": "unknown"}, {"plan": None}):
                with self.subTest(stage=task.stage, patch=patch), self.assertRaises(ValidationError):
                    validate({"answer": "Уточнение", "action": "clarify", **patch}, schema)
            if task.stage != TaskStage.PLANNING:
                with self.assertRaises(ValidationError):
                    validate({"answer": "План", "action": "clarify", "plan": ["Шаг"]}, schema)

    def test_task_rag_schema_keeps_answer_at_top_level_only(self):
        for task in self.tasks:
            for answer in (self.answer, unknown_answer()):
                evidence = deepcopy(answer)
                payload = {"answer": evidence.pop("answer"), "action": "clarify", "rag": evidence}
                schema = task_response_schema(task, rag=True)
                with self.subTest(stage=task.stage, status=answer["status"]):
                    validate(payload, schema)
                    parse_answer(json.dumps({**payload["rag"], "answer": payload["answer"]}), self.context)
                    task_payload = {key: value for key, value in payload.items() if key != "rag"}
                    task.apply_reply("Продолжить", json.dumps(task_payload))
                    payload["rag"]["answer"] = "Лишний ответ"
                    with self.assertRaises(ValidationError):
                        validate(payload, schema)

    def test_inactive_task_has_no_response_schema(self):
        done = TaskState("Завершено", stage=TaskStage.DONE, plan=("Шаг",),
                         results=("Результат",), step=1, validation="Проверено")
        for task in (self.tasks[0].pause(), done):
            with self.subTest(task=task), self.assertRaises(TaskStateError):
                task_response_schema(task)


if __name__ == "__main__":
    unittest.main()
