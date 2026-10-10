"""Переносимые схемы структурированных ответов модели.

Схемы задают форму JSON, а существующие парсеры проверяют смысл: соответствие
цитат источникам, допустимость изменений памяти и связь действия с полями.
Необязательные поля не заменяются обязательным null: это меняет протокол.
"""

from copy import deepcopy

from .task_state import STAGE_ACTIONS, TaskStage, TaskState


def _object(properties: dict, *, required: list[str] | None = None) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties) if required is None else required,
        "additionalProperties": False,
    }


RAG_ANSWER_SCHEMA = _object({
    "status": {"type": "string", "enum": ["answered", "unknown"]},
    "answer": {"type": "string", "minLength": 1},
    "sources": {"type": "array", "items": {"type": "string", "minLength": 1}},
    "quotes": {
        "type": "array",
        "items": _object({
            "chunk_id": {"type": "string", "minLength": 1},
            "text": {"type": "string", "minLength": 1},
        }),
    },
    "clarification": {"type": ["string", "null"]},
})

# Произвольные ключи несовместимы с закрытыми объектами strict=True.
FACTS_SCHEMA = {
    "type": "object",
    "additionalProperties": {"type": ["string", "null"], "minLength": 1},
}
GENERIC_OBJECT_SCHEMA = {"type": "object"}

INVARIANT_SCHEMA = _object({
    "checks": {
        "type": "array",
        "items": _object({
            "id": {"type": "string", "minLength": 1},
            "status": {"type": "string", "enum": ["pass", "conflict", "uncertain"]},
            "reason": {"type": "string", "minLength": 1},
        }),
    },
})

# В протоколе подготовки все поля операции обязательны, включая key/value=null
# для goal/delete. Условия op/field и происхождение quote проверяет _apply_delta.
RAG_PREPARATION_SCHEMA = _object({
    "memory_delta": {
        "type": "array",
        "maxItems": 61,
        "items": _object({
            "op": {"type": "string", "enum": ["set", "delete"]},
            "field": {"type": "string", "enum": ["goal", "clarifications", "constraints", "terms"]},
            "key": {"type": ["string", "null"], "minLength": 1, "maxLength": 100},
            "value": {"type": ["string", "null"], "minLength": 1, "maxLength": 500},
            "quote": {"type": "string", "minLength": 1, "maxLength": 500},
        }),
    },
    "standalone_query": {"type": ["string", "null"], "minLength": 1, "maxLength": 1500},
})


def schema_format(name: str, schema: dict, *, strict: bool = True) -> dict:
    """Сформировать полный response_format, не отдавая изменяемую общую схему."""
    return {
        "type": "json_schema",
        "json_schema": {"name": name, "strict": strict, "schema": deepcopy(schema)},
    }


def task_response_schema(task: TaskState, *, rag: bool = False) -> dict:
    """Разрешить поля и действия текущего этапа активной задачи.

    В planning plan необязателен на уровне схемы: он требуется только для
    действия plan и запрещён для clarify. Эту связь проверяет TaskState.
    Поэтому схема planning должна передаваться с strict=False.
    """
    task.require_active()
    properties = {
        "answer": {"type": "string", "minLength": 1},
        "action": {"type": "string", "enum": list(STAGE_ACTIONS[task.stage])},
    }
    required = ["answer", "action"]
    if task.stage == TaskStage.PLANNING:
        properties["plan"] = {
            "type": "array", "minItems": 1,
            "items": {"type": "string", "minLength": 1},
        }
    if rag:
        evidence = deepcopy(RAG_ANSWER_SCHEMA)
        del evidence["properties"]["answer"]
        evidence["required"].remove("answer")
        properties["rag"] = evidence
        required.append("rag")
    return _object(properties, required=required)


def task_response_format(task: TaskState, *, rag: bool = False) -> dict:
    """Выбрать strict без изменения необязательных полей протокола задачи."""
    return schema_format(
        "task_rag_answer" if rag else "task_answer",
        task_response_schema(task, rag=rag),
        strict=task.stage != TaskStage.PLANNING,
    )
