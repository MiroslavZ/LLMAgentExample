"""Общие фабрики тестовых данных."""

from copy import deepcopy
from unittest.mock import patch

from openai.types.chat import ChatCompletion

from llm_agent.llm_models import LLMModel


def register_model(service, *, token="test-secret", model_id="deepseek-chat"):
    """Явно добавить тестовую модель в каталог без обращения к API."""
    model = LLMModel(
        id="1" * 32, name="DeepSeek", model_id=model_id,
        base_url="https://api.deepseek.com", token=token, timeout=60.0,
    )
    service.models.save(model)
    return model


def create_selected_conversation(service):
    """Создать диалог с явно выбранной тестовой моделью."""
    model = register_model(service)
    return service.select_model(service.create().id, model.id)


def rag_preparation(question, memory, turns, token, model, *, rewrite_enabled, **kwargs):
    """Подготовка без сети и изменений памяти для тестов остальных подсистем."""
    diagnostic = dict(status="success", reason=None, elapsed_seconds=0.0, usage=None)
    rewrite = dict(query=question, **diagnostic) if rewrite_enabled else None
    return dict(memory=deepcopy(memory), rewrite=rewrite, diagnostic=diagnostic)


def patch_rag_preparation(testcase):
    return testcase.enterContext(patch("llm_agent.service.prepare_turn", side_effect=rag_preparation))


def completion(prompt=100, output=30, *, missing=False):
    return ChatCompletion.model_validate({
        "id": "test-completion",
        "created": 0,
        "model": "deepseek-chat",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": "Ответ"},
        }],
        "usage": None if missing else {
            "prompt_tokens": prompt,
            "completion_tokens": output,
            "total_tokens": prompt + output,
            "prompt_cache_hit_tokens": 80,
            "prompt_cache_miss_tokens": prompt - 80,
            "completion_tokens_details": {"reasoning_tokens": 20},
        },
    })
