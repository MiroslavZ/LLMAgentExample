"""Общие фабрики тестовых данных."""

from copy import deepcopy
from unittest.mock import patch

from openai.types.chat import ChatCompletion


def rag_preparation(question, memory, turns, token, model, *, rewrite_enabled):
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
