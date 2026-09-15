"""Общие фабрики тестовых данных."""

from openai.types.chat import ChatCompletion


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
