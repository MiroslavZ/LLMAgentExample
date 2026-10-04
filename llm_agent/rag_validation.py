"""Проверка сохранённой диагностики RAG, включая формат дня 22."""

import math
from dataclasses import asdict

from .models import RAGSettings


def _finite(value: object, *, minimum: float | None = None) -> bool:
    return (type(value) in (int, float) and math.isfinite(value)
            and (minimum is None or value >= minimum))


def _chunk(chunk: object) -> None:
    if (not isinstance(chunk, dict)
            or any(not isinstance(chunk.get(key), str) for key in ("source", "section", "text", "chunk_id"))
            or not _finite(chunk.get("score"))):
        raise ValueError("Некорректный фрагмент RAG")


def validate_rag_context(context: object, snapshot: RAGSettings | None = None) -> None:
    if (not isinstance(context, dict) or not isinstance(context.get("index"), str)
            or not isinstance(context.get("chunks"), list)):
        raise ValueError("Некорректный контекст RAG")
    for chunk in context["chunks"]:
        _chunk(chunk)
    if set(context) == {"index", "chunks"}:
        return
    required = {"index", "chunks", "original_query", "search_query", "settings", "rewrite",
                "candidates", "counts", "retrieval_seconds", "embedding", "corpus_hash"}
    if (set(context) != required
            or any(not isinstance(context[key], str) or not context[key].strip()
                   for key in ("original_query", "search_query", "corpus_hash"))
            or not isinstance(context["settings"], dict)
            or not isinstance(context["embedding"], dict)
            or not _finite(context["retrieval_seconds"], minimum=0)):
        raise ValueError("Некорректная диагностика RAG")
    settings = RAGSettings(**context["settings"])
    settings.validate()
    if context["settings"] != asdict(settings) or snapshot is not None and snapshot != settings:
        raise ValueError("Настройки RAG не совпадают со снимком хода")
    rewrite = context["rewrite"]
    if (not isinstance(rewrite, dict)
            or set(rewrite) != {"status", "reason", "elapsed_seconds", "usage"}
            or rewrite["status"] not in ("disabled", "success", "fallback")
            or not _finite(rewrite["elapsed_seconds"], minimum=0)
            or (rewrite["reason"] is not None and not isinstance(rewrite["reason"], str))
            or settings.rewrite_enabled != (rewrite["status"] != "disabled")
            or (rewrite["status"] == "fallback" and not rewrite["reason"])
            or (rewrite["status"] != "success" and context["search_query"] != context["original_query"])):
        raise ValueError("Некорректная диагностика rewrite")
    usage = rewrite["usage"]
    if usage is not None and (
        not isinstance(usage, dict) or set(usage) != {"prompt_tokens", "completion_tokens", "total_tokens"}
        or any(type(value) is not int or value < 0 for value in usage.values())
        or usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]
    ):
        raise ValueError("Некорректный расход токенов rewrite")
    candidates, counts = context["candidates"], context["counts"]
    if (not isinstance(candidates, list) or len(candidates) > settings.top_k_before
            or not isinstance(counts, dict) or set(counts) != {"found", "passed", "selected"}
            or any(type(value) is not int or value < 0 for value in counts.values())):
        raise ValueError("Некорректные счётчики RAG")
    selected, passed, ids = [], 0, set()
    for rank, candidate in enumerate(candidates, 1):
        _chunk(candidate)
        if (type(candidate.get("original_rank")) is not int or candidate["original_rank"] != rank
                or candidate["chunk_id"] in ids):
            raise ValueError("Некорректный ранг или повторный кандидат RAG")
        ids.add(candidate["chunk_id"])
        if settings.filter_enabled and candidate["score"] < settings.similarity_threshold:
            reason = "below_threshold"
        else:
            passed += 1
            reason = "selected" if len(selected) < settings.top_k_after else "top_k"
            if reason == "selected":
                selected.append({key: value for key, value in candidate.items()
                                 if key not in ("original_rank", "selection_reason")})
        if candidate.get("selection_reason") != reason:
            raise ValueError("Некорректная причина отбора RAG")
    if (selected != context["chunks"]
            or counts != dict(found=len(candidates), passed=passed, selected=len(selected))):
        raise ValueError("Результат RAG не соответствует кандидатам")
