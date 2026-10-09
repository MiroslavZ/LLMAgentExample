"""Воспроизводимое сравнение RAG без доступа к пользовательским диалогам."""

import argparse
import hashlib
import json
import os
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from .agent import BASE_URL, DEFAULT_MODEL
from .cli import ENV_PATH, load_env
from .indexing.store import load_index
from .history import HistoryManager
from .llm_models import LLMModel
from .models import ContextSettings, RAGSettings, RequestOptions, utc_now
from .rag import ROOT, Retriever
from .service import ConversationService

MODES = {
    "baseline": (False, False),
    "filtered": (False, True),
    "rewrite": (True, False),
    "rewrite_filtered": (True, True),
}


def normalize(text: str) -> str:
    """Перенос строки в Markdown не меняет смысл контрольной цитаты."""
    return " ".join(text.split())


def load_questions(path: Path, documents: list) -> list[dict]:
    questions = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(questions, list) or not questions:
        raise ValueError("Нужен непустой JSON-список вопросов")
    sources = {document.source: normalize(document.text) for document in documents}
    for number, question in enumerate(questions, 1):
        if (not isinstance(question, dict) or not isinstance(question.get("question"), str)
                or not question["question"].strip()):
            raise ValueError(f"Вопрос {number}: требуется непустое поле question")
        evidence = question.get("evidence")
        if not isinstance(evidence, list) or (not evidence and question.get("group") != "out_of_corpus"):
            raise ValueError(f"Вопрос {number}: нужны evidence или group=out_of_corpus")
        for item in evidence:
            if (not isinstance(item, dict) or not isinstance(item.get("source"), str)
                    or not isinstance(item.get("quote"), str) or not item["quote"].strip()):
                raise ValueError(f"Вопрос {number}: evidence требует source и непустую quote")
            if normalize(item["quote"]) not in sources.get(item["source"], ""):
                raise ValueError(f"Вопрос {number}: evidence не соответствует снимку индекса")
    return questions


def evidence_metrics(evidence: list[dict], chunks: list[dict]) -> dict:
    hits = [
        {"source": item["source"], "quote": item["quote"], "ranks": [
            rank for rank, chunk in enumerate(chunks, 1)
            if item["source"] == chunk["source"] and normalize(item["quote"]) in normalize(chunk["text"])
        ]}
        for item in evidence
    ]
    return {
        "any_hit": any(item["ranks"] for item in hits) if hits else None,
        "all_hit": all(item["ranks"] for item in hits) if hits else None,
        "evidence": hits,
    }


def citation_metrics(answer: dict | None, context: dict | None) -> dict:
    """Проверка доказательств не является оценкой смысла ответа."""
    metrics = dict(sources_present=None, quotes_present=None,
                   sources_match_context=None, quotes_match_chunks=None)
    if not answer or answer.get("status") != "answered":
        return metrics
    sources, quotes = answer.get("sources", []), answer.get("quotes", [])
    chunks = {chunk["chunk_id"]: chunk for chunk in (context or {}).get("chunks", [])}
    metrics["sources_present"] = bool(sources)
    metrics["quotes_present"] = bool(quotes)
    metrics["sources_match_context"] = bool(sources) and all(
        source.get("chunk_id") in chunks and all(
            source.get(key) == chunks[source["chunk_id"]].get(key)
            for key in ("source", "section")
        ) for source in sources
    )
    source_ids = {source.get("chunk_id") for source in sources}
    quote_ids = {quote.get("chunk_id") for quote in quotes}
    metrics["quotes_match_chunks"] = bool(quotes) and source_ids == quote_ids and all(
        quote.get("chunk_id") in chunks and bool(normalize(quote.get("text", "")))
        and normalize(quote["text"]) in normalize(chunks[quote["chunk_id"]]["text"])
        for quote in quotes
    )
    return metrics


def dialogue_usage(data: object) -> dict:
    """Расход всей изолированной попытки, включая непринятые ответы и repair."""
    messages, _, _, archived = HistoryManager._decode_data(data)
    totals = asdict(archived)
    for message in messages:
        if message["role"] != "assistant":
            continue
        if "usage" not in message:
            totals["missing_responses"] += 1
        else:
            for key, value in message["usage"].items():
                totals[key] += value
    return totals


def summarize(rows: list[dict]) -> dict:
    summary = {}
    for mode in dict.fromkeys(row["mode"] for row in rows):
        selected = [row for row in rows if row["mode"] == mode]
        retrieved = [row for row in selected if row["rag_context"] is not None]
        effective = [row for row in retrieved if not MODES[mode][0]
                     or row["rag_context"].get("rewrite", {}).get("status") == "success"]
        scored = [row for row in effective if row["manual_score"] is not None]
        answered = [row for row in selected if row["status"] == "completed"
                    and (row.get("rag_answer") or {}).get("status") == "answered"]
        unknown = [row for row in selected if row["status"] == "completed"
                   and (row.get("rag_answer") or {}).get("status") == "unknown"]
        semantic = [row for row in answered if row.get("semantic_support") is not None]
        summary[mode] = {
            "runs": len(selected),
            "answers_completed": sum(row["status"] == "completed" for row in selected),
            "mode_answers_completed": sum(row["status"] == "completed" for row in effective),
            "errors": sum(row["status"] not in ("completed", "retrieval_only") for row in selected),
            "retrievals": len(retrieved),
            "evaluated_retrievals": len(effective),
            "rewrite_successes": sum(row["rag_context"].get("rewrite", {}).get("status") == "success" for row in retrieved),
            "rewrite_fallbacks": sum(row["rag_context"].get("rewrite", {}).get("status") == "fallback" for row in retrieved),
            "evidence_questions": sum(bool(row["evidence"]) for row in effective),
            "candidate_hits": sum(row["candidate_evidence"]["any_hit"] is True for row in effective),
            "selected_hits": sum(row["selected_evidence"]["any_hit"] is True for row in effective),
            "empty_contexts": sum(not row["rag_context"]["chunks"] for row in effective),
            "mean_selected_chunks": (
                sum(len(row["rag_context"]["chunks"]) for row in effective) / len(effective)
                if effective else None
            ),
            "manual_scores_count": len(scored),
            "manual_score_sum": sum(row["manual_score"] for row in scored) if scored else None,
            "substantive_answers": len(answered),
            "unknown_answers": len(unknown),
            "substantive_answer_rate": len(answered) / len(selected) if selected else None,
            "citation_rates_among_substantive_answers": {
                key: sum(row.get("citation_metrics", {}).get(key) is True for row in answered) / len(answered)
                if answered else None
                for key in ("sources_present", "quotes_present", "sources_match_context", "quotes_match_chunks")
            },
            "semantic_reviews": len(semantic),
            "fully_supported_answers": sum(row["semantic_support"] == "full" for row in semantic) if semantic else None,
            "refusal_reviews": sum(row.get("refusal_appropriate") is not None for row in unknown),
            "appropriate_refusals": sum(row.get("refusal_appropriate") is True for row in unknown),
            "unnecessary_refusals": sum(row.get("refusal_appropriate") is False for row in unknown),
        }
    return summary


def index_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def ensure_fixed_index(path: Path, digest: str) -> None:
    if index_digest(path) != digest:
        raise ValueError("Индекс изменился во время сравнения; результаты не объединены")


def write_report(path: Path, report: dict) -> None:
    report["summary"] = summarize(report["runs"])
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def evaluation_token(base_url: str, token_env: str | None) -> str:
    """Не отправлять привычный облачный ключ произвольному новому серверу."""
    if token_env:
        token = os.environ.get(token_env, "").strip()
        if not token:
            raise ValueError("Указанная переменная с API-токеном отсутствует или пуста")
        return token
    if base_url.rstrip("/") in (BASE_URL, BASE_URL + "/v1"):
        return os.environ.get("API_KEY", "").strip()
    return ""


def run_evaluation(
    *, questions_path: Path, index_path: Path, output: Path, threshold: float,
    modes: list[str], top_k_before: int = 20, top_k_after: int = 5,
    retrieval_only: bool = False, token: str | None = None, cache: str | None = None,
    model: str = DEFAULT_MODEL, base_url: str = BASE_URL,
    temperature: float = 0.0, max_tokens: int = 512,
) -> dict:
    if not modes or len(set(modes)) != len(modes) or any(mode not in MODES for mode in modes):
        raise ValueError("Выберите различные поддерживаемые режимы сравнения")
    if retrieval_only and any(MODES[mode][0] for mode in modes):
        raise ValueError("Query rewrite требует LLM; для --retrieval-only выберите baseline filtered")
    parameters = RAGSettings(top_k_before=top_k_before, top_k_after=top_k_after, similarity_threshold=threshold)
    parameters.validate()
    options = RequestOptions(model=model, temperature=temperature, max_tokens=max_tokens)
    options.validate()
    evaluation_model = None if retrieval_only else LLMModel(
        uuid5(NAMESPACE_URL, base_url.rstrip("/") + "\n" + model).hex,
        model, model, base_url, token or "",
    )
    index_path = index_path.resolve()
    output = output.resolve()
    if output.exists():
        raise ValueError("Отчёт уже существует; выберите новый --output, чтобы сохранить предыдущий прогон")
    digest = index_digest(index_path)
    metadata, documents, _, _ = load_index(index_path)
    questions = load_questions(questions_path, documents)
    ensure_fixed_index(index_path, digest)
    report = {
        "created_at": utc_now(), "status": "running", "retrieval_only": retrieval_only,
        "index": str(index_path), "index_sha256": digest, "index_metadata": metadata,
        "corpus_documents": [{"source": doc.source, "sha256": hashlib.sha256(doc.text.encode("utf-8")).hexdigest()} for doc in documents],
        "question_overlaps_in_corpus": [
            {"question_number": number, "source": doc.source}
            for number, question in enumerate(questions, 1) for doc in documents
            if normalize(question["question"]) in normalize(doc.text)
        ],
        "questions_file": str(questions_path.resolve()),
        "questions_sha256": index_digest(questions_path),
        "modes": modes, "rag_settings": asdict(parameters), "request_options": asdict(options),
        "base_url": evaluation_model.base_url if evaluation_model is not None else None,
        "system_prompt": "", "isolation": "Fresh dialogue per question/mode; temporary empty stores; no MCP, profile, invariants or prior task memory. RAG dialogue preparation runs normally.",
        "manual_score_scale": "0: incorrect, 1: partial, 2: correct and complete; null: not reviewed",
        "semantic_support_scale": "full / partial / unsupported; null: not reviewed or not applicable. Review answer against quotes and their complete chunks, independently of literal matching.",
        "refusal_appropriate_scale": "true / false; null: not reviewed or not applicable. Review available context and expected evidence.",
        "usage_notes": "request_usage contains final-response callbacks only; dialogue_usage totals include archived rejected/repair responses. Dialogue preparation usage (including rewrite) is separate in rag_preparation and not duplicated in rag_context.rewrite. Missing API usage cannot be reconstructed.",
        "runs": [],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_report(output, report)
    retriever = Retriever(index_path, cache=cache)
    try:
        with tempfile.TemporaryDirectory(prefix="rag-evaluation-", dir=output.parent) as directory:
            isolated = Path(directory)
            service = None if retrieval_only else ConversationService(
                isolated / "conversations", token, invariants_path=isolated / "invariants.json",
            )
            if service is not None:
                service.models.save(evaluation_model)
                service.retriever = retriever
            for number, question in enumerate(questions, 1):
                for mode in modes:
                    ensure_fixed_index(index_path, digest)
                    rewrite, filtered = MODES[mode]
                    settings = RAGSettings(
                        rewrite_enabled=rewrite, filter_enabled=filtered,
                        top_k_before=top_k_before, top_k_after=top_k_after,
                        similarity_threshold=threshold,
                    )
                    row = {
                        "question_number": number, "question": question["question"], "mode": mode,
                        "group": question.get("group", "benchmark"),
                        "expected": question.get("expected"), "evidence": question["evidence"],
                        "manual_score": None, "unsupported_details": None, "review_notes": None,
                        "semantic_support": None, "refusal_appropriate": None, "rag_answer": None,
                        "answer": None, "error": None, "request_usage": [], "dialogue_usage": None,
                    }
                    if retrieval_only:
                        context = retriever.retrieve(question["question"], settings)
                        row.update(status="retrieval_only", rag_context=context, elapsed_seconds=context["retrieval_seconds"])
                    else:
                        conversation = service.create()
                        result = service.send(
                            conversation.id, question["question"], options=options,
                            settings=ContextSettings(rag_enabled=True, **{
                                "rag_" + name: value for name, value in asdict(settings).items()
                            }),
                            on_response=lambda response: row["request_usage"].append(
                                response.response.usage.model_dump() if response.response.usage else None
                            ),
                        )
                        turn = result.turns[-1]
                        context = turn.rag_context
                        row.update(
                            conversation_id=conversation.id, status=turn.status, answer=turn.answer,
                            error=turn.error, rag_context=context, elapsed_seconds=turn.elapsed_seconds,
                            rag_answer=turn.rag_answer,
                            rag_preparation=turn.rag_preparation,
                            dialogue_usage=dialogue_usage(result.working_context),
                        )
                    row["citation_metrics"] = citation_metrics(row["rag_answer"], context)
                    ensure_fixed_index(index_path, digest)
                    if context is not None:
                        if context["corpus_hash"] != metadata["corpus_hash"] or context["embedding"] != metadata["embedding"]:
                            raise ValueError("Метаданные поиска не соответствуют зафиксированному индексу")
                        row["candidate_evidence"] = evidence_metrics(question["evidence"], context["candidates"])
                        row["selected_evidence"] = evidence_metrics(question["evidence"], context["chunks"])
                        row["context_characters"] = sum(len(chunk["text"]) for chunk in context["chunks"])
                        row["context_tokens"] = sum(chunk["token_count"] for chunk in context["chunks"])
                    report["runs"].append(row)
                    write_report(output, report)
                    print(f"{number}/{len(questions)} {mode}: {row['status']}", file=sys.stderr)
        report["status"] = "completed_with_errors" if any(
            row["status"] not in ("completed", "retrieval_only") for row in report["runs"]
        ) else "completed"
        if report["status"] == "completed" and any(
            value["rewrite_fallbacks"] for value in summarize(report["runs"]).values()
        ):
            report["status"] = "completed_with_fallback"
    except (Exception, KeyboardInterrupt):
        report["status"] = "interrupted"
        write_report(output, report)
        raise
    report["finished_at"] = utc_now()
    write_report(output, report)
    return report


def main(argv: list[str] | None = None) -> int:
    load_env(ENV_PATH)
    default_index = Path(os.environ.get("RAG_INDEX_PATH", "data/indexes/structural.sqlite3"))
    if not default_index.is_absolute():
        default_index = ROOT / default_index
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=ROOT / "examples/indexing/questions.json")
    parser.add_argument("--index", type=Path, default=default_index)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--top-k-before", type=int, default=20)
    parser.add_argument("--top-k-after", type=int, default=5)
    parser.add_argument("--modes", nargs="+", choices=MODES)
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=BASE_URL, help="Адрес OpenAI-совместимого API")
    parser.add_argument("--model-token-env", help="Переменная с токеном; API_KEY по умолчанию только для DeepSeek")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args(argv)
    modes = args.modes or (["baseline", "filtered"] if args.retrieval_only else list(MODES))
    try:
        report = run_evaluation(
            questions_path=args.questions, index_path=args.index, output=args.output,
            threshold=args.threshold, modes=modes, top_k_before=args.top_k_before,
            top_k_after=args.top_k_after, retrieval_only=args.retrieval_only,
            token="" if args.retrieval_only else evaluation_token(args.base_url, args.model_token_env),
            cache=args.cache_dir, model=args.model, base_url=args.base_url,
            temperature=args.temperature, max_tokens=args.max_tokens,
        )
    except (ValueError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 1
    except Exception as error:
        # Тела исключений SDK могут содержать секреты; выводим только класс.
        print(f"Сравнение прервано: {type(error).__name__}", file=sys.stderr)
        return 1
    print(json.dumps({"output": str(args.output.resolve()), "status": report["status"], "summary": report["summary"]}, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
