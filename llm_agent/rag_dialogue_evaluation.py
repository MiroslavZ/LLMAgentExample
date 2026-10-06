"""Последовательная проверка диалогового RAG и сохранения памяти задачи."""

import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys

from .agent import DEFAULT_MODEL
from .cli import ENV_PATH, load_env
from .indexing.store import load_index
from .models import ContextSettings, RequestOptions, utc_now
from .rag import ROOT, Retriever
from .rag_evaluation import citation_metrics, ensure_fixed_index, index_digest, normalize
from .service import ConversationService


def load_scenarios(path: Path, documents: list) -> list[dict]:
    scenarios = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(scenarios, list) or len(scenarios) < 2:
        raise ValueError("Нужны минимум два сценария")
    sources = {doc.source for doc in documents}
    identifiers = set()
    for scenario in scenarios:
        if not isinstance(scenario, dict) or not isinstance(scenario.get("id"), str):
            raise ValueError("Сценарий требует строковый id")
        if not scenario["id"].strip() or scenario["id"] in identifiers:
            raise ValueError("Идентификаторы сценариев должны быть непустыми и уникальными")
        identifiers.add(scenario["id"])
        turns = scenario.get("turns")
        if not isinstance(turns, list) or not 10 <= len(turns) <= 15:
            raise ValueError("Сценарий требует 10–15 пользовательских ходов")
        for turn in turns:
            if (not isinstance(turn, dict) or not isinstance(turn.get("question"), str)
                    or not turn["question"].strip()):
                raise ValueError("Каждый ход требует непустой question")
            expected = turn.get("expected_sources")
            if (not isinstance(expected, list) or any(not isinstance(source, str) for source in expected)
                    or not set(expected).issubset(sources)):
                raise ValueError("expected_sources должны присутствовать в снимке индекса")
            checks = turn.get("checks")
            if not isinstance(checks, list) or not checks or any(not isinstance(item, str) or not item.strip() for item in checks):
                raise ValueError("Каждый ход требует непустые checks для ручной оценки")
    return scenarios


def summarize(rows: list[dict]) -> dict:
    answered = [row for row in rows if (row.get("rag_answer") or {}).get("status") == "answered"]
    return {
        "turns": len(rows),
        "completed": sum(row["status"] == "completed" for row in rows),
        "retrievals": sum(row.get("rag_context") is not None for row in rows),
        "substantive_answers": len(answered),
        "unknown_answers": sum((row.get("rag_answer") or {}).get("status") == "unknown" for row in rows),
        "preparation_fallbacks": sum((row.get("rag_preparation") or {}).get("status") == "fallback" for row in rows),
        "rewrite_fallbacks": sum((row.get("rag_context") or {}).get("rewrite", {}).get("status") == "fallback" for row in rows),
        "citation_rates_among_substantive_answers": {
            key: sum(row["citation_metrics"][key] is True for row in answered) / len(answered) if answered else None
            for key in ("sources_present", "quotes_present", "sources_match_context", "quotes_match_chunks")
        },
        "semantic_reviews": sum(row["semantic_support"] is not None for row in rows),
        "goal_reviews": sum(row["goal_retained"] is not None for row in rows),
        "goal_retained": sum(row["goal_retained"] is True for row in rows),
        "memory_reviews": sum(row["memory_correct"] is not None for row in rows),
        "memory_correct": sum(row["memory_correct"] is True for row in rows),
    }


def write_report(path: Path, report: dict) -> None:
    report["summary"] = summarize(report["runs"])
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def run_evaluation(*, scenarios_path: Path, index_path: Path, output: Path,
                   threshold: float = 0.35, token: str | None = None, cache: str | None = None,
                   model: str = DEFAULT_MODEL, max_tokens: int = 2500) -> dict:
    """Один диалог на сценарий, один send на ход; контрольные ответы не уходят LLM."""
    settings = ContextSettings(strategy="window", window_size=2, rag_enabled=True,
                               rag_rewrite_enabled=True, rag_filter_enabled=True,
                               rag_similarity_threshold=threshold)
    options = RequestOptions(model=model, temperature=0.0, max_tokens=max_tokens)
    settings.validate()
    options.validate()
    output, index_path = output.resolve(), index_path.resolve()
    data_dir = output.with_suffix(".data")
    if output.exists() or data_dir.exists():
        raise ValueError("Отчёт или изолированный каталог уже существует; выберите новый --output")
    digest = index_digest(index_path)
    metadata, documents, _, _ = load_index(index_path)
    scenarios = load_scenarios(scenarios_path, documents)
    ensure_fixed_index(index_path, digest)
    report = {
        "created_at": utc_now(), "status": "running", "index": str(index_path),
        "index_sha256": digest, "index_metadata": metadata,
        "corpus_documents": [{"source": doc.source, "sha256": hashlib.sha256(doc.text.encode()).hexdigest()} for doc in documents],
        "scenarios_file": str(scenarios_path.resolve()), "scenarios_sha256": index_digest(scenarios_path),
        "question_overlaps_in_corpus": [
            {"scenario": scenario["id"], "turn": number, "source": doc.source}
            for scenario in scenarios for number, turn in enumerate(scenario["turns"], 1)
            for doc in documents if normalize(turn["question"]) in normalize(doc.text)
        ],
        "settings": asdict(settings), "request_options": asdict(options),
        "data_dir": str(data_dir), "restart_after_turn": 7,
        "isolation": "Separate stores per scenario; no user profiles, MCP, manual memory or invariants.",
        "manual_review": "null means not reviewed. semantic_support: full/partial/unsupported; goal_retained, memory_correct, refusal_appropriate: boolean. Review checks against full answers and memory; citation metrics alone do not prove semantic quality.",
        "restarts": [], "runs": [],
    }
    data_dir.mkdir(parents=True)
    write_report(output, report)
    retriever = Retriever(index_path, cache=cache)

    def create_service(directory: Path) -> ConversationService:
        service = ConversationService(directory / "conversations", token, invariants_path=directory / "invariants.json")
        service.retriever = retriever
        return service

    try:
        for scenario_number, scenario in enumerate(scenarios, 1):
            isolated = data_dir / str(scenario_number)
            service = create_service(isolated)
            conversation = service.create()
            for number, expected in enumerate(scenario["turns"], 1):
                ensure_fixed_index(index_path, digest)
                memory_before = deepcopy(conversation.dialogue_task_memory)
                conversation = service.send(conversation.id, expected["question"], options=options, settings=settings)
                turn = conversation.turns[-1]
                context = turn.rag_context
                ensure_fixed_index(index_path, digest)
                if context is not None and (context["corpus_hash"] != metadata["corpus_hash"] or context["embedding"] != metadata["embedding"]):
                    raise ValueError("Метаданные поиска не соответствуют снимку индекса")
                row = {
                    "scenario": scenario["id"], "turn": number, "conversation_id": conversation.id,
                    "question": expected["question"], "expected_sources": expected["expected_sources"],
                    "checks": expected["checks"], "status": turn.status, "answer": turn.answer,
                    "error": turn.error, "elapsed_seconds": turn.elapsed_seconds,
                    "rag_answer": deepcopy(turn.rag_answer), "rag_context": deepcopy(context),
                    "rag_preparation": deepcopy(turn.rag_preparation),
                    "memory_before": memory_before, "memory_after": deepcopy(conversation.dialogue_task_memory),
                    "citation_metrics": citation_metrics(turn.rag_answer, context),
                    "expected_source_hits": sorted(set(expected["expected_sources"]) & {
                        chunk["source"] for chunk in (context or {}).get("chunks", [])
                    }),
                    "semantic_support": None, "goal_retained": None, "memory_correct": None,
                    "refusal_appropriate": None, "review_notes": None,
                }
                report["runs"].append(row)
                if number == 7:
                    before_restart = deepcopy(conversation.dialogue_task_memory)
                    service = create_service(isolated)
                    conversation = service.get(conversation.id)
                    report["restarts"].append({
                        "scenario": scenario["id"], "after_turn": number,
                        "memory_preserved": before_restart == conversation.dialogue_task_memory,
                        "turns_preserved": len(conversation.turns) == number,
                        "memory_after_reload": deepcopy(conversation.dialogue_task_memory),
                    })
                write_report(output, report)
                print(f"{scenario['id']} {number}/{len(scenario['turns'])}: {turn.status}", file=sys.stderr)
        report["status"] = "completed" if all(row["status"] == "completed" for row in report["runs"]) else "completed_with_errors"
        if any(not item["memory_preserved"] or not item["turns_preserved"] for item in report["restarts"]):
            report["status"] = "completed_with_errors"
        if report["status"] == "completed" and summarize(report["runs"])["preparation_fallbacks"]:
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", type=Path, default=ROOT / "docs/evaluation/day25_scenarios.json")
    parser.add_argument("--index", type=Path, default=Path(os.environ.get("RAG_INDEX_PATH", "data/indexes/structural.sqlite3")))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=2500)
    args = parser.parse_args(argv)
    try:
        token = os.environ.get("API_KEY", "").strip()
        if not token:
            raise ValueError("Для сценарного прогона требуется API_KEY в окружении или .env")
        report = run_evaluation(scenarios_path=args.scenarios,
                                index_path=args.index if args.index.is_absolute() else ROOT / args.index,
                                output=args.output, threshold=args.threshold, token=token,
                                cache=args.cache_dir, model=args.model, max_tokens=args.max_tokens)
    except (ValueError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 1
    except Exception as error:
        print(f"Сценарный прогон прерван: {type(error).__name__}", file=sys.stderr)
        return 1
    print(json.dumps({"output": str(args.output.resolve()), "status": report["status"], "summary": report["summary"]}, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
