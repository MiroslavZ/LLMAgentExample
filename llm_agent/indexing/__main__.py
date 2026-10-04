"""Отдельный CLI: python -m llm_agent.indexing --help."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sqlite3
import sys
import time

from .chunking import load_corpus
from .embeddings import E5, Embedder, embedding_options
from .pipeline import build, compare, corpus_stats, statistics_report
from .store import load_index, search

ROOT = Path(__file__).resolve().parents[2]


def parser():
    p = argparse.ArgumentParser(description="Локальная индексация Markdown/TXT")
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("corpus", "build", "stats", "inspect", "verify", "search", "compare"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--root", type=Path, default=ROOT)
        cmd.add_argument("--corpus", type=Path, default=Path("examples/indexing/corpus.json"))
        cmd.add_argument("--output-dir", type=Path, default=Path("data/indexes"))
        if name not in ("corpus", "compare"):
            cmd.add_argument("--strategy", choices=("fixed", "structural", "all"),
                             default="structural" if name in ("search", "inspect") else "all")
        if name in ("build", "search", "compare"):
            cmd.add_argument("--cache-dir", type=Path)
            cmd.add_argument("--offline", action="store_true")
        if name == "build":
            cmd.add_argument("--model", default=E5, help="Hugging Face ID модели Sentence Transformers")
            cmd.add_argument("--query-prefix", help="Префикс поискового запроса; по умолчанию из модели")
            cmd.add_argument("--passage-prefix", help="Префикс документа; по умолчанию из модели")
            cmd.add_argument("--revision")
            cmd.add_argument("--size", type=int, default=350)
            cmd.add_argument("--overlap", type=int, default=50)
            cmd.add_argument("--batch-size", type=int, default=16)
        if name == "inspect":
            cmd.add_argument("--limit", type=int, default=3)
        if name == "search":
            cmd.add_argument("query")
        if name in ("search", "compare"):
            cmd.add_argument("--top-k", type=int, default=5)
        if name == "compare":
            cmd.add_argument("--questions", type=Path, default=Path("examples/indexing/questions.json"))
            cmd.add_argument("--report", type=Path, default=Path("reports/day21/comparison.json"))
    return p


def run(args):
    root = args.root.resolve()
    resolve = lambda path: path if path.is_absolute() else root / path
    if args.command == "corpus":
        return corpus_stats(load_corpus(root, resolve(args.corpus)))
    output = resolve(args.output_dir)
    strategies = ("fixed", "structural") if getattr(args, "strategy", "all") == "all" else (args.strategy,)
    if args.command == "build":
        if not 0 <= args.overlap < args.size or args.size > 500 or args.batch_size < 1:
            raise ValueError("Нужны 0 <= overlap < size <= 500 и batch-size > 0")
        documents = load_corpus(root, resolve(args.corpus))
        print("Загрузка embedding-модели...", file=sys.stderr)
        started = time.perf_counter()
        prefixes = {key: getattr(args, key) for key in ("query_prefix", "passage_prefix")
                    if getattr(args, key) is not None}
        model = Embedder(args.model, args.revision, args.cache_dir, args.offline, **prefixes)
        result = {"model_load_seconds": time.perf_counter() - started, "indexes": {}}
        for strategy in strategies:
            print(f"Индексация: {strategy}", file=sys.stderr)
            path = output / f"{strategy}.sqlite3"
            build(path, documents, model, strategy, args.size, args.overlap, args.batch_size)
            result["indexes"][strategy] = statistics_report(load_index(path))
        return result
    indexes = {s: load_index(output / f"{s}.sqlite3") for s in strategies}
    if args.command == "verify":
        return {s: {"valid": True, "chunks": len(i[2])} for s, i in indexes.items()}
    if args.command == "stats":
        return {s: dict(**statistics_report(i), file_bytes=(output / f"{s}.sqlite3").stat().st_size)
                for s, i in indexes.items()}
    if args.command == "inspect":
        if args.limit < 1:
            raise ValueError("limit должен быть положительным")
        return {s: [asdict(c) for c in i[2][:args.limit]] for s, i in indexes.items()}
    if args.top_k < 1:
        raise ValueError("top-k должен быть положительным")
    settings = next(iter(indexes.values()))[0]["embedding"]
    model = Embedder(settings["model"], settings["revision"], args.cache_dir, args.offline,
                     **embedding_options(settings))
    if model.metadata != settings:
        raise ValueError("Модель/версии библиотек отличаются от сохранённого индекса; пересоберите индекс")
    if args.command == "search":
        if any(i[0]["embedding"] != settings for i in indexes.values()):
            raise ValueError("Индексы используют разные модели")
        query = model.encode([args.query], query=True)[0]
        return {s: search(i[2], i[3], query, args.top_k) for s, i in indexes.items()}
    questions = json.loads(resolve(args.questions).read_text(encoding="utf-8"))
    report = compare(indexes, questions, model, args.top_k)
    path = resolve(args.report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"report": str(path), "metrics": {s: {k: v for k, v in r.items()
            if k not in ("questions", "statistics")} for s, r in report["strategies"].items()}}


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        result = run(args)
    except (OSError, ValueError, KeyError, TypeError, sqlite3.Error, ImportError) as exc:
        print(f"Ошибка индексации: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
