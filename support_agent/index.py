"""Собрать отдельный индекс поддержки на уже закешированной модели агента."""

import argparse
import json
import os
import sqlite3

from llm_agent.indexing.chunking import load_corpus
from llm_agent.indexing.embeddings import Embedder, embedding_options
from llm_agent.indexing.pipeline import build
from llm_agent.indexing.store import load_index

from .config import ROOT, Settings, project_path


def main() -> int:
    settings = Settings.from_env()
    parser = argparse.ArgumentParser(description="Индекс поддержки: существующий индекс задаёт модель, веса читаются из кеша")
    parser.add_argument("--source-index", default=os.getenv("RAG_INDEX_PATH", "data/indexes/structural.sqlite3"))
    args = parser.parse_args()
    try:
        source = project_path(args.source_index).resolve()
        if source == settings.index_path.resolve():
            raise ValueError("Исходный индекс и индекс поддержки должны различаться.")
        metadata = load_index(source)[0]["embedding"]
        documents = load_corpus(ROOT, ROOT / "docs/support/corpus.json")
        embedder = Embedder(metadata["model"], metadata["revision"], settings.cache, offline=True,
                            **embedding_options(metadata))
        build(settings.index_path, documents, embedder, "structural", size=300, overlap=40)
        report = load_index(settings.index_path)[0]
    except (OSError, ValueError, KeyError, sqlite3.Error) as error:
        parser.exit(1, f"Не удалось собрать индекс поддержки: {error}\nПроверьте исходный индекс и кеш embedding-модели; инструкции в support_agent/README.md.\n")
    print(json.dumps({"index": str(settings.index_path), "documents": len(documents),
                      "chunks": report["chunk_count"], "model": report["embedding"]["model"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
