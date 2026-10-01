"""Сборка индекса и измерение двух стратегий на одном снимке корпуса."""

from datetime import datetime, timezone
import math
import statistics
import time

from .chunking import chunk_documents, corpus_hash
from .store import save_index, search


def corpus_stats(documents):
    chars = sum(len(d.text) for d in documents)
    return dict(documents=len(documents), characters=chars,
                words=sum(len(d.text.split()) for d in documents),
                pages_at_3000_chars=round(chars / 3000, 2),
                sources=[dict(source=d.source, characters=len(d.text), sha256=d.sha256)
                         for d in documents])


def build(path, documents, embedder, strategy, size=350, overlap=50, batch_size=16):
    started = time.perf_counter()
    chunks = chunk_documents(documents, embedder.tokenizer, strategy, size, overlap)
    chunk_seconds = time.perf_counter() - started
    started = time.perf_counter()
    vectors = embedder.encode([c.text for c in chunks], batch_size=batch_size)
    metadata = dict(schema_version=1, created_at=datetime.now(timezone.utc).isoformat(),
                    corpus_hash=corpus_hash(documents), strategy=strategy, size=size,
                    overlap=overlap, chunk_count=len(chunks), embedding=embedder.metadata,
                    chunk_seconds=chunk_seconds, embedding_seconds=time.perf_counter() - started)
    save_index(path, documents, chunks, vectors, metadata)
    return metadata


def statistics_report(index):
    metadata, documents, chunks, _ = index
    sizes = sorted(c.token_count for c in chunks)
    covered = {d.source: bytearray(len(d.text)) for d in documents}
    for c in chunks:
        covered[c.source][c.start:c.end] = b"\1" * (c.end - c.start)
    unique_chars = sum(sum(v) for v in covered.values())
    return dict(metadata=metadata, corpus=corpus_stats(documents), chunks=len(chunks),
                tokens=dict(min=min(sizes), median=statistics.median(sizes),
                            mean=statistics.mean(sizes), p95=sizes[math.ceil(.95*len(sizes))-1],
                            max=max(sizes), total=sum(sizes)),
                short_chunks_under_50=sum(n < 50 for n in sizes),
                covered_characters=unique_chars,
                repeated_characters=sum(len(c.text) for c in chunks) - unique_chars)


def compare(indexes, questions, embedder, top_k=5):
    first = next(iter(indexes.values()))[0]
    for metadata, documents, _, _ in indexes.values():
        for key in ("corpus_hash", "embedding", "size", "overlap"):
            if metadata[key] != first[key]:
                raise ValueError(f"Несовместимые индексы: {key}")
        docs = {d.source: " ".join(d.text.split()) for d in documents}
        if not questions:
            raise ValueError("Набор вопросов пуст")
        for q in questions:
            if not q.get("evidence") or any(
                not e["quote"].strip() or " ".join(e["quote"].split()) not in docs.get(e["source"], "")
                for e in q["evidence"]
            ):
                raise ValueError(f"Устаревший или пустой эталон: {q['question']}")
    if embedder.metadata != first["embedding"]:
        raise ValueError("Настройки embedding-модели не совпадают с индексом")
    queries = embedder.encode([q["question"] for q in questions], query=True)
    report = {}
    for strategy, index in indexes.items():
        rows = []
        for question, vector in zip(questions, queries):
            hits = search(index[2], index[3], vector, top_k)
            relevant = [any(e["source"] == h["source"] and
                           " ".join(e["quote"].split()) in " ".join(h["text"].split())
                           for e in question["evidence"]) for h in hits]
            rank = next((i + 1 for i, yes in enumerate(relevant) if yes), None)
            rows.append(dict(question=question, first_relevant_rank=rank, results=hits))
        report[strategy] = dict(
            statistics=statistics_report(index), questions=rows,
            hit_at_1=sum(r["first_relevant_rank"] == 1 for r in rows) / len(rows),
            hit_at_k=sum(r["first_relevant_rank"] is not None for r in rows) / len(rows),
            mrr_at_k=sum(1 / r["first_relevant_rank"] if r["first_relevant_rank"] else 0
                         for r in rows) / len(rows))
    return dict(top_k=top_k, strategies=report)
