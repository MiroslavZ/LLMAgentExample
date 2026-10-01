"""SQLite со снимками документов и JSON-векторами, атомарная замена файла."""

import json
import hashlib
import math
import os
from pathlib import Path
import sqlite3
import tempfile
from dataclasses import asdict
from contextlib import closing

from .chunking import Chunk, Document, corpus_hash


def validate_vector(vector, dimension):
    if len(vector) != dimension or not all(math.isfinite(x) for x in vector):
        raise ValueError("Некорректная размерность или нечисловой вектор")
    if not math.isclose(sum(x * x for x in vector), 1, abs_tol=1e-4):
        raise ValueError("Вектор должен иметь единичную L2-норму")


def save_index(path, documents, chunks, vectors, metadata):
    if not chunks or len(chunks) != len(vectors):
        raise ValueError("Число векторов не соответствует числу чанков")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".sqlite3.tmp")
    os.close(fd)
    try:
        with closing(sqlite3.connect(temporary)) as db, db:
            db.executescript("""
                CREATE TABLE metadata (data TEXT NOT NULL);
                CREATE TABLE documents (source TEXT PRIMARY KEY, data TEXT NOT NULL);
                CREATE TABLE chunks (position INTEGER PRIMARY KEY, chunk_id TEXT UNIQUE NOT NULL,
                    source TEXT NOT NULL REFERENCES documents(source), data TEXT NOT NULL,
                    vector TEXT NOT NULL);
            """)
            db.execute("INSERT INTO metadata VALUES (?)", (json.dumps(metadata),))
            db.executemany("INSERT INTO documents VALUES (?, ?)",
                           [(d.source, json.dumps(asdict(d), ensure_ascii=False)) for d in documents])
            for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
                validate_vector(vector, metadata["embedding"]["dimension"])
                db.execute("INSERT INTO chunks VALUES (?, ?, ?, ?, ?)",
                           (i, chunk.chunk_id, chunk.source,
                            json.dumps(asdict(chunk), ensure_ascii=False), json.dumps(vector)))
        load_index(temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_index(path):
    path = Path(path).resolve()
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as db:
        if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("Нарушена целостность SQLite")
        metadata = json.loads(db.execute("SELECT data FROM metadata").fetchone()[0])
        if metadata["schema_version"] != 1:
            raise ValueError("Неизвестная версия индекса")
        documents = [Document(**json.loads(row[0])) for row in
                     db.execute("SELECT data FROM documents ORDER BY source")]
        chunks, vectors = [], []
        for chunk_id, source, data, vector in db.execute(
                "SELECT chunk_id, source, data, vector FROM chunks ORDER BY position"):
            chunk = Chunk(**json.loads(data))
            if chunk.chunk_id != chunk_id or chunk.source != source:
                raise ValueError("Нарушена связь чанка с метаданными")
            chunks.append(chunk)
            vectors.append(json.loads(vector))
    if not chunks or len(chunks) != metadata["chunk_count"]:
        raise ValueError("Некорректное число чанков")
    if corpus_hash(documents) != metadata["corpus_hash"]:
        raise ValueError("Не совпадает хеш корпуса")
    by_source = {d.source: d for d in documents}
    for doc in documents:
        if hashlib.sha256(doc.text.encode("utf-8")).hexdigest() != doc.sha256:
            raise ValueError("Не совпадает хеш снимка документа")
    coverage = {d.source: bytearray(len(d.text)) for d in documents}
    for chunk, vector in zip(chunks, vectors):
        validate_vector(vector, metadata["embedding"]["dimension"])
        doc = by_source.get(chunk.source)
        if (doc is None or not 0 <= chunk.start < chunk.end <= len(doc.text)
                or doc.text[chunk.start:chunk.end] != chunk.text
                or chunk.strategy != metadata["strategy"]
                or not 0 < chunk.token_count <= metadata["size"]):
            raise ValueError("Чанк не соответствует снимку документа или настройкам")
        coverage[chunk.source][chunk.start:chunk.end] = b"\1" * (chunk.end - chunk.start)
    for doc in documents:
        if any(not covered and not char.isspace()
               for char, covered in zip(doc.text, coverage[doc.source])):
            raise ValueError("Чанки не покрывают содержательный текст")
    return metadata, documents, chunks, vectors


def search(chunks, vectors, query_vector, top_k=5):
    if top_k < 1 or not vectors:
        raise ValueError("top_k должен быть положительным; индекс непустым")
    validate_vector(query_vector, len(vectors[0]))
    scores = [sum(a * b for a, b in zip(vector, query_vector)) for vector in vectors]
    positions = sorted(range(len(chunks)), key=lambda i: (-scores[i], i))[:top_k]
    return [dict(score=scores[i], **asdict(chunks[i])) for i in positions]
