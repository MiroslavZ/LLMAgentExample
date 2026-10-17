"""Build a bounded review input from Git objects and local lexical retrieval."""

from __future__ import annotations

import json
import re
import sqlite3
import subprocess
from collections import Counter
from pathlib import Path, PurePosixPath

from ai_review import ReviewError

MAX_FILES = 20
MAX_FILE_BYTES = 40_000
MAX_CONTEXT_BYTES = 180_000
MAX_RETRIEVAL_BYTES = 24_000
MAX_CORPUS_BYTES = 3_000_000
MAX_CORPUS_FILES = 400
SOURCES_PER_KIND = 4
CHUNK_CHARS = 3_000
_CODE_ROOTS = {"llm_agent", "tests", "ai_review", "localmcpserver"}
_CODE_SUFFIXES = {".py", ".js", ".ts", ".tsx", ".jsx", ".css", ".html", ".sql", ".sh"}
_EXCLUDED_DIRS = {
    ".git", ".venv", "venv", "env", "node_modules", "data", "evaluation",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "dist", "build",
}
_TRAINING_DOCS = {"chat.md", "tasks.md", "lecture.md", "solutions.md"}
_STOP_WORDS = {
    "def", "class", "return", "from", "import", "self", "none", "true", "false",
    "else", "elif", "with", "for", "and", "not", "try", "except", "raise",
    "str", "int", "bool", "list", "dict", "set", "tuple", "async", "await",
    "diff", "git", "index", "test", "tests", "assert", "pass",
}


def _git(repo: Path, *args: str, input_data: bytes | None = None) -> bytes:
    try:
        result = subprocess.run(
            ["git", "--no-pager", "-C", str(repo), *args], input=input_data,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=45, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ReviewError("Не удалось выполнить Git: проверьте установку и доступ к репозиторию.") from exc
    if result.returncode:
        raise ReviewError(f"Git {args[0]} завершился с ошибкой; проверьте refs и репозиторий.")
    return result.stdout


def _resolve(repo: Path, ref: str) -> str:
    if not ref or ref.startswith("-") or "\x00" in ref:
        raise ReviewError("Некорректная ссылка на Git commit.")
    return _git(repo, "rev-parse", "--verify", "--end-of-options", f"{ref}^{{commit}}").decode().strip()


def _tree(repo: Path, sha: str) -> dict[str, tuple[str, str, int]]:
    entries = {}
    for raw in _git(repo, "ls-tree", "-r", "-l", "-z", sha).split(b"\0"):
        if not raw:
            continue
        metadata, path_bytes = raw.split(b"\t", 1)
        mode, _kind, oid, size = metadata.split()
        try:
            path = path_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ReviewError("Путь в Git не закодирован в UTF-8.") from exc
        entries[path] = (mode.decode(), oid.decode(), int(size) if size != b"-" else 0)
    return entries


def _blobs(repo: Path, oids: list[str]) -> dict[str, bytes]:
    if not oids:
        return {}
    data = _git(repo, "cat-file", "--batch", input_data=("\n".join(dict.fromkeys(oids)) + "\n").encode())
    found = {}
    offset = 0
    while offset < len(data):
        end = data.index(b"\n", offset)
        header = data[offset:end].split()
        if len(header) != 3 or header[1] != b"blob":
            raise ReviewError("Git вернул недоступный или нетекстовый объект вместо blob.")
        size = int(header[2])
        found[header[0].decode()] = data[end + 1:end + 1 + size]
        offset = end + size + 2
    return found


def _excluded(path: str) -> bool:
    parts = PurePosixPath(path.lower()).parts
    name = parts[-1]
    return (
        any(part in _EXCLUDED_DIRS for part in parts[:-1])
        or name.startswith(".env")
        or name in {"credentials", "credentials.json", "id_rsa", "id_ed25519"}
        or PurePosixPath(name).suffix in {".pem", ".key", ".p12", ".pfx"}
        or (parts[0] == "docs" and (name in _TRAINING_DOCS or re.fullmatch(r"day\d+_(?:plan|results)\.md", name) is not None))
    )


def _kind(path: str) -> str | None:
    if _excluded(path):
        return None
    item = PurePosixPath(path.lower())
    if (item.name in {"readme.md", "agents.md"} and (len(item.parts) == 1 or item.parts[0] in _CODE_ROOTS)) or (item.parts[0] == "docs" and item.suffix in {".md", ".rst", ".txt"}):
        return "documentation"
    if item.parts[0] in _CODE_ROOTS and item.suffix in _CODE_SUFFIXES:
        return "code"
    return None


def _text(raw: bytes) -> str | None:
    if b"\0" in raw:
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _changes(repo: Path, base: str, head: str) -> list[dict]:
    parts = _git(repo, "diff", "--no-ext-diff", "--no-textconv", "--name-status", "-z", "--find-renames", base, head).split(b"\0")
    changes = []
    offset = 0
    statuses = {"A": "added", "D": "deleted", "M": "modified", "T": "type_changed", "R": "renamed"}
    while offset < len(parts) and parts[offset]:
        status = parts[offset].decode("ascii")
        path = parts[offset + 1].decode("utf-8")
        offset += 2
        old_path = None
        if status.startswith("R"):
            old_path, path = path, parts[offset].decode("utf-8")
            offset += 1
        changes.append({"path": path, "old_path": old_path, "status": statuses.get(status[0], status)})
    return changes


def _chunks(text: str):
    lines = text.splitlines(keepends=True)
    start = 0
    while start < len(lines):
        end, size = start, 0
        while end < len(lines) and end - start < 60:
            if size and size + len(lines[end]) > CHUNK_CHARS:
                break
            size += len(lines[end])
            end += 1
        yield start + 1, "".join(lines[start:end])[:CHUNK_CHARS]
        start = end


def _query(change: dict, patch: str) -> str:
    identifiers = []
    for line in patch.splitlines():
        if line.startswith(("+", "-")) and not line.startswith(("+++", "---")):
            identifiers.extend(re.findall(r"[\w]{3,}", line[1:], flags=re.UNICODE))
    path_terms = re.findall(r"[\w]{3,}", change["path"], flags=re.UNICODE)
    terms = [term.lower() for term in path_terms + [term for term, _ in Counter(identifiers).most_common(32)]]
    terms += [part for term in terms for part in term.split("_") if len(part) >= 3]
    terms = list(dict.fromkeys(term for term in terms if term not in _STOP_WORDS and not term.isdigit()))[:32]
    return " OR ".join(f'"{term}"' for term in terms)


def _retrieve(repo: Path, sha: str, tree: dict, queries: list[str]) -> tuple[list[dict], dict]:
    stats = {"method": "sqlite_fts5_bm25", "snapshot_sha": sha, "corpus_files": 0,
             "corpus_bytes": 0, "chunks": 0, "queries": len(queries), "omitted_corpus_files": 0}
    selected = []
    for path, (mode, oid, size) in sorted(tree.items()):
        kind = _kind(path)
        if not kind:
            continue
        if mode not in {"100644", "100755"} or size > MAX_FILE_BYTES or len(selected) >= MAX_CORPUS_FILES or stats["corpus_bytes"] + size > MAX_CORPUS_BYTES:
            stats["omitted_corpus_files"] += 1
            continue
        selected.append((path, oid, kind))
        stats["corpus_bytes"] += size
    blobs = _blobs(repo, [oid for _, oid, _ in selected])
    try:
        connection = sqlite3.connect(":memory:")
    except sqlite3.Error as exc:
        raise ReviewError("Не удалось создать локальный индекс SQLite FTS5.") from exc
    try:
        connection.execute("CREATE VIRTUAL TABLE chunks USING fts5(path, text, kind UNINDEXED, start_line UNINDEXED)")
        for path, oid, kind in selected:
            text = _text(blobs[oid])
            if text is None:
                stats["omitted_corpus_files"] += 1
                continue
            stats["corpus_files"] += 1
            for start, chunk in _chunks(text):
                connection.execute("INSERT INTO chunks(path,text,kind,start_line) VALUES (?,?,?,?)", (path, chunk, kind, start))
                stats["chunks"] += 1
        sources = []
        used_bytes = 0
        for kind in ("documentation", "code"):
            kind_bytes = 0
            candidates = {}
            for query in queries:
                if not query:
                    continue
                rows = connection.execute(
                    "SELECT rowid,path,start_line,text,bm25(chunks,2.0,1.0) FROM chunks "
                    "WHERE chunks MATCH ? AND kind=? ORDER BY bm25(chunks,2.0,1.0),rowid LIMIT ?",
                    (query, kind, SOURCES_PER_KIND),
                )
                for row in rows:
                    if row[0] not in candidates or row[4] < candidates[row[0]][4]:
                        candidates[row[0]] = row
            count = 0
            for _, path, start, text, _score in sorted(candidates.values(), key=lambda row: (row[4], row[0])):
                source = {"id": f"R{len(sources) + 1}", "path": path, "start_line": int(start), "text": text, "kind": kind}
                size = len(json.dumps(source, ensure_ascii=False).encode("utf-8"))
                if kind_bytes + size > MAX_RETRIEVAL_BYTES // 2:
                    continue
                sources.append(source)
                used_bytes += size
                kind_bytes += size
                count += 1
                if count == SOURCES_PER_KIND:
                    break
            stats[f"{kind}_sources"] = count
        stats["context_bytes"] = used_bytes
        stats["warnings"] = [f"Не найдены релевантные фрагменты: {kind}." for kind in ("documentation", "code") if not stats[f"{kind}_sources"]]
        if stats["omitted_corpus_files"]:
            stats["warnings"].append("Часть разрешённого корпуса исключена из-за типа или лимита размера.")
        return sources, stats
    except sqlite3.Error as exc:
        raise ReviewError(f"Поиск SQLite FTS5 недоступен: {exc}") from exc
    finally:
        connection.close()


def collect_context(repo: Path, base: str, head: str) -> dict:
    """Read merge-base..head changes and retrieve trusted context at base SHA.

    Only regular UTF-8 blobs enter the model input. Every omitted changed file is
    reported in ``skipped``. No working-tree files or PR Python modules are read.
    """
    repo = Path(repo).resolve()
    base_sha, head_sha = _resolve(repo, base), _resolve(repo, head)
    merge_base = _git(repo, "merge-base", base_sha, head_sha).decode().strip()
    trees = {sha: _tree(repo, sha) for sha in dict.fromkeys((base_sha, merge_base, head_sha))}
    changes = _changes(repo, merge_base, head_sha)
    result = {"base_sha": base_sha, "merge_base_sha": merge_base, "head_sha": head_sha,
              "diff": "", "files": [], "sources": [], "skipped": [], "retrieval": {}}
    queries = []
    content_budget = MAX_CONTEXT_BYTES - MAX_RETRIEVAL_BYTES - 8_192
    for change in changes:
        path, old = change["path"], change["old_path"] or change["path"]
        before_entry, after_entry = trees[merge_base].get(old), trees[head_sha].get(path)
        entries = [entry for entry in (before_entry, after_entry) if entry]
        reason = None
        if _excluded(path) or _excluded(old):
            reason = "Файл исключён правилами корпуса (данные, секреты или учебные материалы)."
        elif any(entry[0] not in {"100644", "100755"} for entry in entries):
            reason = "Символическая ссылка, submodule или другой неподдерживаемый тип."
        elif any(entry[2] > MAX_FILE_BYTES for entry in entries):
            reason = f"Размер версии файла превышает {MAX_FILE_BYTES} байт."
        elif len(result["files"]) >= MAX_FILES:
            reason = f"Достигнут лимит {MAX_FILES} проанализированных файлов."
        if reason:
            result["skipped"].append({"path": path, "reason": reason})
            continue
        blobs = _blobs(repo, [entry[1] for entry in entries])
        before = _text(blobs[before_entry[1]]) if before_entry else None
        after = _text(blobs[after_entry[1]]) if after_entry else None
        if (before_entry and before is None) or (after_entry and after is None):
            result["skipped"].append({"path": path, "reason": "Бинарное содержимое или кодировка, отличная от UTF-8."})
            continue
        patch = _git(repo, "diff", "--no-ext-diff", "--no-textconv", "--no-color", "--unified=3", "--find-renames", merge_base, head_sha, "--", old, *([path] if path != old else [])).decode("utf-8", errors="replace")
        file = {**change, "before": before, "after": after}
        candidate = {"diff": result["diff"] + patch, "files": [*result["files"], file]}
        if len(json.dumps(candidate, ensure_ascii=False).encode("utf-8")) > content_budget:
            result["skipped"].append({"path": path, "reason": "Превышен общий бюджет diff и содержимого файлов."})
            continue
        result.update(candidate)
        queries.append(_query(change, patch))
    result["sources"], result["retrieval"] = _retrieve(repo, base_sha, trees[base_sha], queries)
    result["retrieval"].update(changed_files=len(changes), reviewed_files=len(result["files"]), skipped_files=len(result["skipped"]))
    if len(json.dumps(result, ensure_ascii=False).encode("utf-8")) > MAX_CONTEXT_BYTES:
        raise ReviewError("Список пропусков и метаданные превышают общий бюджет контекста; уменьшите PR.")
    return result
