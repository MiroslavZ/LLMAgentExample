"""Load an explicit text corpus and split it without rewriting source text."""

from __future__ import annotations

import hashlib
import json
import re
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


@dataclass(frozen=True)
class Document:
    source: str
    title: str
    text: str
    sha256: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source: str
    title: str
    section: str
    sections: list[str]
    strategy: str
    text: str
    start: int
    end: int
    token_count: int


_HEADING = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.+?)\s*$")
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")


def _section_spans(text: str) -> list[tuple[int, int, str]]:
    """Recognize ATX headings outside Markdown fenced code blocks."""
    starts: list[tuple[int, str]] = [(0, "")]
    headings: list[tuple[int, str]] = []
    fence: str | None = None
    position = 0
    for line in text.splitlines(keepends=True):
        marker = _FENCE.match(line)
        if fence is not None:
            if marker and marker[1][0] == fence[0] and len(marker[1]) >= len(fence) and not marker[2].strip():
                fence = None
        elif marker:
            fence = marker[1]
        else:
            heading = _HEADING.match(line)
            if heading:
                level = len(heading[1])
                title = re.sub(r"[ \t]+#+[ \t]*$", "", heading[2]).strip()
                while headings and headings[-1][0] >= level:
                    headings.pop()
                headings.append((level, title))
                path = " > ".join(item[1] for item in headings)
                if position == 0:
                    starts[0] = (0, path)
                else:
                    starts.append((position, path))
        position += len(line)
    return [(start, starts[index + 1][0] if index + 1 < len(starts) else len(text), section)
            for index, (start, section) in enumerate(starts)]


def load_corpus(root: Path, manifest: Path) -> list[Document]:
    """Read UTF-8 .md/.txt files explicitly allowed by a JSON manifest.

    Paths are relative to root; traversal and symlink escapes are rejected.
    Source order is canonical and therefore independent of manifest order.
    """
    root = root.resolve()
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("files"), list):
        raise ValueError("Corpus manifest must contain a 'files' list")
    paths: dict[str, Path] = {}
    seen: set[Path] = set()
    for entry in data["files"]:
        if not isinstance(entry, str) or not entry.strip():
            raise ValueError("Manifest file paths must be nonempty strings")
        relative = Path(entry)
        if relative.is_absolute() or relative.drive or ".." in relative.parts:
            raise ValueError(f"Corpus path must stay inside root: {entry}")
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Corpus path escapes root: {entry}")
        if path.suffix.lower() not in {".md", ".txt"}:
            raise ValueError(f"Unsupported corpus file: {entry}")
        if path in seen:
            raise ValueError(f"Duplicate corpus file: {entry}")
        seen.add(path)
        paths[relative.as_posix()] = path
    documents = []
    for source, path in sorted(paths.items()):
        text = path.read_bytes().decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
        if not text.strip():
            continue
        title = path.stem
        if path.suffix.lower() == ".md":
            title = next((section for _, _, section in _section_spans(text) if section), title)
        documents.append(Document(source, title, text, hashlib.sha256(text.encode("utf-8")).hexdigest()))
    return documents


def corpus_hash(documents: list[Document]) -> str:
    entries = sorted((document.source, document.sha256) for document in documents)
    payload = json.dumps(entries, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _count(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def _windows(text: str, tokenizer: Any, size: int, overlap: int) -> Iterator[tuple[int, int]]:
    offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    token_ends = [end for _, end in offsets]
    start = 0
    while start < len(text):
        first_token = bisect_right(token_ends, start)
        next_token = first_token + size
        end = len(text) if next_token >= len(offsets) else offsets[next_token][0]
        # Some byte-level tokenizers give multiple tokens the same Unicode offset.
        # Re-tokenization also catches boundary-dependent tokenization changes.
        if end <= start:
            end = offsets[first_token][1] if first_token < len(offsets) else len(text)
        while end > start and _count(tokenizer, text[start:end]) > size:
            end -= 1
        if end <= start:
            raise ValueError("Chunk size is too small to encode a single source character")
        yield start, end
        if end == len(text):
            break
        next_start = end
        if overlap:
            emitted = tokenizer(text[start:end], add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
            if emitted:
                next_start = start + emitted[max(0, len(emitted) - overlap)][0]
        start = max(start + 1, next_start)


def _paragraphs(text: str) -> Iterator[tuple[int, int]]:
    start = position = 0
    fence: str | None = None
    for line in text.splitlines(keepends=True):
        marker = _FENCE.match(line)
        if marker:
            if fence is None:
                fence = marker[1]
            elif marker[1][0] == fence[0] and len(marker[1]) >= len(fence) and not marker[2].strip():
                fence = None
        position += len(line)
        if not line.strip() and fence is None:
            yield start, position
            start = position
    if start < len(text):
        yield start, len(text)


def _structural_ranges(text: str, tokenizer: Any, size: int, overlap: int) -> Iterator[tuple[int, int]]:
    pending: tuple[int, int] | None = None
    for start, end in _paragraphs(text):
        if pending is not None and _count(tokenizer, text[pending[0]:end]) <= size:
            pending = (pending[0], end)
            continue
        if pending is not None:
            yield pending
            pending = None
        if _count(tokenizer, text[start:end]) > size:
            for local_start, local_end in _windows(text[start:end], tokenizer, size, overlap):
                yield start + local_start, start + local_end
        else:
            pending = (start, end)
    if pending is not None:
        yield pending


def chunk_documents(documents: list[Document], tokenizer: Any, strategy: str,
                    size: int = 350, overlap: int = 50) -> list[Chunk]:
    """Split into token-bounded original substrings with character offsets.

    Structural chunks respect sections and paragraphs; overlap applies only
    when an oversized paragraph requires the fixed-window fallback.
    """
    if strategy not in {"fixed", "structural"}:
        raise ValueError("Chunk strategy must be 'fixed' or 'structural'")
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("Require size > 0 and 0 <= overlap < size")
    result = []
    for document in documents:
        if not document.text.strip():
            continue
        spans = _section_spans(document.text) if Path(document.source).suffix.lower() == ".md" else [(0, len(document.text), "")]
        if strategy == "fixed":
            ranges = list(_windows(document.text, tokenizer, size, overlap))
        else:
            ranges = [(section_start + start, section_start + end)
                      for section_start, section_end, _ in spans
                      for start, end in _structural_ranges(document.text[section_start:section_end], tokenizer, size, overlap)]
        for start, end in ranges:
            text = document.text[start:end]
            if not text.strip():
                continue
            sections = list(dict.fromkeys(section for left, right, section in spans
                                          if left < end and right > start and section))
            identity = json.dumps([document.source, document.sha256, strategy, size, overlap, start, end], ensure_ascii=False)
            chunk_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
            result.append(Chunk(chunk_id, document.source, document.title,
                                sections[0] if sections else "", sections, strategy,
                                text, start, end, _count(tokenizer, text)))
    return result
