"""Bounded UTF-8 file operations within an explicitly configured project root."""

from __future__ import annotations

import difflib
import fnmatch
import hashlib
import json
import os
import re
import stat
import tempfile
import threading
from pathlib import Path, PureWindowsPath


MAX_FILE_BYTES = 256 * 1024
MAX_READ_CHARS = 6000
MAX_READ_PAYLOAD_CHARS = 16_000
MAX_DIFF_CHARS = 6000
MAX_PAGE_CHARS = 6000
MAX_PAGE_SIZE = 200
MAX_WALK_ENTRIES = 20_000
MAX_SEARCH_BYTES = 8 * 1024 * 1024
MAX_SEARCH_LINE_CHARS = 500
MAX_DEPTH = 64

BLOCKED_DIRS = frozenset({
    ".git", ".hg", ".svn", ".venv", "venv", "env", "node_modules",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".idea",
    ".vscode", ".codex", ".agents", ".aws", ".ssh", ".tox", ".nox",
    "dist", "build", "logs",
})
SECRET_NAMES = frozenset({".netrc", ".npmrc", ".pypirc", ".dockercfg"})
SECRET_STEMS = frozenset({"secret", "secrets", "credential", "credentials", "token", "tokens"})
SECRET_SUFFIXES = frozenset({".pem", ".key", ".p12", ".pfx", ".keystore", ".jks", ".secret", ".token"})
SECRET_DATA_SUFFIXES = frozenset({"", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".txt", ".csv", ".tsv", ".xml"})
LOCAL_DATA_SUFFIXES = frozenset({
    ".log", ".db", ".sqlite", ".sqlite3", ".db-wal", ".db-shm",
    ".sqlite-wal", ".sqlite-shm", ".sqlite3-wal", ".sqlite3-shm",
})
_RESERVED_NAMES = {"con", "prn", "aux", "nul"} | {
    f"{prefix}{number}" for prefix in ("com", "lpt") for number in range(1, 10)
}
_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")


class ProjectFileError(ValueError):
    """An expected file-operation failure with a stable, machine-readable code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _forbidden_name(name: str) -> bool:
    lowered = name.casefold()
    suffix = Path(lowered).suffix
    # Names such as tokens.py and secrets.py are normal source files. Only
    # matching data-file names are excluded; this is not a content DLP scanner.
    secret_data = (
        suffix in SECRET_DATA_SUFFIXES
        and bool(set(lowered.lstrip(".").split(".")) & SECRET_STEMS)
    )
    return (
        lowered in BLOCKED_DIRS
        or lowered in SECRET_NAMES
        or lowered.startswith((".env", ".project-files-"))
        or secret_data
        or lowered.startswith(("id_rsa", "id_dsa", "id_ecdsa", "id_ed25519"))
        or suffix in SECRET_SUFFIXES
        or suffix in LOCAL_DATA_SUFFIXES
    )


def _linked(info: os.stat_result) -> bool:
    # Junctions are reparse points on Windows, but not necessarily symlinks.
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    )


def _integer(value: int, name: str, minimum: int, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ProjectFileError("invalid_argument", f"{name}: требуется целое число не меньше {minimum}.")
    if maximum is not None and value > maximum:
        raise ProjectFileError("invalid_argument", f"{name}: максимальное значение {maximum}.")
    return value


def _text_bytes(content: str) -> bytes:
    if not isinstance(content, str):
        raise ProjectFileError("invalid_argument", "Содержимое должно быть строкой.")
    if len(content) > MAX_FILE_BYTES:
        raise ProjectFileError("too_large", f"Размер файла превышает {MAX_FILE_BYTES} байт.")
    try:
        data = content.encode("utf-8")
    except UnicodeError:
        raise ProjectFileError("not_text", "Содержимое не является корректным текстом UTF-8.") from None
    if len(data) > MAX_FILE_BYTES:
        raise ProjectFileError("too_large", f"Размер файла превышает {MAX_FILE_BYTES} байт.")
    _decode(data)
    return data


def _decode(data: bytes) -> str:
    try:
        text = data.decode("utf-8")
    except UnicodeError:
        raise ProjectFileError("not_text", "Поддерживаются только текстовые файлы UTF-8.") from None
    if any(ord(char) < 32 and char not in "\t\r\n\f" for char in text):
        raise ProjectFileError("not_text", "Файл содержит бинарные управляющие символы.")
    return text


class ProjectFiles:
    """Small file service; every public operation enforces the same path policy.

    The lock serializes operations in this instance. Hash rechecks detect normal
    concurrent edits; this is not an OS sandbox against a hostile local process.
    """

    def __init__(self, root: Path) -> None:
        try:
            self.root = Path(root).expanduser().resolve(strict=True)
            if not self.root.is_dir():
                raise ProjectFileError("invalid_path", "Корень проекта должен быть существующим каталогом.")
        except OSError:
            raise ProjectFileError("invalid_path", "Корень проекта недоступен.") from None
        self._lock = threading.RLock()

    def _resolve(self, path: str) -> Path:
        if not isinstance(path, str) or not path or len(path) > 1024 or "\x00" in path:
            raise ProjectFileError("invalid_path", "Укажите непустой относительный путь длиной до 1024 символов.")
        if PureWindowsPath(path).drive or path.startswith(("/", "\\")):
            raise ProjectFileError("invalid_path", "Абсолютные пути и диски недоступны; используйте путь от корня проекта.")
        parts = []
        for part in path.replace("\\", "/").split("/"):
            if part in ("", "."):
                continue
            if (
                part == ".." or part.endswith((" ", "."))
                or any(char in part for char in '<>:"|?*')
                or any(ord(char) < 32 for char in part)
                or part.casefold().split(".", 1)[0] in _RESERVED_NAMES
            ):
                raise ProjectFileError("invalid_path", "Путь содержит запрещённый сегмент, имя устройства или '..'.")
            if _forbidden_name(part):
                raise ProjectFileError("path_forbidden", "Секреты и служебные каталоги недоступны файловым инструментам.")
            parts.append(part)
        candidate = self.root.joinpath(*parts)
        current = self.root
        try:
            for part in parts:
                current = current / part
                try:
                    info = current.lstat()
                except FileNotFoundError:
                    continue
                if _linked(info):
                    raise ProjectFileError("path_forbidden", "Символические ссылки и junction недоступны.")
            if not candidate.resolve().is_relative_to(self.root):
                raise ProjectFileError("path_forbidden", "Путь выходит за пределы проекта.")
        except OSError:
            raise ProjectFileError("io_error", "Не удалось проверить путь внутри проекта.") from None
        return candidate

    def _relative(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    def _read(self, path: Path) -> tuple[bytes, str]:
        # Revalidate immediately before I/O, including every existing parent.
        target = self._resolve(self._relative(path))
        try:
            info = target.stat()
            if not stat.S_ISREG(info.st_mode):
                raise ProjectFileError("not_file", "Указанный путь не является обычным файлом.")
            if info.st_size > MAX_FILE_BYTES:
                raise ProjectFileError("too_large", f"Размер файла превышает {MAX_FILE_BYTES} байт.")
            with target.open("rb") as stream:
                data = stream.read(MAX_FILE_BYTES + 1)
        except FileNotFoundError:
            raise ProjectFileError("not_found", "Файл не найден.") from None
        except OSError:
            raise ProjectFileError("io_error", "Не удалось прочитать файл.") from None
        if len(data) > MAX_FILE_BYTES:
            raise ProjectFileError("too_large", f"Размер файла превышает {MAX_FILE_BYTES} байт.")
        return data, _decode(data)

    @staticmethod
    def _remove_temporary(path: str | None) -> None:
        if path is None:
            return
        try:
            # An edit may have copied a read-only mode before a failed replace.
            if os.path.exists(path):
                os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
                os.unlink(path)
        except OSError:
            # Cleanup must not hide a stale-file or permission error.
            pass

    @staticmethod
    def _pattern(pattern: str) -> str:
        if not isinstance(pattern, str) or not pattern or len(pattern) > 512:
            raise ProjectFileError("invalid_argument", "pattern должен быть непустой маской длиной до 512 символов.")
        normalized = pattern.replace("\\", "/")
        if normalized.startswith("/") or ":" in normalized or ".." in normalized.split("/"):
            raise ProjectFileError("invalid_argument", "pattern должен быть относительной маской без '..'.")
        return normalized

    def _scan(self, path: str, pattern: str) -> tuple[list[Path], dict]:
        base = self._resolve(path)
        if not base.exists():
            raise ProjectFileError("not_found", "Каталог поиска не найден.")
        if not base.is_dir():
            raise ProjectFileError("not_directory", "Для обхода требуется каталог.")
        found: list[Path] = []
        pending = [(base, 0)]
        stats = {"scan_complete": True, "scanned_entries": 0, "skipped_files": {}}
        while pending:
            directory, depth = pending.pop()
            try:
                directory = self._resolve(self._relative(directory))
                with os.scandir(directory) as stream:
                    entries = []
                    for entry in stream:
                        if stats["scanned_entries"] >= MAX_WALK_ENTRIES:
                            stats["scan_complete"] = False
                            break
                        stats["scanned_entries"] += 1
                        entries.append(entry)
                for entry in sorted(entries, key=lambda item: item.name):
                    if _forbidden_name(entry.name):
                        continue
                    try:
                        target = self._resolve(self._relative(Path(entry.path)))
                        info = target.lstat()
                        if stat.S_ISDIR(info.st_mode):
                            if depth >= MAX_DEPTH:
                                stats["scan_complete"] = False
                                self._skip(stats, "depth_limit")
                            else:
                                pending.append((target, depth + 1))
                        elif stat.S_ISREG(info.st_mode):
                            relative = target.relative_to(base).as_posix()
                            if (
                                fnmatch.fnmatchcase(relative, pattern)
                                or fnmatch.fnmatchcase(target.name, pattern)
                                or (pattern.startswith("**/") and fnmatch.fnmatchcase(relative, pattern[3:]))
                            ):
                                found.append(target)
                    except (ProjectFileError, OSError) as exc:
                        reason = getattr(exc, "code", "io_error")
                        self._skip(stats, reason)
                        if reason == "io_error":
                            stats["scan_complete"] = False
                if stats["scanned_entries"] >= MAX_WALK_ENTRIES:
                    if pending:
                        stats["scan_complete"] = False
                    break
            except (ProjectFileError, OSError) as exc:
                stats["scan_complete"] = False
                self._skip(stats, getattr(exc, "code", "io_error"))
        found.sort(key=self._relative)
        return found, stats

    @staticmethod
    def _skip(stats: dict, reason: str) -> None:
        skipped = stats["skipped_files"]
        skipped[reason] = skipped.get(reason, 0) + 1

    @staticmethod
    def _page(items: list, offset: int, limit: int) -> tuple[list, int | None]:
        page = []
        size = 0
        for item in items[offset:offset + limit]:
            item_size = len(json.dumps(item, ensure_ascii=False))
            if page and size + item_size > MAX_PAGE_CHARS:
                break
            page.append(item)
            size += item_size
        next_offset = offset + len(page) if offset + len(page) < len(items) else None
        return page, next_offset

    def list_files(self, path: str = ".", pattern: str = "*", limit: int = 100, offset: int = 0) -> dict:
        _integer(limit, "limit", 1, MAX_PAGE_SIZE)
        _integer(offset, "offset", 0)
        pattern = self._pattern(pattern)
        with self._lock:
            found, stats = self._scan(path, pattern)
            files, next_offset = self._page([self._relative(item) for item in found], offset, limit)
            return {
                "files": files, "next_offset": next_offset,
                "truncated": next_offset is not None or not stats["scan_complete"],
                "total_files": len(found) if stats["scan_complete"] else None,
                **stats,
            }

    def search_text(
        self, query: str, path: str = ".", pattern: str = "*", case_sensitive: bool = True,
        limit: int = 100, offset: int = 0,
    ) -> dict:
        if not isinstance(query, str) or not query or len(query) > 4096 or "\n" in query or "\r" in query:
            raise ProjectFileError("invalid_argument", "query должен быть непустой строкой до 4096 символов без переносов.")
        if not isinstance(case_sensitive, bool):
            raise ProjectFileError("invalid_argument", "case_sensitive должен быть логическим значением.")
        _integer(limit, "limit", 1, MAX_PAGE_SIZE)
        _integer(offset, "offset", 0)
        pattern = self._pattern(pattern)
        needle = query if case_sensitive else query.casefold()
        with self._lock:
            found, stats = self._scan(path, pattern)
            matches = []
            total_matches = 0
            scanned_files = 0
            scanned_bytes = 0
            for target in found:
                try:
                    size = self._resolve(self._relative(target)).stat().st_size
                    if size > MAX_FILE_BYTES:
                        self._skip(stats, "too_large")
                        continue
                    if scanned_bytes + size > MAX_SEARCH_BYTES:
                        stats["scan_complete"] = False
                        self._skip(stats, "search_byte_limit")
                        break
                    # Charge attempted reads too: binary/non-UTF-8 files must
                    # not bypass the byte budget when decoding is rejected.
                    scanned_bytes += size
                    data, text = self._read(target)
                except ProjectFileError as exc:
                    self._skip(stats, exc.code)
                    if exc.code in {"io_error", "not_found"}:
                        stats["scan_complete"] = False
                    continue
                except OSError:
                    self._skip(stats, "io_error")
                    stats["scan_complete"] = False
                    continue
                scanned_bytes += max(0, len(data) - size)
                scanned_files += 1
                for number, line in enumerate(text.splitlines(), 1):
                    haystack = line if case_sensitive else line.casefold()
                    if needle not in haystack:
                        continue
                    if offset <= total_matches < offset + limit + 1:
                        matches.append({
                            "path": self._relative(target), "line": number,
                            "text": line[:MAX_SEARCH_LINE_CHARS],
                            "text_truncated": len(line) > MAX_SEARCH_LINE_CHARS,
                        })
                    total_matches += 1
            page, _ = self._page(matches, 0, limit)
            next_offset = offset + len(page) if offset + len(page) < total_matches else None
            return {
                "matches": page, "next_offset": next_offset,
                "truncated": next_offset is not None or not stats["scan_complete"],
                "total_matches": total_matches if stats["scan_complete"] else None,
                "matched_lines": total_matches, "scanned_files": scanned_files,
                "scanned_bytes": scanned_bytes, **stats,
            }

    def read_file(self, path: str, start_line: int = 1, end_line: int | None = None) -> dict:
        _integer(start_line, "start_line", 1)
        if end_line is not None:
            _integer(end_line, "end_line", start_line)
        with self._lock:
            target = self._resolve(path)
            data, text = self._read(target)
            lines = text.splitlines(keepends=True)
            total = len(lines)
            if start_line > max(total, 1):
                raise ProjectFileError("invalid_argument", f"Начальная строка выходит за пределы файла ({total} строк).")
            requested_end = min(end_line if end_line is not None else total, total)
            selected = []
            numbered = []
            size = 0
            payload_size = 0
            for number, line in enumerate(lines[start_line - 1:requested_end], start_line):
                annotated = f"{number}: {line}"
                encoded_size = sum(len(json.dumps(value, ensure_ascii=False)) - 2 for value in (line, annotated))
                if size + len(line) > MAX_READ_CHARS or payload_size + encoded_size > MAX_READ_PAYLOAD_CHARS:
                    if not selected:
                        raise ProjectFileError("line_too_long", "Строка не помещается в ограниченный ответ чтения.")
                    break
                selected.append(line)
                numbered.append(annotated)
                size += len(line)
                payload_size += encoded_size
            actual_end = start_line + len(selected) - 1
            truncated = actual_end < requested_end
            return {
                "path": self._relative(target), "content": "".join(selected),
                "sha256": hashlib.sha256(data).hexdigest(),
                "start_line": start_line, "end_line": actual_end, "total_lines": total,
                "truncated": truncated, "next_start_line": actual_end + 1 if truncated else None,
                "numbered_content": "".join(numbered),
            }

    @staticmethod
    def _diff(path: str, before: str, after: str, creating: bool) -> tuple[str, bool]:
        chunks = []
        size = 0
        for line in difflib.unified_diff(
            before.splitlines(keepends=True), after.splitlines(keepends=True),
            fromfile="/dev/null" if creating else f"a/{path}", tofile=f"b/{path}",
        ):
            chunk = line
            if not chunk.endswith("\n"):
                chunk += "\n\\ No newline at end of file\n"
            if size + len(chunk) > MAX_DIFF_CHARS:
                remaining = MAX_DIFF_CHARS - size
                chunks.append(chunk[:remaining])
                return "".join(chunks), True
            chunks.append(chunk)
            size += len(chunk)
        return "".join(chunks), False

    def _write_result(self, target: Path, before: str, after: str, data: bytes, creating: bool) -> dict:
        path = self._relative(target)
        changed = creating or before != after
        diff, truncated = self._diff(path, before, after, creating)
        return {
            "path": path, "changed": changed, "sha256": hashlib.sha256(data).hexdigest(),
            "diff": diff, "diff_truncated": truncated, "bytes_written": len(data) if changed else 0,
        }

    def create_file(self, path: str, content: str) -> dict:
        data = _text_bytes(content)
        with self._lock:
            target = self._resolve(path)
            if target.exists():
                raise ProjectFileError("already_exists", "Файл уже существует; прочитайте его и используйте edit_file.")
            temporary = None
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                target = self._resolve(path)
                descriptor, temporary = tempfile.mkstemp(prefix=".project-files-", suffix=".tmp", dir=target.parent)
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(data)
                self._resolve(path)
                # Windows rename refuses an existing destination. On POSIX a
                # hard link gives the same no-overwrite publication guarantee.
                if os.name == "nt":
                    os.rename(temporary, target)
                else:
                    os.link(temporary, target)
            except FileExistsError:
                raise ProjectFileError("already_exists", "Файл появился во время создания; существующее содержимое сохранено.") from None
            except OSError:
                raise ProjectFileError("io_error", "Не удалось атомарно создать файл в выбранном каталоге.") from None
            finally:
                self._remove_temporary(temporary)
            return self._write_result(target, "", content, data, True)

    def edit_file(self, path: str, old_text: str, new_text: str, expected_sha256: str) -> dict:
        if not isinstance(old_text, str) or not old_text:
            raise ProjectFileError("invalid_argument", "old_text должен быть непустым точным фрагментом файла.")
        if not isinstance(new_text, str):
            raise ProjectFileError("invalid_argument", "new_text должен быть строкой.")
        if not isinstance(expected_sha256, str) or not _SHA256.fullmatch(expected_sha256):
            raise ProjectFileError("invalid_argument", "Передайте sha256 полного файла из read_file.")
        with self._lock:
            target = self._resolve(path)
            original, before = self._read(target)
            if hashlib.sha256(original).hexdigest() != expected_sha256.lower():
                raise ProjectFileError("stale_file", "Файл изменился после чтения; прочитайте его заново перед правкой.")
            first = before.find(old_text)
            if first == -1:
                raise ProjectFileError("text_not_found", "old_text не найден; скопируйте точный фрагмент из read_file.")
            if before.find(old_text, first + 1) != -1:
                raise ProjectFileError("ambiguous_match", "old_text встречается больше одного раза; добавьте контекст для однозначной замены.")
            after = before.replace(old_text, new_text, 1)
            data = _text_bytes(after)
            if data == original:
                return self._write_result(target, before, after, data, False)
            temporary = None
            try:
                descriptor, temporary = tempfile.mkstemp(prefix=".project-files-", suffix=".tmp", dir=target.parent)
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(data)
                current, _ = self._read(target)
                if current != original:
                    raise ProjectFileError("stale_file", "Файл изменился во время подготовки правки; ничего не записано.")
                os.chmod(temporary, stat.S_IMODE(target.stat().st_mode))
                self._resolve(path)
                os.replace(temporary, target)
            except OSError:
                raise ProjectFileError("io_error", "Не удалось атомарно сохранить изменение.") from None
            finally:
                self._remove_temporary(temporary)
            return self._write_result(target, before, after, data, False)
