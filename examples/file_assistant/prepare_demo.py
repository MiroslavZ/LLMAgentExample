"""Готовит фиксированную копию примера; сброс возможен только с --reset."""

import argparse
import json
import os
import re
import shutil
import stat
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BASELINE = Path(__file__).resolve().parent / "baseline"
DEMO_DIRECTORY = REPOSITORY_ROOT / "data" / "file-assistant-demo"
PROJECT_ROOT = DEMO_DIRECTORY / "project"
MARKER = DEMO_DIRECTORY / ".demo-owner.json"
OWNER = {
    "owner": "LLMAgentExample/file-assistant-demo",
    "version": 1,
    "project_root": str(PROJECT_ROOT),
}
SOURCE_FILES = ("client.py", "service.py", "worker.py")


class DemoError(ValueError):
    """Небезопасное или неизвестное состояние демонстрационной папки."""


def reject_link(path: Path) -> None:
    """Не допускает symlink и Windows junction/reparse point, включая битые ссылки."""
    try:
        info = path.lstat()
    except FileNotFoundError:
        return
    if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
        raise DemoError(f"Ссылки и junction недопустимы: {path}")


def validate_paths() -> None:
    expected = REPOSITORY_ROOT / "data" / "file-assistant-demo" / "project"
    if PROJECT_ROOT != expected:
        raise DemoError("Путь демо не совпадает с фиксированным каталогом проекта.")
    for path in (*reversed(PROJECT_ROOT.parents), PROJECT_ROOT, MARKER):
        reject_link(path)
    if PROJECT_ROOT.resolve() != expected:
        raise DemoError("Фактический путь демо не совпадает с ожидаемым.")


def validate_marker() -> None:
    try:
        owner = json.loads(MARKER.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise DemoError("Папка демо не содержит корректный маркер владельца; содержимое сохранено.") from error
    if owner != OWNER:
        raise DemoError("Маркер принадлежит другой папке или версии демо; содержимое сохранено.")


def validate_tree(root: Path) -> None:
    """Проверяет всё дерево до рекурсивного удаления или чтения результатов."""
    reject_link(root)
    if not root.is_dir():
        raise DemoError(f"Нет каталога проекта: {root}")
    for directory, folders, files in os.walk(root, followlinks=False):
        for name in (*folders, *files):
            path = Path(directory) / name
            reject_link(path)
            mode = path.lstat().st_mode
            if not stat.S_ISDIR(mode) and not stat.S_ISREG(mode):
                raise DemoError(f"Обнаружен необычный объект файловой системы: {path}")


def prepare(*, reset: bool = False) -> Path:
    validate_paths()
    validate_tree(BASELINE)
    if DEMO_DIRECTORY.exists():
        validate_marker()
    else:
        DEMO_DIRECTORY.mkdir(parents=True)
        with MARKER.open("x", encoding="utf-8") as marker:
            json.dump(OWNER, marker, ensure_ascii=False, indent=2)
            marker.write("\n")

    if PROJECT_ROOT.exists():
        validate_tree(PROJECT_ROOT)
        if not reset:
            print("Демо уже подготовлено; существующие файлы оставлены без изменений.")
            return PROJECT_ROOT
        # Путь фиксирован, маркер проверен, всё удаляемое дерево проверено выше.
        shutil.rmtree(PROJECT_ROOT)
    shutil.copytree(BASELINE, PROJECT_ROOT)
    print("Демо восстановлено из baseline." if reset else "Демо подготовлено из baseline.")
    return PROJECT_ROOT


def check_result() -> bool:
    """Проверяет артефакты и сохранность исходников, не оценивает семантику ответа LLM."""
    validate_paths()
    validate_marker()
    validate_tree(PROJECT_ROOT)
    report_path = PROJECT_ROOT / "docs" / "api-usage.md"
    readme_path = PROJECT_ROOT / "README.md"
    report = report_path.read_text(encoding="utf-8") if report_path.is_file() else ""
    readme = readme_path.read_text(encoding="utf-8") if readme_path.is_file() else ""
    checks = {
        "Отчёт docs/api-usage.md существует и непустой": bool(report.strip()),
        "Отчёт упоминает ApiClient и все три исходника": (
            "ApiClient" in report and all(name in report for name in SOURCE_FILES)
        ),
        "README изменён относительно baseline": (
            bool(readme) and readme != (BASELINE / "README.md").read_text(encoding="utf-8")
        ),
        "README содержит timeout рядом с числом 30 и retries рядом с числом 3": (
            re.search(r"\btimeout\b[^\n]{0,60}?\b30\b", readme, re.IGNORECASE) is not None
            and re.search(r"\bretries\b[^\n]{0,60}?\b3\b", readme, re.IGNORECASE) is not None
        ),
        "Три исходника побайтно совпадают с baseline": all(
            (PROJECT_ROOT / name).is_file()
            and (PROJECT_ROOT / name).read_bytes() == (BASELINE / name).read_bytes()
            for name in SOURCE_FILES
        ),
    }
    print("Структурная проверка без LLM; полноту отчёта и смысл документации проверьте по коду.")
    for description, passed in checks.items():
        print(f"[{'OK' if passed else 'FAIL'}] {description}")
    return all(checks.values())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--reset", action="store_true", help="Явно пересоздать только собственную копию демо")
    action.add_argument("--check", action="store_true", help="Проверить структуру результатов без вызова модели")
    args = parser.parse_args()
    try:
        if args.check:
            return 0 if check_result() else 1
        print(f"Корень для --project-root: {prepare(reset=args.reset)}")
        return 0
    except (DemoError, OSError, UnicodeError) as error:
        print(f"Ошибка: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
