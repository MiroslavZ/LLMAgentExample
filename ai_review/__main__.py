"""Запуск: python -m ai_review --base origin/main --head HEAD."""

import argparse
import json
import os
from pathlib import Path
import sys

from . import ReviewError
from .context import collect_context
from .review import DEFAULT_MODEL, generate_review, render_markdown


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", default="HEAD")
    parser.add_argument("--output", type=Path, default=Path("ai_review/output"))
    parser.add_argument("--model", default=os.environ.get("DEEPSEEK_MODEL") or DEFAULT_MODEL)
    parser.add_argument("--retrieve-only", action="store_true", help="Собрать Git/RAG контекст без запроса к модели")
    args = parser.parse_args(argv)
    try:
        args.output.mkdir(parents=True, exist_ok=True)
        # Старый успешный отчёт не должен попасть в артефакты неудачного прогона.
        for name in ("review.md", "result.json", "context.json"):
            (args.output / name).unlink(missing_ok=True)
        key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not args.retrieve_only and not key.strip():
            raise ReviewError("Добавьте DEEPSEEK_API_KEY в GitHub Actions Secrets или переменные окружения.")
        context = collect_context(args.repo, args.base, args.head)
        (args.output / "context.json").write_text(json.dumps(context, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Контекст: {len(context['files'])} файлов, {len(context['sources'])} RAG-источников, "
              f"{len(context['skipped'])} пропусков.")
        if args.retrieve_only:
            print("Контекст сохранён; обращений к DeepSeek не было.")
            return 0
        if context["files"]:
            result = generate_review(context, key, args.model)
        else:
            result = {"summary": "Нет доступных текстовых изменений для анализа; проверьте сведения о пропусках.",
                      "findings": [], "model": "не вызывалась", "usage": {}}
        (args.output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        (args.output / "review.md").write_text(render_markdown(result, context), encoding="utf-8")
        print("Ревью сохранено в review.md и result.json.")
        return 0
    except ReviewError as error:
        print(f"Ошибка ревью: {error}", file=sys.stderr)
        return 1
    except (OSError, ValueError):
        print("Не удалось подготовить ревью. Проверьте параметры и доступ к файлам.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
