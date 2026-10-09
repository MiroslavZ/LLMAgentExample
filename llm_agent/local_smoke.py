"""Три запроса к явно выбранной локальной модели: python -m llm_agent.local_smoke."""

import argparse
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

from .llm_models import discover_models
from .models import RequestOptions, utc_now
from .service import ConversationService


PROMPTS = (
    ("simple", "Сколько будет 17 + 25? Ответь только числом."),
    ("medium", "Объясни на русском различие списка и кортежа в Python. Дай по одному примеру использования. Не более 100 слов."),
    ("complex", "Напиши функцию Python unique_in_order(items), которая убирает дубликаты, сохраняя порядок. Элементы могут быть словарями и списками. Не изменяй вход. Приведи 3 assert-теста, включая пустой вход и словари, и укажи сложность алгоритма. Отвечай кратко."),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True, help="Точный API ID; остальные модели не вызываются")
    parser.add_argument("--token-env", help="Имя переменной с токеном; локальному серверу не требуется")
    parser.add_argument("--output", type=Path, required=True, help="Новый JSON-файл результатов без токена")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Отчёт уже существует; выберите другой --output")
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    token = os.environ.get(args.token_env, "") if args.token_env else ""
    if args.token_env and not token:
        parser.error("Переменная с токеном отсутствует или пуста")
    available = discover_models(args.base_url, token)
    if args.model not in available:
        parser.error("Указанная модель отсутствует в ответе /models")
    report = {"created_at": utc_now(), "base_url": args.base_url, "model": args.model, "runs": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="local-llm-smoke-") as directory:
        service = ConversationService(Path(directory) / "conversations")
        selected = service.models.import_models(args.base_url, token, [args.model])[0]
        for difficulty, prompt in PROMPTS:
            conversation = service.create()
            service.select_model(conversation.id, selected.id)
            result = service.send(conversation.id, prompt, options=RequestOptions(temperature=0, max_tokens=1400))
            turn = result.turns[-1]
            report["runs"].append({"difficulty": difficulty, **asdict(turn)})
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"{difficulty}: {turn.status}, {turn.elapsed_seconds:.2f} s", flush=True)
            if turn.status != "completed" or not turn.answer or not turn.answer.strip():
                raise SystemExit(turn.error or "Модель не вернула непустой ответ")


if __name__ == "__main__":
    main()
