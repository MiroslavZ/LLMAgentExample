import argparse
import os
from pathlib import Path

from openai import OpenAI

ENV_PATH = Path(__file__).resolve().parent / ".env"
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-chat"
RESPONSE_FORMATS = {
    "text": {"type": "text"},
    "object": {"type": "json_object"},
    "schema": {"type": "json_schema"},
}


def load_env(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Файл {path} не найден")

    with path.open(encoding="utf-8") as env_file:
        for line in env_file:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Запрос к LLM через DeepSeek API")
    parser.add_argument("--prompt", required=True, help="Текст пользовательского запроса")
    parser.add_argument(
        "--max-tokens",
        type=int,
        metavar="N",
        help="Максимальное число токенов в ответе",
    )
    parser.add_argument(
        "--stop-sequences",
        nargs="*",
        default=None,
        metavar="STOP",
        help="Необязательные стоп-последовательности (одна или несколько строк)",
    )
    parser.add_argument(
        "--response-format",
        choices=("text", "schema", "object"),
        default="text",
        help="Формат ответа модели (по умолчанию: text)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_env(ENV_PATH)

    api_key = os.environ.get("API_KEY")
    if not api_key:
        raise SystemExit("Переменная API_KEY не найдена в .env")

    request = {
        "model": MODEL,
        "messages": [{"role": "user", "content": args.prompt}],
        "response_format": RESPONSE_FORMATS[args.response_format],
    }
    if args.max_tokens is not None:
        request["max_tokens"] = args.max_tokens
    if args.stop_sequences:
        request["stop"] = args.stop_sequences

    client = OpenAI(api_key=api_key, base_url=BASE_URL)
    response = client.chat.completions.create(**request)
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
