import argparse
import json
import os
import time
from pathlib import Path

from openai import OpenAI
from rich.console import Console, RenderableType
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

ENV_PATH = Path(__file__).resolve().parent / ".env"
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-chat"
RESPONSE_FORMATS = {
    "text": {"type": "text"},
    "object": {"type": "json_object"},
    "schema": {"type": "json_schema"},
}

console = Console()


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


def print_request_info(args: argparse.Namespace) -> None:
    meta = Table.grid(padding=(0, 2))
    meta.add_column(style="bold dim")
    meta.add_column()
    meta.add_row("Модель", MODEL)
    meta.add_row("Формат", args.response_format)
    if args.max_tokens is not None:
        meta.add_row("Max tokens", str(args.max_tokens))
    if args.stop_sequences:
        meta.add_row("Stop", ", ".join(args.stop_sequences))

    console.print(meta)
    console.print()
    console.print(
        Panel(args.prompt, title="[bold]Запрос[/bold]", border_style="cyan", padding=(1, 2))
    )


def render_content(content: str, response_format: str) -> RenderableType:
    if not content:
        return "[dim]Пустой ответ[/dim]"

    if response_format in ("object", "schema"):
        try:
            pretty = json.dumps(json.loads(content), ensure_ascii=False, indent=2)
        except json.JSONDecodeError:
            pretty = content
        return Syntax(pretty, "json", theme="monokai", word_wrap=True)

    return Markdown(content)


def print_response(response, elapsed: float, response_format: str) -> None:
    choice = response.choices[0]
    content = choice.message.content or ""

    console.print(
        Panel(
            render_content(content, response_format),
            title="[bold green]Ответ[/bold green]",
            border_style="green",
            padding=(1, 2),
        )
    )

    usage = response.usage
    meta = Table.grid(padding=(0, 2))
    meta.add_column(style="dim")
    meta.add_column()
    if usage is not None:
        meta.add_row(
            "Токены",
            f"{usage.prompt_tokens} → {usage.completion_tokens} (всего {usage.total_tokens})",
        )
    meta.add_row("Причина остановки", choice.finish_reason or "—")
    meta.add_row("Время", f"{elapsed:.2f} с")
    console.print()
    console.print(meta)


def main() -> None:
    args = parse_args()
    load_env(ENV_PATH)

    api_key = os.environ.get("API_KEY")
    if not api_key:
        raise SystemExit("Переменная API_KEY не найдена в .env")

    print_request_info(args)

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
    started = time.perf_counter()
    with console.status("[bold cyan]Ожидание ответа модели…", spinner="dots"):
        response = client.chat.completions.create(**request)
    elapsed = time.perf_counter() - started

    print_response(response, elapsed, args.response_format)


if __name__ == "__main__":
    main()
