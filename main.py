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
META_PROMPT_SYSTEM = (
    "Составь оптимальный промпт для решения следующей задачи. "
    "Верни только текст промпта без пояснений и комментариев."
)
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
    parser.add_argument("--system", help="Текст системного промпта")
    parser.add_argument("--user", required=True, help="Текст пользовательского запроса")
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
    parser.add_argument(
        "--meta-prompt",
        action="store_true",
        help="Сначала сгенерировать оптимальный промпт, затем выполнить основной запрос",
    )
    return parser.parse_args()


def print_request_info(
    args: argparse.Namespace,
    *,
    system: str | None = None,
    user: str | None = None,
    stage: str | None = None,
) -> None:
    meta = Table.grid(padding=(0, 2))
    meta.add_column(style="bold dim")
    meta.add_column()
    meta.add_row("Модель", MODEL)
    meta.add_row("Формат", args.response_format)
    if stage:
        meta.add_row("Этап", stage)
    if args.max_tokens is not None:
        meta.add_row("Max tokens", str(args.max_tokens))
    if args.stop_sequences:
        meta.add_row("Stop", ", ".join(args.stop_sequences))

    console.print(meta)
    console.print()
    if system:
        console.print(
            Panel(system, title="[bold]Система[/bold]", border_style="magenta", padding=(1, 2))
        )
        console.print()
    console.print(
        Panel(user or args.user, title="[bold]Пользователь[/bold]", border_style="cyan", padding=(1, 2))
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


def build_messages(system: str | None, user: str) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    return messages


def build_request(
    args: argparse.Namespace,
    messages: list[dict[str, str]],
    *,
    response_format: str | None = None,
) -> dict:
    request = {
        "model": MODEL,
        "messages": messages,
        "response_format": RESPONSE_FORMATS[response_format or args.response_format],
    }
    if args.max_tokens is not None:
        request["max_tokens"] = args.max_tokens
    if args.stop_sequences:
        request["stop"] = args.stop_sequences
    return request


def call_model(client: OpenAI, request: dict, status: str = "Ожидание ответа модели…") -> tuple:
    started = time.perf_counter()
    with console.status(f"[bold cyan]{status}", spinner="dots"):
        response = client.chat.completions.create(**request)
    return response, time.perf_counter() - started


def print_response(response, elapsed: float, response_format: str, *, title: str = "Ответ") -> None:
    choice = response.choices[0]
    content = choice.message.content or ""

    console.print(
        Panel(
            render_content(content, response_format),
            title=f"[bold green]{title}[/bold green]",
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

    client = OpenAI(api_key=api_key, base_url=BASE_URL)

    if args.meta_prompt:
        print_request_info(
            args,
            system=META_PROMPT_SYSTEM,
            user=args.user,
            stage="1/2 — генерация промпта",
        )
        meta_request = build_request(
            args,
            build_messages(META_PROMPT_SYSTEM, args.user),
            response_format="text",
        )
        meta_response, meta_elapsed = call_model(
            client, meta_request, status="Генерация оптимального промпта…"
        )
        optimized_prompt = meta_response.choices[0].message.content or ""
        print_response(meta_response, meta_elapsed, "text", title="Сгенерированный промпт")
        console.print()

        print_request_info(
            args,
            system=args.system,
            user=optimized_prompt,
            stage="2/2 — основной запрос",
        )
        main_request = build_request(args, build_messages(args.system, optimized_prompt))
        response, elapsed = call_model(client, main_request)
    else:
        print_request_info(args, system=args.system, user=args.user)
        request = build_request(args, build_messages(args.system, args.user))
        response, elapsed = call_model(client, request)

    print_response(response, elapsed, args.response_format)


if __name__ == "__main__":
    main()
