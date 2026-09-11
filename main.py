import argparse
import json
import os
from pathlib import Path

from rich.console import Console, RenderableType
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from agent import DEFAULT_MODEL, META_PROMPT_SYSTEM, Agent, RequestResult
from history import DEFAULT_HISTORY_PATH

ENV_PATH = Path(__file__).resolve().parent / ".env"

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
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Имя модели (по умолчанию: {DEFAULT_MODEL})",
    )
    parser.add_argument("--system", help="Системный промпт диалога (сохраняется, если ещё не задан)")
    parser.add_argument("--user", required=True, help="Текст пользовательского запроса")
    parser.add_argument(
        "--max-tokens",
        type=int,
        metavar="N",
        help="Максимальное число токенов в ответе",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        metavar="T",
        help="Температура сэмплирования (0–2)",
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
    parser.add_argument(
        "--history",
        type=Path,
        default=DEFAULT_HISTORY_PATH,
        metavar="PATH",
        help=f"Путь к JSON-файлу истории (по умолчанию: ./{DEFAULT_HISTORY_PATH})",
    )
    return parser.parse_args()


def print_request_info(
    args: argparse.Namespace,
    *,
    system: str | None = None,
    user: str | None = None,
    stage: str | None = None,
    response_format: str | None = None,
) -> None:
    meta = Table.grid(padding=(0, 2))
    meta.add_column(style="bold dim")
    meta.add_column()
    meta.add_row("Модель", args.model)
    meta.add_row("Формат", response_format or args.response_format)
    if stage:
        meta.add_row("Этап", stage)
    if args.max_tokens is not None:
        meta.add_row("Max tokens", str(args.max_tokens))
    if args.temperature is not None:
        meta.add_row("Temperature", str(args.temperature))
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
        Panel(
            user if user is not None else args.user,
            title="[bold]Пользователь[/bold]",
            border_style="cyan",
            padding=(1, 2),
        )
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


def print_response(result: RequestResult, response_format: str, *, title: str = "Ответ") -> None:
    response = result.response
    choice = response.choices[0]

    console.print(
        Panel(
            render_content(result.content, response_format),
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
            "Токены за запрос",
            f"вход {usage.prompt_tokens} → выход {usage.completion_tokens} (всего {usage.total_tokens})",
        )
    else:
        meta.add_row("Токены за запрос", "API не вернул статистику")
    total = result.dialogue_usage
    meta.add_row(
        "Токены за диалог" + (" (неполные данные)" if total.missing_responses else ""),
        f"вход {total.prompt_tokens} → выход {total.completion_tokens} (всего {total.total_tokens})",
    )
    if total.missing_responses:
        meta.add_row("Ответов без статистики", str(total.missing_responses))
    meta.add_row("Причина остановки", choice.finish_reason or "—")
    meta.add_row("Время", f"{result.elapsed:.2f} с")
    console.print()
    console.print(meta)


def main() -> None:
    args = parse_args()
    load_env(ENV_PATH)

    api_key = os.environ.get("API_KEY")
    if not api_key:
        raise SystemExit("Переменная API_KEY не найдена в .env")

    agent = Agent(api_key, history_path=args.history)
    saved_system = agent.history.get_system_prompt()
    if saved_system is not None:
        args.system = saved_system
    request_options = {
        "user": args.user,
        "model": args.model,
        "system": args.system,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "stop_sequences": args.stop_sequences,
        "response_format": args.response_format,
    }

    if args.meta_prompt:
        print_request_info(
            args,
            system=META_PROMPT_SYSTEM,
            user=args.user,
            stage="1/2 — генерация промпта",
            response_format="text",
        )
        with console.status("[bold cyan]Генерация промпта и выполнение запроса…", spinner="dots"):
            meta_result, result = agent.request_with_meta_prompt(**request_options)
        print_response(meta_result, "text", title="Сгенерированный промпт")
        console.print()

        print_request_info(
            args,
            system=args.system,
            user=meta_result.content,
            stage="2/2 — основной запрос",
        )
    else:
        print_request_info(args, system=args.system, user=args.user)
        with console.status("[bold cyan]Ожидание ответа модели…", spinner="dots"):
            result = agent.request(**request_options)

    print_response(result, args.response_format)


if __name__ == "__main__":
    main()
