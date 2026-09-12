import argparse
import json
import os
from pathlib import Path

from rich.console import Console, RenderableType
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from agent import (
    DEFAULT_COMPRESS_EVERY, DEFAULT_LAST_MESSAGES, DEFAULT_MODEL,
    META_PROMPT_SYSTEM, Agent, CompressionResult, RequestResult,
)
from history import DEFAULT_HISTORY_PATH

ENV_PATH = Path(__file__).resolve().parent / ".env"

console = Console()


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Значение должно быть больше нуля")
    return number


def nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("Значение должно быть не меньше нуля")
    return number


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
        type=positive_int,
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
    parser.add_argument(
        "--context-limit", type=positive_int, metavar="N",
        help="Лимит контекста в токенах для сравнения с входом (задаётся явно)",
    )
    parser.add_argument(
        "--last-messages", type=nonnegative_int, default=DEFAULT_LAST_MESSAGES, metavar="N",
        help=f"Число последних сообщений, сохраняемых при сжатии (по умолчанию: {DEFAULT_LAST_MESSAGES})",
    )
    parser.add_argument(
        "--compress-every", type=positive_int, default=DEFAULT_COMPRESS_EVERY, metavar="N",
        help=f"Сжимать при накоплении N сообщений сверх last-messages (по умолчанию: {DEFAULT_COMPRESS_EVERY})",
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


def add_context_info(
    table: Table, prompt_tokens: int, context_limit: int | None,
    max_tokens: int | None = None,
) -> None:
    if context_limit is None:
        return
    table.add_row(
        "Вход / заданный лимит",
        f"{prompt_tokens} / {context_limit} ({prompt_tokens / context_limit:.1%})",
    )
    remaining = context_limit - prompt_tokens
    table.add_row("Лимит минус вход", str(remaining))
    if max_tokens is not None:
        remaining -= max_tokens
        table.add_row("Остаток после резерва max_tokens", str(remaining))
    if remaining < 0:
        table.add_row("Контекст", Text("Вход с резервом превышает заданный лимит", style="yellow"))


def print_response(
    result: RequestResult, response_format: str, *, title: str = "Ответ",
    context_limit: int | None = None, max_tokens: int | None = None,
) -> None:
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
        meta.add_row("Контекст запроса: вход по API", str(usage.prompt_tokens))
        add_context_info(meta, usage.prompt_tokens, context_limit, max_tokens)
        meta.add_row(
            "Токены за запрос",
            f"вход {usage.prompt_tokens} → выход {usage.completion_tokens} (всего {usage.total_tokens})",
        )
        cache_hit = getattr(usage, "prompt_cache_hit_tokens", None)
        cache_miss = getattr(usage, "prompt_cache_miss_tokens", None)
        if cache_hit is not None:
            meta.add_row("Из входа: кэш", str(cache_hit))
        if cache_miss is not None:
            meta.add_row("Из входа: без кэша", str(cache_miss))
        reasoning = getattr(getattr(usage, "completion_tokens_details", None), "reasoning_tokens", None)
        if reasoning is not None:
            meta.add_row("Из выхода: рассуждения", str(reasoning))
    else:
        meta.add_row("Токены за запрос", "API не вернул статистику")
    total = result.dialogue_usage
    dialogue_label = Text("Суммарный расход диалога")
    if total.missing_responses:
        dialogue_label.append(" (неполные данные)", style="bold yellow not dim")
    dialogue_tokens = Text(f"вход {total.prompt_tokens} → выход {total.completion_tokens} (всего ")
    dialogue_tokens.append(str(total.total_tokens))
    dialogue_tokens.append(")")
    meta.add_row(
        dialogue_label,
        dialogue_tokens,
    )
    if total.missing_responses:
        meta.add_row("Ответов без статистики", str(total.missing_responses))
    meta.add_row("Причина остановки", choice.finish_reason or "—")
    meta.add_row("Время", f"{result.elapsed:.2f} с")
    console.print()
    console.print(meta)

    console.print("[dim]Расход диалога включает повторную отправку истории; это не размер контекста.[/dim]")


def print_compression(result: CompressionResult) -> None:
    console.print(
        f"[bold cyan]Сжатие истории выполнено:[/bold cyan] "
        f"сообщений заменено на summary — {result.messages_compressed}."
    )
    if result.usage is None:
        console.print("[dim]Токены до / после: API не вернул статистику.[/dim]")
        return
    console.print(
        f"Токены по API: до (вход сжатия) {result.usage.prompt_tokens} "
        f"→ после (выход summary) {result.usage.completion_tokens}."
    )
    console.print(
        "[dim]Вход включает инструкцию, JSON и предыдущее summary; "
        "это не размер всей истории до / после.[/dim]"
    )


def main() -> None:
    args = parse_args()
    load_env(ENV_PATH)

    api_key = os.environ.get("API_KEY")
    if not api_key:
        raise SystemExit("Переменная API_KEY не найдена в .env")

    agent = Agent(
        api_key, history_path=args.history,
        last_messages=args.last_messages, compress_every=args.compress_every,
        on_compression=print_compression,
    )
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
        print_response(
            meta_result, "text", title="Сгенерированный промпт",
            context_limit=args.context_limit, max_tokens=args.max_tokens,
        )
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

    print_response(
        result, args.response_format,
        context_limit=args.context_limit, max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
