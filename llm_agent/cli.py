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

from .agent import (
    DEFAULT_MODEL,
    META_PROMPT_SYSTEM, Agent, CompressionResult, RequestResult,
)
from .history import DEFAULT_HISTORY_PATH, HistoryManager
from .branch_history import BranchHistoryManager
from .context_strategy import SUPPORTED_STRATEGIES
from .memory import (
    DEFAULT_MEMORY_PATH, MEMORY_LAYERS,
    MemorySnapshot, MemoryStorageError, MemoryStore, cli_memory_scope,
)
from .profile import ProfileStorageError, ProfileStore, UserProfile

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"

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
    parser.add_argument("--user", help="Текст запроса; необязателен для управления ветками, памятью и профилями")
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
        "--last-messages", type=nonnegative_int, metavar="N",
        help="Число последних сообщений, сохраняемых при сжатии; требует --compress-every (по умолчанию сжатие выключено)",
    )
    parser.add_argument(
        "--compress-every", type=positive_int, metavar="N",
        help="Сжимать при накоплении N сообщений сверх last-messages; требует --last-messages (по умолчанию сжатие выключено)",
    )
    parser.add_argument(
        "--strategy", choices=SUPPORTED_STRATEGIES,
        help="Стратегия контекста: window, facts или branch",
    )
    parser.add_argument(
        "--window-size", type=positive_int, metavar="N",
        help="Число последних сообщений, включая текущий запрос; требует --strategy window или facts",
    )
    parser.add_argument(
        "--branch", type=BranchHistoryManager.validate_name, metavar="NAME",
        help="Переключиться на существующую ветку (по умолчанию: сохранённая активная ветка, сначала main)",
    )
    branch_action = parser.add_mutually_exclusive_group()
    branch_action.add_argument(
        "--checkpoint", type=BranchHistoryManager.validate_name, metavar="NAME",
        help="Сохранить checkpoint: после успешного запроса с --user, иначе из текущего диалога",
    )
    branch_action.add_argument(
        "--create-branch", type=BranchHistoryManager.validate_name, metavar="NAME",
        help="Создать и активировать ветку; требует --from-checkpoint",
    )
    branch_action.add_argument(
        "--list-branches", action="store_true",
        help="Показать ветки, активную ветку и checkpoints без обращения к API",
    )
    parser.add_argument(
        "--from-checkpoint", type=BranchHistoryManager.validate_name, metavar="NAME",
        help="Checkpoint, от которого создаётся новая ветка",
    )
    parser.add_argument(
        "--memory-db", type=Path, default=DEFAULT_MEMORY_PATH, metavar="PATH",
        help=f"База рабочей и долговременной памяти (по умолчанию: {DEFAULT_MEMORY_PATH})",
    )
    memory_action = parser.add_mutually_exclusive_group()
    memory_action.add_argument(
        "--memory-set", nargs=3, metavar=("LAYER", "KEY", "VALUE"),
        help="Сохранить запись: LAYER — working или long_term",
    )
    memory_action.add_argument(
        "--memory-delete", nargs=2, metavar=("LAYER", "KEY"),
        help="Удалить запись из working или long_term",
    )
    memory_action.add_argument(
        "--memory-clear-working", action="store_true",
        help="Очистить рабочую память текущего диалога/ветки",
    )
    parser.add_argument(
        "--memory-show", action="store_true",
        help="Показать три слоя памяти без обращения к API (до запроса, если указан --user)",
    )
    parser.add_argument(
        "--profiles-db", type=Path, metavar="PATH",
        help="База профилей (по умолчанию: profiles.sqlite3 рядом с --memory-db)",
    )
    profile_selection = parser.add_mutually_exclusive_group()
    profile_selection.add_argument(
        "--profile", metavar="ID",
        help="Выбрать сохранённый профиль для текущего диалога/ветки и следующих запросов",
    )
    profile_selection.add_argument(
        "--profile-clear", action="store_true",
        help="Отключить профиль текущего диалога/ветки",
    )
    profile_edit = parser.add_mutually_exclusive_group()
    profile_edit.add_argument(
        "--profile-import", type=Path, metavar="PATH",
        help="Создать или заменить профиль из JSON; для выбора добавьте --profile ID",
    )
    profile_edit.add_argument(
        "--profile-delete", metavar="ID",
        help="Удалить профиль и отключить его во всех диалогах/ветках",
    )
    profile_view = parser.add_mutually_exclusive_group()
    profile_view.add_argument(
        "--profile-list", action="store_true",
        help="Показать сохранённые профили без обращения к API",
    )
    profile_view.add_argument(
        "--profile-show", action="store_true",
        help="Показать выбранный профиль текущего диалога/ветки (null, если профиль отключён)",
    )
    args = parser.parse_args()
    if args.profiles_db is None:
        args.profiles_db = args.memory_db.with_name("profiles.sqlite3")
    profile_options = (
        args.profile, args.profile_clear, args.profile_import, args.profile_delete,
        args.profile_list, args.profile_show,
    )
    if any(value is not None and not value.strip() for value in (args.profile, args.profile_delete)):
        parser.error("ID профиля должен быть непустой строкой")
    if args.profile_delete is not None and args.profile_delete == args.profile:
        parser.error("Нельзя одновременно удалить и выбрать один профиль")
    if args.memory_show and (args.profile_list or args.profile_show):
        parser.error("--memory-show нельзя совмещать с --profile-list или --profile-show")
    memory_edit = args.memory_set or args.memory_delete
    if memory_edit:
        layer = memory_edit[0]
        if layer not in MEMORY_LAYERS:
            parser.error("LAYER должен быть working или long_term")
        if any(not value.strip() for value in memory_edit[1:]):
            parser.error("Ключ и значение памяти должны быть непустыми строками")
    branch_options = (args.branch, args.checkpoint, args.create_branch, args.from_checkpoint, args.list_branches)
    if any(branch_options) and args.strategy != "branch":
        parser.error("Аргументы управления ветками требуют --strategy branch")
    if (args.create_branch is None) != (args.from_checkpoint is None):
        parser.error("Для создания ветки укажите вместе --create-branch и --from-checkpoint")
    if args.create_branch is not None and args.branch is not None:
        parser.error("--create-branch уже активирует новую ветку; не совмещайте его с --branch")
    if args.list_branches and args.user is not None:
        parser.error("--list-branches не совмещается с --user")
    if args.user is None:
        if not any(branch_options) and not any(profile_options) and not (
            memory_edit or args.memory_show or args.memory_clear_working
        ):
            parser.error("Укажите --user или операцию управления ветками, памятью или профилями")
        if args.meta_prompt or args.system is not None:
            parser.error("--meta-prompt и --system требуют --user")
    if args.strategy in ("window", "facts") and args.window_size is None:
        parser.error(f"Для --strategy {args.strategy} необходимо указать --window-size")
    if args.window_size is not None and args.strategy not in ("window", "facts"):
        parser.error("--window-size требует --strategy window или facts")
    if args.strategy is not None and (args.last_messages is not None or args.compress_every is not None):
        parser.error("--strategy нельзя совмещать с --last-messages или --compress-every")
    if (args.last_messages is None) != (args.compress_every is None):
        console.print(
            "[yellow]Предупреждение: для работы сжатия истории необходимо указать оба аргумента: "
            "--last-messages и --compress-every. Агент продолжит работу без сжатия.[/yellow]"
        )
    return args


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


def print_memory(history: HistoryManager, scope: str, snapshot: MemorySnapshot) -> None:
    """Показать содержимое слоёв, не создавая API-клиент и не меняя историю."""
    data = {
        "short_term": {
            "scope": scope,
            "history": str(history.path),
            "branch": getattr(history, "active_branch", None),
            "messages": history.get_messages(),
        },
        "working": snapshot.working,
        "long_term": snapshot.long_term,
    }
    console.print(Syntax(json.dumps(data, ensure_ascii=False, indent=2), "json", word_wrap=True))


def load_profile(path: Path) -> UserProfile:
    """Проверить импорт целиком до изменения сохранённого профиля."""
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as error:
        raise ValueError(f"Файл профиля не найден: {path}") from error
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"Некорректный JSON профиля: {path}") from error
    return UserProfile.from_dict(data)


def apply_profile_options(
    args: argparse.Namespace, scope: str, imported: UserProfile | None,
) -> UserProfile | None:
    store = ProfileStore(args.profiles_db)
    status = None
    if imported is not None:
        store.save(imported)
        status = f"Профиль сохранён: {imported.id}"
    elif args.profile_delete is not None:
        store.delete(args.profile_delete)
        status = f"Профиль удалён: {args.profile_delete}"
    if args.profile is not None:
        store.select(scope, args.profile)
        status = f"Профиль выбран: {args.profile}"
    elif args.profile_clear:
        store.select(scope, None)
        status = "Профиль текущего диалога/ветки отключён"
    profile = store.selected(scope)
    if args.profile_list:
        data = [saved.to_dict() for saved in store.list()]
        console.print(Syntax(json.dumps(data, ensure_ascii=False, indent=2), "json", word_wrap=True))
    elif args.profile_show:
        data = profile.to_dict() if profile is not None else None
        console.print(Syntax(json.dumps(data, ensure_ascii=False, indent=2), "json", word_wrap=True))
    elif status and not args.memory_show:
        console.print(Text(status))
    return profile


def main() -> None:
    args = parse_args()
    imported = load_profile(args.profile_import) if args.profile_import is not None else None
    history = None
    if args.strategy == "branch":
        history = BranchHistoryManager(args.history, branch=args.branch)
        # Проверяем имя до запроса, чтобы не тратить API на заведомо неверную команду.
        if args.checkpoint is not None and args.checkpoint in history.list_checkpoints():
            raise ValueError(f"Checkpoint уже существует: {args.checkpoint}")
        if args.create_branch is not None:
            history.create_branch(args.create_branch, from_checkpoint=args.from_checkpoint)
        if not (args.memory_show or args.profile_list or args.profile_show):
            console.print(Text(f"Активная ветка: {history.active_branch}"))
        if args.list_branches:
            console.print(Text("Ветки: " + ", ".join(history.list_branches())))
            console.print(Text("Checkpoints: " + (", ".join(history.list_checkpoints()) or "нет")))
        if args.user is None:
            if args.checkpoint is not None:
                history.create_checkpoint(args.checkpoint)
                console.print(Text(f"Checkpoint сохранён: {args.checkpoint}"))
    memory_action = args.memory_set or args.memory_delete or args.memory_clear_working or args.memory_show
    profile_action = (
        args.profile is not None or args.profile_clear or imported is not None
        or args.profile_delete is not None or args.profile_list or args.profile_show
    )
    if args.user is None and not (memory_action or profile_action):
        return

    scope = cli_memory_scope(args.history, getattr(history, "active_branch", None))
    profile = apply_profile_options(args, scope, imported) if args.user is not None or profile_action else None
    if args.user is None and not memory_action:
        return
    history = history or HistoryManager(args.history)
    store = MemoryStore(args.memory_db)
    if args.memory_set:
        layer, key, value = args.memory_set
        store.remember(scope, layer, key, value)
        console.print(Text(f"Память сохранена: {layer}, {key}"))
    elif args.memory_delete:
        layer, key = args.memory_delete
        store.forget(scope, layer, key)
        console.print(Text(f"Запись памяти удалена: {layer}, {key}"))
    elif args.memory_clear_working:
        store.clear_working(scope)
        console.print(Text("Рабочая память текущего диалога/ветки очищена"))
    memory = store.snapshot(scope)
    if args.memory_show:
        print_memory(history, scope, memory)
    if args.user is None:
        return
    load_env(ENV_PATH)

    api_key = os.environ.get("API_KEY")
    if not api_key:
        raise SystemExit("Переменная API_KEY не найдена в .env")

    agent = Agent(
        api_key, history_path=args.history,
        last_messages=args.last_messages, compress_every=args.compress_every,
        on_compression=print_compression,
        strategy=args.strategy, window_size=args.window_size,
        memory=memory, profile=profile,
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
    if args.checkpoint is not None:
        agent.history.create_checkpoint(args.checkpoint)
        console.print(Text(f"Checkpoint сохранён: {args.checkpoint}"))


def run() -> None:
    """Запустить CLI с выводом ожидаемых ошибок без traceback."""
    try:
        main()
    except (ValueError, OSError, MemoryStorageError, ProfileStorageError) as error:
        raise SystemExit(str(error)) from error
