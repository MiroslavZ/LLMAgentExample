import argparse
import json
import math
import os
import re
import sys
from dataclasses import asdict, replace
from pathlib import Path
from uuid import uuid4

from rich.console import Console, RenderableType
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from .agent import CompressionResult, RequestResult
from .llm_models import LLMModel, ModelStorageError, ModelConnectionError, discover_models
from .history import HistoryManager
from .invariants import (
    InvariantSet, InvariantStorageError,
)
from .memory import (
    MEMORY_LAYERS, MemorySnapshot, MemoryStorageError,
)
from .profile import ProfileStorageError, UserProfile
from .task_state import CONTINUE_TASK, TaskState
from .mcp_config import MCPServer, MCPStorageError
from .mcp_client import MCPConnectionError
from .mcp_tools import MCPToolError
from .tool_events import ToolCallRecord
from .models import Conversation, ContextSettings, RequestOptions, Turn
from .service import ConversationService
from .storage import DEFAULT_DATA_DIR, ConversationBusyError, ConversationStorageError

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
RAG_SETTING_FIELDS = (
    "rag_enabled", "rag_rewrite_enabled", "rag_filter_enabled",
    "rag_top_k_before", "rag_top_k_after", "rag_similarity_threshold",
)

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


def cosine_threshold(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not -1 <= number <= 1:
        raise argparse.ArgumentTypeError("Порог cosine должен быть конечным числом от -1 до 1")
    return number


def load_env(path: Path) -> None:
    if not path.exists():
        return

    with path.open(encoding="utf-8") as env_file:
        for line in env_file:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Чат с выбранной OpenAI-совместимой LLM")
    parser.add_argument(
        "--model",
        help="ID записи каталога или уникальный API ID; без флага используется выбор диалога",
    )
    model_action = parser.add_mutually_exclusive_group()
    model_action.add_argument("--models-list", action="store_true", help="Показать каталог моделей без токенов")
    model_action.add_argument("--models-discover", metavar="BASE_URL", help="Получить список моделей сервера без добавления")
    model_action.add_argument("--models-import", metavar="BASE_URL", help="Добавить список моделей сервера без дубликатов")
    model_action.add_argument("--model-add", metavar="API_ID", help="Добавить модель вручную; требуется --base-url")
    parser.add_argument("--base-url", help="Адрес OpenAI-совместимого API для --model-add")
    parser.add_argument("--model-name", help="Отображаемое имя при --model-add (по умолчанию API ID)")
    parser.add_argument("--model-token-env", help="Имя переменной окружения с токеном при добавлении/загрузке моделей")
    parser.add_argument("--system", help="Системный промпт диалога (сохраняется, если ещё не задан)")
    parser.add_argument("--user", help="Текст запроса; необязателен для управления диалогами, памятью, профилями, задачей и инвариантами")
    parser.add_argument("--batch", action="store_true", help="Один запрос для планировщика: --conversation и --user; результат в JSON")
    rag = parser.add_mutually_exclusive_group()
    rag.add_argument("--rag", dest="rag_enabled", action="store_true", default=None,
                     help="Включить поиск по базе знаний; без флага сохраняется режим диалога")
    rag.add_argument("--no-rag", dest="rag_enabled", action="store_false",
                     help="Отключить поиск по базе знаний")
    rewrite = parser.add_mutually_exclusive_group()
    rewrite.add_argument("--rag-rewrite", dest="rag_rewrite_enabled", action="store_true", default=None,
                         help="Переписывать поисковый запрос отдельным вызовом модели")
    rewrite.add_argument("--no-rag-rewrite", dest="rag_rewrite_enabled", action="store_false",
                         help="Искать по исходному вопросу")
    relevance = parser.add_mutually_exclusive_group()
    relevance.add_argument("--rag-filter", dest="rag_filter_enabled", action="store_true", default=None,
                           help="Отсекать кандидатов ниже порога cosine similarity")
    relevance.add_argument("--no-rag-filter", dest="rag_filter_enabled", action="store_false",
                           help="Не отсеивать отдельные чанки; проверка достаточности контекста остаётся включённой")
    parser.add_argument("--rag-top-k-before", type=positive_int, metavar="N",
                        help="Количество кандидатов поиска до отбора")
    parser.add_argument("--rag-top-k-after", type=positive_int, metavar="N",
                        help="Максимум фрагментов после отбора; не больше --rag-top-k-before")
    parser.add_argument("--rag-similarity-threshold", type=cosine_threshold, metavar="SCORE",
                        help="Порог достаточности контекста от -1 до 1; с --rag-filter также отсекает чанки")
    parser.add_argument("--mcp-url", help="Сохранить общий MCP-сервер для CLI и веба (ID: cli)")
    parser.add_argument("--mcp-token-env", default="", help="Имя переменной с MCP Bearer-токеном (не сам токен)")
    task_action = parser.add_mutually_exclusive_group()
    task_action.add_argument(
        "--task-start", metavar="TITLE",
        help="Создать задачу на этапе planning; добавьте --user для первого шага через API",
    )
    task_action.add_argument(
        "--task-show", action="store_true",
        help="Показать сохранённое состояние задачи в JSON без обращения к API",
    )
    task_action.add_argument(
        "--task-pause", action="store_true",
        help="Приостановить задачу, сохранив её этап и текущий шаг, без обращения к API",
    )
    task_action.add_argument(
        "--task-resume", action="store_true",
        help="Снять паузу задачи без обращения к API; добавьте --user для следующего шага",
    )
    task_action.add_argument(
        "--task-approve", action="store_true",
        help="Утвердить сохранённый план без обращения к API; добавьте --user для выполнения первого шага",
    )
    task_action.add_argument(
        "--task-continue", action="store_true",
        help="Выполнить следующий шаг сохранённой задачи через API без повторного описания",
    )
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
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--conversation", "--history", dest="conversation",
        metavar="ID_OR_PATH",
        help="ID или путь к существующему общему диалогу",
    )
    selection.add_argument("--new-conversation", action="store_true", help="Создать общий диалог")
    parser.add_argument("--data-dir", type=Path, help="Общий каталог диалогов (по умолчанию data/conversations проекта)")
    parser.add_argument("--list-conversations", action="store_true", help="Показать ID, названия и пути диалогов в JSON")
    parser.add_argument("--show-conversation", action="store_true", help="Показать выбранный диалог целиком в JSON")
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
        "--strategy", choices=("full", "window", "facts", "summary"),
        help="Стратегия нового диалога; у существующего по умолчанию сохраняется",
    )
    parser.add_argument(
        "--window-size", type=positive_int, metavar="N",
        help="Число последних сообщений, включая текущий запрос; требует --strategy window или facts",
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
        help="Очистить рабочую память текущего диалога",
    )
    parser.add_argument(
        "--memory-show", action="store_true",
        help="Показать три слоя памяти без обращения к API (до запроса, если указан --user)",
    )
    profile_selection = parser.add_mutually_exclusive_group()
    profile_selection.add_argument(
        "--profile", metavar="ID",
        help="Выбрать сохранённый профиль для текущего диалога и следующих запросов",
    )
    profile_selection.add_argument(
        "--profile-clear", action="store_true",
        help="Отключить профиль текущего диалога",
    )
    profile_edit = parser.add_mutually_exclusive_group()
    profile_edit.add_argument(
        "--profile-import", type=Path, metavar="PATH",
        help="Создать или заменить профиль из JSON; для выбора добавьте --profile ID",
    )
    profile_edit.add_argument(
        "--profile-delete", metavar="ID",
        help="Удалить профиль и отключить его во всех диалогах",
    )
    profile_view = parser.add_mutually_exclusive_group()
    profile_view.add_argument(
        "--profile-list", action="store_true",
        help="Показать сохранённые профили без обращения к API",
    )
    profile_view.add_argument(
        "--profile-show", action="store_true",
        help="Показать выбранный профиль текущего диалога (null, если профиль отключён)",
    )
    parser.add_argument(
        "--invariants-file", type=Path, metavar="PATH",
        help="Файл правил (по умолчанию invariants.json рядом с каталогом диалогов)",
    )
    parser.add_argument(
        "--invariants-import", type=Path, metavar="PATH",
        help="Проверить JSON и заменить общие правила в --invariants-file без обращения к API",
    )
    parser.add_argument(
        "--invariants-show", action="store_true",
        help="Показать действующие инварианты в JSON без обращения к API",
    )
    args = parser.parse_args()
    model_management = args.models_list or args.models_discover or args.models_import or args.model_add
    if args.model_add and not args.base_url:
        parser.error("--model-add требует --base-url")
    if (args.base_url or args.model_name) and not args.model_add:
        parser.error("--base-url и --model-name используются с --model-add")
    if args.model_token_env and not (args.model_add or args.models_discover or args.models_import):
        parser.error("--model-token-env требует добавления или загрузки моделей")
    if model_management and (args.user is not None or args.conversation or args.new_conversation or args.batch):
        parser.error("Управление каталогом моделей выполняется отдельной командой")
    rag_changed = any(getattr(args, name) is not None for name in RAG_SETTING_FIELDS)
    if args.batch:
        if not args.conversation or args.user is None:
            parser.error("--batch требует явные --conversation и --user")
        forbidden = (
            args.new_conversation, args.list_conversations, args.show_conversation,
            args.memory_set, args.memory_delete, args.memory_clear_working, args.memory_show,
            args.profile, args.profile_clear, args.profile_import, args.profile_delete,
            args.profile_list, args.profile_show, args.invariants_import, args.invariants_show,
            args.task_start, args.task_show, args.task_pause, args.task_resume,
            args.task_approve, args.task_continue, args.mcp_url, args.mcp_token_env,
        )
        if any(value is not None and value is not False and value != "" for value in forbidden):
            parser.error("--batch нельзя совмещать с командами управления или просмотра")
    if args.conversation is not None and not args.conversation.strip():
        parser.error("Укажите ID или путь диалога")
    if args.user is not None and not args.user.strip():
        parser.error("Введите непустое сообщение")
    task_options = (
        args.task_start is not None, args.task_show, args.task_pause,
        args.task_resume, args.task_approve, args.task_continue,
    )
    if args.task_start is not None and not args.task_start.strip():
        parser.error("Название задачи должно быть непустой строкой")
    if args.user is not None and (args.task_show or args.task_pause or args.task_continue):
        parser.error("--task-show, --task-pause и --task-continue не совмещаются с --user")
    if args.task_show and (args.memory_show or args.profile_show or args.profile_list):
        parser.error("--task-show нельзя совмещать с другими командами просмотра")
    if args.task_continue:
        if not args.conversation:
            parser.error("--task-continue требует --conversation ID_OR_PATH")
        args.user = CONTINUE_TASK
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
    if model_management and (
        args.model or any(profile_options) or any(task_options) or memory_edit
        or args.memory_show or args.memory_clear_working or args.invariants_import
        or args.invariants_show or args.list_conversations or args.show_conversation
        or rag_changed or args.mcp_url or args.strategy
    ):
        parser.error("Управление каталогом моделей выполняется отдельной командой")
    if memory_edit:
        layer = memory_edit[0]
        if layer not in MEMORY_LAYERS:
            parser.error("LAYER должен быть working или long_term")
        if any(not value.strip() for value in memory_edit[1:]):
            parser.error("Ключ и значение памяти должны быть непустыми строками")
    if args.invariants_show and (
        args.user is not None or any(profile_options) or any(task_options)
        or memory_edit or args.memory_show or args.memory_clear_working or args.new_conversation or rag_changed
    ):
        parser.error("--invariants-show совмещается только с --invariants-import и выбором файла")
    if args.user is None:
        if not any(profile_options) and not any(task_options) and not (
            args.invariants_import or args.invariants_show or args.new_conversation or model_management or args.model
            or args.list_conversations or args.show_conversation or
            rag_changed or
            memory_edit or args.memory_show or args.memory_clear_working
        ):
            parser.error("Укажите --user или операцию с диалогами, памятью, профилями, задачей или инвариантами")
        if args.meta_prompt or args.system is not None:
            parser.error("--meta-prompt и --system требуют --user")
    if args.strategy in ("window", "facts") and args.window_size is None and args.conversation is None:
        parser.error(f"Для --strategy {args.strategy} необходимо указать --window-size")
    if args.window_size is not None and args.strategy not in ("window", "facts"):
        parser.error("--window-size требует --strategy window или facts")
    if args.strategy not in (None, "summary") and (args.last_messages is not None or args.compress_every is not None):
        parser.error("--strategy нельзя совмещать с --last-messages или --compress-every")
    if (args.last_messages is None) != (args.compress_every is None):
        parser.error("Укажите вместе --last-messages и --compress-every")
    if args.mcp_token_env and not args.mcp_url:
        parser.error("--mcp-token-env требует --mcp-url")
    views = (args.list_conversations, args.show_conversation)
    if any(views) and (
        args.user is not None or any(task_options) or any(profile_options) or memory_edit
        or args.memory_show or args.memory_clear_working or args.invariants_import
        or args.invariants_show or args.new_conversation or all(views) or args.mcp_url
        or rag_changed
    ):
        parser.error("Просмотр диалогов нельзя совмещать с другими операциями")
    if args.list_conversations and args.conversation:
        parser.error("--list-conversations не требует выбора диалога; используйте --data-dir")
    requires_conversation = (
        args.show_conversation or args.task_show or args.task_pause or args.task_resume
        or args.task_approve or args.task_continue or args.memory_set or args.memory_delete
        or args.memory_show or args.memory_clear_working or args.profile or args.profile_clear
        or args.profile_show
        or rag_changed
    )
    if requires_conversation and not (args.conversation or args.new_conversation or args.user is not None or args.task_start):
        parser.error("Выберите --conversation ID_OR_PATH или создайте --new-conversation")
    if args.show_conversation and not args.conversation:
        parser.error("--show-conversation требует --conversation ID_OR_PATH")
    if args.conversation and re.fullmatch(r"[0-9a-f]{32}", args.conversation) is None:
        path = Path(args.conversation).expanduser().resolve()
        if path.suffix != ".json" or re.fullmatch(r"[0-9a-f]{32}", path.stem) is None:
            parser.error("Путь должен указывать на файл диалога <ID>.json; старый формат истории не поддерживается")
        if args.data_dir is not None and args.data_dir.expanduser().resolve() != path.parent:
            parser.error("Путь диалога находится вне --data-dir")
        args.data_dir = path.parent
        args.conversation = path.stem
    args.data_dir = (args.data_dir or DEFAULT_DATA_DIR).expanduser().resolve()
    if args.conversation and not (args.data_dir / f"{args.conversation}.json").is_file():
        parser.error("Диалог не найден. Для создания используйте --new-conversation")
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
    meta.add_row("Модель", args.model or "Выбранная в диалоге")
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
    choice = response.choices[0] if response.choices else None

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
    meta.add_row("Причина остановки", (choice.finish_reason if choice is not None else None) or "—")
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


def print_memory(conversation: Conversation, path: Path, snapshot: MemorySnapshot) -> None:
    messages, summary, facts, _ = HistoryManager._decode_data(conversation.working_context)
    data = {
        "short_term": {
            "scope": conversation.id,
            "history": str(path),
            "messages": [{"role": item["role"], "content": item["content"]} for item in messages],
            "summary": summary,
            "facts": facts,
        },
        "working": snapshot.working,
        "long_term": snapshot.long_term,
        "dialogue_task_memory": conversation.dialogue_task_memory,
    }
    print_json(data)


def print_task_state(state: TaskState | None, *, json_only: bool = False) -> None:
    if json_only:
        data = state.to_dict() if state is not None else None
        console.print(
            json.dumps(data, ensure_ascii=False, indent=2),
            markup=False, highlight=False, soft_wrap=True,
        )
    elif state is not None:
        details = Table.grid(padding=(0, 2))
        details.add_column(style="bold dim")
        details.add_column()
        details.add_row("Задача", Text(state.title))
        details.add_row("Этап", state.stage.value + (" · пауза" if state.paused else ""))
        details.add_row("Текущий шаг", Text(state.current_step))
        if state.awaiting_approval:
            details.add_row("План на утверждение", Text("\n".join(
                f"{index}. {step}" for index, step in enumerate(state.plan, start=1)
            )))
        details.add_row("Ожидаемое действие", Text(state.expected_action))
        console.print(Panel(details, title="Состояние задачи", border_style="cyan"))


def load_profile(path: Path) -> UserProfile:
    """Проверить импорт целиком до изменения сохранённого профиля."""
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as error:
        raise ValueError(f"Файл профиля не найден: {path}") from error
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"Некорректный JSON профиля: {path}") from error
    return UserProfile.from_dict(data)


def print_tool_call(record: ToolCallRecord) -> None:
    status = "ошибка" if record.is_error else "результат получен"
    console.print(Panel(
        Syntax(json.dumps({"arguments": record.arguments, "result": json.loads(record.result)},
                          ensure_ascii=False, indent=2), "json", word_wrap=True),
        title=Text(f"MCP · {record.server_name} / {record.tool_name} · {status}"),
    ))


def print_json(value: object) -> None:
    console.print(json.dumps(value, ensure_ascii=False, indent=2),
                  markup=False, highlight=False, soft_wrap=True)


def context_settings(args: argparse.Namespace, current: ContextSettings) -> ContextSettings:
    updates = {name: getattr(args, name) for name in RAG_SETTING_FIELDS if getattr(args, name) is not None}
    if args.strategy is not None:
        updates["strategy"] = args.strategy
    if args.window_size is not None:
        updates["window_size"] = args.window_size
    if args.last_messages is not None:
        updates.update(strategy="summary", last_messages=args.last_messages,
                       compress_every=args.compress_every)
    settings = replace(current, **updates)
    settings.validate()
    return settings


def apply_task_options(args: argparse.Namespace, service: ConversationService, conversation: Conversation) -> Conversation:
    if args.task_start is not None:
        return service.start_task(conversation.id, args.task_start)
    if args.task_pause:
        return service.pause_task(conversation.id)
    if args.task_resume:
        return service.resume_task(conversation.id)
    if args.task_approve:
        return service.approve_task_plan(conversation.id)
    if args.task_continue:
        state = conversation.task_state
        if state is None:
            raise ValueError("Задача не создана; используйте --task-start TITLE")
        if state.stage == "done":
            raise ValueError("Задача уже завершена; создайте новую через --task-start TITLE")
        if state.paused:
            raise ValueError("Задача на паузе; сначала снимите паузу через --task-resume")
        if state.awaiting_approval:
            raise ValueError("План ожидает утверждения; используйте --task-approve или отправьте правки через --user")
    return conversation


def send_message(args: argparse.Namespace, conversation: Conversation, options: RequestOptions) -> None:
    load_env(ENV_PATH)
    service = ConversationService(args.data_dir, invariants_path=args.invariants_file)
    selected = service.get_model(conversation.id, options.model)
    args.model = selected.name
    settings = context_settings(args, conversation.settings)
    if args.batch:
        result = service.send(
            conversation.id, args.user, system_prompt=args.system or "", options=options,
            settings=settings, expected_settings=conversation.settings,
        )
        turn = result.turns[-1]
        payload = json.dumps({"conversation_id": result.id, **asdict(turn)}, ensure_ascii=False) + "\n"
        # Перенаправленный stdout Windows может иметь ANSI-кодировку: JSONL всегда UTF-8.
        if hasattr(sys.stdout, "buffer"):
            sys.stdout.buffer.write(payload.encode("utf-8"))
            sys.stdout.buffer.flush()
        else:
            sys.stdout.write(payload)
        if turn.status != "completed" or turn.error or any(call.is_error for call in turn.tool_calls):
            raise SystemExit(1)
        return
    responses: list[RequestResult] = []
    compressions: list[CompressionResult] = []
    print_request_info(args, system=conversation.system_prompt or args.system, user=args.user)
    with console.status("[bold cyan]Ожидание ответа модели…", spinner="dots"):
        result = service.send(
            conversation.id, args.user, system_prompt=args.system or "", options=options,
            settings=settings, expected_settings=conversation.settings,
            on_response=responses.append, on_compression=compressions.append,
        )
    for compression in compressions:
        print_compression(compression)
    turn = result.turns[-1]
    mode = (turn.rag_settings.mode_label if turn.rag_settings is not None else "С RAG") if turn.rag_enabled else "Без RAG"
    console.print(Text(f"Режим: {mode}"))
    print_rag_sources(turn)
    for tool_call in turn.tool_calls:
        print_tool_call(tool_call)
    for index, response in enumerate(responses):
        is_meta = len(responses) == 2 and index == 0
        print_response(
            response, "text" if is_meta else args.response_format,
            title="Сгенерированный промпт" if is_meta else "Ответ",
            context_limit=args.context_limit, max_tokens=args.max_tokens,
        )
    if not responses:
        for title, content in (("Сгенерированный промпт", turn.meta_prompt), ("Ответ", turn.answer)):
            if content is not None:
                console.print(Panel(render_content(content, args.response_format if title == "Ответ" else "text"), title=title))
    print_task_state(result.task_state)
    if turn.error:
        raise ValueError(turn.error)


def print_rag_sources(turn: Turn) -> None:
    if turn.rag_preparation is not None:
        diagnostic = turn.rag_preparation
        console.print(Text(f"Подготовка диалога: {diagnostic['status']} · {diagnostic['elapsed_seconds']:.2f} с"))
        if diagnostic["reason"]:
            console.print(Text(diagnostic["reason"]))
        if diagnostic["usage"] is not None:
            console.print(Text(f"Токены подготовки: {diagnostic['usage']['total_tokens']}"))
    if turn.rag_context is None:
        return
    context = turn.rag_context
    sources = Table(title="Найденные фрагменты RAG · диагностика")
    for column in ("№", "Источник", "Раздел", "Чанк", "Cosine similarity"):
        sources.add_column(column)
    for number, chunk in enumerate(context["chunks"], start=1):
        sources.add_row(str(number), Text(chunk["source"]), Text(chunk["section"] or "—"),
                        Text(str(chunk["chunk_id"])), f"{chunk['score']:.3f}")
    console.print(Text(f"Индекс: {context['index']}"))
    console.print(sources)
    console.print("Результаты поиска не равны использованным доказательствам. Источники и цитаты ответа приведены в самом ответе.")
    if not context["chunks"]:
        console.print("После отбора релевантных фрагментов не осталось. Для нового RAG-ответа требуется уточнение вопроса.")
    if "original_query" not in context:
        return
    console.print(Text(f"Исходный вопрос: {context['original_query']}\nПоисковый запрос: {context['search_query']}"))
    settings = context["settings"]
    threshold = str(settings["similarity_threshold"])
    filter_label = "включён" if settings["filter_enabled"] else "выключен"
    console.print(f"Top-K: {settings['top_k_before']} → {settings['top_k_after']} · Порог cosine: {threshold} · Фильтр чанков: {filter_label}")
    counts = context["counts"]
    console.print(f"Найдено: {counts['found']} · Прошло порог: {counts['passed']} · Передано модели: {counts['selected']}")
    rewrite = context["rewrite"]
    status = {"disabled": "выключен", "success": "выполнен", "fallback": "использован исходный вопрос"}
    console.print(f"Query rewrite: {status[rewrite['status']]} · {rewrite['elapsed_seconds']:.2f} с")
    if rewrite.get("reason"):
        console.print(Text(f"Причина fallback: {rewrite['reason']}"))
    usage = rewrite.get("usage")
    if usage is not None:
        console.print(f"Токены rewrite: вход {usage['prompt_tokens']} → выход {usage['completion_tokens']} (всего {usage['total_tokens']})")
    elif rewrite["status"] != "disabled":
        console.print("Токены rewrite: API не вернул статистику")
    console.print(f"Время поиска и отбора: {context['retrieval_seconds']:.2f} с")
    embedding = context["embedding"]
    console.print(Text(f"Эмбеддинги: {embedding.get('model', '—')} · ревизия {embedding.get('revision', '—')}\nХеш корпуса: {context['corpus_hash']}"))
    candidates = Table(title="Кандидаты до отбора")
    for column in ("Ранг", "Источник", "Чанк", "Cosine", "Отбор"):
        candidates.add_column(column)
    reasons = {"selected": "Передан модели", "below_threshold": "Ниже порога", "top_k": "За пределами top-K"}
    for chunk in context["candidates"]:
        candidates.add_row(str(chunk["original_rank"]), Text(chunk["source"]), Text(str(chunk["chunk_id"])),
                           f"{chunk['score']:.3f}", reasons[chunk["selection_reason"]])
    console.print(candidates)
    console.print("Cosine similarity — сходство с поисковым запросом, не вероятность правильного ответа.")


def manage_models(args: argparse.Namespace, service: ConversationService) -> bool:
    """Управление каталогом без создания диалога и раскрытия токенов."""
    if not (args.models_list or args.models_discover or args.models_import or args.model_add):
        return False
    token = ""
    if args.model_token_env:
        load_env(ENV_PATH)
        token = os.environ.get(args.model_token_env, "").strip()
        if not token:
            raise ValueError("Указанная переменная с токеном отсутствует или пуста")
    if args.models_discover or args.models_import:
        url = args.models_discover or args.models_import
        ids = discover_models(url, token)
        if args.models_discover:
            print_json(ids)
            return True
        service.models.import_models(url, token, ids)
    elif args.model_add:
        service.models.save(LLMModel(
            id=uuid4().hex, name=args.model_name or args.model_add,
            model_id=args.model_add, base_url=args.base_url, token=token,
        ))
    print_json([
        {"id": model.id, "name": model.name, "model_id": model.model_id,
         "base_url": model.base_url, "has_token": bool(model.token),
         "timeout": model.timeout, "json_mode": model.json_mode,
         "tools_enabled": model.tools_enabled}
        for model in service.models.list()
    ])
    return True


def main() -> None:
    args = parse_args()
    options = RequestOptions(
        meta_prompt=args.meta_prompt, temperature=args.temperature, max_tokens=args.max_tokens,
        model=args.model, stop_sequences=args.stop_sequences, response_format=args.response_format,
    )
    options.validate()
    imported_profile = load_profile(args.profile_import) if args.profile_import else None
    imported_invariants = None
    if args.invariants_import:
        try:
            imported_invariants = InvariantSet.from_json(args.invariants_import.read_text(encoding="utf-8-sig"))
        except FileNotFoundError as error:
            raise ValueError(f"Файл инвариантов не найден: {args.invariants_import}") from error
    service = ConversationService(args.data_dir, token=None, invariants_path=args.invariants_file)
    if manage_models(args, service):
        return
    if imported_invariants is not None:
        service.invariants.save(imported_invariants)
    if args.invariants_show:
        print_json(service.get_invariants().to_dict())
        return
    if args.list_conversations:
        print_json([
            {"id": item.id, "title": item.title, "path": str(service.store.path(item.id))}
            for item in service.list_conversations()
        ])
        if service.storage_errors:
            raise ConversationStorageError("\n".join(service.storage_errors))
        return

    conversation = service.get(args.conversation) if args.conversation else None
    if conversation is None and (args.new_conversation or args.user is not None or args.task_start is not None):
        conversation = service.create()
    if args.show_conversation:
        print_json(asdict(conversation))
        return
    if args.user is not None:
        # Validate common rules before making local changes or loading the API environment.
        service.get_invariants()
    if imported_profile is not None:
        if conversation is not None:
            service.save_profile(conversation.id, imported_profile)
        else:
            service.profiles.save(imported_profile)
    if args.profile_delete:
        if conversation is not None:
            service.delete_profile(conversation.id, args.profile_delete)
        else:
            service.profiles.delete(args.profile_delete)
    if args.profile_list:
        print_json([profile.to_dict() for profile in service.profiles.list()])
    if conversation is None:
        if args.model:
            raise ValueError("Для выбора модели укажите --conversation или --new-conversation")
        return

    if args.model:
        selected = service.models.resolve(args.model)
        conversation = service.select_model(conversation.id, selected.id)
    conversation = apply_task_options(args, service, conversation)
    if args.profile:
        service.select_profile(conversation.id, args.profile)
    elif args.profile_clear:
        service.select_profile(conversation.id, None)
    if args.memory_set:
        service.remember_memory(conversation.id, *args.memory_set)
    elif args.memory_delete:
        service.forget_memory(conversation.id, *args.memory_delete)
    elif args.memory_clear_working:
        service.clear_working_memory(conversation.id)
    if args.mcp_url:
        service.mcp_servers.save(MCPServer("cli", "MCP", args.mcp_url, args.mcp_token_env, enabled=True))
    if args.task_show:
        print_task_state(conversation.task_state, json_only=True)
    elif args.profile_show:
        profile = service.get_profile(conversation.id)
        print_json(profile.to_dict() if profile is not None else None)
    elif args.memory_show:
        print_memory(conversation, service.store.path(conversation.id), service.get_memory(conversation.id))
    elif not args.profile_list and not args.batch:
        console.print(Text(f"Диалог: {conversation.id}\nФайл: {service.store.path(conversation.id)}"), soft_wrap=True)
        if args.user is None:
            print_task_state(conversation.task_state)
    if args.user is not None:
        send_message(args, conversation, options)
    else:
        settings = context_settings(args, conversation.settings)
        if settings != conversation.settings:
            service.update_settings(conversation.id, settings, expected_settings=conversation.settings)


def run() -> None:
    """Запустить CLI с выводом ожидаемых ошибок без traceback."""
    try:
        main()
    except (ValueError, OSError, KeyError, MemoryStorageError, ProfileStorageError, InvariantStorageError,
            MCPConnectionError, MCPToolError, MCPStorageError, ConversationBusyError,
            ConversationStorageError, ModelStorageError, ModelConnectionError) as error:
        raise SystemExit(str(error)) from error
