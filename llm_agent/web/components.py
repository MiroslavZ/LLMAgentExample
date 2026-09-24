"""Компоненты представления без работы с хранилищем и моделью."""

import re
from collections.abc import Callable
from datetime import datetime

from nicegui import ui

from ..models import Turn

STRATEGIES = {
    "full": "Вся история",
    "window": "Скользящее окно",
    "facts": "Факты текущего диалога",
    "summary": "Сжатие истории",
}
STRATEGY_HELP = {
    "full": "Агент получает всю переписку этого диалога.",
    "window": "Агент получает последние N сообщений. Системный промпт всегда остаётся в контексте.",
    "facts": "Агент извлекает факты только из текущего диалога и получает последние N сообщений. Извлечение — дополнительный запрос к модели. Эти факты остаются в краткосрочном контексте диалога.",
    "summary": "Старые сообщения объединяются в краткое содержание. Последние сообщения передаются полностью. Сжатие — дополнительный запрос к модели.",
}


def format_time(value: str) -> str:
    try:
        return datetime.fromisoformat(value).astimezone().strftime("%H:%M")
    except (ValueError, TypeError):
        return ""


def empty_chat() -> None:
    with ui.column().classes("empty-chat"):
        with ui.element("div").classes("welcome-icon"):
            ui.icon("auto_awesome", size="30px")
        ui.label("С чего начнём?").classes("welcome-title")
        ui.label("Задайте агенту роль и отправьте первое сообщение.").classes("welcome-description")
        with ui.row().classes("welcome-tags"):
            ui.label("Обсудить идею")
            ui.label("Разобраться в задаче")
            ui.label("Написать код")


def _tool_code(content: str, *, language: str) -> None:
    code = ui.code(content, language=language).classes("w-full")
    # Результат MCP может сам содержать Markdown-код. Длинная ограда сохраняет
    # его буквальным текстом, не позволяя закрыть блок и отобразить HTML.
    fence = "`" * max(3, 1 + max((len(run[0]) for run in re.finditer(r"`+", content)), default=0))
    code.markdown.bind_content_from(code, "content", lambda value: f"{fence}{language}\n{value}\n{fence}")


def render_turn(turn: Turn, *, restore: Callable[[], None], busy: bool) -> None:
    with ui.column().classes("chat-turn"):
        with ui.column().classes("user-message"):
            with ui.row().classes("message-heading"):
                ui.label("Вы").classes("message-author")
                ui.label(format_time(turn.created_at)).classes("message-time")
            # Пользовательский ввод отображаем буквально, включая пробелы и HTML.
            ui.label(turn.user).classes("user-text")
            ui.button(icon="content_copy", on_click=lambda: ui.clipboard.write(turn.user)).props(
                'flat round dense size=sm aria-label="Копировать сообщение"'
            ).classes("copy-user-message").tooltip("Копировать сообщение")
        if turn.meta_prompt is not None:
            with ui.expansion("Сгенерированный промпт", icon="auto_fix_high").classes("meta-result"):
                ui.markdown(turn.meta_prompt).classes("message-markdown")
        for call in turn.tool_calls:
            status = "Ошибка" if call.is_error else "Успешно"
            with ui.expansion(
                f"{call.server_name} · {call.tool_name} · {status}",
                icon="error_outline" if call.is_error else "build",
            ).classes("mcp-tool"):
                ui.label(f"{status} · {call.elapsed_seconds:.1f} с").classes("memory-description")
                ui.label("Аргументы").classes("font-medium text-sm")
                _tool_code(call.arguments, language="json")
                ui.label("Результат").classes("font-medium text-sm")
                _tool_code(call.result, language="text")
        if turn.answer is not None:
            with ui.column().classes("assistant-message"):
                with ui.row().classes("message-heading"):
                    ui.icon("auto_awesome", size="17px").classes("text-primary")
                    ui.label("Агент").classes("message-author")
                    if turn.elapsed_seconds is not None:
                        ui.label(f"{turn.elapsed_seconds:.1f} с").classes("message-time")
                if turn.answer:
                    ui.markdown(turn.answer).classes("message-markdown")
                else:
                    ui.label("Модель вернула пустой ответ.").classes("muted")
        if turn.status == "running":
            with ui.row().classes("working-message"):
                ui.spinner("dots", size="24px")
                ui.label("Агент готовит ответ…")
        elif turn.error or turn.status in {"error", "partial", "interrupted"}:
            with ui.column().classes("error-message"):
                with ui.row().classes("items-center gap-2"):
                    ui.icon("info_outline", size="18px")
                    ui.label(turn.error or "Запрос был прерван. Его результат неизвестен.")
                if turn.memory_updated:
                    ui.label("Краткосрочный контекст диалога был обновлён до ошибки.").classes("text-xs")
                ui.button("Вернуть текст в поле ввода", icon="edit_note", on_click=restore).props(
                    "flat dense no-caps"
                ).set_enabled(not busy)
