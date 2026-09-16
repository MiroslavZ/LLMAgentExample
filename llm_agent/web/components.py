"""Компоненты представления без работы с хранилищем и моделью."""

from collections.abc import Callable
from datetime import datetime

from nicegui import ui

from ..models import Turn

STRATEGIES = {
    "full": "Вся история",
    "window": "Скользящее окно",
    "facts": "Память фактов",
    "summary": "Сжатие истории",
}
STRATEGY_HELP = {
    "full": "Агент получает всю переписку этого диалога.",
    "window": "Агент получает последние N сообщений. Системный промпт всегда остаётся в контексте.",
    "facts": "Агент запоминает важные факты и получает последние N сообщений. Обновление памяти — дополнительный запрос к модели.",
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


def render_turn(turn: Turn, *, restore: Callable[[], None], busy: bool) -> None:
    with ui.column().classes("chat-turn"):
        with ui.column().classes("user-message"):
            with ui.row().classes("message-heading"):
                ui.label("Вы").classes("message-author")
                ui.label(format_time(turn.created_at)).classes("message-time")
            # Пользовательский ввод отображаем буквально, включая пробелы и HTML.
            ui.label(turn.user).classes("user-text")
        if turn.meta_prompt is not None:
            with ui.expansion("Сгенерированный промпт", icon="auto_fix_high").classes("meta-result"):
                ui.markdown(turn.meta_prompt).classes("message-markdown")
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
                    ui.label("Память диалога была обновлена до ошибки.").classes("text-xs")
                ui.button("Вернуть текст в поле ввода", icon="edit_note", on_click=restore).props(
                    "flat dense no-caps"
                ).set_enabled(not busy)
