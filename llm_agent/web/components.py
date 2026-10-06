"""Компоненты представления без работы с хранилищем и моделью."""

import re
from collections.abc import Callable
from datetime import datetime

from nicegui import ui

from ..models import Turn
from ..rag_answer import render_answer

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


def render_rag_context(context: dict) -> None:
    chunks = context["chunks"]
    with ui.expansion(f"Найденные фрагменты RAG · {len(chunks)}", icon="library_books").classes("w-full"):
        ui.label(f"Индекс: {context['index']}").classes("memory-description")
        ui.label("Результаты поиска не равны использованным доказательствам. Источники и цитаты ответа приведены в самом ответе.").classes("setting-note")
        if not chunks:
            ui.label("После отбора релевантных фрагментов не осталось. Для нового RAG-ответа требуется уточнение вопроса.").classes("setting-note")
        for number, chunk in enumerate(chunks, start=1):
            with ui.expansion(f"[{number}] {chunk['source']}", icon="description").classes("w-full"):
                ui.label(f"Раздел: {chunk['section'] or '—'}").classes("memory-description")
                ui.label(f"Чанк: {chunk['chunk_id']} · Cosine similarity: {chunk['score']:.3f}").classes("memory-description")
                ui.label(chunk['text']).classes("whitespace-pre-wrap break-words w-full")
        if "original_query" not in context:
            return
        with ui.expansion("Диагностика поиска и отбора", icon="manage_search").classes("w-full"):
            ui.label(f"Исходный вопрос: {context['original_query']}").classes("whitespace-pre-wrap break-words")
            ui.label(f"Поисковый запрос: {context['search_query']}").classes("whitespace-pre-wrap break-words")
            settings = context["settings"]
            threshold = str(settings["similarity_threshold"])
            filter_label = "включён" if settings["filter_enabled"] else "выключен"
            ui.label(
                f"Top-K: {settings['top_k_before']} → {settings['top_k_after']} · Порог cosine: {threshold} · Фильтр чанков: {filter_label}"
            ).classes("memory-description")
            counts = context["counts"]
            ui.label(
                f"Найдено: {counts['found']} · Прошло порог: {counts['passed']} · Передано модели: {counts['selected']}"
            ).classes("memory-description")
            rewrite = context["rewrite"]
            status = {"disabled": "выключен", "success": "выполнен", "fallback": "использован исходный вопрос"}
            ui.label(f"Query rewrite: {status[rewrite['status']]} · {rewrite['elapsed_seconds']:.2f} с").classes("memory-description")
            if rewrite.get("reason"):
                ui.label(f"Причина fallback: {rewrite['reason']}").classes("memory-description")
            usage = rewrite.get("usage")
            if usage is not None:
                ui.label(
                    f"Токены rewrite: вход {usage['prompt_tokens']} → выход {usage['completion_tokens']} (всего {usage['total_tokens']})"
                ).classes("memory-description")
            elif rewrite["status"] != "disabled":
                ui.label("Токены rewrite: API не вернул статистику").classes("memory-description")
            ui.label(f"Время поиска и отбора: {context['retrieval_seconds']:.2f} с").classes("memory-description")
            embedding = context["embedding"]
            ui.label(f"Эмбеддинги: {embedding.get('model', '—')} · ревизия {embedding.get('revision', '—')}").classes("memory-description break-words")
            ui.label(f"Хеш корпуса: {context['corpus_hash']}").classes("memory-description break-words")
            ui.label("Cosine similarity — сходство с поисковым запросом, не вероятность правильного ответа.").classes("setting-note")
            reasons = {"selected": "Передан модели", "below_threshold": "Ниже порога", "top_k": "За пределами top-K"}
            ui.table(columns=[
                {"name": "rank", "label": "Ранг", "field": "rank", "align": "left"},
                {"name": "source", "label": "Источник", "field": "source", "align": "left"},
                {"name": "chunk_id", "label": "Чанк", "field": "chunk_id", "align": "left"},
                {"name": "score", "label": "Cosine", "field": "score", "align": "left"},
                {"name": "reason", "label": "Отбор", "field": "reason", "align": "left"},
            ], rows=[{
                "rank": chunk["original_rank"], "source": chunk["source"], "chunk_id": chunk["chunk_id"],
                "score": f"{chunk['score']:.3f}", "reason": reasons[chunk["selection_reason"]],
            } for chunk in context["candidates"]], row_key="chunk_id").classes("w-full").props("dense flat wrap-cells")


def render_turn(turn: Turn, *, restore: Callable[[], None], busy: bool) -> None:
    with ui.column().classes("chat-turn"):
        with ui.column().classes("user-message"):
            with ui.row().classes("message-heading"):
                ui.label("Вы").classes("message-author")
                ui.label(format_time(turn.created_at)).classes("message-time")
                mode = (turn.rag_settings.mode_label if turn.rag_settings is not None else "С RAG") if turn.rag_enabled else "Без RAG"
                ui.label(mode).classes("message-time")
            # Пользовательский ввод отображаем буквально, включая пробелы и HTML.
            ui.label(turn.user).classes("user-text")
            ui.button(icon="content_copy", on_click=lambda: ui.clipboard.write(turn.user)).props(
                'flat round dense size=sm aria-label="Копировать сообщение"'
            ).classes("copy-user-message").tooltip("Копировать сообщение")
        if turn.meta_prompt is not None:
            with ui.expansion("Сгенерированный промпт", icon="auto_fix_high").classes("meta-result"):
                ui.markdown(turn.meta_prompt).classes("message-markdown")
        if turn.rag_preparation is not None:
            diagnostic = turn.rag_preparation
            with ui.expansion("Подготовка диалога", icon="psychology").classes("w-full"):
                ui.label(f"{diagnostic['status']} · {diagnostic['elapsed_seconds']:.2f} с")
                if diagnostic["reason"]:
                    ui.label(diagnostic["reason"])
                if diagnostic["usage"] is not None:
                    ui.label(f"Токены подготовки: {diagnostic['usage']['total_tokens']}")
        if turn.rag_context is not None:
            render_rag_context(turn.rag_context)
        for step, call in enumerate(turn.tool_calls, start=1):
            status = "Ошибка" if call.is_error else "Успешно"
            with ui.expansion(
                f"{step}. {call.server_name} · {call.tool_name} · {status}",
                icon="error_outline" if call.is_error else "build",
            ).classes("mcp-tool"):
                ui.label(f"{status} · {call.elapsed_seconds:.1f} с").classes("memory-description")
                ui.label("Аргументы").classes("font-medium text-sm")
                _tool_code(call.arguments, language="json")
                ui.label("Результат").classes("font-medium text-sm")
                _tool_code(call.result, language="text")
            for attachment in call.attachments:
                ui.button(
                    f"Скачать {attachment.filename}", icon="download",
                    on_click=lambda _, file=attachment: ui.download.content(
                        file.text.encode("utf-8"), filename=file.filename, media_type="text/plain; charset=utf-8",
                    ),
                ).props("outline no-caps").classes("self-start")
        if turn.answer is not None:
            with ui.column().classes("assistant-message"):
                with ui.row().classes("message-heading"):
                    ui.icon("auto_awesome", size="17px").classes("text-primary")
                    ui.label("Агент").classes("message-author")
                    if turn.elapsed_seconds is not None:
                        ui.label(f"{turn.elapsed_seconds:.1f} с").classes("message-time")
                if turn.answer:
                    answer = (render_answer(turn.rag_answer, turn.options.response_format)
                              if turn.rag_answer is not None else turn.answer)
                    ui.markdown(answer).classes("message-markdown")
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
