"""Запуск локального чата: python -m llm_agent.web."""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from nicegui import ui

from ..service import ConversationService
from .jobs import RequestRunner
from .page import ChatPage

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "conversations"


def create_app(service: ConversationService, *, token_available: bool) -> RequestRunner:
    """Собрать маршруты; сервис можно подменить для проверки без API."""
    runner = RequestRunner(service)
    ui.add_css((Path(__file__).with_name("styles.css")).read_text(encoding="utf-8"), shared=True)

    @ui.page("/", title="Агент · Чат", language="ru")
    def chat(conversation: str | None = None) -> None:
        ui.colors(primary="#6658d9", secondary="#8176e4", accent="#6658d9", positive="#20856a")
        ChatPage(service, runner, token_available=token_available, conversation_id=conversation).build()

    return runner


def main() -> None:
    parser = argparse.ArgumentParser(description="Локальный чат с ИИ-агентом на NiceGUI")
    parser.add_argument("--host", default="127.0.0.1", help="Адрес сервера (по умолчанию 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8080, help="Порт сервера (по умолчанию 8080)")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Каталог диалогов")
    parser.add_argument("--no-browser", action="store_true", help="Не открывать браузер автоматически")
    args = parser.parse_args()
    load_dotenv(PROJECT_ROOT / ".env")
    token = os.getenv("API_KEY", "").strip()
    service = ConversationService(args.data_dir, token=token or None)
    create_app(service, token_available=bool(token))
    ui.run(
        host=args.host, port=args.port, title="Агент · Чат", favicon="✦",
        language="ru", show=not args.no_browser, reload=False,
    )
