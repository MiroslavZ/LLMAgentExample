"""Read-only MCP через Streamable HTTP для сервиса поддержки LLMAgent."""

import argparse
from pathlib import Path
from typing import Any

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

if __package__:
    from .store import DEFAULT_DATA_PATH, SupportDataError, SupportStore
else:
    # Прямой запуск server.py из каталога сервера, в том числе через его venv.
    from store import DEFAULT_DATA_PATH, SupportDataError, SupportStore


def create_server(data_path: Path = DEFAULT_DATA_PATH) -> MCPServer:
    store = SupportStore(data_path)
    server = MCPServer(
        name="llm-agent-support",
        version="1.0.0",
        instructions="Демонстрационные пользователи и тикеты LLMAgent. Только чтение.",
    )
    annotations = ToolAnnotations(
        read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=False,
    )

    def read(method, *args):
        try:
            return method(*args)
        except SupportDataError as error:
            raise ToolError(str(error)) from None

    @server.tool(annotations=annotations, structured_output=True)
    def list_users() -> dict[str, Any]:
        """Список демонстрационных пользователей: id, имя и краткое описание."""
        return read(store.list_users)

    @server.tool(annotations=annotations, structured_output=True)
    def list_user_tickets(user_id: str) -> dict[str, Any]:
        """Краткие карточки обращений только выбранного пользователя."""
        return read(store.list_user_tickets, user_id)

    @server.tool(annotations=annotations, structured_output=True)
    def get_support_context(user_id: str, ticket_id: str | None = None) -> dict[str, Any]:
        """Конфигурация пользователя и его тикет; без ticket_id — только пользователь.

        Возвращает фиксированные демонстрационные наблюдения, а не живую диагностику.
        Чужой или неизвестный тикет недоступен. Секретов и операций записи нет.
        """
        return read(store.get_support_context, user_id, ticket_id)

    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="MCP-контекст поддержки LLMAgent (демо)")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("Порт должен быть от 1 до 65535")
    create_server().run(
        transport="streamable-http", host="127.0.0.1", port=args.port,
        streamable_http_path="/mcp", stateless_http=True, json_response=True,
    )


if __name__ == "__main__":
    main()
