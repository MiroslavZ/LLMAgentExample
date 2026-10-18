"""Данные клиентов доступны сервису только через существующий MCP-клиент."""

from llm_agent.mcp_client import MCPConnectionError, call_tool
from llm_agent.mcp_config import MCPServer

from .errors import SupportError


class SupportCRM:
    def __init__(self, server: MCPServer) -> None:
        self.server = server

    async def _call(self, name: str, arguments: dict) -> dict:
        try:
            result = await call_tool(self.server, name, arguments, timeout=15)
        except MCPConnectionError as error:
            raise SupportError(str(error)) from None
        if result.is_error:
            # Не выводим произвольный текст сервера или его исключения.
            raise SupportError("Не удалось получить данные пользователя или обращения. Проверьте выбранный тикет.", 404)
        data = result.structured_content
        if not isinstance(data, dict):
            raise SupportError("Сервер поддержки вернул некорректные данные.", 502)
        return data

    async def list_users(self) -> list[dict]:
        data = await self._call("list_users", {})
        users = data.get("users")
        if not isinstance(users, list) or any(
            not isinstance(item, dict) or any(not isinstance(item.get(key), str) for key in ("id", "name", "summary"))
            for item in users
        ):
            raise SupportError("Сервер поддержки вернул некорректный список пользователей.", 502)
        return users

    async def list_tickets(self, user_id: str) -> list[dict]:
        data = await self._call("list_user_tickets", {"user_id": user_id})
        tickets = data.get("tickets")
        if not isinstance(tickets, list) or any(
            not isinstance(item, dict) or item.get("user_id") != user_id
            or any(not isinstance(item.get(key), str) for key in ("id", "subject", "status"))
            for item in tickets
        ):
            raise SupportError("Сервер поддержки вернул некорректный список обращений.", 502)
        return tickets

    async def context(self, user_id: str, ticket_id: str | None) -> dict:
        data = await self._call("get_support_context", {"user_id": user_id, "ticket_id": ticket_id})
        user, ticket = data.get("user"), data.get("ticket")
        if (not isinstance(user, dict) or user.get("id") != user_id
                or (ticket_id is None and ticket is not None)
                or (ticket_id is not None and (
                    not isinstance(ticket, dict) or ticket.get("id") != ticket_id or ticket.get("user_id") != user_id
                ))):
            raise SupportError("Контекст обращения не соответствует выбранному пользователю.", 502)
        return data
