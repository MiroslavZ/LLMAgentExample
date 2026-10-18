"""Серверная конфигурация; ключ и параметры модели не передаются браузеру."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from llm_agent.mcp_config import MCPServer

ROOT = Path(__file__).resolve().parents[1]
HISTORY_MESSAGES = 10
DEEPSEEK_URL = "https://api.deepseek.com"


def project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


@dataclass(frozen=True)
class Settings:
    api_key: str = field(default="", repr=False)
    model: str = "deepseek-flash"
    index_path: Path = ROOT / "data/support/index/structural.sqlite3"
    database_path: Path = ROOT / "data/support/conversations.sqlite3"
    cache: str | None = None
    mcp_url: str = "http://127.0.0.1:8001/mcp"
    mcp_token_env: str = ""

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv(ROOT / ".env")
        cache = os.getenv("RAG_CACHE_DIR")
        return cls(
            api_key=(os.getenv("DEEPSEEK_API_KEY") or os.getenv("API_KEY") or "").strip(),
            model=os.getenv("SUPPORT_DEEPSEEK_MODEL", "deepseek-flash").strip(),
            index_path=project_path(os.getenv("SUPPORT_INDEX_PATH", "data/support/index/structural.sqlite3")),
            database_path=project_path(os.getenv("SUPPORT_DATABASE_PATH", "data/support/conversations.sqlite3")),
            cache=str(project_path(cache)) if cache else None,
            mcp_url=os.getenv("SUPPORT_MCP_URL", "http://127.0.0.1:8001/mcp"),
            mcp_token_env=os.getenv("SUPPORT_MCP_TOKEN_ENV", ""),
        )

    def mcp_server(self) -> MCPServer:
        return MCPServer("support", "Поддержка LLMAgent", self.mcp_url, self.mcp_token_env, True)
