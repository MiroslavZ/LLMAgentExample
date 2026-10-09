"""Общие параметры OpenAI-совместимых подключений, включая локальную сеть."""

from ipaddress import ip_address
from urllib.parse import urlsplit

from openai import DefaultHttpxClient


def connection_options(base_url: str, token: str | None) -> dict:
    """Для локальных адресов обходить системный HTTP-прокси.

    Публичные серверы сохраняют настройки proxy/CA окружения. Пустой ключ
    заменяется служебным значением SDK, а не ключом из OPENAI_API_KEY.
    """
    options = {"base_url": base_url, "api_key": token or "local-no-auth"}
    host = urlsplit(base_url).hostname or ""
    local = host.lower() == "localhost" or host.lower().endswith(".localhost")
    try:
        address = ip_address(host)
        local = local or address.is_private or address.is_loopback or address.is_link_local
    except ValueError:
        pass
    if local:
        options["http_client"] = DefaultHttpxClient(trust_env=False)
    return options
