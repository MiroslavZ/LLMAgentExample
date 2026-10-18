"""Запуск: python -m support_agent."""

import argparse

import uvicorn

from .web import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Сервис поддержки пользователей LLMAgent")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    args = parser.parse_args()
    uvicorn.run(create_app(), host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
