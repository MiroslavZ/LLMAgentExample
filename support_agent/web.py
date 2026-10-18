"""HTTP API и статический интерфейс; cookie изолирует демонстрационные диалоги."""

import re
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from .config import Settings
from .errors import SupportError
from .service import SupportService

COOKIE_NAME = "llmagent_support_session"
STATIC = Path(__file__).with_name("static")
Identifier = Annotated[str, Field(pattern=r"^[a-zA-Z0-9_-]{1,64}$")]


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    user_id: Identifier
    ticket_id: Identifier | None = None
    message: str = Field(min_length=1, max_length=4000)


def create_app(service: SupportService | None = None) -> FastAPI:
    support = service if service is not None else SupportService(Settings.from_env())
    app = FastAPI(title="Поддержка LLMAgent", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.support = support

    @app.middleware("http")
    async def browser_session(request: Request, call_next):
        session = request.cookies.get(COOKIE_NAME, "")
        is_new = re.fullmatch(r"[0-9a-f]{32}", session) is None
        request.state.session_id = uuid.uuid4().hex if is_new else session
        response = await call_next(request)
        if is_new:
            response.set_cookie(COOKIE_NAME, request.state.session_id, httponly=True, samesite="strict",
                                secure=request.url.scheme == "https", max_age=30 * 24 * 3600)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.exception_handler(SupportError)
    async def support_error(request: Request, error: SupportError):
        return JSONResponse({"detail": str(error)}, status_code=error.status_code)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, error: RequestValidationError):
        return JSONResponse({"detail": "Проверьте пользователя, обращение и текст вопроса (до 4000 символов)."}, status_code=422)

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, error: Exception):
        # Не возвращаем тело SDK-ошибки, содержимое окружения или traceback.
        return JSONResponse({"detail": "Не удалось обработать запрос. Повторите позже."}, status_code=500)

    @app.get("/")
    async def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/users")
    async def users():
        return {"users": await support.crm.list_users()}

    @app.get("/api/tickets")
    async def tickets(user_id: Annotated[Identifier, Query()]):
        return {"tickets": await support.crm.list_tickets(user_id)}

    @app.get("/api/history")
    async def history(request: Request, user_id: Annotated[Identifier, Query()],
                      ticket_id: Annotated[Identifier | None, Query()] = None):
        return {"turns": await support.history(request.state.session_id, user_id, ticket_id)}

    @app.delete("/api/history")
    async def clear(request: Request, user_id: Annotated[Identifier, Query()],
                    ticket_id: Annotated[Identifier | None, Query()] = None):
        await support.clear(request.state.session_id, user_id, ticket_id)
        return {"ok": True}

    @app.post("/api/chat")
    async def chat(request: Request, payload: ChatRequest):
        return await support.reply(request.state.session_id, payload.user_id, payload.ticket_id, payload.message)

    @app.get("/api/health")
    async def health():
        return {"status": "ok", "demo": True}

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app
