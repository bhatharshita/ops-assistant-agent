"""FastAPI service.

    GET  /health        liveness + DB status + which providers are active
    POST /chat          run the agent, return the full answer as JSON
    POST /chat/stream   run the agent, stream progress + tokens as server-sent events

Run locally:  uvicorn ops_assistant.api.main:app --reload
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..config import get_settings
from ..service import OpsAssistant, result_payload

log = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)


def sse(event: dict[str, Any]) -> str:
    """Format one server-sent event. The event type doubles as the SSE `event:` name."""
    return f"event: {event['type']}\ndata: {json.dumps(event, default=str)}\n\n"


def create_app(assistant_factory: Callable[[], OpsAssistant] | None = None) -> FastAPI:
    factory = assistant_factory or (lambda: OpsAssistant(get_settings()))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.assistant = factory()
        yield
        app.state.assistant.close()

    app = FastAPI(title="Ops Assistant", version="0.1.0", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"],
                       allow_methods=["*"], allow_headers=["*"])

    @app.get("/health")
    def health(request: Request) -> dict[str, Any]:
        assistant: OpsAssistant = request.app.state.assistant
        try:
            stats, db_ok = assistant.store.stats(), True
        except Exception as exc:  # keep /health answering even when the DB is down
            log.warning("health check DB error: %s", exc)
            stats, db_ok = {}, False
        return {
            "status": "ok" if db_ok else "degraded",
            "database": db_ok,
            "llm_provider": assistant.mode,
            "embedding_provider": assistant.settings.resolved_embedding_provider(),
            "retrieval_mode": assistant.settings.retrieval_mode,
            "index": stats,
        }

    @app.post("/chat")
    def chat(body: ChatRequest, request: Request) -> dict[str, Any]:
        return request.app.state.assistant.ask(body.question)

    @app.post("/chat/stream")
    async def chat_stream(body: ChatRequest, request: Request) -> StreamingResponse:
        assistant: OpsAssistant = request.app.state.assistant

        async def events() -> AsyncIterator[str]:
            t0 = time.perf_counter()
            final: dict[str, Any] = {}
            try:
                async for mode, chunk in assistant.graph.astream(
                        {"question": body.question}, stream_mode=["custom", "values"]):
                    if mode == "custom":
                        yield sse(chunk)
                    else:
                        final = chunk  # full state after each step; the last one wins
                yield sse({"type": "done", **result_payload(final, time.perf_counter() - t0)})
            except Exception as exc:
                log.exception("agent run failed")
                yield sse({"type": "error", "message": str(exc)})

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache",
                                          "X-Accel-Buffering": "no"})

    return app


app = create_app()
