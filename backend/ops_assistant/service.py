"""Wiring: build the store, retriever, LLM and compiled graph from settings."""

from __future__ import annotations

import time
from typing import Any

from .agent.graph import AgentDeps, build_graph
from .config import Settings
from .embeddings import get_embedder
from .llm import get_chat_model
from .retrieval import Retriever
from .store import PgStore


class OpsAssistant:
    def __init__(self, settings: Settings, store: PgStore | None = None):
        self.settings = settings
        self.store = store or PgStore(settings.database_url, settings.embedding_dim)
        self.retriever = Retriever(self.store, get_embedder(settings),
                                   settings.retrieval_mode, settings.top_k)
        self.deps = AgentDeps(settings, self.store, self.retriever, get_chat_model(settings))
        self.graph = build_graph(self.deps)

    @property
    def mode(self) -> str:
        return self.settings.resolved_llm_provider()

    def ask(self, question: str) -> dict[str, Any]:
        """Run the whole graph once (no streaming). Used by evals and the /chat endpoint."""
        t0 = time.perf_counter()
        state = self.graph.invoke({"question": question})
        return result_payload(state, time.perf_counter() - t0)

    def close(self) -> None:
        self.store.close()


def result_payload(state: dict[str, Any], seconds: float) -> dict[str, Any]:
    return {
        "answer": state.get("answer", ""),
        "route": state.get("route"),
        "plan": state.get("plan"),
        "citations": state.get("citations", []),
        "tool_results": state.get("tool_results", []),
        "latency_ms": round(seconds * 1000, 1),
    }
