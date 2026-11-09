"""The LangGraph multi-agent flow.

    START -> router --(out_of_scope)--------------------------> writer -> END
                    \\-> retriever -> analyst <-> tools (loop, capped) -> writer

- router     classifies the question (logs / issues / graph / mixed / out_of_scope)
- retriever  hybrid RAG over pgvector + full-text, filtered by route
- analyst    tool-calling agent; decides which tools to call with which arguments
- tools      executes the tool calls and records results + extra sources
- writer     streams the final grounded answer with [n] citations

Every node also emits progress events through LangGraph's custom stream
(`get_stream_writer`), which the API forwards to the UI as server-sent events.
With no API key the LLM-backed steps fall back to `offline.py`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from ..config import Settings
from ..retrieval import Retriever
from ..store import PgStore
from ..tools import ToolBox, to_json
from . import offline
from .prompts import ANALYST_SYSTEM, OUT_OF_SCOPE_ANSWER, ROUTER_SYSTEM, WRITER_SYSTEM
from .state import AgentState, Route

log = logging.getLogger(__name__)
SOURCE_FILTER = {"logs": "log", "issues": "issue"}  # graph/mixed search everything


@dataclass
class AgentDeps:
    settings: Settings
    store: PgStore
    retriever: Retriever
    llm: BaseChatModel | None  # None = offline mode


class RouteDecision(BaseModel):
    route: Route = Field(description="Which route to take")
    plan: str = Field(description="One sentence on how to answer")


def _emit(event: dict[str, Any]) -> None:
    get_stream_writer()(event)


def _text(content: Any) -> str:
    """Chat model content can be a string or a list of content blocks (Anthropic)."""
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in content if isinstance(b, dict))


def build_graph(deps: AgentDeps):
    toolbox = ToolBox(deps.store, deps.retriever)
    tools = {t.name: t for t in toolbox.as_langchain_tools()}
    llm = deps.llm
    max_calls = deps.settings.max_tool_calls

    # ----------------------------------------------------------------- nodes
    def router(state: AgentState) -> dict[str, Any]:
        route, plan = offline.route_question(state["question"])
        if llm is not None:
            try:
                decision = llm.with_structured_output(RouteDecision).invoke(
                    [SystemMessage(ROUTER_SYSTEM), HumanMessage(state["question"])])
                route, plan = decision.route, decision.plan
            except Exception:  # bad JSON, provider hiccup: keep the rule-based route
                log.exception("LLM router failed; using rule-based route")
        _emit({"type": "route", "route": route, "plan": plan})
        return {"route": route, "plan": plan, "tool_calls_made": 0}

    def retriever(state: AgentState) -> dict[str, Any]:
        hits = deps.retriever.search(state["question"],
                                     source_type=SOURCE_FILTER.get(state["route"]))
        sources = [h.to_dict() for h in hits]
        _emit({"type": "sources", "sources": sources})
        return {"sources": sources}

    def analyst(state: AgentState) -> dict[str, Any]:
        if llm is None:
            msg = offline.analyst_message(state["question"], state["route"],
                                          already_called=state.get("tool_calls_made", 0) > 0)
            return {"messages": [msg]}
        new: list[Any] = []
        history = list(state.get("messages", []))
        if not history:
            context = "\n\n".join(f"- {s['title']}: {s['content'][:400]}"
                                  for s in state.get("sources", []))
            new = [SystemMessage(ANALYST_SYSTEM.format(route=state["route"], plan=state["plan"],
                                                       max_calls=max_calls, context=context)),
                   HumanMessage(state["question"])]
        ai = llm.bind_tools(list(tools.values())).invoke(history + new)
        return {"messages": [*new, ai]}

    def run_tools(state: AgentState) -> dict[str, Any]:
        last = state["messages"][-1]
        messages, results, sources = [], [], []
        for call in last.tool_calls:
            _emit({"type": "tool_call", "name": call["name"], "args": call["args"]})
            try:
                result = tools[call["name"]].func(**call["args"])
            except Exception as exc:  # report tool errors to the model instead of crashing
                result = {"error": f"{type(exc).__name__}: {exc}"}
            if call["name"] == "search_knowledge_base":
                sources += result.get("hits", [])
            results.append({"name": call["name"], "args": call["args"], "result": result})
            messages.append(ToolMessage(content=to_json(result), tool_call_id=call["id"],
                                        name=call["name"]))
            _emit({"type": "tool_result", "name": call["name"], "result": result})
        if sources:
            _emit({"type": "sources", "sources": sources})
        return {"messages": messages, "tool_results": results, "sources": sources,
                "tool_calls_made": state.get("tool_calls_made", 0) + len(last.tool_calls)}

    def writer(state: AgentState) -> dict[str, Any]:
        emit = get_stream_writer()
        if state["route"] == "out_of_scope":
            emit({"type": "token", "text": OUT_OF_SCOPE_ANSWER})
            return {"answer": OUT_OF_SCOPE_ANSWER, "citations": []}

        seen: set[str] = set()
        citations = []
        for s in state.get("sources", []):
            if s["chunk_id"] not in seen and len(citations) < 8:
                seen.add(s["chunk_id"])
                citations.append(s)
        tool_results = state.get("tool_results", [])

        if llm is None:
            answer = offline.compose_answer(state["question"], citations, tool_results)
            for word in answer.split(" "):  # stream word by word like an LLM would
                emit({"type": "token", "text": word + " "})
            return {"answer": answer, "citations": citations}

        sources_txt = "\n\n".join(
            f"[{n}] ({s['source_type']}) {s['title']}\n{s['content'][:800]}"
            for n, s in enumerate(citations, 1)) or "(none)"
        tools_txt = "\n".join(f"{t['name']}({t['args']}) -> {to_json(t['result'], 1500)}"
                              for t in tool_results) or "(none)"
        prompt = [SystemMessage(WRITER_SYSTEM.format(sources=sources_txt,
                                                     tool_results=tools_txt)),
                  HumanMessage(state["question"])]
        answer = ""
        for chunk in llm.stream(prompt):
            text = _text(chunk.content)
            if text:
                answer += text
                emit({"type": "token", "text": text})
        return {"answer": answer, "citations": citations}

    # ----------------------------------------------------------------- edges
    def after_router(state: AgentState) -> str:
        return "writer" if state["route"] == "out_of_scope" else "retriever"

    def after_analyst(state: AgentState) -> str:
        last = state["messages"][-1]
        wants_tools = isinstance(last, AIMessage) and bool(last.tool_calls)
        return "tools" if wants_tools and state.get("tool_calls_made", 0) < max_calls \
            else "writer"

    g = StateGraph(AgentState)
    g.add_node("router", router)
    g.add_node("retriever", retriever)
    g.add_node("analyst", analyst)
    g.add_node("tools", run_tools)
    g.add_node("writer", writer)
    g.add_edge(START, "router")
    g.add_conditional_edges("router", after_router, ["retriever", "writer"])
    g.add_edge("retriever", "analyst")
    g.add_conditional_edges("analyst", after_analyst, ["tools", "writer"])
    g.add_edge("tools", "analyst")
    g.add_edge("writer", END)
    return g.compile()
