"""Exercise the LLM code path (structured routing, tool calling, streaming writer)
with a scripted fake chat model, so it is tested without an API key."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from ops_assistant.agent.graph import AgentDeps, build_graph
from ops_assistant.embeddings import HashingEmbedder
from ops_assistant.retrieval import Retriever

from .conftest import offline_settings


class ScriptedChatModel(BaseChatModel):
    """Returns the next scripted AIMessage on every call and records the prompts."""

    script: list[AIMessage]
    prompts: list[list[BaseMessage]] = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedChatModel:
        return self  # tool schemas are irrelevant for a scripted model

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.prompts.append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=self.script.pop(0))])


def tool_call(name: str, args: dict, call_id: str) -> dict:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def test_llm_router_analyst_tools_and_writer(fake_store):
    llm = ScriptedChatModel(script=[
        # 1. router: structured output arrives as a tool call named after the schema
        AIMessage(content="", tool_calls=[tool_call(
            "RouteDecision", {"route": "logs", "plan": "count OOMKilled by node"}, "r1")]),
        # 2. analyst asks for a tool
        AIMessage(content="", tool_calls=[tool_call(
            "log_stats", {"group_by": "node", "error_signature": "OOMKilled"}, "c1")]),
        # 3. analyst is satisfied
        AIMessage(content="done"),
        # 4. writer
        AIMessage(content="node-4 had the most OOMKilled containers (24 lines) [1]."),
    ], prompts=[])
    settings = offline_settings()
    deps = AgentDeps(settings, fake_store, Retriever(fake_store, HashingEmbedder(256)), llm)
    graph = build_graph(deps)

    events: list[dict] = []
    final: dict = {}
    for mode, chunk in graph.stream({"question": "Which node had the most OOMKilled?"},
                                    stream_mode=["custom", "values"]):
        if mode == "custom":
            events.append(chunk)
        else:
            final = chunk

    assert final["route"] == "logs"
    assert final["answer"].startswith("node-4 had the most")
    assert [t["name"] for t in final["tool_results"]] == ["log_stats"]
    # the analyst's second call saw the tool result as a ToolMessage
    assert isinstance(llm.prompts[2][-1], ToolMessage)
    # the writer prompt contained the numbered source and the tool output
    writer_system = llm.prompts[3][0].content
    assert "[1]" in writer_system and "node-4" in writer_system
    assert [e["type"] for e in events][:2] == ["route", "sources"]
    assert "".join(e["text"] for e in events if e["type"] == "token") == final["answer"]


def test_router_falls_back_to_rules_when_llm_output_is_invalid(fake_store):
    llm = ScriptedChatModel(script=[
        AIMessage(content="not a tool call"),  # router: unparseable -> rule-based route
        AIMessage(content="done"),  # analyst
        AIMessage(content="answer"),  # writer
    ], prompts=[])
    deps = AgentDeps(offline_settings(), fake_store,
                     Retriever(fake_store, HashingEmbedder(256)), llm)
    state = build_graph(deps).invoke({"question": "How was the etcd issue fixed?"})
    assert state["route"] == "issues"
    assert state["answer"] == "answer"
