"""Tools the analyst agent can call. Each is a plain function plus a LangChain wrapper.

The plain functions (`ToolBox` methods) are reused by the MCP server, so the
agent and external MCP clients run exactly the same code.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from .retrieval import Retriever
from .store import PgStore


class SearchArgs(BaseModel):
    query: str = Field(description="What to look for, in natural language or exact error text")
    source_type: Literal["issue", "log", "any"] = Field(
        "any", description="Restrict to GitHub issues, logs, or search both")


class LogStatsArgs(BaseModel):
    group_by: Literal["component", "level", "error_signature", "node"] = Field(
        "component", description="Column to count by")
    component: str | None = Field(None, description="Only this component, e.g. 'kubelet'")
    level: Literal["INFO", "WARN", "ERROR"] | None = Field(None, description="Only this level")
    contains: str | None = Field(None, description="Only lines whose message contains this text")
    error_signature: str | None = Field(
        None, description="Only lines with this normalised error, e.g. 'OOMKilled', "
                          "'CrashLoopBackOff', 'context deadline exceeded'")
    limit: int = Field(10, ge=1, le=50)


class GraphArgs(BaseModel):
    entity: str = Field(description="A component (kubelet), error (OOMKilled) or issue id "
                                    "(issue:1006)")


class IssueArgs(BaseModel):
    number: int = Field(description="GitHub issue number")


class ToolBox:
    def __init__(self, store: PgStore, retriever: Retriever):
        self.store = store
        self.retriever = retriever

    def search_knowledge_base(self, query: str, source_type: str = "any") -> dict[str, Any]:
        hits = self.retriever.search(query, source_type=None if source_type == "any"
                                     else source_type)
        return {"hits": [h.to_dict() for h in hits]}

    def log_stats(self, group_by: str = "component", component: str | None = None,
                  level: str | None = None, contains: str | None = None,
                  limit: int = 10, error_signature: str | None = None) -> dict[str, Any]:
        rows = self.store.log_stats(group_by, component, level, contains, limit,
                                    error_signature)
        return {"group_by": group_by,
                "filters": {"component": component, "level": level, "contains": contains,
                            "error_signature": error_signature},
                "rows": rows}

    def graph_neighbors(self, entity: str) -> dict[str, Any]:
        return self.store.kg_neighbors(entity)

    def get_issue(self, number: int) -> dict[str, Any]:
        doc = self.store.get_document(f"issue:{number}")
        return doc or {"error": f"issue #{number} not found"}

    def as_langchain_tools(self) -> list[BaseTool]:
        return [
            StructuredTool.from_function(
                self.search_knowledge_base, name="search_knowledge_base", args_schema=SearchArgs,
                description="Semantic + keyword search over ingested GitHub issues and logs. "
                            "Use for 'why', 'how to fix' and 'has anyone seen' questions."),
            StructuredTool.from_function(
                self.log_stats, name="log_stats", args_schema=LogStatsArgs,
                description="Exact counts over structured log lines (SQL GROUP BY). Use for "
                            "'how many', 'which component/node has the most', 'when' questions."),
            StructuredTool.from_function(
                self.graph_neighbors, name="graph_neighbors", args_schema=GraphArgs,
                description="Knowledge-graph lookup: what issues, components and errors are "
                            "linked to an entity. Use for 'related', 'which components' "
                            "and 'what else' questions."),
            StructuredTool.from_function(
                self.get_issue, name="get_issue", args_schema=IssueArgs,
                description="Fetch the full text of one GitHub issue by number."),
        ]


def to_json(result: Any, max_chars: int = 4000) -> str:
    """Serialise a tool result for the LLM, truncated so one tool can't flood the context."""
    text = json.dumps(result, default=str)
    return text if len(text) <= max_chars else text[:max_chars] + "...(truncated)"
