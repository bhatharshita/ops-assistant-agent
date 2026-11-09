"""Shared state passed between LangGraph nodes.

Fields annotated with a reducer (operator.add / add_messages) are *appended to*
when a node returns them; all other fields are overwritten.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

Route = Literal["logs", "issues", "graph", "mixed", "out_of_scope"]


class AgentState(TypedDict, total=False):
    question: str
    route: Route
    plan: str
    sources: Annotated[list[dict[str, Any]], operator.add]  # retrieved chunks (citations)
    messages: Annotated[list[AnyMessage], add_messages]  # analyst <-> tools conversation
    tool_results: Annotated[list[dict[str, Any]], operator.add]
    tool_calls_made: int
    citations: list[dict[str, Any]]  # de-duplicated, numbered sources used by the writer
    answer: str
