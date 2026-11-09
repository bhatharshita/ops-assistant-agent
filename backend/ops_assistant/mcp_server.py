"""MCP server exposing the agent's tools to any MCP client (Claude Desktop, IDEs, ...).

It wraps the same `ToolBox` the LangGraph analyst uses, so behaviour is identical.

    python -m ops_assistant.mcp_server            # stdio transport (default)

Example client config (e.g. claude_desktop_config.json):
    {"mcpServers": {"ops-assistant": {
        "command": "/path/to/backend/.venv/bin/python",
        "args": ["-m", "ops_assistant.mcp_server"],
        "env": {"DATABASE_URL": "postgresql://ops:ops@localhost:5432/ops"}}}}
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Literal

from mcp.server.mcpserver import MCPServer

from .config import get_settings
from .embeddings import get_embedder
from .retrieval import Retriever
from .store import PgStore
from .tools import ToolBox

mcp = MCPServer(
    "ops-assistant",
    instructions="Query a software project's ingested logs, GitHub issues and knowledge graph.",
)


@lru_cache
def toolbox() -> ToolBox:
    # Created lazily so importing this module (e.g. in tests) needs no database.
    s = get_settings()
    store = PgStore(s.database_url, s.embedding_dim)
    return ToolBox(store, Retriever(store, get_embedder(s), s.retrieval_mode, s.top_k))


@mcp.tool()
def log_stats(
    group_by: Literal["component", "level", "error_signature", "node"] = "component",
    component: str | None = None,
    level: Literal["INFO", "WARN", "ERROR"] | None = None,
    contains: str | None = None,
    error_signature: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    """Count log lines grouped by a column, with optional filters. Answers 'how many' and
    'which component/node has the most' questions with exact numbers."""
    return toolbox().log_stats(group_by, component, level, contains, limit, error_signature)


@mcp.tool()
def graph_neighbors(entity: str) -> dict[str, Any]:
    """Knowledge-graph neighbourhood of a component (kubelet), error (OOMKilled) or
    issue (issue:1006): linked nodes with relation types and weights, plus related issues."""
    return toolbox().graph_neighbors(entity)


@mcp.tool()
def search_knowledge_base(
    query: str, source_type: Literal["issue", "log", "any"] = "any"
) -> dict[str, Any]:
    """Hybrid semantic + keyword search over ingested GitHub issues and logs."""
    return toolbox().search_knowledge_base(query, source_type)


if __name__ == "__main__":
    mcp.run()
