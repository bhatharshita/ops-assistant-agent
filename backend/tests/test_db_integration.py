"""Integration tests against a real PostgreSQL + pgvector (set TEST_DATABASE_URL)."""

import pytest

from ops_assistant.embeddings import HashingEmbedder
from ops_assistant.retrieval import Retriever
from ops_assistant.service import OpsAssistant

from .conftest import offline_settings

pytestmark = pytest.mark.db


def test_ingest_populates_every_table(db_store):
    stats = db_store.stats()
    assert stats["documents"] == 31  # 24 issues + 7 log files
    assert stats["log_lines"] == 448
    assert stats["chunks"] > 31 and stats["kg_nodes"] > 0 and stats["kg_edges"] > 0


def test_log_stats_counts_and_filters(db_store):
    rows = db_store.log_stats("node", error_signature="OOMKilled")
    assert (rows[0]["key"], rows[0]["count"]) == ("node-4", 24)
    by_component = db_store.log_stats("component", level="error")  # level is case-insensitive
    assert by_component[0]["key"] == "kubelet"
    with pytest.raises(ValueError):
        db_store.log_stats("message; DROP TABLE log_lines")  # not a whitelisted column


def test_dense_and_keyword_search(db_store):
    emb = HashingEmbedder(db_store.dim)
    retriever = Retriever(db_store, emb, mode="hybrid", top_k=5)
    hits = retriever.search("etcd mvcc database space exceeded defrag", source_type="issue")
    assert hits[0].document_id == "issue:1003"
    assert all(h.source_type == "issue" for h in hits)
    assert db_store.search_keyword("SystemdCgroup", 3)[0].document_id == "issue:1001"
    assert db_store.search_keyword("!!! ''' ::", 3) == []  # punctuation-only query is safe


def test_knowledge_graph_neighbors(db_store):
    result = db_store.kg_neighbors("etcd")
    assert result["found"]
    linked = {n["id"] for n in result["neighbors"]}
    assert {"issue:1003", "issue:1017", "error:mvcc: database space exceeded"} <= linked
    assert not db_store.kg_neighbors("no-such-thing")["found"]


def test_agent_answers_from_real_database(db_store):
    assistant = OpsAssistant(offline_settings(database_url=db_store.dsn), store=db_store)
    result = assistant.ask("Which node had the most OOMKilled containers?")
    assert "node-4" in result["answer"]
    result = assistant.ask("Which issues are related to etcd?")
    assert "#1003" in result["answer"]


async def test_mcp_server_tool_call(db_store, monkeypatch):
    from ops_assistant import mcp_server
    from ops_assistant.tools import ToolBox

    toolbox = ToolBox(db_store, Retriever(db_store, HashingEmbedder(db_store.dim)))
    monkeypatch.setattr(mcp_server, "toolbox", lambda: toolbox)
    names = {t.name for t in await mcp_server.mcp.list_tools()}
    assert names == {"log_stats", "graph_neighbors", "search_knowledge_base"}
    result = await mcp_server.mcp.call_tool("log_stats", {"group_by": "node",
                                                          "error_signature": "OOMKilled"})
    assert "node-4" in result.content[0].text
