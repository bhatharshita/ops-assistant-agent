import pytest

from ops_assistant.agent import offline
from ops_assistant.retrieval import reciprocal_rank_fusion
from ops_assistant.service import OpsAssistant
from ops_assistant.store import Hit

from .conftest import offline_settings


def _hit(cid: str) -> Hit:
    return Hit(chunk_id=cid, document_id=cid.split("#")[0], source_type="issue", title=cid,
               url=None, content="", score=0.0)


def test_rrf_rewards_documents_found_by_both_retrievers():
    dense = [_hit("a#0"), _hit("b#0"), _hit("c#0")]
    keyword = [_hit("c#0"), _hit("d#0")]
    fused = [h.chunk_id for h in reciprocal_rank_fusion([dense, keyword])]
    assert fused[0] == "c#0"  # rank 3 + rank 1 beats a single rank 1
    assert set(fused) == {"a#0", "b#0", "c#0", "d#0"}


@pytest.mark.parametrize("question,route", [
    ("Which node had the most OOMKilled containers?", "logs"),
    ("How was the etcd database space exceeded problem fixed?", "issues"),
    ("Which issues are related to etcd?", "graph"),
    ("What errors does the kubelet log most often and how do I fix them?", "mixed"),
    ("What is the capital of France?", "out_of_scope"),
])
def test_offline_router(question, route):
    assert offline.route_question(question)[0] == route


def test_offline_tool_planning_extracts_arguments():
    calls = offline.plan_tool_calls("Which node had the most OOMKilled containers?", "logs")
    assert calls == [{"name": "log_stats",
                      "args": {"group_by": "node", "error_signature": "OOMKilled"}}]
    calls = offline.plan_tool_calls("What other issues are linked to issue #1006?", "graph")
    assert calls == [{"name": "graph_neighbors", "args": {"entity": "issue:1006"}}]
    assert offline.plan_tool_calls("Tell me about #1015", "issues")[0]["name"] == "get_issue"


def test_graph_runs_logs_route_end_to_end(fake_store):
    assistant = OpsAssistant(offline_settings(), store=fake_store)
    result = assistant.ask("Which node had the most OOMKilled containers?")
    assert result["route"] == "logs"
    assert [t["name"] for t in result["tool_results"]] == ["log_stats"]
    assert "node-4" in result["answer"] and "24" in result["answer"]
    log_call = next(c for c in fake_store.calls if c[0] == "log_stats")
    assert log_call[1][0] == "node"  # group_by chosen by the analyst


def test_graph_cites_sources_for_issue_questions(fake_store):
    assistant = OpsAssistant(offline_settings(), store=fake_store)
    result = assistant.ask("How was the etcd database space exceeded problem fixed?")
    assert result["route"] == "issues"
    assert result["citations"][0]["document_id"] == "issue:1003"
    assert "defrag" in result["answer"] and "[1]" in result["answer"]


def test_out_of_scope_skips_retrieval(fake_store):
    assistant = OpsAssistant(offline_settings(), store=fake_store)
    result = assistant.ask("What is the capital of France?")
    assert result["route"] == "out_of_scope"
    assert fake_store.calls == []  # no retrieval, no tools


def test_tool_budget_is_enforced(fake_store):
    assistant = OpsAssistant(offline_settings(max_tool_calls=0), store=fake_store)
    result = assistant.ask("Which node had the most OOMKilled containers?")
    assert result["tool_results"] == []
