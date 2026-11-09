from ops_assistant.embeddings import HashingEmbedder
from ops_assistant.ingest.chunking import chunk_issue, chunk_logs, split_text
from ops_assistant.ingest.graph import build_graph
from ops_assistant.ingest.loaders import load_issues, load_log_file, parse_log_line
from ops_assistant.ingest.signatures import (
    canonical_component,
    find_components,
    find_error_signatures,
)

from .conftest import SAMPLE_DIR


def test_split_text_respects_size_and_overlaps():
    text = "\n\n".join(f"paragraph {i} " + "word " * 40 for i in range(10))
    chunks = split_text(text, size=300, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 300 + 60 for c in chunks)  # overlap tail may add a little
    assert "paragraph 9" in chunks[-1]


def test_chunk_issue_prefixes_every_chunk_with_header():
    text = "Issue #7: Broken thing\nState: open | Labels: none\n\n" + "\n\n".join(
        "detail " * 60 for _ in range(4))
    chunks = chunk_issue("issue:7", text, size=400, overlap=40)
    assert len(chunks) > 1
    assert all(c.content.startswith("Issue #7: Broken thing") for c in chunks)
    assert [c.id for c in chunks][:2] == ["issue:7#0", "issue:7#1"]


def test_parse_structured_and_unstructured_log_lines():
    line = parse_log_line("2026-10-14T03:00:00.000Z ERROR apiserver node-1 context deadline "
                          "exceeded", "log:x", "x")
    assert line.component == "kube-apiserver"  # alias mapped to canonical name
    assert line.level == "ERROR"
    assert line.error_signature == "context deadline exceeded"

    loghub = parse_log_line("nova-api 2017-05-16 00:00:00 2931 WARNING nova.compute oops",
                            "log:nova", "nova")
    assert loghub.level == "WARN" and loghub.component == "nova"
    assert parse_log_line("   ", "log:x", "x") is None


def test_signature_and_component_extraction():
    text = "kubelet: pod OOMKilled, then CrashLoopBackOff; kube-apiserver was fine"
    assert find_error_signatures(text) == ["OOMKilled", "CrashLoopBackOff"]
    assert set(find_components(text)) == {"kubelet", "kube-apiserver"}
    # 'apiserver' inside 'kube-apiserver' must not count twice / as another alias
    assert find_components("kube-scheduler") == ["kube-scheduler"]
    assert canonical_component("DNS") == "coredns"
    assert canonical_component("nonsense") is None


def test_sample_data_loads_and_builds_graph():
    issues = load_issues(SAMPLE_DIR / "issues.json")
    assert len(issues) == 24
    doc, lines = load_log_file(SAMPLE_DIR / "logs" / "kubelet.log")
    assert doc.id == "log:kubelet.log" and len(lines) > 100

    nodes, edges = build_graph(issues, lines)
    assert nodes["component:kubelet"] == ("component", "kubelet")
    assert ("issue:1003", "component:etcd", "AFFECTS") in edges
    assert ("issue:1003", "error:mvcc: database space exceeded", "MENTIONS") in edges
    assert edges[("error:OOMKilled", "component:kubelet", "SEEN_IN")] == 38


def test_log_digest_chunks_summarise_counts():
    doc, lines = load_log_file(SAMPLE_DIR / "logs" / "kubelet.log")
    chunks = chunk_logs(doc.id, lines)
    digest = next(c for c in chunks if "OOMKilled" in c.content and "#digest" in c.id)
    assert "38 ERROR lines" in digest.content
    assert "node-4" in digest.content


def test_hashing_embedder_is_deterministic_and_normalised():
    e = HashingEmbedder(128)
    a, b = e.embed(["etcd database space exceeded", "etcd database space exceeded"])
    assert (a == b).all()
    assert abs(float((a * a).sum()) - 1.0) < 1e-5
    near, far = e.embed(["etcd database space", "ingress 502 gateway"])
    assert float(a @ near) > float(a @ far)
