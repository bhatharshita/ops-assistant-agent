import json

import pytest
from fastapi.testclient import TestClient

from ops_assistant.api.main import create_app
from ops_assistant.service import OpsAssistant

from .conftest import offline_settings


@pytest.fixture
def client(fake_store):
    app = create_app(lambda: OpsAssistant(offline_settings(), store=fake_store))
    with TestClient(app) as c:
        yield c


def parse_sse(body: str) -> list[dict]:
    events = []
    for block in body.strip().split("\n\n"):
        data = next(line[6:] for line in block.splitlines() if line.startswith("data: "))
        events.append(json.loads(data))
    return events


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["llm_provider"] == "offline"
    assert body["index"]["documents"] == 1


def test_chat_returns_answer_and_citations(client):
    body = client.post("/chat", json={"question": "How was the etcd issue fixed?"}).json()
    assert body["route"] == "issues"
    assert body["citations"][0]["document_id"] == "issue:1003"
    assert body["latency_ms"] >= 0


def test_chat_validates_input(client):
    assert client.post("/chat", json={"question": ""}).status_code == 422


def test_stream_emits_progress_tokens_and_done(client):
    resp = client.post("/chat/stream",
                       json={"question": "Which node had the most OOMKilled containers?"})
    assert resp.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(resp.text)
    types = [e["type"] for e in events]
    assert types[0] == "route"
    assert "sources" in types and "tool_call" in types and "tool_result" in types
    assert types[-1] == "done"
    streamed = "".join(e["text"] for e in events if e["type"] == "token")
    assert streamed.strip() == events[-1]["answer"].strip()
