"""Shared fixtures.

Unit tests use `FakeStore`, an in-memory stand-in with the same read methods as
PgStore, so the agent graph and API can be tested without a database.

Integration tests (marked `db`) need a disposable PostgreSQL + pgvector database:
    TEST_DATABASE_URL=postgresql://ops:ops@localhost:5432/ops_test pytest
They are skipped when TEST_DATABASE_URL is not set. They DROP and recreate the tables.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from ops_assistant.config import Settings
from ops_assistant.embeddings import HashingEmbedder
from ops_assistant.store import Hit

SAMPLE_DIR = Path(__file__).resolve().parents[2] / "data" / "sample"


def offline_settings(**overrides: Any) -> Settings:
    base = dict(llm_provider="offline", embedding_provider="hashing", embedding_dim=256,
                database_url=os.environ.get("TEST_DATABASE_URL", "postgresql://unused"),
                data_dir=str(SAMPLE_DIR))
    return Settings(_env_file=None, **{**base, **overrides})


class FakeStore:
    """Returns canned results; records calls so tests can assert on tool arguments."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []
        self.hit = Hit(chunk_id="issue:1003#0", document_id="issue:1003", source_type="issue",
                       title="#1003 etcd 'mvcc: database space exceeded'", url=None,
                       content="Issue #1003\n\nComment 1: Resolution: run etcdctl defrag.",
                       score=0.9)

    def search_dense(self, *args, **kwargs):
        self.calls.append(("search_dense", args, kwargs))
        return [self.hit]

    def search_keyword(self, *args, **kwargs):
        self.calls.append(("search_keyword", args, kwargs))
        return [self.hit]

    def log_stats(self, *args, **kwargs):
        self.calls.append(("log_stats", args, kwargs))
        return [{"key": "node-4", "count": 24, "first_seen": "t0", "last_seen": "t1",
                 "example": "OOMKilled"}]

    def kg_neighbors(self, entity):
        self.calls.append(("kg_neighbors", (entity,), {}))
        return {"entity": entity, "found": True,
                "node": {"id": "component:etcd", "kind": "component", "label": "etcd"},
                "neighbors": [{"id": "issue:1003", "kind": "issue", "label": "#1003",
                               "relation": "AFFECTS", "weight": 1}],
                "related_issues": [{"id": "issue:1003", "label": "#1003"}]}

    def get_document(self, doc_id):
        return {"id": doc_id, "title": "#1015 CoreDNS loop",
                "content": "Comment 1: Resolution: fix resolv.conf"}

    def stats(self):
        return {"documents": 1, "chunks": 1, "log_lines": 0, "kg_nodes": 0, "kg_edges": 0}

    def close(self):
        pass


@pytest.fixture
def fake_store() -> FakeStore:
    return FakeStore()


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder(256)


@pytest.fixture(scope="session")
def db_store():
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL not set")
    from ops_assistant.ingest.pipeline import ingest
    from ops_assistant.store import PgStore

    settings = offline_settings(database_url=url)
    store = PgStore(url, settings.embedding_dim)
    ingest(store, HashingEmbedder(settings.embedding_dim), settings,
           SAMPLE_DIR / "issues.json", SAMPLE_DIR / "logs", reset=True)
    yield store
    store.close()
