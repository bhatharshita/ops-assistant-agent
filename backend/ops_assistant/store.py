"""All PostgreSQL access lives here (pgvector search, full-text search, log SQL, graph).

Keeping SQL in one class means the agent, API, MCP server and evals share one
tested data layer, and tests can swap in a fake with the same methods.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from importlib import resources
from typing import Any

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool


@dataclass
class Document:
    id: str
    source_type: str  # 'issue' | 'log'
    title: str
    url: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str | None = None


@dataclass
class Chunk:
    id: str
    document_id: str
    source_type: str
    content: str


@dataclass
class LogLine:
    document_id: str
    ts: str | None
    level: str | None
    component: str | None
    node: str | None
    message: str
    error_signature: str | None = None


@dataclass
class Hit:
    """A retrieved chunk plus the metadata the UI needs to show it as a source."""

    chunk_id: str
    document_id: str
    source_type: str
    title: str
    url: str | None
    content: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Group-by columns the log_stats tool may use. Whitelisting columns (instead of
# letting the LLM write SQL) keeps the tool safe from injection by construction.
LOG_GROUP_COLUMNS = {"component", "level", "error_signature", "node"}


class PgStore:
    def __init__(self, dsn: str, dim: int):
        self.dsn = dsn
        self.dim = dim
        self._pool: ConnectionPool | None = None

    # ------------------------------------------------------------------ setup
    def init_schema(self, reset: bool = False) -> None:
        sql = resources.files("ops_assistant").joinpath("schema.sql").read_text()
        with psycopg.connect(self.dsn, autocommit=True) as conn:
            if reset:
                conn.execute(
                    "DROP TABLE IF EXISTS kg_edges, kg_nodes, log_lines, chunks, documents CASCADE"
                )
            conn.execute(sql.replace("{dim}", str(self.dim)))

    @property
    def pool(self) -> ConnectionPool:
        if self._pool is None:
            self._pool = ConnectionPool(
                self.dsn,
                min_size=1,
                max_size=5,
                # report timestamps in UTC regardless of the server's local timezone
                kwargs={"row_factory": dict_row, "options": "-c timezone=UTC"},
                configure=register_vector,
                open=True,
            )
        return self._pool

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool = None

    def ping(self) -> bool:
        with self.pool.connection() as conn:
            return conn.execute("SELECT 1 AS ok").fetchone()["ok"] == 1

    # ---------------------------------------------------------------- writes
    def upsert_documents(self, docs: Iterable[Document]) -> None:
        with self.pool.connection() as conn, conn.cursor() as cur:
            cur.executemany(
                """INSERT INTO documents (id, source_type, title, url, metadata, created_at)
                   VALUES (%s, %s, %s, %s, %s, %s)
                   ON CONFLICT (id) DO UPDATE SET title = EXCLUDED.title, url = EXCLUDED.url,
                       metadata = EXCLUDED.metadata, created_at = EXCLUDED.created_at""",
                [
                    (d.id, d.source_type, d.title, d.url, json.dumps(d.metadata), d.created_at)
                    for d in docs
                ],
            )

    def replace_chunks(self, chunks: Sequence[Chunk], embeddings: np.ndarray) -> None:
        """Delete old chunks of these documents and insert the new ones (idempotent re-ingest)."""
        doc_ids = sorted({c.document_id for c in chunks})
        with self.pool.connection() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM chunks WHERE document_id = ANY(%s)", (doc_ids,))
            cur.executemany(
                """INSERT INTO chunks (id, document_id, source_type, content, embedding)
                   VALUES (%s, %s, %s, %s, %s)""",
                [
                    (c.id, c.document_id, c.source_type, c.content, emb)
                    for c, emb in zip(chunks, embeddings, strict=True)
                ],
            )

    def replace_log_lines(self, lines: Sequence[LogLine]) -> None:
        doc_ids = sorted({line.document_id for line in lines})
        with self.pool.connection() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM log_lines WHERE document_id = ANY(%s)", (doc_ids,))
            cur.executemany(
                """INSERT INTO log_lines
                   (document_id, ts, level, component, node, message, error_signature)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                [
                    (x.document_id, x.ts, x.level, x.component, x.node, x.message,
                     x.error_signature)
                    for x in lines
                ],
            )

    def upsert_graph(
        self, nodes: dict[str, tuple[str, str]], edges: dict[tuple[str, str, str], int]
    ) -> None:
        """nodes: id -> (kind, label); edges: (src, dst, kind) -> weight."""
        with self.pool.connection() as conn, conn.cursor() as cur:
            cur.executemany(
                """INSERT INTO kg_nodes (id, kind, label) VALUES (%s, %s, %s)
                   ON CONFLICT (id) DO NOTHING""",
                [(nid, kind, label) for nid, (kind, label) in nodes.items()],
            )
            cur.executemany(
                """INSERT INTO kg_edges (src, dst, kind, weight) VALUES (%s, %s, %s, %s)
                   ON CONFLICT (src, dst, kind) DO UPDATE SET weight = EXCLUDED.weight""",
                [(s, d, k, w) for (s, d, k), w in edges.items()],
            )

    # ----------------------------------------------------------------- reads
    def search_dense(
        self, query_vec: np.ndarray, k: int, source_type: str | None = None
    ) -> list[Hit]:
        """Nearest neighbours by cosine distance (pgvector `<=>`, HNSW index)."""
        with self.pool.connection() as conn:
            rows = conn.execute(
                """SELECT c.id, c.document_id, c.source_type, d.title, d.url, c.content,
                          1 - (c.embedding <=> %(q)s) AS score
                   FROM chunks c JOIN documents d ON d.id = c.document_id
                   WHERE %(st)s::text IS NULL OR c.source_type = %(st)s
                   ORDER BY c.embedding <=> %(q)s
                   LIMIT %(k)s""",
                {"q": np.asarray(query_vec, dtype=np.float32), "st": source_type, "k": k},
            ).fetchall()
        return [_hit(r) for r in rows]

    def search_keyword(self, query: str, k: int, source_type: str | None = None) -> list[Hit]:
        """Postgres full-text search. Terms are OR-ed so partial matches still rank."""
        with self.pool.connection() as conn:
            rows = conn.execute(
                """WITH q AS (
                       -- stem the question with the same config as the index, then OR
                       -- the quoted lexemes together ("pod | crash | oomkil")
                       SELECT string_agg(quote_literal(lexeme), ' | ')::tsquery AS tsq
                       FROM unnest(to_tsvector('english', %(text)s))
                   )
                   SELECT c.id, c.document_id, c.source_type, d.title, d.url, c.content,
                          ts_rank_cd(c.tsv, q.tsq) AS score
                   FROM chunks c JOIN documents d ON d.id = c.document_id, q
                   WHERE c.tsv @@ q.tsq
                     AND (%(st)s::text IS NULL OR c.source_type = %(st)s)
                   ORDER BY score DESC
                   LIMIT %(k)s""",
                {"text": query, "st": source_type, "k": k},
            ).fetchall()
        return [_hit(r) for r in rows]

    def log_stats(
        self,
        group_by: str = "component",
        component: str | None = None,
        level: str | None = None,
        contains: str | None = None,
        limit: int = 10,
        error_signature: str | None = None,
    ) -> list[dict[str, Any]]:
        if group_by not in LOG_GROUP_COLUMNS:
            raise ValueError(f"group_by must be one of {sorted(LOG_GROUP_COLUMNS)}")
        # group_by is whitelisted above, so formatting it into SQL is safe;
        # every user-controlled value goes through a bound parameter.
        sql = f"""
            SELECT {group_by} AS key, count(*) AS count,
                   min(ts) AS first_seen, max(ts) AS last_seen,
                   (array_agg(message ORDER BY ts DESC))[1] AS example
            FROM log_lines
            WHERE (%(component)s::text IS NULL OR component = %(component)s)
              AND (%(level)s::text IS NULL OR level = upper(%(level)s))
              AND (%(contains)s::text IS NULL OR message ILIKE '%%' || %(contains)s || '%%')
              AND (%(sig)s::text IS NULL OR error_signature = %(sig)s)
            GROUP BY {group_by}
            ORDER BY count DESC
            LIMIT %(limit)s"""
        params = {"component": component, "level": level, "contains": contains,
                  "sig": error_signature, "limit": min(limit, 50)}
        with self.pool.connection() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [
            {**r, "first_seen": _iso(r["first_seen"]), "last_seen": _iso(r["last_seen"])}
            for r in rows
        ]

    def kg_neighbors(self, entity: str, limit: int = 15) -> dict[str, Any]:
        """Find the node best matching `entity` and return its 1-hop neighbourhood,
        plus 2-hop issues reached through it (e.g. error -> component -> other issues)."""
        with self.pool.connection() as conn:
            node = conn.execute(
                """SELECT id, kind, label FROM kg_nodes
                   WHERE lower(label) = lower(%(e)s) OR lower(id) = lower(%(e)s)
                      OR label ILIKE '%%' || %(e)s || '%%'
                   ORDER BY (lower(label) = lower(%(e)s)) DESC, length(label)
                   LIMIT 1""",
                {"e": entity.strip()},
            ).fetchone()
            if node is None:
                return {"entity": entity, "found": False, "neighbors": [], "related_issues": []}
            neighbors = conn.execute(
                """SELECT n.id, n.kind, n.label, e.kind AS relation, e.weight
                   FROM kg_edges e JOIN kg_nodes n
                     ON n.id = CASE WHEN e.src = %(id)s THEN e.dst ELSE e.src END
                   WHERE e.src = %(id)s OR e.dst = %(id)s
                   ORDER BY e.weight DESC, n.id LIMIT %(limit)s""",
                {"id": node["id"], "limit": limit},
            ).fetchall()
            related = conn.execute(
                """SELECT DISTINCT i.id, i.label
                   FROM kg_edges e1
                   JOIN kg_edges e2 ON e2.dst = e1.dst AND e2.src <> %(id)s
                   JOIN kg_nodes i ON i.id = e2.src AND i.kind = 'issue'
                   WHERE e1.src = %(id)s AND %(kind)s = 'issue'
                   UNION
                   SELECT i.id, i.label FROM kg_edges e
                   JOIN kg_nodes i ON i.id = e.src AND i.kind = 'issue'
                   WHERE e.dst = %(id)s
                   ORDER BY 1 LIMIT %(limit)s""",
                {"id": node["id"], "kind": node["kind"], "limit": limit},
            ).fetchall()
        return {"entity": entity, "found": True, "node": node, "neighbors": neighbors,
                "related_issues": related}

    def get_document(self, doc_id: str) -> dict[str, Any] | None:
        with self.pool.connection() as conn:
            doc = conn.execute(
                "SELECT id, source_type, title, url, metadata FROM documents WHERE id = %s",
                (doc_id,),
            ).fetchone()
            if doc is None:
                return None
            chunks = conn.execute(
                "SELECT content FROM chunks WHERE document_id = %s ORDER BY id", (doc_id,)
            ).fetchall()
        return {**doc, "content": "\n".join(c["content"] for c in chunks)}

    def stats(self) -> dict[str, int]:
        with self.pool.connection() as conn:
            row = conn.execute(
                """SELECT (SELECT count(*) FROM documents) AS documents,
                          (SELECT count(*) FROM chunks) AS chunks,
                          (SELECT count(*) FROM log_lines) AS log_lines,
                          (SELECT count(*) FROM kg_nodes) AS kg_nodes,
                          (SELECT count(*) FROM kg_edges) AS kg_edges"""
            ).fetchone()
        return dict(row)


def _hit(r: dict[str, Any]) -> Hit:
    return Hit(
        chunk_id=r["id"], document_id=r["document_id"], source_type=r["source_type"],
        title=r["title"], url=r["url"], content=r["content"], score=float(r["score"]),
    )


def _iso(value: Any) -> str | None:
    return value.isoformat() if value is not None else None
