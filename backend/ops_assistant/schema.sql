-- Schema for the ops assistant. Applied by `ops_assistant.db.init_schema()`.
-- {dim} is replaced with EMBEDDING_DIM at runtime.

CREATE EXTENSION IF NOT EXISTS vector;

-- Source documents: one row per GitHub issue or per log file.
CREATE TABLE IF NOT EXISTS documents (
    id          TEXT PRIMARY KEY,          -- e.g. "issue:118734" or "log:kubelet.log"
    source_type TEXT NOT NULL,             -- 'issue' | 'log'
    title       TEXT NOT NULL,
    url         TEXT,
    metadata    JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at  TIMESTAMPTZ
);

-- Retrieval units. Each chunk has a dense vector and a full-text vector, so the
-- same table serves both pgvector similarity search and keyword (BM25-like) search.
CREATE TABLE IF NOT EXISTS chunks (
    id          TEXT PRIMARY KEY,          -- "<document id>#<n>"
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    source_type TEXT NOT NULL,
    content     TEXT NOT NULL,
    embedding   vector({dim}) NOT NULL,
    tsv         tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
);
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_tsv_gin ON chunks USING gin (tsv);

-- Structured log lines, used by the `log_stats` tool for exact counts
-- (vector search is bad at "how many"; SQL is good at it).
CREATE TABLE IF NOT EXISTS log_lines (
    id              BIGSERIAL PRIMARY KEY,
    document_id     TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ts              TIMESTAMPTZ,
    level           TEXT,
    component       TEXT,
    node            TEXT,
    message         TEXT NOT NULL,
    error_signature TEXT
);
CREATE INDEX IF NOT EXISTS log_lines_component ON log_lines (component, level);
CREATE INDEX IF NOT EXISTS log_lines_signature ON log_lines (error_signature);

-- Knowledge graph: a property graph stored as two relational tables.
-- Node kinds: issue, component, error. Edge kinds: AFFECTS, MENTIONS, SEEN_IN.
CREATE TABLE IF NOT EXISTS kg_nodes (
    id    TEXT PRIMARY KEY,                -- "component:kubelet", "error:OOMKilled", "issue:42"
    kind  TEXT NOT NULL,
    label TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS kg_edges (
    src    TEXT NOT NULL REFERENCES kg_nodes(id) ON DELETE CASCADE,
    dst    TEXT NOT NULL REFERENCES kg_nodes(id) ON DELETE CASCADE,
    kind   TEXT NOT NULL,
    weight INT NOT NULL DEFAULT 1,         -- how many times the relation was observed
    PRIMARY KEY (src, dst, kind)
);
CREATE INDEX IF NOT EXISTS kg_edges_dst ON kg_edges (dst);
