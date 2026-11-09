"""Ingestion pipeline: load -> chunk -> embed -> store (+ log lines + graph)."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

from ..config import Settings
from ..embeddings import Embedder
from ..store import Chunk, LogLine, PgStore
from .chunking import chunk_issue, chunk_logs
from .graph import build_graph
from .loaders import load_issues, load_log_file

log = logging.getLogger(__name__)


@dataclass
class IngestReport:
    documents: int
    chunks: int
    log_lines: int
    kg_nodes: int
    kg_edges: int
    seconds: float


def ingest(
    store: PgStore,
    embedder: Embedder,
    settings: Settings,
    issues_path: Path | None,
    logs_dir: Path | None,
    reset: bool = False,
) -> IngestReport:
    t0 = time.perf_counter()
    store.init_schema(reset=reset)

    issues = load_issues(issues_path) if issues_path else []
    log_docs, all_lines = [], []
    if logs_dir:
        for path in sorted(logs_dir.glob("*.log")):
            doc, lines = load_log_file(path)
            log_docs.append((doc, lines))
            all_lines.extend(lines)

    chunks: list[Chunk] = []
    for rec in issues:
        chunks += chunk_issue(rec.document.id, rec.text, settings.chunk_size,
                              settings.chunk_overlap)
    for doc, lines in log_docs:
        chunks += chunk_logs(doc.id, lines)

    log.info("embedding %d chunks with %s", len(chunks), type(embedder).__name__)
    embeddings = embedder.embed([c.content for c in chunks]) if chunks else []

    store.upsert_documents([r.document for r in issues] + [d for d, _ in log_docs])
    if chunks:
        store.replace_chunks(chunks, embeddings)
    lines: list[LogLine] = all_lines
    if lines:
        store.replace_log_lines(lines)
    nodes, edges = build_graph(issues, all_lines)
    store.upsert_graph(nodes, edges)

    return IngestReport(
        documents=len(issues) + len(log_docs), chunks=len(chunks), log_lines=len(lines),
        kg_nodes=len(nodes), kg_edges=len(edges), seconds=round(time.perf_counter() - t0, 2),
    )
