"""Retriever: dense (pgvector) or hybrid (dense + Postgres full-text, fused with RRF).

Why hybrid? Ops questions are full of exact tokens - error strings, flags, pod and
node names - that keyword search matches perfectly while embeddings blur them.
Reciprocal Rank Fusion combines the two ranked lists without having to calibrate
their very different score scales: score(d) = sum over lists of 1 / (k + rank).
"""

from __future__ import annotations

from collections.abc import Sequence

from .embeddings import Embedder
from .store import Hit, PgStore


def reciprocal_rank_fusion(result_lists: Sequence[Sequence[Hit]], k: int = 60) -> list[Hit]:
    scores: dict[str, float] = {}
    best: dict[str, Hit] = {}
    for results in result_lists:
        for rank, hit in enumerate(results, start=1):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (k + rank)
            best.setdefault(hit.chunk_id, hit)
    fused = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    return [Hit(**{**best[cid].to_dict(), "score": round(scores[cid], 5)}) for cid in fused]


class Retriever:
    def __init__(self, store: PgStore, embedder: Embedder, mode: str = "hybrid", top_k: int = 5):
        self.store = store
        self.embedder = embedder
        self.mode = mode
        self.top_k = top_k

    def search(self, query: str, k: int | None = None,
               source_type: str | None = None) -> list[Hit]:
        k = k or self.top_k
        qvec = self.embedder.embed([query])[0]
        if self.mode == "dense":
            return self.store.search_dense(qvec, k, source_type)
        candidates = k * 4  # over-fetch from each list so fusion has something to re-rank
        dense = self.store.search_dense(qvec, candidates, source_type)
        keyword = self.store.search_keyword(query, candidates, source_type)
        return reciprocal_rank_fusion([dense, keyword])[:k]
