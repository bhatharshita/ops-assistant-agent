"""Pluggable embedding providers.

- OpenAICompatibleEmbedder: any OpenAI-style `/embeddings` endpoint (OpenAI,
  Mistral `mistral-embed`, Azure OpenAI, Ollama, vLLM, ...). Key read from env.
- HashingEmbedder: local fallback with no model download. It hashes word
  unigrams/bigrams into a fixed-size vector ("feature hashing"). It only captures
  lexical overlap, not meaning, but it is deterministic and free, which makes it
  ideal for tests, CI and an offline demo.
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol

import httpx
import numpy as np

from .config import Settings

TOKEN_RE = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an array of shape (len(texts), dim), L2-normalised."""
        ...


def _normalise(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(norms == 0, 1.0, norms)


class HashingEmbedder:
    def __init__(self, dim: int = 512):
        self.dim = dim

    def _bucket(self, feature: str) -> tuple[int, float]:
        # md5 (not Python's hash()) so vectors are identical across processes.
        h = int.from_bytes(hashlib.md5(feature.encode()).digest()[:8], "little")
        return h % self.dim, (1.0 if (h >> 63) & 1 else -1.0)

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = TOKEN_RE.findall(text.lower())
            features = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:], strict=False)]
            for f in features:
                idx, sign = self._bucket(f)
                out[row, idx] += sign
        # sublinear term frequency so one repeated word does not dominate
        out = np.sign(out) * np.log1p(np.abs(out))
        return _normalise(out)


class OpenAICompatibleEmbedder:
    def __init__(self, api_key: str, base_url: str, model: str, dim: int, batch_size: int = 64):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.dim = dim
        self.batch_size = batch_size

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors: list[list[float]] = []
        with httpx.Client(timeout=60) as client:
            for start in range(0, len(texts), self.batch_size):
                payload: dict = {"model": self.model,
                                 "input": texts[start:start + self.batch_size]}
                # text-embedding-3-* can shorten vectors server-side to match our column.
                if self.model.startswith("text-embedding-3"):
                    payload["dimensions"] = self.dim
                resp = client.post(
                    f"{self.base_url}/embeddings",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=payload,
                )
                resp.raise_for_status()
                data = sorted(resp.json()["data"], key=lambda d: d["index"])
                vectors.extend(d["embedding"] for d in data)
        m = np.asarray(vectors, dtype=np.float32)
        if m.shape[1] != self.dim:
            raise ValueError(
                f"Model returned {m.shape[1]}-d vectors but EMBEDDING_DIM={self.dim}. "
                "Set EMBEDDING_DIM to match and re-run ingestion with --reset."
            )
        return _normalise(m)


def get_embedder(settings: Settings) -> Embedder:
    if settings.resolved_embedding_provider() == "openai":
        key = settings.embedding_api_key or settings.openai_api_key
        if not key:
            raise ValueError("EMBEDDING_PROVIDER=openai needs EMBEDDING_API_KEY or OPENAI_API_KEY")
        return OpenAICompatibleEmbedder(
            key, settings.embedding_base_url, settings.embedding_model, settings.embedding_dim
        )
    return HashingEmbedder(settings.embedding_dim)
