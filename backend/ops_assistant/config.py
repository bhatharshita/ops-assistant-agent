"""Application settings, read from environment variables (or a .env file).

Everything that changes between laptop, Docker and CI lives here, so the rest of
the code never calls os.environ directly.
"""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Database -----------------------------------------------------------
    database_url: str = "postgresql://ops:ops@localhost:5432/ops"

    # --- LLM ----------------------------------------------------------------
    # "auto" picks the first provider whose API key is set, else "offline".
    # "offline" runs the whole graph with deterministic rule-based nodes, so the
    # project works end to end (and in CI) without any API key.
    llm_provider: Literal["auto", "openai", "anthropic", "offline"] = "auto"
    openai_api_key: str | None = None
    openai_base_url: str | None = None  # e.g. https://api.mistral.ai/v1 or an Ollama URL
    openai_model: str = "gpt-4o-mini"
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-haiku-5-5"
    llm_temperature: float = 0.0

    # --- Embeddings ---------------------------------------------------------
    # "openai" calls any OpenAI-compatible /embeddings endpoint (OpenAI, Mistral, ...).
    # "hashing" is a local, dependency-free fallback (feature hashing, no download).
    embedding_provider: Literal["auto", "openai", "hashing"] = "auto"
    embedding_api_key: str | None = None  # falls back to OPENAI_API_KEY
    embedding_base_url: str = "https://api.openai.com/v1"
    embedding_model: str = "text-embedding-3-small"
    # Must match the vector size the model returns. text-embedding-3-* can be shortened
    # server-side to any size; mistral-embed is always 1024.
    embedding_dim: int = 512

    # --- Retrieval ----------------------------------------------------------
    retrieval_mode: Literal["dense", "hybrid"] = "hybrid"
    top_k: int = 5
    chunk_size: int = 800
    chunk_overlap: int = 120

    # --- Data -------------------------------------------------------------
    data_dir: str = "../data/sample"  # default location of issues.json + logs/

    # --- Agent --------------------------------------------------------------
    max_tool_calls: int = 4

    def resolved_llm_provider(self) -> str:
        if self.llm_provider != "auto":
            return self.llm_provider
        if self.anthropic_api_key:
            return "anthropic"
        if self.openai_api_key:
            return "openai"
        return "offline"

    def resolved_embedding_provider(self) -> str:
        if self.embedding_provider != "auto":
            return self.embedding_provider
        return "openai" if (self.embedding_api_key or self.openai_api_key) else "hashing"


@lru_cache
def get_settings() -> Settings:
    return Settings()
