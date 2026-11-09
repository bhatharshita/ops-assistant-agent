"""Chat-model factory. Returns None in offline mode (no API key configured)."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from .config import Settings


def get_chat_model(settings: Settings) -> BaseChatModel | None:
    provider = settings.resolved_llm_provider()
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=settings.anthropic_model, api_key=settings.anthropic_api_key,
                             temperature=settings.llm_temperature, max_tokens=1024)
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        # base_url lets the same client talk to Mistral, Azure-compatible gateways or Ollama.
        return ChatOpenAI(model=settings.openai_model, api_key=settings.openai_api_key,
                          base_url=settings.openai_base_url,
                          temperature=settings.llm_temperature)
    return None
