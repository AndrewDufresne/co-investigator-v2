"""LLM Gateway — unified interface for DeepSeek (primary) and other LLMs.

Uses LangChain's ChatOpenAI with DeepSeek's OpenAI-compatible API.
"""

from __future__ import annotations

from typing import Any
from langchain_openai import ChatOpenAI
from src.config import get_settings


# Agent role → model config mapping (all default to DeepSeek)
MODEL_ROUTING: dict[str, dict[str, str]] = {
    "planning":         {"model": "deepseek-reasoner"},
    "narrative":        {"model": "deepseek-chat"},
    "compliance_judge": {"model": "deepseek-reasoner"},
    "crime_detection":  {"model": "deepseek-reasoner"},
    "typology":         {"model": "deepseek-chat"},
    "evaluation":       {"model": "deepseek-chat"},
    "default":          {"model": "deepseek-chat"},
}


def get_llm(role: str = "default", streaming: bool = True, **overrides: Any) -> ChatOpenAI:
    """Create a ChatOpenAI instance routed to the appropriate model for the given agent role.

    Args:
        role: Agent role key (e.g., "narrative", "planning").
        streaming: Whether to enable token-level streaming (default True).
        **overrides: Additional kwargs passed to ChatOpenAI.

    Returns:
        A configured ChatOpenAI instance pointing to DeepSeek.
    """
    settings = get_settings()
    route = MODEL_ROUTING.get(role, MODEL_ROUTING["default"])

    kwargs: dict[str, Any] = dict(
        model=route["model"],
        base_url=settings.deepseek_base_url,
        api_key=settings.deepseek_api_key,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        streaming=streaming,
    )
    kwargs.update(overrides)
    return ChatOpenAI(**kwargs)
