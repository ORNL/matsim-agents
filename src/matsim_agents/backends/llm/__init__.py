"""Language-model backend selection."""

from __future__ import annotations

from typing import TYPE_CHECKING

from matsim_agents.backends.llm.provider import get_chat_model

if TYPE_CHECKING:
    # The stable public type for an LLM backend is LangChain's BaseChatModel.
    # Import here so callers can write `backends.llm.LLMBackend` without
    # pulling in the full langchain dependency at parse time.
    from langchain_core.language_models import BaseChatModel as LLMBackend

__all__ = ["LLMBackend", "get_chat_model"]
