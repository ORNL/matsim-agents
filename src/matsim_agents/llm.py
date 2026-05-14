"""LLM provider factory.

Keeps the rest of the codebase agnostic to which chat model is in use.
Supported providers (set ``MATSIM_LLM_PROVIDER`` or pass ``provider=...``):

    ollama     - local Ollama daemon (default for local development)
    vllm       - vLLM server with an OpenAI-compatible /v1 endpoint
    openai     - hosted OpenAI API
    anthropic  - hosted Anthropic API

For vLLM, set the server URL via ``MATSIM_VLLM_BASE_URL`` (default
``http://localhost:8000/v1``) and, if your server requires it,
``MATSIM_VLLM_API_KEY`` (default ``"EMPTY"``).

For Ollama, set ``MATSIM_OLLAMA_BASE_URL`` to point at a non-default host
(default ``http://localhost:11434``).
"""

from __future__ import annotations

import os
from typing import Any

from langchain_core.language_models import BaseChatModel

DEFAULT_PROVIDER = "ollama"
DEFAULT_MODELS = {
    "ollama": "llama3.1:8b",
    "vllm": "meta-llama/Llama-3.1-8B-Instruct",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-sonnet-latest",
}


def get_chat_model(
    provider: str | None = None,
    model: str | None = None,
    temperature: float = 0.0,
    base_url: str | None = None,
    api_key: str | None = None,
    **kwargs: Any,
) -> BaseChatModel:
    """Return a configured LangChain chat model.

    Parameters
    ----------
    provider:
        One of ``"ollama"``, ``"vllm"``, ``"openai"``, ``"anthropic"``.
        Defaults to ``$MATSIM_LLM_PROVIDER`` or ``"ollama"``.
    model:
        Provider-specific model identifier. Sensible defaults are used
        when omitted.
    base_url:
        Override the server URL (Ollama or vLLM).
    api_key:
        Override the API key (OpenAI, Anthropic, or a secured vLLM server).
    """
    provider = (provider or os.environ.get("MATSIM_LLM_PROVIDER", DEFAULT_PROVIDER)).lower()
    model = model or DEFAULT_MODELS.get(provider)

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        url = base_url or os.environ.get("MATSIM_OLLAMA_BASE_URL")
        if url:
            kwargs.setdefault("base_url", url)
        return ChatOllama(model=model, temperature=temperature, **kwargs)

    if provider == "vllm":
        # vLLM exposes an OpenAI-compatible API; reuse ChatOpenAI with a custom base_url.
        from langchain_openai import ChatOpenAI

        url = base_url or os.environ.get("MATSIM_VLLM_BASE_URL", "http://localhost:8000/v1")
        key = api_key or os.environ.get("MATSIM_VLLM_API_KEY", "EMPTY")
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            base_url=url,
            api_key=key,
            **kwargs,
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        if api_key:
            kwargs.setdefault("api_key", api_key)
        return ChatOpenAI(model=model, temperature=temperature, **kwargs)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        if api_key:
            kwargs.setdefault("api_key", api_key)
        return ChatAnthropic(model=model, temperature=temperature, **kwargs)

    raise ValueError(
        f"Unknown LLM provider: {provider!r}. "
        f"Expected one of: {sorted(DEFAULT_MODELS)}."
    )
