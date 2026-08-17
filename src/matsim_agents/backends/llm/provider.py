"""LLM provider factory.

Keeps the rest of the codebase agnostic to which chat model is in use.
Supported providers (set ``MATSIM_LLM_PROVIDER`` or pass ``provider=...``):

    ollama       - local Ollama daemon (default for local development)
    vllm         - vLLM server with an OpenAI-compatible /v1 endpoint
    openai       - hosted OpenAI API
    anthropic    - hosted Anthropic API
    huggingface  - local HuggingFace transformers pipeline (no server needed)

For vLLM, set the server URL via ``MATSIM_VLLM_BASE_URL`` (default
``http://localhost:8000/v1``) and, if your server requires it,
``MATSIM_VLLM_API_KEY`` (default ``"EMPTY"``).

For Ollama, set ``MATSIM_OLLAMA_BASE_URL`` to point at a non-default host
(default ``http://localhost:11434``).

For HuggingFace local inference, set ``MATSIM_HF_MODEL_PATH`` to a local
model directory (default: the ``model`` argument). Uses ``device_map="auto"``
to spread across all available GPUs. Install extras: ``pip install -e .[huggingface]``.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult


class ChatVLLM(BaseChatModel):
    """Minimal LangChain chat model that talks to a vLLM (or any OpenAI-compatible)
    server using the ``openai`` package directly.  This avoids the ``langchain_openai``
    dependency which is not present in the HPC conda environment.
    """

    model: str
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    temperature: float = 0.0
    max_completion_tokens: int = 2048

    @property
    def _llm_type(self) -> str:
        return "vllm"

    def _convert_messages(self, messages: list[BaseMessage]) -> list[dict]:
        role_map = {
            "human": "user",
            "ai": "assistant",
            "system": "system",
            "tool": "tool",
        }
        result = []
        for m in messages:
            role = role_map.get(m.type, "user")
            result.append({"role": role, "content": m.content})
        return result

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        import openai

        client = openai.OpenAI(base_url=self.base_url, api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model,
            messages=self._convert_messages(messages),
            temperature=self.temperature,
            max_completion_tokens=self.max_completion_tokens,
            stop=stop,
            **kwargs,
        )
        content = response.choices[0].message.content or ""
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        import openai

        client = openai.OpenAI(base_url=self.base_url, api_key=self.api_key)
        stream = client.chat.completions.create(
            model=self.model,
            messages=self._convert_messages(messages),
            temperature=self.temperature,
            max_completion_tokens=self.max_completion_tokens,
            stop=stop,
            stream=True,
            **kwargs,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                yield ChatGenerationChunk(message=AIMessageChunk(content=delta))


DEFAULT_PROVIDER = "ollama"
DEFAULT_MODELS = {
    "ollama": "llama3.1:8b",
    "vllm": "meta-llama/Llama-3.1-8B-Instruct",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-sonnet-latest",
    "huggingface": "Qwen/Qwen2.5-72B-Instruct",
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
        # vLLM exposes an OpenAI-compatible API.  Use the built-in ChatVLLM wrapper
        # which talks directly to the openai package — no langchain_openai required.
        url = base_url or os.environ.get("MATSIM_VLLM_BASE_URL", "http://localhost:8000/v1")
        key = api_key or os.environ.get("MATSIM_VLLM_API_KEY", "EMPTY")
        return ChatVLLM(model=model, temperature=temperature, base_url=url, api_key=key)

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

    if provider == "huggingface":
        from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

        model_path = os.environ.get("MATSIM_HF_MODEL_PATH") or model
        pipeline_kwargs: dict[str, Any] = {
            "max_new_tokens": kwargs.pop("max_new_tokens", 2048),
            "do_sample": temperature > 0.0,
            # Suppress warnings from model's generation_config by not passing
            # conflicting/irrelevant parameters. Only add temperature if sampling.
        }
        if temperature > 0.0:
            pipeline_kwargs["temperature"] = temperature
            # top_p, top_k are only relevant when do_sample=True
            if "top_p" not in pipeline_kwargs:
                pipeline_kwargs["top_p"] = 0.9
        hf_pipeline = HuggingFacePipeline.from_model_id(
            model_id=model_path,
            task="text-generation",
            device_map="auto",
            pipeline_kwargs=pipeline_kwargs,
            **kwargs,
        )
        return ChatHuggingFace(llm=hf_pipeline)

    raise ValueError(
        f"Unknown LLM provider: {provider!r}. Expected one of: {sorted(DEFAULT_MODELS)}."
    )
