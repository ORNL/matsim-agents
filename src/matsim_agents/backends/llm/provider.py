"""LLM provider factory.

Keeps the rest of the codebase agnostic to which chat model is in use.
Supported providers (set ``MATSIM_LLM_PROVIDER`` or pass ``provider=...``):

    ollama       - local Ollama daemon (default for local development)
    vllm         - vLLM server with an OpenAI-compatible /v1 endpoint
    openai       - hosted OpenAI API
    anthropic    - hosted Anthropic API
    huggingface  - local HuggingFace transformers pipeline (no server needed)

For vLLM, set the server URL via ``MATSIM_VLLM_BASE_URL`` (default
``http://localhost:8000/v1``), request timeout via
``MATSIM_VLLM_TIMEOUT_SECONDS`` (default ``3600``), and, if your server requires
it, ``MATSIM_VLLM_API_KEY`` (default ``"EMPTY"``).
``MATSIM_VLLM_MAX_TOKENS`` defaults to ``4096``. Token limits must be positive
integers and timeouts finite positive seconds; explicit arguments take priority.

For Ollama, set ``MATSIM_OLLAMA_BASE_URL`` to point at a non-default host
(default ``http://localhost:11434``).

For HuggingFace local inference, set ``MATSIM_HF_MODEL_PATH`` to a local
model directory (default: the ``model`` argument). Uses ``device_map="auto"``
to spread across all available GPUs, or across the devices a resource plan
assigns to the model (:func:`limit_in_process_devices`). Each model is loaded
once per process and shared by every caller; replies contain only newly
generated text.
Install extras: ``pip install -e .[huggingface]``.
"""

from __future__ import annotations

import functools
import logging
import os
from collections.abc import Iterator
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field

from matsim_agents.backends.llm.replies import split_reasoning

log = logging.getLogger(__name__)


class ChatVLLM(BaseChatModel):
    """Minimal LangChain chat model that talks to a vLLM (or any OpenAI-compatible)
    server using the ``openai`` package directly.  This avoids the ``langchain_openai``
    dependency which is not present in the HPC conda environment.
    """

    model: str
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    temperature: float = 0.0
    top_p: float | None = None
    seed: int | None = None
    max_completion_tokens: int = Field(default=4096, gt=0)
    request_timeout: float = Field(default=3600.0, gt=0, allow_inf_nan=False)

    @property
    def _llm_type(self) -> str:
        return "vllm"

    def _sampling(self) -> dict[str, Any]:
        return {
            name: value
            for name, value in (("top_p", self.top_p), ("seed", self.seed))
            if value is not None
        }

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

        client = openai.OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.request_timeout,
        )
        response = client.chat.completions.create(
            model=self.model,
            messages=self._convert_messages(messages),
            temperature=self.temperature,
            max_completion_tokens=self.max_completion_tokens,
            stop=stop,
            **{**self._sampling(), **kwargs},
        )
        choice = response.choices[0]
        # With a reasoning parser vLLM returns the reasoning separately
        # (`reasoning`, formerly `reasoning_content`); without one it precedes
        # </think> in the content. Either way the content is only the answer.
        reasoning = next(
            (
                value
                for value in (
                    getattr(choice.message, "reasoning", None),
                    getattr(choice.message, "reasoning_content", None),
                )
                if isinstance(value, str) and value
            ),
            None,
        )
        content = choice.message.content or ""
        if reasoning is None:
            content, reasoning = split_reasoning(content)
        finish = choice.finish_reason if isinstance(choice.finish_reason, str) else None
        usage = getattr(response, "usage", None)
        tokens = [getattr(usage, name, None) for name in ("prompt_tokens", "completion_tokens")]
        message = AIMessage(
            content=content,
            additional_kwargs={"reasoning_content": reasoning} if reasoning else {},
            response_metadata={"finish_reason": finish, "model_name": self.model},
            usage_metadata=(
                {
                    "input_tokens": tokens[0],
                    "output_tokens": tokens[1],
                    "total_tokens": tokens[0] + tokens[1],
                }
                if all(isinstance(count, int) for count in tokens)
                else None
            ),
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        import openai

        client = openai.OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.request_timeout,
        )
        stream = client.chat.completions.create(
            model=self.model,
            messages=self._convert_messages(messages),
            temperature=self.temperature,
            max_completion_tokens=self.max_completion_tokens,
            stop=stop,
            stream=True,
            **{**self._sampling(), **kwargs},
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                yield ChatGenerationChunk(message=AIMessageChunk(content=delta))


# Loaded in-process models keyed by model path, load options, and assigned
# devices. Every ``get_chat_model`` call reuses the same weights; only
# generation settings differ between callers.
_HUGGINGFACE_PIPELINES: dict[tuple[str, str, str], Any] = {}

# Devices (and optional memory budgets) a resource plan assigns to an in-process
# model, keyed by model path; see matsim_agents.backends.llm.servers.
_IN_PROCESS_DEVICES: dict[str, dict[int, float | None]] = {}


def _loaded_placements(model_path: str) -> set[str]:
    return {key[2] for key in _HUGGINGFACE_PIPELINES if key[0] == model_path}


def limit_in_process_devices(model_path: str, fractions: dict[int, float | None]) -> None:
    """Restrict an in-process model to these device ordinals.

    A device mapped to a fraction may use at most that share of its memory; a
    device mapped to ``None`` may use its free memory. Other devices and the
    CPU are not used. A model already loaded in this process with a different
    assignment is not moved: that raises instead of silently keeping the old
    devices or loading a second copy.
    """

    if not fractions:
        raise ValueError(f"in-process model {model_path} needs at least one device")
    placement = repr(sorted(fractions.items()))
    loaded = _loaded_placements(model_path) - {placement}
    if loaded:
        raise ValueError(
            f"in-process model {model_path} is already loaded in this process on a different "
            f"device assignment ({sorted(loaded)}); use one assignment per process"
        )
    _IN_PROCESS_DEVICES[model_path] = dict(fractions)


def _huggingface_model_path(model: str | None) -> str:
    """Model directory to load; a model placed by a resource plan is authoritative.

    ``MATSIM_HF_MODEL_PATH`` would otherwise replace it with a snapshot that has
    no registered devices and would load with unrestricted placement.
    """

    override = os.environ.get("MATSIM_HF_MODEL_PATH")
    if model is not None and model in _IN_PROCESS_DEVICES:
        if override and override != model:
            log.warning(
                "Ignoring MATSIM_HF_MODEL_PATH=%s: the resource plan places in-process model %s",
                override,
                model,
            )
        return model
    return override or str(model)


def _max_memory(fractions: dict[int, float | None]) -> dict[int, int]:
    import torch

    backend = torch.cuda if torch.cuda.is_available() else getattr(torch, "xpu", None)
    if backend is None or not backend.is_available():
        raise RuntimeError("in-process LLM devices were assigned, but no GPU is available")
    budget: dict[int, int] = {}
    for device, fraction in fractions.items():
        total = int(backend.get_device_properties(device).total_memory)
        mem_get_info = getattr(backend, "mem_get_info", None)
        free = int(mem_get_info(device)[0]) if mem_get_info is not None else total
        budget[device] = min(free, int(total * fraction)) if fraction is not None else free
    return budget


def _load_huggingface_pipeline(model_path: str, load_kwargs: dict[str, Any]) -> Any:
    from langchain_huggingface import HuggingFacePipeline

    devices = _IN_PROCESS_DEVICES.get(model_path)
    placement = repr(sorted(devices.items())) if devices is not None else "auto"
    key = (model_path, repr(sorted(load_kwargs.items())), placement)
    if key not in _HUGGINGFACE_PIPELINES:
        if _loaded_placements(model_path) - {placement}:
            raise ValueError(
                f"in-process model {model_path} is already loaded in this process on "
                f"different devices than {placement}"
            )
        kwargs = dict(load_kwargs)
        if devices is not None:
            model_kwargs = dict(kwargs.pop("model_kwargs", None) or {})
            model_kwargs["max_memory"] = _max_memory(devices)
            kwargs["model_kwargs"] = model_kwargs
        _HUGGINGFACE_PIPELINES[key] = HuggingFacePipeline.from_model_id(
            model_id=model_path,
            task="text-generation",
            device_map="auto",
            **kwargs,
        )
    return _HUGGINGFACE_PIPELINES[key]


@functools.cache
def _generation_pipeline_class(base: type) -> type:
    """Subclass *base* so its ``pipeline_kwargs`` apply to every generation call.

    ``HuggingFacePipeline`` applies generation settings only when the
    transformers pipeline is built; a shared, already-loaded pipeline would
    otherwise ignore each caller's temperature and token limit.
    """

    def _with_generation(self: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
        merged = {**(self.pipeline_kwargs or {}), **kwargs.get("pipeline_kwargs", {})}
        return {**kwargs, "pipeline_kwargs": merged}

    def _generate(self, prompts, stop=None, run_manager=None, **kwargs):
        return base._generate(self, prompts, stop, run_manager, **_with_generation(self, kwargs))

    def _stream(self, prompt, stop=None, run_manager=None, **kwargs):
        return base._stream(self, prompt, stop, run_manager, **_with_generation(self, kwargs))

    return type(f"Generation{base.__name__}", (base,), {"_generate": _generate, "_stream": _stream})


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
    max_tokens: int | None = None,
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
    max_tokens:
        Most tokens one reply may generate, including a reasoning model's
        reasoning; the provider's default when omitted.
    """
    provider = (provider or os.environ.get("MATSIM_LLM_PROVIDER", DEFAULT_PROVIDER)).lower()
    model = model or DEFAULT_MODELS.get(provider)

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        url = base_url or os.environ.get("MATSIM_OLLAMA_BASE_URL")
        if url:
            kwargs.setdefault("base_url", url)
        if max_tokens is not None:
            kwargs.setdefault("num_predict", max_tokens)
        return ChatOllama(model=model, temperature=temperature, **kwargs)

    if provider == "vllm":
        # vLLM exposes an OpenAI-compatible API.  Use the built-in ChatVLLM wrapper
        # which talks directly to the openai package — no langchain_openai required.
        url = base_url or os.environ.get("MATSIM_VLLM_BASE_URL", "http://localhost:8000/v1")
        key = api_key or os.environ.get("MATSIM_VLLM_API_KEY", "EMPTY")
        if max_tokens is None:
            max_tokens = (
                kwargs.pop("max_completion_tokens")
                if "max_completion_tokens" in kwargs
                else os.environ.get("MATSIM_VLLM_MAX_TOKENS", 4096)
            )
        request_timeout = (
            kwargs.pop("request_timeout")
            if "request_timeout" in kwargs
            else os.environ.get("MATSIM_VLLM_TIMEOUT_SECONDS", 3600)
        )
        return ChatVLLM(
            model=model,
            temperature=temperature,
            base_url=url,
            api_key=key,
            max_completion_tokens=max_tokens,
            request_timeout=request_timeout,
            top_p=kwargs.get("top_p"),
            seed=kwargs.get("seed"),
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        if api_key:
            kwargs.setdefault("api_key", api_key)
        if max_tokens is not None:
            kwargs.setdefault("max_tokens", max_tokens)
        return ChatOpenAI(model=model, temperature=temperature, **kwargs)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        if api_key:
            kwargs.setdefault("api_key", api_key)
        if max_tokens is not None:
            kwargs.setdefault("max_tokens", max_tokens)
        return ChatAnthropic(model=model, temperature=temperature, **kwargs)

    if provider == "huggingface":
        from langchain_huggingface import ChatHuggingFace

        model_path = _huggingface_model_path(model)
        default_new_tokens = kwargs.pop("max_new_tokens", 2048)
        generation: dict[str, Any] = {
            "max_new_tokens": max_tokens if max_tokens is not None else default_new_tokens,
            "do_sample": temperature > 0.0,
            # Chat replies must not repeat the rendered prompt.
            "return_full_text": False,
        }
        if temperature > 0.0:
            generation["temperature"] = temperature
            generation["top_p"] = 0.9
        loaded = _load_huggingface_pipeline(model_path, kwargs)
        llm = _generation_pipeline_class(type(loaded))(
            pipeline=loaded.pipeline,
            model_id=model_path,
            model_kwargs=loaded.model_kwargs,
            pipeline_kwargs=generation,
            batch_size=loaded.batch_size,
        )
        return ChatHuggingFace(llm=llm, tokenizer=loaded.pipeline.tokenizer)

    raise ValueError(
        f"Unknown LLM provider: {provider!r}. Expected one of: {sorted(DEFAULT_MODELS)}."
    )
