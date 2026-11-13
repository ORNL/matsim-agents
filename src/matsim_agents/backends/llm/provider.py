"""LLM provider factory.

Keeps the rest of the codebase agnostic to which chat model is in use.
Supported providers (set ``MATSIM_LLM_PROVIDER`` or pass ``provider=...``):

    ollama       - local Ollama daemon (default for local development)
    vllm         - vLLM server with an OpenAI-compatible /v1 endpoint
    mag          - Genesis Mission Model Access Gateway
    copilot      - models available to a GitHub Copilot account (Copilot SDK)
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

For Copilot, sign in once with ``copilot login`` (or set
``COPILOT_GITHUB_TOKEN``) and pass an explicit model ID from the account's
catalog (:func:`list_copilot_models`). Install extras: ``pip install -e .[copilot]``.

For HuggingFace local inference, set ``MATSIM_HF_MODEL_PATH`` to a local
model directory (default: the ``model`` argument). Uses ``device_map="auto"``
to spread across all available GPUs, or across the devices a resource plan
assigns to the model (:func:`limit_in_process_devices`). Each model is loaded
once per process and shared by every caller; replies contain only newly
generated text.
Install extras: ``pip install -e .[huggingface]``.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import os
from collections.abc import Callable, Coroutine, Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field, SecretStr

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
    max_retries: int = Field(default=0, ge=0)
    default_headers: dict[str, str] = Field(default_factory=dict)
    capture_response_headers: bool = False

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

        client_kwargs = {
            "base_url": self.base_url,
            "api_key": self.api_key,
            "timeout": self.request_timeout,
            "max_retries": self.max_retries,
        }
        if self.default_headers:
            client_kwargs["default_headers"] = self.default_headers
        client = openai.OpenAI(**client_kwargs)
        create = (
            client.chat.completions.with_raw_response.create
            if self.capture_response_headers
            else client.chat.completions.create
        )
        raw_response = create(
            model=self.model,
            messages=self._convert_messages(messages),
            temperature=self.temperature,
            max_completion_tokens=self.max_completion_tokens,
            stop=stop,
            **{**self._sampling(), **kwargs},
        )
        response = raw_response.parse() if self.capture_response_headers else raw_response
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
            response_metadata={
                "finish_reason": finish,
                "model_name": self.model,
                **self._provider_metadata(
                    raw_response.headers if self.capture_response_headers else None
                ),
            },
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

    def _provider_metadata(self, headers: Any) -> dict[str, Any]:
        applied = headers.get("x-amsc-mag-applied") if headers is not None else None
        return {"provider_metadata": {"mag_applied": applied}} if applied else {}

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        import openai

        client_kwargs = {
            "base_url": self.base_url,
            "api_key": self.api_key,
            "timeout": self.request_timeout,
            "max_retries": self.max_retries,
        }
        if self.default_headers:
            client_kwargs["default_headers"] = self.default_headers
        client = openai.OpenAI(**client_kwargs)
        create = (
            client.chat.completions.with_raw_response.create
            if self.capture_response_headers
            else client.chat.completions.create
        )
        raw_stream = create(
            model=self.model,
            messages=self._convert_messages(messages),
            temperature=self.temperature,
            max_completion_tokens=self.max_completion_tokens,
            stop=stop,
            stream=True,
            **{**self._sampling(), **kwargs},
        )
        stream = raw_stream.parse() if self.capture_response_headers else raw_stream
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                yield ChatGenerationChunk(message=AIMessageChunk(content=delta))


class ChatMAG(ChatVLLM):
    """LangChain chat model for the OpenAI-compatible Genesis Mission MAG API."""

    base_url: str = "https://mag.genesis.americansciencecloud.org/v1"
    capture_response_headers: bool = True

    @property
    def _llm_type(self) -> str:
        return "mag"


_COPILOT_DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


def _run_coroutine(factory: Callable[[], Coroutine[Any, Any, Any]]) -> Any:
    """Run a coroutine from synchronous code, even when this thread has an event loop."""

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(factory())
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(factory())).result()


def _copilot_client(github_token: str | None) -> Any:
    from copilot import CopilotClient

    options = {
        "github_token": github_token,
        "use_logged_in_user": github_token is None,
    }
    base_directory = os.environ.get("COPILOT_HOME")
    if base_directory:
        options["base_directory"] = base_directory
    return CopilotClient(**options)


def list_copilot_models(github_token: str | None = None) -> list[str]:
    """Model IDs the authenticated GitHub Copilot account may use."""

    async def _list() -> list[str]:
        client = _copilot_client(github_token)
        await client.start()
        try:
            return [model.id for model in await client.list_models()]
        finally:
            await client.stop()

    return _run_coroutine(_list)


class ChatCopilot(BaseChatModel):
    """LangChain chat model backed by the GitHub Copilot SDK.

    Each call starts a Copilot runtime, opens a tool-free session whose system
    prompt is replaced by the caller's system messages, sends one prompt, and
    stops the runtime. Authentication uses the Copilot CLI's signed-in account
    (or ``COPILOT_GITHUB_TOKEN``/``GH_TOKEN``/``GITHUB_TOKEN``) unless
    ``github_token`` is set. The SDK exposes no sampling controls, so replies
    use the model's default decoding and are not guaranteed to be deterministic.
    ``max_completion_tokens`` is passed to the runtime as a model limit, but
    Copilot-hosted models do not enforce it; replies that exceed it are logged
    and flagged in ``provider_metadata["output_token_limit_exceeded"]``.
    """

    model: str = Field(min_length=1)
    max_completion_tokens: int = Field(default=4096, gt=0)
    request_timeout: float = Field(default=600.0, gt=0, allow_inf_nan=False)
    github_token: SecretStr | None = Field(default=None, repr=False)

    @property
    def _llm_type(self) -> str:
        return "copilot"

    @staticmethod
    def _text(message: BaseMessage) -> str:
        if not isinstance(message.content, str):
            raise ValueError("the Copilot provider supports text message content only")
        return message.content

    def _prompt(self, messages: list[BaseMessage]) -> tuple[str, str]:
        system = "\n\n".join(self._text(m) for m in messages if m.type == "system")
        turns = [m for m in messages if m.type != "system"]
        if not turns:
            raise ValueError("the Copilot provider needs at least one non-system message")
        if len(turns) == 1 and turns[0].type == "human":
            prompt = self._text(turns[0])
        else:
            labels = {"human": "User", "ai": "Assistant", "tool": "Tool"}
            prompt = "\n\n".join(
                f"{labels.get(m.type, m.type.title())}:\n{self._text(m)}" for m in turns
            )
        return system or _COPILOT_DEFAULT_SYSTEM_PROMPT, prompt

    async def _acomplete(self, messages: list[BaseMessage]) -> ChatResult:
        from copilot.session import ModelCapabilitiesOverride, ModelLimitsOverride
        from copilot.session_events import AssistantMessageData, AssistantUsageData

        system, prompt = self._prompt(messages)
        token = self.github_token.get_secret_value() if self.github_token else None
        usages: list[Any] = []
        client = _copilot_client(token)
        await client.start()
        try:
            session = await client.create_session(
                model=self.model,
                system_message={"mode": "replace", "content": system},
                model_capabilities=ModelCapabilitiesOverride(
                    limits=ModelLimitsOverride(max_output_tokens=self.max_completion_tokens)
                ),
                tools=[],
                available_tools=[],
                mcp_servers={},
                included_builtin_agents=[],
                enable_skills=False,
                skip_custom_instructions=True,
                enable_config_discovery=False,
                enable_session_telemetry=False,
                streaming=False,
            )
            session.on(
                lambda event: (
                    usages.append(event.data)
                    if isinstance(event.data, AssistantUsageData)
                    and not getattr(event, "agent_id", None)
                    else None
                )
            )
            response = await session.send_and_wait(prompt, timeout=self.request_timeout)
        finally:
            await client.stop()
        data = getattr(response, "data", None)
        if not isinstance(data, AssistantMessageData):
            raise RuntimeError(f"Copilot model {self.model!r} returned no assistant message")
        usage = next(
            (u for u in reversed(usages) if data.api_call_id and u.api_call_id == data.api_call_id),
            usages[-1] if usages else None,
        )
        input_tokens = getattr(usage, "input_tokens", None)
        output_tokens = getattr(usage, "output_tokens", None)
        provider_metadata: dict[str, str] = {}
        if isinstance(output_tokens, int) and output_tokens > self.max_completion_tokens:
            log.warning(
                "Copilot model %s produced %d output tokens, above the requested %d; "
                "Copilot-hosted models do not enforce the output-token limit",
                self.model,
                output_tokens,
                self.max_completion_tokens,
            )
            provider_metadata["output_token_limit_exceeded"] = (
                f"{output_tokens}>{self.max_completion_tokens}"
            )
        message = AIMessage(
            content=data.content,
            additional_kwargs=(
                {"reasoning_content": data.reasoning_text} if data.reasoning_text else {}
            ),
            response_metadata={
                "finish_reason": getattr(usage, "finish_reason", None),
                "model_name": data.model or self.model,
                **({"provider_metadata": provider_metadata} if provider_metadata else {}),
            },
            usage_metadata=(
                {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                }
                if isinstance(input_tokens, int) and isinstance(output_tokens, int)
                else None
            ),
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    @staticmethod
    def _check_call_options(stop: list[str] | None, kwargs: dict[str, Any]) -> None:
        if stop:
            raise ValueError("the Copilot provider does not support stop sequences")
        unsupported = sorted(
            name
            for name, value in kwargs.items()
            if not (name == "temperature" and value in (None, 0, 0.0))
        )
        if unsupported:
            raise ValueError(
                "the Copilot SDK exposes no sampling controls; unsupported call options: "
                + ", ".join(unsupported)
            )

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self._check_call_options(stop, kwargs)
        return _run_coroutine(lambda: self._acomplete(messages))

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        self._check_call_options(stop, kwargs)
        return await self._acomplete(messages)


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
SUPPORTED_PROVIDERS = frozenset((*DEFAULT_MODELS, "mag", "copilot"))


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
        Override the server URL (Ollama, vLLM, or MAG).
    api_key:
        Override the API key (OpenAI, Anthropic, MAG, or a secured vLLM server),
        or a GitHub token for Copilot (default: the Copilot CLI's signed-in account).
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
        max_retries = (
            kwargs.pop("max_retries")
            if "max_retries" in kwargs
            else os.environ.get("MATSIM_VLLM_MAX_RETRIES", 0)
        )
        return ChatVLLM(
            model=model,
            temperature=temperature,
            base_url=url,
            api_key=key,
            max_completion_tokens=max_tokens,
            request_timeout=request_timeout,
            max_retries=max_retries,
            top_p=kwargs.get("top_p"),
            seed=kwargs.get("seed"),
        )

    if provider == "mag":
        if not model:
            raise ValueError("MAG requires an explicit model ID from the account's /models catalog")
        key = api_key or os.environ.get("AMSC_TOKEN")
        if not key:
            raise ValueError("MAG requires AMSC_TOKEN or an explicit api_key")
        url = base_url or os.environ.get(
            "MATSIM_MAG_BASE_URL", "https://mag.genesis.americansciencecloud.org/v1"
        )
        route = (
            kwargs.pop("mag_provider")
            if "mag_provider" in kwargs
            else os.environ.get("MATSIM_MAG_PROVIDER")
        )
        headers = {"X-AmSC-MAG-Provider": route} if route else {}
        request_timeout = kwargs.pop("request_timeout", 3600)
        if max_tokens is None:
            max_tokens = kwargs.pop("max_completion_tokens", 4096)
        return ChatMAG(
            model=model,
            temperature=temperature,
            base_url=url,
            api_key=key,
            max_completion_tokens=max_tokens,
            request_timeout=request_timeout,
            top_p=kwargs.get("top_p"),
            seed=kwargs.get("seed"),
            default_headers=headers,
        )

    if provider == "copilot":
        if not model:
            raise ValueError(
                "Copilot requires an explicit model ID from the account's Copilot model catalog"
            )
        if base_url:
            raise ValueError("the Copilot provider does not use base_url")
        if temperature != 0.0 or {"top_p", "seed"} & kwargs.keys():
            raise ValueError(
                "the Copilot SDK exposes no sampling controls; use the default temperature"
            )
        return ChatCopilot(
            model=model,
            max_completion_tokens=max_tokens if max_tokens is not None else 4096,
            request_timeout=kwargs.pop("request_timeout", 600),
            github_token=SecretStr(api_key) if api_key else None,
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
        f"Unknown LLM provider: {provider!r}. Expected one of: {sorted(SUPPORTED_PROVIDERS)}."
    )
