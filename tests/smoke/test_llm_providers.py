"""Smoke tests for the LLM provider factory.

These tests run without any GPU, real model weights, or network access.
They verify that:
  1. get_chat_model() instantiates the correct class for each provider.
  2. The huggingface provider can be patched and invoked end-to-end.
  3. Unknown providers raise ValueError.
  4. Provider selection via env var works correctly.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage

# ── provider instantiation ────────────────────────────────────────────────────


class TestGetChatModelInstantiation:
    """get_chat_model() returns the right class without calling remote services."""

    def test_ollama_provider(self):
        """ChatOllama is returned for the 'ollama' provider."""
        mock_ollama = MagicMock()
        with (
            patch("matsim_agents.backends.llm.provider.ChatOllama", mock_ollama, create=True),
            patch.dict("sys.modules", {"langchain_ollama": MagicMock(ChatOllama=mock_ollama)}),
        ):
            from matsim_agents.backends.llm.provider import get_chat_model

            get_chat_model(provider="ollama", model="test:model")
            mock_ollama.assert_called_once()

    def test_vllm_provider(self):
        """ChatVLLM is returned for the 'vllm' provider (OpenAI-compatible, no langchain_openai)."""
        from matsim_agents.backends.llm.provider import ChatVLLM, get_chat_model

        result = get_chat_model(
            provider="vllm",
            model="Qwen/Qwen2.5-72B-Instruct",
            base_url="http://localhost:8000/v1",
            api_key="EMPTY",
        )
        assert isinstance(result, ChatVLLM)
        assert result.model == "Qwen/Qwen2.5-72B-Instruct"
        assert result.base_url == "http://localhost:8000/v1"
        assert result.max_completion_tokens == 4096
        assert result.request_timeout == 3600
        assert result.max_retries == 0

    def test_vllm_completion_tokens_from_environment(self, monkeypatch):
        from matsim_agents.backends.llm.provider import get_chat_model

        monkeypatch.setenv("MATSIM_VLLM_MAX_TOKENS", "4096")
        result = get_chat_model(provider="vllm", model="test-model")

        assert result.max_completion_tokens == 4096

    @pytest.mark.parametrize("environment", ["4096", "invalid"])
    def test_vllm_explicit_completion_tokens_override_environment(self, monkeypatch, environment):
        from matsim_agents.backends.llm.provider import get_chat_model

        monkeypatch.setenv("MATSIM_VLLM_MAX_TOKENS", environment)
        result = get_chat_model(
            provider="vllm",
            model="test-model",
            max_completion_tokens=2048,
        )

        assert result.max_completion_tokens == 2048

    def test_vllm_timeout_from_environment(self, monkeypatch):
        from matsim_agents.backends.llm.provider import get_chat_model

        monkeypatch.setenv("MATSIM_VLLM_TIMEOUT_SECONDS", "7200")
        result = get_chat_model(provider="vllm", model="test-model")

        assert result.request_timeout == 7200

    def test_vllm_retries_from_environment(self, monkeypatch):
        from matsim_agents.backends.llm.provider import get_chat_model

        monkeypatch.setenv("MATSIM_VLLM_MAX_RETRIES", "2")
        result = get_chat_model(provider="vllm", model="test-model")

        assert result.max_retries == 2

    @pytest.mark.parametrize("environment", ["7200", "invalid"])
    def test_vllm_explicit_timeout_overrides_environment(self, monkeypatch, environment):
        from matsim_agents.backends.llm.provider import get_chat_model

        monkeypatch.setenv("MATSIM_VLLM_TIMEOUT_SECONDS", environment)
        result = get_chat_model(
            provider="vllm",
            model="test-model",
            request_timeout=1800,
        )

        assert result.request_timeout == 1800

    @pytest.mark.parametrize("variable", ["MATSIM_VLLM_MAX_TOKENS", "MATSIM_VLLM_TIMEOUT_SECONDS"])
    def test_invalid_vllm_environment_without_override_raises(self, monkeypatch, variable):
        from matsim_agents.backends.llm.provider import get_chat_model

        monkeypatch.setenv(variable, "invalid")
        with pytest.raises(ValueError):
            get_chat_model(provider="vllm", model="test-model")

    @pytest.mark.parametrize("source", ["environment", "explicit", "direct"])
    @pytest.mark.parametrize(
        ("field", "variable", "value"),
        [
            ("max_completion_tokens", "MATSIM_VLLM_MAX_TOKENS", 0),
            ("max_completion_tokens", "MATSIM_VLLM_MAX_TOKENS", -1),
            ("max_completion_tokens", "MATSIM_VLLM_MAX_TOKENS", 1.5),
            ("request_timeout", "MATSIM_VLLM_TIMEOUT_SECONDS", 0),
            ("request_timeout", "MATSIM_VLLM_TIMEOUT_SECONDS", -1),
            ("request_timeout", "MATSIM_VLLM_TIMEOUT_SECONDS", float("nan")),
            ("request_timeout", "MATSIM_VLLM_TIMEOUT_SECONDS", float("inf")),
        ],
    )
    def test_invalid_vllm_limits_fail_during_configuration(
        self, monkeypatch, source, field, variable, value
    ):
        from matsim_agents.backends.llm.provider import ChatVLLM, get_chat_model

        kwargs = {}
        if source == "environment":
            monkeypatch.setenv(variable, str(value))
        else:
            kwargs[field] = value
        with pytest.raises(ValueError, match=field):
            if source == "direct":
                ChatVLLM(model="test-model", **kwargs)
            else:
                get_chat_model(provider="vllm", model="test-model", **kwargs)

    @pytest.mark.parametrize("source", ["environment", "explicit"])
    def test_vllm_positive_string_limits_parse_uniformly(self, monkeypatch, source):
        from matsim_agents.backends.llm.provider import get_chat_model

        kwargs = {}
        if source == "environment":
            monkeypatch.setenv("MATSIM_VLLM_MAX_TOKENS", "123")
            monkeypatch.setenv("MATSIM_VLLM_TIMEOUT_SECONDS", "0.5")
        else:
            kwargs = {"max_completion_tokens": "123", "request_timeout": "0.5"}
        model = get_chat_model(provider="vllm", model="test-model", **kwargs)
        assert model.max_completion_tokens == 123
        assert model.request_timeout == 0.5

    def test_vllm_passes_timeout_to_client(self):
        from matsim_agents.backends.llm.provider import ChatVLLM

        response = MagicMock()
        response.choices[0].message.content = "local response"
        client = MagicMock()
        client.chat.completions.create.return_value = response
        with patch("openai.OpenAI", return_value=client) as openai_client:
            result = ChatVLLM(model="test-model").invoke([HumanMessage(content="hello")])

        openai_client.assert_called_once_with(
            base_url="http://localhost:8000/v1",
            api_key="EMPTY",
            timeout=3600,
            max_retries=0,
        )
        assert result.content == "local response"

    def test_mag_provider_routes_through_gateway_and_sets_provider_header(self, monkeypatch):
        from matsim_agents.backends.llm.provider import ChatMAG, get_chat_model

        monkeypatch.setenv("AMSC_TOKEN", "test-token")
        response = MagicMock()
        response.choices[0].message.content = "gateway response"
        client = MagicMock()
        raw_response = MagicMock()
        raw_response.parse.return_value = response
        raw_response.headers = {"x-amsc-mag-applied": "provider=openai"}
        client.chat.completions.with_raw_response.create.return_value = raw_response
        with patch("openai.OpenAI", return_value=client) as openai_client:
            model = get_chat_model(
                provider="mag",
                model="catalog-model",
                mag_provider="openai",
            )
            result = model.invoke([HumanMessage(content="hello")])

        assert isinstance(model, ChatMAG)
        openai_client.assert_called_once_with(
            base_url="https://mag.genesis.americansciencecloud.org/v1",
            api_key="test-token",
            timeout=3600,
            max_retries=0,
            default_headers={"X-AmSC-MAG-Provider": "openai"},
        )
        assert result.content == "gateway response"
        assert result.response_metadata["provider_metadata"] == {"mag_applied": "provider=openai"}
        with pytest.raises(ValueError, match="explicit model ID"):
            get_chat_model(provider="mag", model=None, api_key="test-token")

    def test_mag_explicit_none_suppresses_environment_route(self, monkeypatch):
        from matsim_agents.backends.llm.provider import get_chat_model

        monkeypatch.setenv("MATSIM_MAG_PROVIDER", "environment-route")
        model = get_chat_model(
            provider="mag", model="catalog-model", api_key="test-token", mag_provider=None
        )
        assert model.default_headers == {}

    def test_openai_provider(self):
        mock_openai = MagicMock()
        with patch.dict("sys.modules", {"langchain_openai": MagicMock(ChatOpenAI=mock_openai)}):
            from matsim_agents.backends.llm.provider import get_chat_model

            get_chat_model(provider="openai", model="gpt-4o-mini")
            mock_openai.assert_called_once()

    def test_anthropic_provider(self):
        mock_anthropic = MagicMock()
        with patch.dict(
            "sys.modules", {"langchain_anthropic": MagicMock(ChatAnthropic=mock_anthropic)}
        ):
            from matsim_agents.backends.llm.provider import get_chat_model

            get_chat_model(provider="anthropic", model="claude-3-5-sonnet-latest")
            mock_anthropic.assert_called_once()

    def test_unknown_provider_raises(self):
        from matsim_agents.backends.llm.provider import get_chat_model

        with pytest.raises(ValueError, match="Unknown LLM provider"):
            get_chat_model(provider="nonexistent_provider")

    def test_env_var_selects_provider(self, monkeypatch):
        """MATSIM_LLM_PROVIDER env var is respected."""
        monkeypatch.setenv("MATSIM_LLM_PROVIDER", "vllm")
        monkeypatch.setenv("MATSIM_VLLM_BASE_URL", "http://localhost:8000/v1")
        from matsim_agents.backends.llm.provider import ChatVLLM, get_chat_model

        result = get_chat_model()
        assert isinstance(result, ChatVLLM)
        assert result.base_url == "http://localhost:8000/v1"


# ── copilot provider ──────────────────────────────────────────────────────────


@pytest.fixture
def fake_copilot_sdk(monkeypatch):
    """Install a stand-in ``copilot`` SDK that records clients, sessions, and prompts."""

    import sys
    from dataclasses import dataclass
    from types import ModuleType, SimpleNamespace

    @dataclass
    class AssistantMessageData:
        content: str
        model: str | None = None
        reasoning_text: str | None = None
        api_call_id: str | None = None

    @dataclass
    class AssistantUsageData:
        model: str
        api_call_id: str | None = None
        finish_reason: str | None = None
        input_tokens: int | None = None
        output_tokens: int | None = None

    @dataclass
    class ModelLimitsOverride:
        max_output_tokens: int | None = None

    @dataclass
    class ModelCapabilitiesOverride:
        limits: ModelLimitsOverride | None = None

    state = SimpleNamespace(
        clients=[], sessions=[], prompts=[], reply="copilot reply", finish_reason="stop"
    )

    class FakeSession:
        def __init__(self):
            self.handlers = []

        def on(self, handler):
            self.handlers.append(handler)
            return lambda: self.handlers.remove(handler)

        async def send_and_wait(self, prompt, *, timeout):
            state.prompts.append((prompt, timeout))
            if isinstance(state.reply, Exception):
                raise state.reply
            events = [
                SimpleNamespace(
                    agent_id="sub-agent",
                    data=AssistantUsageData(model="other", finish_reason="stop"),
                ),
                SimpleNamespace(
                    agent_id=None,
                    data=AssistantUsageData(
                        model="served-model",
                        api_call_id="call-1",
                        finish_reason=state.finish_reason,
                        input_tokens=11,
                        output_tokens=7,
                    ),
                ),
            ]
            for event in events:
                for handler in self.handlers:
                    handler(event)
            return SimpleNamespace(
                data=AssistantMessageData(
                    content=state.reply, model="served-model", api_call_id="call-1"
                )
            )

    class CopilotClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.started = self.stopped = False
            state.clients.append(self)

        async def start(self):
            self.started = True

        async def stop(self):
            self.stopped = True

        async def list_models(self):
            return [SimpleNamespace(id="catalog-model"), SimpleNamespace(id="other-model")]

        async def create_session(self, **kwargs):
            state.sessions.append(kwargs)
            return FakeSession()

    modules = {
        "copilot": ModuleType("copilot"),
        "copilot.session": ModuleType("copilot.session"),
        "copilot.session_events": ModuleType("copilot.session_events"),
    }
    modules["copilot"].CopilotClient = CopilotClient
    modules["copilot.session"].ModelCapabilitiesOverride = ModelCapabilitiesOverride
    modules["copilot.session"].ModelLimitsOverride = ModelLimitsOverride
    modules["copilot.session_events"].AssistantMessageData = AssistantMessageData
    modules["copilot.session_events"].AssistantUsageData = AssistantUsageData
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return state


class TestCopilotProvider:
    def test_factory_requires_explicit_model_and_rejects_unsupported_settings(self):
        from matsim_agents.backends.llm.provider import ChatCopilot, get_chat_model

        model = get_chat_model(provider="copilot", model="catalog-model", max_tokens=8192)
        assert isinstance(model, ChatCopilot)
        assert model.max_completion_tokens == 8192
        assert model.github_token is None
        with pytest.raises(ValueError, match="explicit model ID"):
            get_chat_model(provider="copilot")
        with pytest.raises(ValueError, match="base_url"):
            get_chat_model(provider="copilot", model="catalog-model", base_url="http://x/v1")
        with pytest.raises(ValueError, match="sampling controls"):
            get_chat_model(provider="copilot", model="catalog-model", temperature=0.7)
        with pytest.raises(ValueError, match="sampling controls"):
            get_chat_model(provider="copilot", model="catalog-model", seed=1)

    def test_invoke_uses_tool_free_session_and_signed_in_account(
        self, fake_copilot_sdk, monkeypatch
    ):
        from langchain_core.messages import SystemMessage

        from matsim_agents.backends.llm.provider import get_chat_model

        monkeypatch.delenv("COPILOT_HOME", raising=False)
        model = get_chat_model(provider="copilot", model="catalog-model", max_tokens=8192)
        result = model.invoke([SystemMessage(content="Be terse."), HumanMessage(content="hi")])

        assert result.content == "copilot reply"
        assert result.response_metadata["model_name"] == "served-model"
        assert result.response_metadata["finish_reason"] == "stop"
        assert result.usage_metadata == {"input_tokens": 11, "output_tokens": 7, "total_tokens": 18}
        (client,) = fake_copilot_sdk.clients
        assert client.kwargs == {"github_token": None, "use_logged_in_user": True}
        assert client.started and client.stopped
        (session,) = fake_copilot_sdk.sessions
        assert session["model"] == "catalog-model"
        assert session["system_message"] == {"mode": "replace", "content": "Be terse."}
        assert session["model_capabilities"].limits.max_output_tokens == 8192
        assert session["tools"] == session["available_tools"] == []
        assert session["mcp_servers"] == {}
        assert session["included_builtin_agents"] == []
        assert fake_copilot_sdk.prompts == [("hi", 600)]

    def test_runtime_uses_explicit_copilot_home(self, fake_copilot_sdk, monkeypatch):
        from matsim_agents.backends.llm.provider import get_chat_model

        monkeypatch.setenv("COPILOT_HOME", "/tmp/copilot-state")
        model = get_chat_model(provider="copilot", model="catalog-model")
        model.invoke([HumanMessage(content="hi")])

        (client,) = fake_copilot_sdk.clients
        assert client.kwargs == {
            "github_token": None,
            "use_logged_in_user": True,
            "base_directory": "/tmp/copilot-state",
        }

    def test_explicit_token_disables_signed_in_account_and_is_not_shown(
        self, fake_copilot_sdk, monkeypatch
    ):
        from matsim_agents.backends.llm.provider import get_chat_model

        monkeypatch.delenv("COPILOT_HOME", raising=False)
        model = get_chat_model(provider="copilot", model="catalog-model", api_key="secret-token")
        assert "secret-token" not in repr(model)
        model.invoke([HumanMessage(content="hi")])

        (client,) = fake_copilot_sdk.clients
        assert client.kwargs == {"github_token": "secret-token", "use_logged_in_user": False}

    def test_multi_turn_history_is_rendered_and_runtime_stops_on_failure(self, fake_copilot_sdk):
        from matsim_agents.backends.llm.provider import get_chat_model

        model = get_chat_model(provider="copilot", model="catalog-model")
        model.invoke(
            [
                HumanMessage(content="one"),
                AIMessage(content="two"),
                HumanMessage(content="three"),
            ]
        )
        assert fake_copilot_sdk.prompts[0][0] == "User:\none\n\nAssistant:\ntwo\n\nUser:\nthree"
        assert fake_copilot_sdk.sessions[0]["system_message"]["content"]

        fake_copilot_sdk.reply = TimeoutError("no reply")
        with pytest.raises(TimeoutError):
            model.invoke([HumanMessage(content="hi")])
        assert fake_copilot_sdk.clients[-1].stopped

    def test_sync_and_async_calls_work_inside_an_event_loop(self, fake_copilot_sdk):
        import asyncio

        from matsim_agents.backends.llm.provider import get_chat_model

        model = get_chat_model(provider="copilot", model="catalog-model")

        async def call_both():
            sync_reply = model.invoke([HumanMessage(content="sync")])
            async_reply = await model.ainvoke([HumanMessage(content="async")])
            return sync_reply.content, async_reply.content

        assert asyncio.run(call_both()) == ("copilot reply", "copilot reply")

    def test_list_copilot_models_reads_account_catalog(self, fake_copilot_sdk):
        from matsim_agents.backends.llm.provider import list_copilot_models

        assert list_copilot_models() == ["catalog-model", "other-model"]
        assert fake_copilot_sdk.clients[0].stopped

    def test_token_limit_termination_is_reported_as_truncated(self, fake_copilot_sdk):
        from matsim_agents.backends.llm.provider import get_chat_model
        from matsim_agents.backends.llm.replies import invoke_model

        fake_copilot_sdk.finish_reason = "length"
        model = get_chat_model(provider="copilot", model="catalog-model", max_tokens=8)
        reply = invoke_model(model, [HumanMessage(content="hi")])

        assert reply.text == "copilot reply"
        assert reply.finish_reason == "length"
        assert reply.completion_tokens == 7
        assert reply.truncated
        assert reply.provider_metadata is None

    def test_unenforced_output_token_limit_overrun_is_flagged(self, fake_copilot_sdk):
        from matsim_agents.backends.llm.provider import get_chat_model
        from matsim_agents.backends.llm.replies import invoke_model

        model = get_chat_model(provider="copilot", model="catalog-model", max_tokens=4)
        reply = invoke_model(model, [HumanMessage(content="hi")])

        assert reply.provider_metadata == {"output_token_limit_exceeded": "7>4"}

    @pytest.mark.parametrize(
        "call",
        [
            lambda model, messages: model.invoke(messages, temperature=0.7),
            lambda model, messages: model.bind(seed=1).invoke(messages),
            lambda model, messages: model.invoke(messages, top_p=0.9),
        ],
    )
    def test_per_call_sampling_options_are_rejected_before_runtime_starts(
        self, fake_copilot_sdk, call
    ):
        import asyncio

        from matsim_agents.backends.llm.provider import get_chat_model

        model = get_chat_model(provider="copilot", model="catalog-model")
        messages = [HumanMessage(content="hi")]
        with pytest.raises(ValueError, match="no sampling controls"):
            call(model, messages)
        with pytest.raises(ValueError, match="no sampling controls"):
            asyncio.run(model.ainvoke(messages, temperature=0.7))
        assert fake_copilot_sdk.clients == []
        assert model.invoke(messages, temperature=0.0).content == "copilot reply"


# ── huggingface provider ──────────────────────────────────────────────────────


class _FakeTextGenerationPipeline:
    """Stand-in for a loaded transformers text-generation pipeline."""

    task = "text-generation"
    tokenizer = object()

    def __init__(self):
        self.calls: list[dict] = []


class _FakeHuggingFacePipeline:
    """Stand-in for ``langchain_huggingface.HuggingFacePipeline``."""

    loads: list[tuple[str, dict]] = []

    def __init__(
        self, pipeline=None, model_id=None, model_kwargs=None, pipeline_kwargs=None, batch_size=4
    ):
        self.pipeline = pipeline
        self.model_id = model_id
        self.model_kwargs = model_kwargs
        self.pipeline_kwargs = pipeline_kwargs
        self.batch_size = batch_size

    @classmethod
    def from_model_id(cls, model_id, task, device_map=None, **kwargs):
        cls.loads.append((model_id, {"task": task, "device_map": device_map, **kwargs}))
        return cls(pipeline=_FakeTextGenerationPipeline(), model_id=model_id, model_kwargs={})

    def _generate(self, prompts, stop=None, run_manager=None, **kwargs):
        self.pipeline.calls.append(kwargs["pipeline_kwargs"])
        return ["reply" for _ in prompts]

    def _stream(self, prompt, stop=None, run_manager=None, **kwargs):
        self.pipeline.calls.append(kwargs["pipeline_kwargs"])
        return iter(["reply"])


class TestHuggingFaceProvider:
    """The huggingface provider loads each model once and keeps per-caller settings."""

    @pytest.fixture(autouse=True)
    def _fake_module(self, monkeypatch, tmp_path):
        from matsim_agents.backends.llm import provider

        monkeypatch.setattr(provider, "_HUGGINGFACE_PIPELINES", {})
        _FakeHuggingFacePipeline.loads = []
        fake_chat = FakeListChatModel(responses=["2 + 2 equals 4."])
        module = MagicMock(HuggingFacePipeline=_FakeHuggingFacePipeline)
        module.ChatHuggingFace.return_value = fake_chat
        monkeypatch.setenv("MATSIM_HF_MODEL_PATH", str(tmp_path))
        with patch.dict("sys.modules", {"langchain_huggingface": module}):
            self.module = module
            self.model_path = str(tmp_path)
            yield

    def _llm(self, **kwargs):
        from matsim_agents.backends.llm.provider import get_chat_model

        get_chat_model(provider="huggingface", model=self.model_path, **kwargs)
        return self.module.ChatHuggingFace.call_args.kwargs["llm"]

    def test_loads_text_generation_pipeline_across_devices(self):
        self._llm()
        assert _FakeHuggingFacePipeline.loads == [
            (self.model_path, {"task": "text-generation", "device_map": "auto"})
        ]
        tokenizer = self.module.ChatHuggingFace.call_args.kwargs["tokenizer"]
        assert tokenizer is _FakeTextGenerationPipeline.tokenizer

    def test_model_is_loaded_once_per_process(self):
        first = self._llm(temperature=0.0)
        second = self._llm(temperature=0.7)
        assert len(_FakeHuggingFacePipeline.loads) == 1
        assert first.pipeline is second.pipeline

    def test_greedy_generation_settings(self):
        llm = self._llm(temperature=0.0)
        llm._generate(["prompt"])
        settings = llm.pipeline.calls[-1]
        assert settings["do_sample"] is False
        assert settings["return_full_text"] is False
        assert "temperature" not in settings

    def test_sampling_generation_settings_apply_per_caller(self):
        greedy = self._llm(temperature=0.0)
        sampling = self._llm(temperature=0.7, max_new_tokens=128)
        sampling._generate(["prompt"])
        settings = sampling.pipeline.calls[-1]
        assert settings["do_sample"] is True
        assert settings["temperature"] == pytest.approx(0.7)
        assert settings["max_new_tokens"] == 128
        assert "top_p" in settings
        greedy._stream("prompt")
        assert greedy.pipeline.calls[-1]["do_sample"] is False

    def test_call_time_settings_override_defaults(self):
        llm = self._llm(temperature=0.0)
        llm._generate(["prompt"], pipeline_kwargs={"max_new_tokens": 8})
        assert llm.pipeline.calls[-1]["max_new_tokens"] == 8
        assert llm.pipeline.calls[-1]["return_full_text"] is False

    def test_hf_inference_roundtrip(self):
        """End-to-end: mocked HF provider returns a valid AIMessage."""
        from matsim_agents.backends.llm.provider import get_chat_model

        model = get_chat_model(provider="huggingface", model=self.model_path)
        result = model.invoke([HumanMessage(content="What is 2 + 2?")])
        assert isinstance(result, AIMessage)
        assert result.content


# ── stub LLM end-to-end invocation ───────────────────────────────────────────


class TestStubLLMInference:
    """Verify that a FakeListChatModel works as a matsim-agents drop-in."""

    def test_invoke_returns_ai_message(self, fake_llm):
        result = fake_llm.invoke([HumanMessage(content="Hello")])
        assert isinstance(result, AIMessage)
        assert result.content == "OK"

    def test_multi_turn(self):
        llm = FakeListChatModel(responses=["First", "Second", "Third"])
        for expected in ["First", "Second", "Third"]:
            msg = llm.invoke([HumanMessage(content="q")])
            assert msg.content == expected
