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
        with patch("matsim_agents.llm.ChatOllama", mock_ollama, create=True), \
             patch.dict("sys.modules", {"langchain_ollama": MagicMock(ChatOllama=mock_ollama)}):
            from matsim_agents.llm import get_chat_model
            get_chat_model(provider="ollama", model="test:model")
            mock_ollama.assert_called_once()

    def test_vllm_provider(self):
        """ChatOpenAI is returned for the 'vllm' provider (OpenAI-compatible)."""
        mock_openai = MagicMock()
        with patch.dict("sys.modules", {"langchain_openai": MagicMock(ChatOpenAI=mock_openai)}):
            from matsim_agents.llm import get_chat_model
            get_chat_model(provider="vllm", model="Qwen/Qwen2.5-72B-Instruct",
                           base_url="http://localhost:8000/v1", api_key="EMPTY")
            mock_openai.assert_called_once()

    def test_openai_provider(self):
        mock_openai = MagicMock()
        with patch.dict("sys.modules", {"langchain_openai": MagicMock(ChatOpenAI=mock_openai)}):
            from matsim_agents.llm import get_chat_model
            get_chat_model(provider="openai", model="gpt-4o-mini")
            mock_openai.assert_called_once()

    def test_anthropic_provider(self):
        mock_anthropic = MagicMock()
        with patch.dict("sys.modules", {
            "langchain_anthropic": MagicMock(ChatAnthropic=mock_anthropic)
        }):
            from matsim_agents.llm import get_chat_model
            get_chat_model(provider="anthropic", model="claude-3-5-sonnet-latest")
            mock_anthropic.assert_called_once()

    def test_unknown_provider_raises(self):
        from matsim_agents.llm import get_chat_model
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            get_chat_model(provider="nonexistent_provider")

    def test_env_var_selects_provider(self, monkeypatch):
        """MATSIM_LLM_PROVIDER env var is respected."""
        monkeypatch.setenv("MATSIM_LLM_PROVIDER", "vllm")
        monkeypatch.setenv("MATSIM_VLLM_BASE_URL", "http://localhost:8000/v1")
        mock_openai = MagicMock()
        with patch.dict("sys.modules", {"langchain_openai": MagicMock(ChatOpenAI=mock_openai)}):
            from matsim_agents.llm import get_chat_model
            get_chat_model()
            mock_openai.assert_called_once()


# ── huggingface provider ──────────────────────────────────────────────────────

class TestHuggingFaceProvider:
    """The huggingface provider builds correctly when the pipeline is mocked."""

    def _make_hf_mocks(self):
        """Return patched langchain_huggingface module with FakeListChatModel stand-in."""
        fake_pipeline = MagicMock()
        fake_pipeline.from_model_id.return_value = fake_pipeline

        # ChatHuggingFace wraps the pipeline; we make it return a FakeListChatModel
        # so we can call .invoke() in test_hf_inference.
        fake_chat = FakeListChatModel(responses=["2 + 2 equals 4."])
        mock_module = MagicMock()
        mock_module.HuggingFacePipeline = fake_pipeline
        mock_module.ChatHuggingFace.return_value = fake_chat
        return mock_module, fake_pipeline, fake_chat

    def test_hf_provider_calls_from_model_id(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MATSIM_HF_MODEL_PATH", str(tmp_path))
        mock_module, fake_pipeline, _ = self._make_hf_mocks()
        with patch.dict("sys.modules", {"langchain_huggingface": mock_module}):
            from matsim_agents.llm import get_chat_model
            get_chat_model(provider="huggingface", model=str(tmp_path))
        fake_pipeline.from_model_id.assert_called_once()
        call_kwargs = fake_pipeline.from_model_id.call_args[1]
        assert call_kwargs.get("task") == "text-generation"
        assert call_kwargs.get("device_map") == "auto"

    def test_hf_pipeline_kwargs_no_sampling(self, monkeypatch, tmp_path):
        """With temperature=0, do_sample=False and temperature not in pipeline_kwargs."""
        monkeypatch.setenv("MATSIM_HF_MODEL_PATH", str(tmp_path))
        mock_module, fake_pipeline, _ = self._make_hf_mocks()
        with patch.dict("sys.modules", {"langchain_huggingface": mock_module}):
            from matsim_agents.llm import get_chat_model
            get_chat_model(provider="huggingface", model=str(tmp_path), temperature=0.0)
        kwargs = fake_pipeline.from_model_id.call_args[1]
        pkwargs = kwargs.get("pipeline_kwargs", {})
        assert pkwargs.get("do_sample") is False
        assert "temperature" not in pkwargs

    def test_hf_pipeline_kwargs_with_sampling(self, monkeypatch, tmp_path):
        """With temperature>0, do_sample=True and temperature/top_p included."""
        monkeypatch.setenv("MATSIM_HF_MODEL_PATH", str(tmp_path))
        mock_module, fake_pipeline, _ = self._make_hf_mocks()
        with patch.dict("sys.modules", {"langchain_huggingface": mock_module}):
            from matsim_agents.llm import get_chat_model
            get_chat_model(provider="huggingface", model=str(tmp_path), temperature=0.7)
        kwargs = fake_pipeline.from_model_id.call_args[1]
        pkwargs = kwargs.get("pipeline_kwargs", {})
        assert pkwargs.get("do_sample") is True
        assert pkwargs.get("temperature") == pytest.approx(0.7)
        assert "top_p" in pkwargs

    def test_hf_inference_roundtrip(self, monkeypatch, tmp_path):
        """End-to-end: mocked HF provider returns a valid AIMessage."""
        monkeypatch.setenv("MATSIM_HF_MODEL_PATH", str(tmp_path))
        mock_module, _, fake_chat = self._make_hf_mocks()
        with patch.dict("sys.modules", {"langchain_huggingface": mock_module}):
            from matsim_agents.llm import get_chat_model
            model = get_chat_model(provider="huggingface", model=str(tmp_path))

        result = model.invoke([HumanMessage(content="What is 2 + 2?")])
        assert isinstance(result, AIMessage)
        assert result.content  # non-empty


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
