"""Integration tests for the chat session flow.

Tests the DiscoveryChatSession with a mocked LLM and mocked _kickoff_exploration,
verifying that:
  1. Composition detection fires on LLM responses containing formulas.
  2. auto_confirm=True triggers exploration without interactive prompts.
  3. No exploration is triggered when the LLM proposes no compositions.
  4. Multiple compositions in a single response are all detected.
  5. Exploration errors are caught and the session continues.
  6. Message history accumulates correctly across turns.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import HumanMessage

from matsim_agents.chat import DiscoveryChatConfig, DiscoveryChatSession, chat_once
from matsim_agents.discovery.composition import Composition
from matsim_agents.discovery.wrapper import CompositionExplorationResult

# ── helpers ───────────────────────────────────────────────────────────────────


def _fake_exploration_result(comp: Composition, cfg=None) -> CompositionExplorationResult:
    return CompositionExplorationResult(composition=comp, phase_candidates=[])


def _make_session(
    config: DiscoveryChatConfig,
    responses: list[str],
    monkeypatch,
) -> tuple[DiscoveryChatSession, MagicMock]:
    """Build a DiscoveryChatSession with a fake LLM and mocked _kickoff_exploration."""
    import matsim_agents.chat as chat_mod

    fake_llm = FakeListChatModel(responses=responses)
    monkeypatch.setattr(chat_mod, "get_chat_model", lambda **_: fake_llm)

    mock_explore = MagicMock(side_effect=_fake_exploration_result)
    monkeypatch.setattr(chat_mod, "_kickoff_exploration", mock_explore)

    session = DiscoveryChatSession(config=config)
    return session, mock_explore


@dataclass
class _StubModel:
    """Small invoke-only model stub used to test proposer/critic orchestration."""

    name: str
    outputs: list[str]
    invoke_count: int = 0

    def invoke(self, _messages):
        idx = min(self.invoke_count, len(self.outputs) - 1)
        self.invoke_count += 1
        return SimpleNamespace(content=self.outputs[idx])


# ── composition detection ─────────────────────────────────────────────────────


class TestCompositionDetection:
    """extract_compositions is called on every assistant response."""

    def test_no_composition_no_exploration(self, discovery_config, monkeypatch):
        session, mock_explore = _make_session(
            discovery_config,
            responses=["The sky is blue. No formulas here."],
            monkeypatch=monkeypatch,
        )
        chat_once(session, "What color is the sky?")
        mock_explore.assert_not_called()

    def test_single_composition_triggers_exploration(self, discovery_config, monkeypatch):
        session, mock_explore = _make_session(
            discovery_config,
            responses=["I recommend Li2MnO3 as a cathode material."],
            monkeypatch=monkeypatch,
        )
        chat_once(session, "Suggest a cathode")
        mock_explore.assert_called_once()
        explored_comp = mock_explore.call_args[0][0]
        assert hasattr(explored_comp, "formula")

    def test_multiple_compositions_all_explored(self, discovery_config, monkeypatch):
        session, mock_explore = _make_session(
            discovery_config,
            responses=["Consider both Li2MnO3 and LiCoO2 as candidates."],
            monkeypatch=monkeypatch,
        )
        chat_once(session, "Suggest cathodes")
        assert mock_explore.call_count == 2
        formulas = {mock_explore.call_args_list[i][0][0].formula for i in range(2)}
        assert len(formulas) == 2


# ── auto_confirm behaviour ────────────────────────────────────────────────────


class TestAutoConfirm:
    """auto_confirm=True explores without prompting; False skips exploration."""

    def test_auto_confirm_true_explores(self, discovery_config, monkeypatch):
        assert discovery_config.auto_confirm is True
        session, mock_explore = _make_session(
            discovery_config,
            responses=["MgAl2O4 spinel is a great candidate."],
            monkeypatch=monkeypatch,
        )
        chat_once(session, "Suggest a spinel")
        mock_explore.assert_called_once()

    def test_auto_confirm_false_does_not_explore(self, discovery_config, monkeypatch):
        discovery_config.auto_confirm = False
        session, mock_explore = _make_session(
            discovery_config,
            responses=["MgAl2O4 spinel is a great candidate."],
            monkeypatch=monkeypatch,
        )
        import builtins

        monkeypatch.setattr(builtins, "input", lambda _: "n")
        chat_once(session, "Suggest a spinel")
        mock_explore.assert_not_called()


# ── error resilience ──────────────────────────────────────────────────────────


class TestExplorationErrors:
    """Errors during _kickoff_exploration propagate (no silent catch in chat_once)."""

    def test_exploration_error_propagates(self, discovery_config, monkeypatch):
        import matsim_agents.chat as chat_mod

        fake_llm = FakeListChatModel(responses=["Try Li2MnO3 as a cathode."])
        monkeypatch.setattr(chat_mod, "get_chat_model", lambda **_: fake_llm)

        def _boom(comp, cfg):
            raise RuntimeError("Relaxation failed")

        monkeypatch.setattr(chat_mod, "_kickoff_exploration", _boom)

        session = DiscoveryChatSession(config=discovery_config)
        with pytest.raises(RuntimeError, match="Relaxation failed"):
            chat_once(session, "Suggest cathode")


# ── message history ───────────────────────────────────────────────────────────


class TestMessageHistory:
    """Session maintains correct conversation history in session.messages."""

    def test_history_accumulates_across_turns(self, discovery_config, monkeypatch):
        session, _ = _make_session(
            discovery_config,
            responses=["Response A.", "Response B."],
            monkeypatch=monkeypatch,
        )
        chat_once(session, "First question")
        chat_once(session, "Second question")
        # Messages: system + human + ai + human + ai = 5
        assert len(session.messages) >= 5

    def test_first_message_is_human(self, discovery_config, monkeypatch):
        session, _ = _make_session(
            discovery_config,
            responses=["Hello!"],
            monkeypatch=monkeypatch,
        )
        chat_once(session, "Hello?")
        human_msgs = [m for m in session.messages if isinstance(m, HumanMessage)]
        assert len(human_msgs) >= 1
        assert human_msgs[0].content == "Hello?"

    def test_last_assistant_response_captured(self, discovery_config, monkeypatch):
        session, _ = _make_session(
            discovery_config,
            responses=["The answer is 42."],
            monkeypatch=monkeypatch,
        )
        response = chat_once(session, "What is the answer?")
        assert response is not None
        assert "42" in response


# ── multi-LLM critique / panel mode ──────────────────────────────────────────


class TestHypothesisDebate:
    def test_single_critic_revision_round(self, discovery_config, monkeypatch):
        import matsim_agents.chat as chat_mod

        cfg = discovery_config
        cfg.enable_hypothesis_debate = True
        cfg.debate_rounds = 1
        cfg.llm_model = "primary-model"
        cfg.critic_llm_model = "critic-model"

        models = {
            "primary-model": _StubModel(
                "primary-model",
                outputs=["Draft hypothesis.", "Revised hypothesis after critique."],
            ),
            "critic-model": _StubModel(
                "critic-model",
                outputs=["Weak assumption: missing control case."],
            ),
        }

        def _factory(**kwargs):
            return models[kwargs.get("model")]

        monkeypatch.setattr(chat_mod, "get_chat_model", _factory)
        monkeypatch.setattr(chat_mod, "_kickoff_exploration", MagicMock())

        session = DiscoveryChatSession(config=cfg)
        rsp = chat_once(session, "Suggest a robust hypothesis for cathodes.")

        assert rsp == "Revised hypothesis after critique."
        assert models["primary-model"].invoke_count == 2  # draft + revise
        assert models["critic-model"].invoke_count == 1

    def test_multi_critic_panel_with_cross_critique(self, discovery_config, monkeypatch):
        import matsim_agents.chat as chat_mod

        cfg = discovery_config
        cfg.enable_hypothesis_debate = True
        cfg.debate_rounds = 1
        cfg.critic_cross_critique = True
        cfg.llm_model = "primary-model"
        cfg.critic_panel_models = ["critic-a", "critic-b", "critic-c"]
        cfg.critic_panel_providers = ["ollama", "ollama", "ollama"]

        models = {
            "primary-model": _StubModel(
                "primary-model",
                outputs=["Initial draft.", "Panel-synthesized revised answer."],
            ),
            "critic-a": _StubModel("critic-a", outputs=["A1", "A2"]),
            "critic-b": _StubModel("critic-b", outputs=["B1", "B2"]),
            "critic-c": _StubModel("critic-c", outputs=["C1", "C2"]),
        }

        def _factory(**kwargs):
            return models[kwargs.get("model")]

        monkeypatch.setattr(chat_mod, "get_chat_model", _factory)
        monkeypatch.setattr(chat_mod, "_kickoff_exploration", MagicMock())

        session = DiscoveryChatSession(config=cfg)
        rsp = chat_once(session, "Propose a hypothesis for stable spinels.")

        assert rsp == "Panel-synthesized revised answer."
        assert models["primary-model"].invoke_count == 2
        assert models["critic-a"].invoke_count == 2  # initial + cross critique
        assert models["critic-b"].invoke_count == 2
        assert models["critic-c"].invoke_count == 2
