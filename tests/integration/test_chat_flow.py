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
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import HumanMessage

from matsim_agents.discovery.composition import Composition
from matsim_agents.discovery.wrapper import CompositionExplorationResult
from matsim_agents.orchestration.interactive_chat import (
    DiscoveryChatConfig,
    DiscoveryChatSession,
    chat_once,
)

# ── helpers ───────────────────────────────────────────────────────────────────


def _fake_exploration_result(comp: Composition, cfg=None, **_) -> CompositionExplorationResult:
    return CompositionExplorationResult(composition=comp, phase_candidates=[])


def _make_session(
    config: DiscoveryChatConfig,
    responses: list[str],
    monkeypatch,
) -> tuple[DiscoveryChatSession, MagicMock]:
    """Build a DiscoveryChatSession with a fake LLM and mocked _kickoff_exploration."""
    import matsim_agents.orchestration.interactive_chat as chat_mod

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


class TestLLMOnly:
    @pytest.mark.parametrize("custom_prompt", [None, "Custom: offer structure relaxation."])
    def test_prompt_matches_session_capabilities_after_clear(
        self, monkeypatch, custom_prompt, tmp_path
    ):
        from matsim_agents.orchestration.interactive_chat import (
            ATOMISTIC_EXPLORATION_GUIDANCE,
            LLM_ONLY_MODE_INSTRUCTION,
        )

        kwargs = {"system_prompt": custom_prompt} if custom_prompt else {}
        cfg = DiscoveryChatConfig(llm_only=True, output_dir=str(tmp_path), **kwargs)
        session, _ = _make_session(cfg, responses=["Hypothesis."], monkeypatch=monkeypatch)
        chat_once(session, "Discuss materials")
        initial_prompt = session.messages[0].content
        assert LLM_ONLY_MODE_INSTRUCTION in initial_prompt
        if custom_prompt:
            assert initial_prompt.startswith(custom_prompt)
        else:
            assert ATOMISTIC_EXPLORATION_GUIDANCE not in initial_prompt
        chat_once(session, "/clear")
        assert session.messages[0].content == initial_prompt

    def test_config_does_not_require_mlip_inputs(self):
        cfg = DiscoveryChatConfig(llm_only=True)

        assert cfg.llm_only is True
        assert cfg.trigger_active_learning_on_high_uq is False

    def test_formulas_do_not_trigger_atomistic_tools(self, monkeypatch, tmp_path):
        cfg = DiscoveryChatConfig(llm_only=True, auto_confirm=True, output_dir=str(tmp_path))
        session, mock_explore = _make_session(
            cfg,
            responses=["Li2MnO3 is a useful candidate."],
            monkeypatch=monkeypatch,
        )

        response = chat_once(session, "Compare LiCoO2 and Li2MnO3")

        assert "Li2MnO3" in response
        mock_explore.assert_not_called()
        assert session.seen_compositions == set()

    @pytest.mark.parametrize(
        ("command", "expected"),
        [
            ("/relax candidate.cif", "relaxation is unavailable"),
            ("/al Li2MnO3", "Active learning is unavailable"),
        ],
    )
    def test_atomistic_commands_are_rejected(self, command, expected, monkeypatch, tmp_path):
        import matsim_agents.orchestration.interactive_chat as chat_mod

        cfg = DiscoveryChatConfig(llm_only=True, output_dir=str(tmp_path))
        session = DiscoveryChatSession(config=cfg)
        relax = MagicMock()
        active_learning = MagicMock()
        monkeypatch.setattr(chat_mod, "_run_single_structure_relaxation", relax)
        monkeypatch.setattr(chat_mod, "_run_active_learning_for_formula", active_learning)

        response = chat_once(session, command)

        assert expected in response
        relax.assert_not_called()
        active_learning.assert_not_called()


# ── error resilience ──────────────────────────────────────────────────────────


class TestExplorationErrors:
    """Errors during _kickoff_exploration propagate (no silent catch in chat_once)."""

    def test_exploration_error_propagates(self, discovery_config, monkeypatch):
        import matsim_agents.orchestration.interactive_chat as chat_mod

        fake_llm = FakeListChatModel(responses=["Try Li2MnO3 as a cathode."])
        monkeypatch.setattr(chat_mod, "get_chat_model", lambda **_: fake_llm)

        def _boom(comp, cfg, **_):
            raise RuntimeError("Relaxation failed")

        monkeypatch.setattr(chat_mod, "_kickoff_exploration", _boom)

        session = DiscoveryChatSession(config=discovery_config)
        with pytest.raises(RuntimeError, match="Relaxation failed"):
            chat_once(session, "Suggest cathode")

    def test_an_interrupted_exploration_is_offered_again_on_resume(
        self, discovery_config, monkeypatch
    ):
        import matsim_agents.orchestration.interactive_chat as chat_mod

        session, _ = _make_session(
            discovery_config, responses=["Try Li2MnO3."], monkeypatch=monkeypatch
        )
        monkeypatch.setattr(
            chat_mod, "_kickoff_exploration", MagicMock(side_effect=KeyboardInterrupt)
        )
        with pytest.raises(KeyboardInterrupt):
            chat_once(session, "Suggest a cathode.")
        assert "Li2MnO3" not in session.seen_compositions

        _, mock_explore = _make_session(
            discovery_config, responses=["Li2MnO3 again."], monkeypatch=monkeypatch
        )
        resumed = DiscoveryChatSession.resume(discovery_config, session.transcript.directory)
        chat_once(resumed, "Retry.")
        assert [c.args[0].formula for c in mock_explore.call_args_list] == ["Li2MnO3"]
        assert len(resumed.explorations) == 1

    def test_an_interrupted_confirmation_is_offered_again(self, discovery_config, monkeypatch):
        import builtins

        discovery_config.auto_confirm = False
        session, mock_explore = _make_session(
            discovery_config, responses=["Try Li2MnO3."], monkeypatch=monkeypatch
        )

        def interrupt(_prompt):
            raise KeyboardInterrupt

        monkeypatch.setattr(builtins, "input", interrupt)
        with pytest.raises(KeyboardInterrupt):
            chat_once(session, "Suggest a cathode.")
        mock_explore.assert_not_called()
        resumed = DiscoveryChatSession.resume(discovery_config, session.transcript.directory)
        assert "Li2MnO3" not in resumed.seen_compositions


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


class TestResumableSession:
    def test_resume_restores_conversation_and_discovery_state(self, discovery_config, monkeypatch):
        session, _ = _make_session(
            discovery_config, responses=["Try Li2MnO3."], monkeypatch=monkeypatch
        )
        chat_once(session, "Suggest a cathode.")
        directory = session.transcript.directory
        assert (directory / "session.json").is_file()

        _, mock_explore = _make_session(
            discovery_config, responses=["Li2MnO3 again, or LiCoO2."], monkeypatch=monkeypatch
        )
        resumed = DiscoveryChatSession.resume(discovery_config, directory.name)
        assert [m.content for m in resumed.messages] == [m.content for m in session.messages]
        assert resumed.seen_compositions == session.seen_compositions
        assert len(resumed.explorations) == 1
        assert resumed.transcript.count == session.transcript.count

        chat_once(resumed, "Anything else?")
        assert mock_explore.call_count == 1  # only LiCoO2; Li2MnO3 was already seen
        assert len(resumed.explorations) == 2
        assert resumed.transcript.directory == directory

    @pytest.mark.parametrize("saved_llm_only", [True, False])
    def test_resume_uses_the_current_capabilities(
        self, discovery_config, monkeypatch, saved_llm_only
    ):
        from matsim_agents.orchestration.interactive_chat import LLM_ONLY_MODE_INSTRUCTION

        llm_only = DiscoveryChatConfig(llm_only=True, output_dir=discovery_config.output_dir)
        saved, current = (
            (llm_only, discovery_config) if saved_llm_only else (discovery_config, llm_only)
        )
        session, _ = _make_session(saved, responses=["Hypothesis."], monkeypatch=monkeypatch)
        chat_once(session, "Discuss materials")
        assert (LLM_ONLY_MODE_INSTRUCTION in session.messages[0].content) is saved_llm_only

        _make_session(current, responses=["More."], monkeypatch=monkeypatch)
        resumed = DiscoveryChatSession.resume(current, session.transcript.directory)
        chat_once(resumed, "Continue")
        assert (LLM_ONLY_MODE_INSTRUCTION in resumed.messages[0].content) is not saved_llm_only
        assert [m.content for m in resumed.messages[1:3]] == [
            m.content for m in session.messages[1:3]
        ]

    def test_resume_of_missing_session_fails(self, discovery_config):
        with pytest.raises(FileNotFoundError, match="no saved chat session"):
            DiscoveryChatSession.resume(discovery_config, "does-not-exist")

    def test_the_resume_hint_works_from_another_output_directory(
        self, discovery_config, monkeypatch, tmp_path
    ):
        import builtins
        import shlex
        from dataclasses import replace

        import matsim_agents.orchestration.interactive_chat as chat_mod

        _make_session(discovery_config, responses=["Hello."], monkeypatch=monkeypatch)
        printed = []
        monkeypatch.setattr(chat_mod, "_print", lambda text, *_a, **_k: printed.append(text))
        replies = iter(["Hi.", "exit"])
        monkeypatch.setattr(builtins, "input", lambda _prompt: next(replies))
        session = chat_mod.run_chat(discovery_config)
        [hint] = [text for text in printed if "Resume with" in text]
        [argument] = shlex.split(hint.split("--resume ", 1)[1].removesuffix("[/dim]"))
        assert Path(argument) == session.transcript.directory.resolve()
        elsewhere = replace(discovery_config, output_dir=str(tmp_path / "elsewhere"))
        resumed = DiscoveryChatSession.resume(elsewhere, argument)
        assert resumed.transcript.directory == Path(argument)


# ── multi-LLM critique / panel mode ──────────────────────────────────────────


class TestHypothesisDebate:
    def test_single_critic_revision_round(self, discovery_config, monkeypatch):
        import matsim_agents.orchestration.interactive_chat as chat_mod

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
        import matsim_agents.orchestration.interactive_chat as chat_mod

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


class TestTranscript:
    def test_session_transcript_records_every_message(self, discovery_config, monkeypatch):
        import matsim_agents.orchestration.interactive_chat as chat_mod

        cfg = discovery_config
        cfg.enable_hypothesis_debate = True
        cfg.llm_model = "primary-model"
        cfg.critic_llm_model = "critic-model"
        models = {
            "primary-model": _StubModel(
                "primary-model", outputs=["<think>plan</think>Draft.", "Revised."]
            ),
            "critic-model": _StubModel("critic-model", outputs=["Weak."]),
        }
        monkeypatch.setattr(chat_mod, "get_chat_model", lambda **kw: models[kw["model"]])

        session = DiscoveryChatSession(config=cfg)
        assert chat_once(session, "Suggest a hypothesis.") == "Revised."
        assert session.messages[-1].content == "Revised."
        chat_once(session, "/clear")

        records = session.transcript.contributions()
        assert [(r.id, r.speaker) for r in records] == [
            ("0001-user", "user"),
            ("0002-draft", "proposer"),
            ("0003-critique", "critic_1"),
            ("0004-revision", "proposer"),
            ("0005-user", "user"),
            ("0006-tool", "matsim-agents"),
        ]
        assert records[1].text == "Draft." and records[1].reasoning == "plan"
        assert records[2].model == "critic-model"
        directory = session.transcript.directory
        assert directory.parent == Path(cfg.output_dir) / "chat"
        assert sorted(p.stem for p in (directory / "prompts").iterdir()) == [
            "0002-draft",
            "0003-critique",
            "0004-revision",
        ]
        markdown = (directory / "transcript.md").read_text()
        assert markdown.startswith("# Discovery chat")
        assert "<summary>Reasoning (4 characters)</summary>" in markdown
