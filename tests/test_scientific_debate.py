from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from matsim_agents.workflows.debate import (
    DebateParticipant,
    ScientificDebateConfig,
    run_scientific_debate,
)


class _Model:
    def __init__(self, name: str):
        self.name = name
        self.prompts = []

    def invoke(self, messages):
        self.prompts.append(messages[-1].content)
        return SimpleNamespace(content=f"argument from {self.name} call {len(self.prompts)}")


def test_models_debate_each_other_for_user_selected_rounds(tmp_path):
    models = {}

    def factory(*, provider, model, base_url):
        models[model] = _Model(model)
        return models[model]

    cfg = ScientificDebateConfig(
        hypothesis="A metastable Si phase is stabilized by pressure.",
        rounds=3,
        output_root=str(tmp_path),
        participants=[
            DebateParticipant(name="theorist", provider="vllm", model="model-a"),
            DebateParticipant(name="skeptic", provider="ollama", model="model-b"),
            DebateParticipant(name="experimentalist", provider="openai", model="model-c"),
        ],
    )
    result = run_scientific_debate(cfg, model_factory=factory)
    assert result.rounds_completed == 3
    assert len(result.turns) == 9
    assert [turn.participant for turn in result.turns[:6]] == [
        "theorist",
        "skeptic",
        "experimentalist",
        "skeptic",
        "experimentalist",
        "theorist",
    ]
    assert "argument from model-a" in models["model-b"].prompts[-1]
    transcript = json.loads((tmp_path / result.run_id / "debate_transcript.json").read_text())
    assert len(transcript["turns"]) == 9
    assert transcript["synthesis"] == result.synthesis
    dialogue = json.loads((tmp_path / result.run_id / "dialogue.json").read_text())
    model_contributions = [entry for entry in dialogue if entry["kind"].startswith("model_")]
    identifiers = [entry["contribution_id"] for entry in model_contributions]
    assert len(model_contributions) == 10
    assert len(identifiers) == len(set(identifiers))
    assert all(identifier for identifier in identifiers)


def test_debate_rejects_duplicate_names_and_unknown_synthesizer():
    participant = DebateParticipant(name="same", provider="vllm", model="a")
    with pytest.raises(ValueError, match="unique"):
        ScientificDebateConfig(hypothesis="h", participants=[participant, participant])
    with pytest.raises(ValueError, match="synthesis_participant"):
        ScientificDebateConfig(
            hypothesis="h",
            participants=[
                participant,
                DebateParticipant(name="other", provider="vllm", model="b"),
            ],
            synthesis_participant="missing",
        )
