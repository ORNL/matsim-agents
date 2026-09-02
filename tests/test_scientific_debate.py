from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from matsim_agents.workflows.debate import (
    DebateParticipant,
    ScientificDebateConfig,
    run_scientific_debate,
)


class _Model:
    def __init__(self, name: str):
        self.name = name
        self.prompts = []
        self.system_prompts = []

    def invoke(self, messages):
        self.system_prompts.append(messages[0].content)
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
    assert len(result.verdicts) == 3
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
    assert len(model_contributions) == 12
    assert len(identifiers) == len(set(identifiers))
    assert all(identifier for identifier in identifiers)
    assert len({models[name].system_prompts[0] for name in models}) == 1


def test_designated_synthesis_remains_available_for_role_based_debate(tmp_path):
    models = {}

    def factory(*, provider, model, base_url):
        models[model] = _Model(model)
        return models[model]

    result = run_scientific_debate(
        ScientificDebateConfig(
            hypothesis="Candidate X has high ZT.",
            output_root=str(tmp_path),
            debate_mode="role_based",
            synthesis_method="designated_model",
            synthesis_participant="theorist",
            participants=[
                DebateParticipant(name="theorist", provider="vllm", model="a", role="theorist"),
                DebateParticipant(
                    name="experimentalist",
                    provider="vllm",
                    model="b",
                    role="experimentalist",
                ),
            ],
        ),
        model_factory=factory,
    )
    assert len(result.verdicts) == 1
    assert result.verdicts[0].participant == "theorist"
    assert result.synthesis_turn_id == result.verdicts[0].contribution_id


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


@pytest.mark.parametrize(
    "name,mode,method,verdict_count",
    [
        ("equal-independent.yaml", "equal", "independent_verdicts", 2),
        ("equal-designated.yaml", "equal", "designated_model", 1),
        ("role-based-independent.yaml", "role_based", "independent_verdicts", 2),
        ("role-based-designated.yaml", "role_based", "designated_model", 1),
    ],
)
def test_documented_debate_modalities_match_the_config_contract(
    name, mode, method, verdict_count, tmp_path
):
    raw = yaml.safe_load((Path("examples/debate") / name).read_text())
    raw["output_root"] = str(tmp_path)
    config = ScientificDebateConfig.model_validate(raw)
    result = run_scientific_debate(
        config,
        model_factory=lambda *, provider, model, base_url: _Model(model),
    )
    assert config.debate_mode == mode
    assert config.synthesis_method == method
    assert len(result.verdicts) == verdict_count
