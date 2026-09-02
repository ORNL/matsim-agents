from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from matsim_agents.discovery.composition import parse_composition
from matsim_agents.discovery.wrapper import CompositionExplorationResult
from matsim_agents.workflows.debate import DebateParticipant
from matsim_agents.workflows.investigation import (
    HypothesisDiscussionConfig,
    InvestigationConfig,
    MultiLLMDebateDiscussionConfig,
    SingleLLMDiscussionConfig,
    run_investigation,
)
from matsim_agents.workflows.phase_exploration import PhaseExplorationWorkflowResult


class _ProposalModel:
    def __init__(self, composition: str):
        self.composition = composition
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        return SimpleNamespace(
            content=json.dumps(
                {
                    "hypothesis": f"{self.composition} may provide high thermoelectric ZT.",
                    "proposed_compositions": [self.composition],
                    "scientific_rationale": f"Test transport and stability of {self.composition}.",
                    "assumptions": ["Comparable synthesis conditions"],
                }
            )
        )


def _phase_result(composition, _policy, _directory):
    parsed = parse_composition(composition)
    assert parsed is not None
    return PhaseExplorationWorkflowResult(
        composition=composition,
        initial=CompositionExplorationResult(composition=parsed, phase_candidates=[]),
    )


def test_agentic_investigation_can_use_single_llm_discussion(tmp_path):
    config = InvestigationConfig(
        objective="Find a high-ZT material.",
        output_root=str(tmp_path),
        hypothesis_discussion=HypothesisDiscussionConfig(
            mode="single_llm",
            single_llm=SingleLLMDiscussionConfig(provider="vllm", model="single"),
        ),
    )
    result = run_investigation(
        config,
        phase_runner=_phase_result,
        model_factory=lambda **_: _ProposalModel("SnSe"),
    )
    assert result.hypothesis.discussion_mode == "single_llm"
    assert result.hypothesis.proposed_compositions == ["SnSe"]
    assert [exploration.composition for exploration in result.explorations] == ["SnSe"]


def test_agentic_investigation_can_use_equal_multi_llm_debate(tmp_path):
    models = {"model-a": _ProposalModel("SnSe"), "model-b": _ProposalModel("Mg3Sb2")}
    config = InvestigationConfig(
        objective="Find a stable, abundant high-ZT material near 800 K.",
        output_root=str(tmp_path),
        hypothesis_discussion=HypothesisDiscussionConfig(
            mode="multi_llm_debate",
            multi_llm_debate=MultiLLMDebateDiscussionConfig(
                rounds=2,
                debate_mode="equal",
                synthesis_method="independent_verdicts",
                participants=[
                    DebateParticipant(name="a", provider="vllm", model="model-a"),
                    DebateParticipant(name="b", provider="vllm", model="model-b"),
                ],
            ),
        ),
    )
    result = run_investigation(
        config,
        phase_runner=_phase_result,
        model_factory=lambda *, provider, model, base_url: models[model],
    )
    assert result.hypothesis.discussion_mode == "multi_llm_debate"
    assert result.hypothesis.discussion_run_id
    assert result.hypothesis.proposed_compositions == ["SnSe", "Mg3Sb2"]
    assert len(result.hypothesis.source_contribution_ids) == 2
    assert [exploration.composition for exploration in result.explorations] == [
        "SnSe",
        "Mg3Sb2",
    ]
    assert all(model.calls == 3 for model in models.values())


@pytest.mark.parametrize(
    "name,mode",
    [("single_llm.yaml", "single_llm"), ("multi_llm_debate.yaml", "multi_llm_debate")],
)
def test_documented_investigation_discussion_configs_are_valid(name, mode, tmp_path):
    raw = yaml.safe_load((Path("examples/investigation") / name).read_text())
    raw["output_root"] = str(tmp_path)
    config = InvestigationConfig.model_validate(raw)
    assert config.hypothesis_discussion is not None
    assert config.hypothesis_discussion.mode == mode
