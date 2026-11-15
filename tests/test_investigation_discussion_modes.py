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
    build_hypothesis_from_discussion,
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
                    "response": f"Investigate {self.composition}.",
                    "hypothesis": f"{self.composition} may provide high thermoelectric ZT.",
                    "proposed_compositions": [self.composition],
                    "scientific_rationale": f"Test transport and stability of {self.composition}.",
                    "assumptions": ["Comparable synthesis conditions"],
                    "scientific_hypotheses": [
                        {
                            "id": "thermoelectric-claim",
                            "claim": f"{self.composition} may provide high thermoelectric ZT.",
                            "prediction": "Transport measurements will show enhanced ZT.",
                            "status": "proposed",
                            "reason": "No numerical evidence has been supplied.",
                            "evidence_ids": ["investigation-objective"],
                        }
                    ],
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


def test_agentic_investigation_can_use_single_llm_discussion(tmp_path, monkeypatch):
    monkeypatch.setenv("FRONTIER_API_KEY", "test-secret")
    factory_calls = []

    def model_factory(**kwargs):
        factory_calls.append(kwargs)
        return _ProposalModel("SnSe")

    config = InvestigationConfig(
        objective="Find a high-ZT material.",
        output_root=str(tmp_path),
        hypothesis_discussion=HypothesisDiscussionConfig(
            mode="single_llm",
            single_llm=SingleLLMDiscussionConfig(
                provider="openai",
                model="frontier-model",
                base_url="https://api.example/v1",
                max_completion_tokens=2048,
                api_key_env="FRONTIER_API_KEY",
            ),
        ),
    )
    result = run_investigation(
        config,
        phase_runner=_phase_result,
        model_factory=model_factory,
    )
    assert result.hypothesis.discussion_mode == "single_llm"
    assert result.hypothesis.proposed_compositions == ["SnSe"]
    assert factory_calls == [
        {
            "provider": "openai",
            "model": "frontier-model",
            "base_url": "https://api.example/v1",
            "api_key": "test-secret",
            "max_tokens": 2048,
        }
    ]
    assert "test-secret" not in result.model_dump_json()
    assert [exploration.composition for exploration in result.explorations] == ["SnSe"]


def test_single_llm_followup_preserves_hypothesis_lineage_and_evidence(tmp_path):
    previous_state = [
        {
            "id": "thermoelectric-claim",
            "claim": "SnSe may have high thermoelectric ZT.",
            "prediction": "Transport measurements will show enhanced ZT.",
            "status": "unresolved",
            "reason": "Awaiting a DFT band-gap comparison.",
            "evidence_ids": ["investigation-objective"],
        }
    ]
    evidence = {
        "evidence_id": "dft-gap-001",
        "evidence_level": "dft",
        "outcome": "measured",
        "method": {"backend": "Quantum Espresso"},
        "values": [{"name": "band_gap", "value": 0.9, "unit": "eV"}],
    }
    mlip_evidence = {
        "evidence_id": "mlip-ranking-001",
        "evidence_level": "mlip_relaxation",
        "outcome": "measured",
        "method": {"backend": "MACE"},
        "values": [{"name": "energy_above_hull", "value": 0.02, "unit": "eV/atom"}],
    }
    raw_reply = json.dumps(
        {
            "hypothesis": "SnSe may retain high thermoelectric ZT.",
            "proposed_compositions": ["SnSe"],
            "scientific_rationale": "The measured DFT gap is consistent with the target.",
            "assumptions": [],
            "scientific_hypotheses": [
                {
                    **previous_state[0],
                    "status": "supported_by_dft",
                    "reason": "The supplied DFT band-gap result supports this limited claim.",
                    "evidence_ids": ["dft-gap-001"],
                },
                {
                    "id": "mlip-energy-ranking",
                    "claim": "The MLIP ranking may identify the lower-energy polymorph.",
                    "prediction": "The converged MLIP relaxation will rank polymorph A lower.",
                    "status": "supported_by_mlip",
                    "reason": (
                        "The supplied measured MLIP energy ranking supports this limited claim."
                    ),
                    "evidence_ids": ["mlip-ranking-001"],
                },
            ],
        }
    )
    prompts = []

    class FollowUpModel:
        def invoke(self, messages):
            prompts.extend(messages)
            return SimpleNamespace(content=raw_reply)

    hypothesis = build_hypothesis_from_discussion(
        "Find a high-ZT material.",
        HypothesisDiscussionConfig(
            mode="single_llm",
            single_llm=SingleLLMDiscussionConfig(provider="vllm", model="proposal"),
        ),
        output_root=str(tmp_path),
        previous={
            "hypothesis": {"scientific_hypotheses": previous_state},
            "evidence_records": [evidence, mlip_evidence],
            "explorations": [
                {
                    "composition": {"formula": "SnSe"},
                    "initial": {
                        "relaxations": [
                            {
                                "converged": True,
                                "final_energy_eV": -1.25,
                                "final_max_force_eV_per_A": 0.01,
                            },
                            {"converged": False, "final_energy_eV": -1.0},
                        ]
                    },
                }
            ],
        },
        model_factory=lambda **_kwargs: FollowUpModel(),
    )

    assert hypothesis.scientific_hypotheses[0]["id"] == "thermoelectric-claim"
    assert hypothesis.scientific_hypotheses[0]["status"] == "supported_by_dft"
    assert hypothesis.scientific_hypotheses[1]["status"] == "supported_by_mlip"
    assert "thermoelectric-claim" in prompts[-1].content
    assert "dft-gap-001" in prompts[-1].content
    assert "prior-mlip-" in prompts[-1].content
    assert "measured converged MLIP relaxation; not DFT" in prompts[-1].content
    audit = tmp_path / hypothesis.discussion_run_id
    assert json.loads((audit / "contributions.jsonl").read_text())["text"] == raw_reply
    saved_prompt = json.loads((audit / "prompts" / "proposal-001.json").read_text())
    assert "Previous investigation results and evidence" in saved_prompt[1]["content"]


def test_invalid_single_llm_reply_is_audited_before_it_fails(tmp_path):
    class InvalidModel:
        def invoke(self, _messages):
            return SimpleNamespace(content="not JSON")

    with pytest.raises(json.JSONDecodeError):
        build_hypothesis_from_discussion(
            "Find a high-ZT material.",
            HypothesisDiscussionConfig(
                mode="single_llm",
                single_llm=SingleLLMDiscussionConfig(provider="vllm", model="proposal"),
            ),
            output_root=str(tmp_path),
            model_factory=lambda **_kwargs: InvalidModel(),
        )

    [run] = list(tmp_path.glob("hypothesis-*"))
    [contribution] = (run / "contributions.jsonl").read_text().splitlines()
    assert json.loads(contribution)["text"] == "not JSON"
    assert (run / "prompts" / "proposal-001.json").is_file()
    events = [json.loads(line)["event"] for line in (run / "events.jsonl").read_text().splitlines()]
    assert "hypothesis_validation_failed" in events


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
        model_factory=lambda *, provider, model, base_url, max_tokens: models[model],
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
