from __future__ import annotations

import json

import pytest

from benchmarks.e3.mlip_exploration import (
    _llm_action_prompt,
    load_action_history,
    run_generation_round,
    select_llm_guided_actions,
    select_rule_based_actions,
)
from matsim_agents.backends.mlip.results import RelaxationResult
from matsim_agents.discovery.composition import parse_composition
from matsim_agents.discovery.seeds import SeedGenerationAction
from matsim_agents.discovery.wrapper import CompositionExplorationResult


def _action(action_id, formula, formula_units, **kwargs):
    return SeedGenerationAction(
        action_id=action_id,
        formula=formula,
        formula_units=formula_units,
        **kwargs,
    )


def _history_record(action):
    return {
        "action_id": action.action_id,
        "action": action.model_dump(mode="json"),
        "evidence_id": f"evidence-{action.action_id}",
        "formula": action.formula,
        "formula_units": action.formula_units,
        "generated": 4,
        "attempted": 4,
        "converged": 3,
        "failed": 1,
        "backend_errors": 0,
        "nonconverged": 1,
        "unrelaxed": 0,
        "outcome_class": "usable_minimum",
    }


def test_rule_policy_is_deterministic_and_prioritizes_coverage():
    prior = _action("prior", "Si", 4, max_aflow_candidates=4)
    larger = _action("larger", "Si", 16, max_aflow_candidates=4)
    other_composition = _action("other", "NaCl", 4, pyxtal_count=4)
    history = [_history_record(prior)]

    chosen = select_rule_based_actions([prior, larger, other_composition], history, n_actions=1)

    assert chosen == [other_composition]
    assert (
        select_rule_based_actions([prior, larger, other_composition], history, n_actions=1)
        == chosen
    )


def test_action_history_excludes_current_and_future_rounds(tmp_path):
    strategy_dir = tmp_path / "rule_based_adaptation"
    for round_index in (1, 2, 3):
        round_dir = strategy_dir / f"round-{round_index:02d}"
        round_dir.mkdir(parents=True)
        action = _action(f"action-{round_index}", "Si", 4, max_aflow_candidates=2)
        record = _history_record(action)
        record["evidence_id"] = f"round-{round_index}"
        (round_dir / "round-manifest.json").write_text(
            json.dumps(
                {
                    "strategy": "rule_based_adaptation",
                    "status": "complete",
                    "actions": [{"status": "complete", **record}],
                }
            )
        )

    history = load_action_history(tmp_path, "rule_based_adaptation", before_round_index=3)

    assert [record["evidence_id"] for record in history] == ["round-1", "round-2"]


def test_llm_policy_enforces_allowlist_and_evidence_citations():
    first = _action("first", "Si", 4, max_aflow_candidates=2)
    second = _action("second", "Si", 8, pyxtal_count=2)

    class FakeChat:
        model_id = "test-model"

        def __init__(self, selection, evidence_ids):
            self.selection = selection
            self.evidence_ids = evidence_ids

        def invoke(self, prompt):
            assert "DFT evidence" in prompt
            return type(
                "Reply",
                (),
                {
                    "content": json.dumps(
                        {
                            "selected_action_ids": self.selection,
                            "cited_evidence_ids": self.evidence_ids,
                            "rationale": "The larger size tier is underexplored.",
                        }
                    )
                },
            )()

    selected, cited, rationale = select_llm_guided_actions(
        FakeChat(["second"], ["campaign-design"]),
        [first, second],
        [],
        n_actions=1,
    )
    assert selected == [second]
    assert cited == ["campaign-design"]
    assert rationale

    selected, cited, _ = select_llm_guided_actions(
        FakeChat(["second"], ["evidence-first"]),
        [first, second],
        [_history_record(first)],
        n_actions=1,
    )
    assert selected == [second]
    assert cited == ["evidence-first"]
    assert '"nonconverged": 1' in _llm_action_prompt(
        [second], [_history_record(first)], n_actions=1
    )

    with pytest.raises(ValueError, match="outside the allow-listed"):
        select_llm_guided_actions(
            FakeChat(["not-allowed"], ["campaign-design"]),
            [first, second],
            [],
            n_actions=1,
        )


def test_generation_round_uses_explicit_actions_and_mace_medium(tmp_path, monkeypatch):
    action = SeedGenerationAction(
        action_id="fu4-aflow12-pyxtal3",
        formula="NaCl",
        formula_units=4,
        max_aflow_candidates=12,
        pyxtal_count=3,
        random_seed=19,
    )
    calls = []

    def fake_explore(formula, **kwargs):
        calls.append((formula, kwargs))
        return CompositionExplorationResult(
            composition=parse_composition(formula),
            phase_candidates=[],
            relaxations=[
                RelaxationResult(
                    structure_path="seed.extxyz",
                    optimized_structure_path="relaxed.extxyz",
                    trajectory_path="trajectory.traj",
                    log_csv_path="log.csv",
                    final_energy_eV=-1.0,
                    final_max_force_eV_per_A=0.1,
                    num_steps=200,
                    converged=False,
                )
            ],
            outcome_class="generation_failure",
        )

    monkeypatch.setattr("benchmarks.e3.mlip_exploration.explore_composition", fake_explore)
    results = run_generation_round(
        "llm_guided_adaptation",
        2,
        [action],
        tmp_path,
        mlp_device="cuda",
        fmax=0.03,
    )

    assert len(results) == 1
    formula, kwargs = calls[0]
    assert formula == action.formula
    assert kwargs["generation_action"] == action
    assert kwargs["mlip_backend"] == "mace"
    assert kwargs["mace_model"] == "medium-omat-0"
    assert kwargs["maxiter"] == 200
    assert kwargs["fmax"] == 0.03
    assert kwargs["relax_cell"] is False
    assert "max_relaxations" not in kwargs

    round_dir = tmp_path / "llm_guided_adaptation" / "round-02"
    manifest = json.loads((round_dir / "round-manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["actions"][0]["aflow_candidate_cap"] == 12
    assert manifest["actions"][0]["pyxtal_requested"] == 3
    assert manifest["actions"][0]["failed"] == 1
    assert manifest["actions"][0]["nonconverged"] == 1
    saved_result = round_dir / manifest["actions"][0]["result_path"]
    assert json.loads(saved_result.read_text())["relaxations"][0]["converged"] is False


def test_generation_round_selects_rule_actions_from_strategy_local_history(tmp_path, monkeypatch):
    actions = [
        _action("small", "Si", 4, max_aflow_candidates=2),
        _action("large", "Si", 16, max_aflow_candidates=2),
    ]
    calls = []

    def fake_explore(formula, **kwargs):
        calls.append(kwargs["generation_action"].action_id)
        return CompositionExplorationResult(
            composition=parse_composition(formula),
            phase_candidates=[],
            outcome_class="generation_failure",
        )

    monkeypatch.setattr("benchmarks.e3.mlip_exploration.explore_composition", fake_explore)

    run_generation_round(
        "rule_based_adaptation",
        1,
        None,
        tmp_path,
        action_space=actions,
        n_actions=1,
    )
    run_generation_round(
        "rule_based_adaptation",
        2,
        None,
        tmp_path,
        action_space=actions,
        n_actions=1,
    )

    assert calls == ["large", "small"]
    second_manifest = json.loads(
        (tmp_path / "rule_based_adaptation" / "round-02" / "round-manifest.json").read_text()
    )
    prior_evidence_id = second_manifest["selection"]["cited_evidence_ids"][1]
    assert prior_evidence_id != "campaign-design"
    assert len(prior_evidence_id) == 16


def test_generation_round_rejects_duplicate_action_ids(tmp_path):
    action = SeedGenerationAction(
        action_id="same",
        formula="Si",
        formula_units=4,
        max_aflow_candidates=1,
    )
    with pytest.raises(ValueError, match="IDs must be unique"):
        run_generation_round(
            "rule_based_adaptation",
            1,
            [action, action],
            tmp_path,
        )
    assert not (tmp_path / "rule_based_adaptation").exists()
