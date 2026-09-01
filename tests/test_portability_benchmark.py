from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.portability.compare import compare_runs
from benchmarks.portability.run import (
    build_plan,
    resolved_configuration,
    validate_inputs,
    validate_llm_check,
)
from benchmarks.portability.suites import (
    active_learning_contract,
    investigation_contract,
    relaxation_contract,
)
from benchmarks.portability.validate import validate_run


@pytest.mark.parametrize("facility", ["frontier", "aurora", "perlmutter"])
def test_facility_overlays_preserve_scientific_configuration(facility):
    config = resolved_configuration(facility)
    assert config["science"]["active_learning"]["retrain"] is False
    assert config["science"]["active_learning"]["promote_model"] is False
    assert config["deployment"]["facility"] == facility
    assert validate_inputs(config) == []


def test_fixed_active_learning_plan_is_small_and_non_mutating():
    config = resolved_configuration("frontier")
    plan = build_plan(config, "active-learning", "qe")
    assert plan == [
        {
            "stage": "active-learning",
            "fixed_candidate_pool": True,
            "candidate_count": 4,
            "selected_count": 2,
            "retrain": False,
            "promote_model": False,
            "backend": "qe",
        }
    ]


def test_llm_discussion_plan_has_proposal_critique_and_revision():
    plan = build_plan(resolved_configuration("aurora"), "llm-discussion", "qe")
    assert plan == [
        {
            "stage": "llm-discussion",
            "turns": ["proposal", "critique", "revision"],
            "phase_dispatch": True,
        }
    ]


def test_executable_scientific_contracts(tmp_path):
    structure = Path("benchmarks/portability/structures/Si.vasp")
    relaxation = relaxation_contract(structure, tmp_path / "relaxation")
    active_learning = active_learning_contract(tmp_path / "active-learning")
    investigation = investigation_contract(tmp_path / "investigation")
    assert relaxation["converged"] is True
    assert active_learning["selected_candidate_ids"] == [1, 3]
    assert active_learning["retrain"] is False
    assert investigation["discussion_turns"] == 3
    assert investigation["composition_count"] == 1


def _write_run(path: Path, *, facility: str, commit: str, digest: str) -> None:
    path.mkdir()
    config = resolved_configuration(facility)
    (path / "environment.json").write_text(
        json.dumps({"git_commit": commit, "structure_sha256": digest}), encoding="utf-8"
    )
    (path / "resolved_config.json").write_text(json.dumps(config), encoding="utf-8")
    (path / "result.json").write_text(
        json.dumps({"facility": facility, "status": "passed"}), encoding="utf-8"
    )


def test_validation_and_comparison_reject_changed_scientific_inputs(tmp_path):
    frontier = tmp_path / "frontier"
    aurora = tmp_path / "aurora"
    _write_run(frontier, facility="frontier", commit="abc", digest="one")
    _write_run(aurora, facility="aurora", commit="abc", digest="two")
    assert validate_run(frontier) == []
    assert compare_runs([frontier, aurora]) == ["runs used different structure bytes"]


def test_live_portability_requires_complete_llm_qualification(tmp_path):
    run = tmp_path / "llm-check"
    run.mkdir()
    stages = {
        name: True
        for name in ("readiness", "load", "generation", "structured", "discussion", "distributed")
    }
    (run / "result.json").write_text(
        json.dumps({"run_id": "qualified", "status": "complete", "stages": stages})
    )
    (run / "model_identity.json").write_text(
        json.dumps({"provider": "vllm", "model": "model", "base_url": "http://node:8000/v1"})
    )
    qualification = validate_llm_check(run)
    assert qualification["run_id"] == "qualified"
    assert qualification["model"]["model"] == "model"
