from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.portability.compare import compare_runs
from benchmarks.portability.qualification import execute_compute_qualification
from benchmarks.portability.run import (
    build_plan,
    resolved_configuration,
    validate_inputs,
    validate_llm_check,
)
from benchmarks.portability.suites import (
    active_learning_contract,
    active_learning_loop_contract,
    dft_allocation_contract,
    investigation_contract,
    phase_exploration_contract,
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


def test_production_contracts_cover_resume_allocation_and_independent_phase(tmp_path):
    structure = Path("benchmarks/portability/structures/Si.vasp")
    active_learning = active_learning_loop_contract(structure, tmp_path / "active-learning")
    allocation = dft_allocation_contract(tmp_path / "allocation")
    phase = phase_exploration_contract(tmp_path / "phase")
    assert active_learning["labelled_count"] == 2
    assert active_learning["resume_avoided_duplicate_iteration"] is True
    assert allocation["disjoint_node_groups"] is True
    assert phase["used_llm_investigation"] is False


def test_compute_qualification_requires_and_executes_real_workflow_modes(tmp_path):
    mlip = tmp_path / "mlip.yaml"
    qe = tmp_path / "qe.yaml"
    mlip.write_text("mode: mlip\nstructure_path: replaced\n")
    qe.write_text(
        "mode: dft\nstructure_path: replaced\ndft:\n  backend: qe\n  pseudo_dir: /shared/pseudo\n"
    )

    def fake_runner(config):
        backend = "qe" if config.mode.value == "dft" else "uma"
        stage_payload = {"backend": backend, "converged": True}
        stage = SimpleNamespace(
            energy_eV=-10.0,
            max_force_eV_per_A=0.01,
            converged=True,
            model_dump=lambda **_: stage_payload,
        )
        return SimpleNamespace(
            status=SimpleNamespace(value="complete"),
            run_id=f"{backend}-run",
            run_directory=str(tmp_path / backend),
            final_structure_path=str(tmp_path / f"{backend}.vasp"),
            stages=[stage],
        )

    summary = execute_compute_qualification(
        structure=Path("benchmarks/portability/structures/Si.vasp"),
        output=tmp_path / "compute",
        relaxation_configs=[mlip, qe],
        runner=fake_runner,
    )
    assert summary["status"] == "passed"
    assert [case["mode"] for case in summary["cases"]] == ["mlip", "dft"]


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
    (path / "scientific_summary.json").write_text(
        json.dumps({"qualification": "contract"}), encoding="utf-8"
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
