from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.portability.compare import compare_runs
from benchmarks.portability.run import build_plan, resolved_configuration, validate_inputs
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

