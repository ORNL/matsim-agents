"""Controller-loop regressions for the rolling E4 submitter (no PBS needed)."""

from __future__ import annotations

import json

import pytest

from benchmarks.e4.aurora_dft import rolling_submit


def _plan(tmp_path, *, after=()):
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"tiers": []}\n')
    jobs = [
        {
            "job_key": "first",
            "material_class": "fcc_hea",
            "experiment": "strong",
            "nodes": 16,
            "segment_id": "all",
            "manifest": str(manifest),
            "manifest_sha256": rolling_submit._sha256(manifest),
            "run_id": "run-first",
            "walltime_hours": 1,
            "node_hours_cap": 16,
            "job_name": "e4-first",
            "after_job_keys": list(after),
            "after_pbs_job_ids": [],
        }
    ]
    plan = {
        "execution_authorized": True,
        "authorization_id": "approved",
        "allocation": "acct",
        "queue": "capacity",
        "max_total_user_jobs": 10,
        "combined_reserved_node_hours": 16,
        "jobs": jobs,
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    return path, manifest


def test_single_segment_tiers_use_dispatcher_all_segment():
    assert rolling_submit._segment_id({"segment": 1}, 1) == "all"
    assert rolling_submit._segment_id({"segment": 2}, 3) == "segment-2"


def test_completed_jobs_are_not_requeried_from_expired_pbs_history(tmp_path, monkeypatch):
    plan_path, _ = _plan(tmp_path)
    plan = json.loads(plan_path.read_text())
    plan["jobs"].append(
        {
            **plan["jobs"][0],
            "job_key": "second",
            "after_job_keys": ["first"],
            "run_id_prefix": "run-second",
        }
    )
    plan_path.write_text(json.dumps(plan))
    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps(
            {
                "plan_sha256": rolling_submit._sha256(plan_path),
                "submitted": {"first": {"pbs_job_id": "1.pbs"}},
                "blocked": {},
                "completed": {"first": {"pbs_job_id": "1.pbs"}},
            }
        )
    )
    queried = []
    monkeypatch.setattr(
        rolling_submit, "_job_status", lambda job_id: queried.append(job_id) or ("unknown", None)
    )
    monkeypatch.setattr(rolling_submit, "_current_user_job_count", lambda user: 0)
    submitted = []

    class Result:
        returncode, stdout, stderr = 0, "2.pbs\n", ""

    monkeypatch.setattr(
        rolling_submit.subprocess, "run", lambda cmd, **kw: submitted.append(cmd) or Result()
    )
    assert rolling_submit._run_controller_loop(plan_path, state_path, once=True) == 0
    assert queried == []
    assert len(submitted) == 1
    assert "second" in json.loads(state_path.read_text())["submitted"]


def test_controller_refuses_manifest_changed_after_planning(tmp_path, monkeypatch):
    plan_path, manifest = _plan(tmp_path)
    manifest.write_text('{"tiers": ["edited"]}\n')
    monkeypatch.setattr(rolling_submit, "_current_user_job_count", lambda user: 0)
    monkeypatch.setattr(
        rolling_submit.subprocess,
        "run",
        lambda *a, **kw: pytest.fail("qsub must not run for a changed manifest"),
    )
    with pytest.raises(ValueError, match="changed after planning"):
        rolling_submit._run_controller_loop(plan_path, tmp_path / "state.json", once=True)
