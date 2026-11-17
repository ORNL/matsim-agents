"""Durable accounting regressions; no GPU allocations or scientific jobs."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.e3 import production


@pytest.fixture
def clock(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(production.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(production.time, "time", lambda: now[0] + 100_000)
    return now


def test_failed_attempts_consume_time_and_worker_uses_supervisor_balance(tmp_path, clock):
    path = tmp_path / "execution-time.json"
    with (
        pytest.raises(OSError, match="interrupted"),
        production._time_budget_attempt(path, limit_seconds=1700),
    ):
        clock[0] += 300
        raise OSError("interrupted")
    with production._time_budget_attempt(path, limit_seconds=1700) as attempt:
        assert attempt["reserved_seconds"] == 1400
        clock[0] += 20
        assert production._execution_allowance(tmp_path, attempt["attempt_id"]) == 1380
    state = json.loads(path.read_text())
    assert production._time_budget_consumed(state) == 320
    assert [row["status"] for row in state["attempts"]] == ["failed", "complete"]


def test_hard_interruption_retains_full_reservation(tmp_path):
    path = tmp_path / "execution-time.json"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os\nfrom pathlib import Path\n"
            "from benchmarks.e3.production import _time_budget_attempt\n"
            f"with _time_budget_attempt(Path({str(path)!r}), limit_seconds=1700):\n"
            "    os._exit(9)\n",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 9, result.stderr
    state = json.loads(path.read_text())
    assert production._time_budget_consumed(state) == 1700
    assert state["attempts"][0]["status"] == "running"
    with (
        pytest.raises(TimeoutError, match="unfinished attempts"),
        production._time_budget_attempt(path, limit_seconds=1700),
    ):
        pytest.fail("an interrupted attempt must not obtain a new allowance")


def test_direct_replicate_retries_share_one_execution_budget(tmp_path, clock, monkeypatch):
    deadlines = []

    def fail(*args, round_deadline, **kwargs):
        deadlines.append(round_deadline - clock[0])
        clock[0] += 250
        raise OSError("worker failed")

    monkeypatch.setattr(production, "_run_replicate", fail)
    for _ in range(2):
        with pytest.raises(OSError, match="worker failed"):
            production.run_replicate(
                tmp_path / "manifest.json",
                tmp_path,
                "phosphorene",
                "llm_guided_adaptation",
                20261113,
                round_index=1,
            )
    assert deadlines == [1800, 1550]


def test_supervised_replicate_does_not_double_charge(tmp_path, clock, monkeypatch):
    round_dir = tmp_path / "phosphorene/llm_guided_adaptation/seed-20261113/round-01"
    path = round_dir / "execution-time.json"
    with production._time_budget_attempt(path, limit_seconds=1700) as attempt:
        clock[0] += 10

        def finish(*args, round_deadline, **kwargs):
            assert round_deadline - clock[0] == 1690
            clock[0] += 200
            return {"status": "complete"}

        monkeypatch.setattr(production, "_run_replicate", finish)
        production.run_replicate(
            tmp_path / "manifest.json",
            tmp_path,
            "phosphorene",
            "llm_guided_adaptation",
            20261113,
            round_index=1,
            round_attempt_id=attempt["attempt_id"],
        )
    state = json.loads(path.read_text())
    assert len(state["attempts"]) == 1
    assert production._time_budget_consumed(state) == 210


def test_initial_generation_retries_also_keep_consumed_time(tmp_path, monkeypatch, clock):
    manifest = tmp_path / "manifest.json"
    production._atomic_json(manifest, production.build_campaign_manifest())
    monkeypatch.setattr(production, "_chat_model", lambda: object())

    def fail(*args, **kwargs):
        clock[0] += 300
        raise OSError("generation interrupted")

    monkeypatch.setattr(production, "_materialize_round", fail)
    for _ in range(2):
        with pytest.raises(OSError, match="generation interrupted"):
            production.prepare_round(manifest, tmp_path, "lifepo4", 1)
    round_dir = tmp_path / "lifepo4/llm_guided_adaptation/seed-20261113/round-01"
    state = json.loads((round_dir / "generation-time.json").read_text())
    assert production._time_budget_consumed(state) == 600
    assert [row["reserved_seconds"] for row in state["attempts"]] == [1800, 1500]


@pytest.fixture
def preflight(tmp_path, monkeypatch, clock):
    """Synthetic geometry identities isolate accounting, not atomistic generation."""
    manifest_path = tmp_path / "manifest.json"
    production._atomic_json(manifest_path, production.build_campaign_manifest())
    round_dir = tmp_path / "lifepo4/llm_guided_adaptation/seed-20261113/round-01"
    round_dir.mkdir(parents=True)
    tasks = []
    for i in range(64):
        structure = round_dir / f"structure-{i}.extxyz"
        structure.write_text(f"geometry identity {i}")
        tasks.append(
            {
                "candidate_id": str(i),
                "structure_path": str(structure),
                "structure_file_sha256": production._sha256_file(structure),
                "structure_sha256": structure.stem,
            }
        )
    production._write_jsonl(round_dir / "tasks.jsonl", tasks)
    receipt = {
        "generation_seconds": 100,
        "manifest_sha256": production._sha256_file(manifest_path),
        "generator_sha256": production._sha256_file(
            production.REPO / "benchmarks/e3/structure_generation.py"
        ),
        "catalog_sha256": production._sha256_file(production.REPO / "benchmarks/e3/feasibility.py"),
        "acquisition_sha256": production._sha256_file(Path(production.__file__)),
        "structures": {row["candidate_id"]: row["structure_file_sha256"] for row in tasks},
    }
    production._atomic_json(round_dir / "generation-preflight.json", receipt)
    return SimpleNamespace(manifest=manifest_path, root=tmp_path, round_dir=round_dir)


@pytest.mark.parametrize(
    "prior_evidence", [None, "attempt", "summary", "later_round", "changed_input"]
)
def test_fresh_cpu_preflight_can_register_hard_deadline_policy(
    preflight, monkeypatch, prior_evidence
):
    replay = preflight
    if prior_evidence == "attempt":
        (replay.round_dir / "tasks/candidate/attempt-0001").mkdir(parents=True)
    elif prior_evidence == "summary":
        production._atomic_json(replay.round_dir / "round-summary.json", {"status": "complete"})
    elif prior_evidence == "later_round":
        (replay.round_dir.parent / "round-02").mkdir()
    elif prior_evidence == "changed_input":
        (replay.round_dir / "structure-0.extxyz").write_text("changed")

    def reached_execution():
        raise RuntimeError("verified preflight reached execution")

    monkeypatch.setattr(production, "_verified_checkpoint_sha256", reached_execution)
    message = (
        "requires a fresh stream"
        if prior_evidence in {"attempt", "summary", "later_round"}
        else "preflight structure changed"
        if prior_evidence == "changed_input"
        else "verified preflight reached execution"
    )
    with pytest.raises((ValueError, RuntimeError), match=message):
        production.run_replicate(
            replay.manifest,
            replay.root,
            "lifepo4",
            "llm_guided_adaptation",
            20261113,
            hard_optimization_seconds=300,
            round_index=1,
        )
    policy_path = replay.round_dir.parent / "execution-policy.json"
    assert policy_path.exists() == (prior_evidence is None)
    if prior_evidence is None:
        policy = json.loads(policy_path.read_text())
        assert policy["hard_optimization_seconds"] == 300
        assert policy["manifest_sha256"] == production._sha256_file(replay.manifest)


def test_preflight_refuses_gpu_allocation_after_combined_budget_exhaustion(preflight, clock):
    replay = preflight
    with production._time_budget_attempt(
        replay.round_dir / "execution-time.json",
        limit_seconds=1700,
    ):
        clock[0] += 1699
    assert production._preflight_receipt(replay.round_dir, replay.manifest)
    with production._time_budget_attempt(
        replay.round_dir / "execution-time.json",
        limit_seconds=1700,
    ) as attempt:
        assert production._preflight_receipt(
            replay.round_dir,
            replay.manifest,
            execution_attempt_id=attempt["attempt_id"],
        )
        with pytest.raises(TimeoutError, match="generation plus prior execution"):
            production._preflight_receipt(replay.round_dir, replay.manifest)


@pytest.fixture
def reservation(tmp_path):
    path = tmp_path / "resource-budget.json"
    production.reserve_gpu_hours(path, "submission-test", reserved_gpu_seconds=460800)
    production.bind_gpu_reservation(path, "submission-test", "123456")
    return path


def test_reservation_cannot_be_reused_by_another_job(reservation):
    production.bind_gpu_reservation(reservation, "submission-test", "123456")
    with pytest.raises(ValueError, match="different Slurm"):
        production.bind_gpu_reservation(reservation, "submission-test", "123457")


@pytest.mark.parametrize(
    "row",
    [
        "123456|RUNNING|100|gres/gpu=64\n",
        "123456|COMPLETING|100|gres/gpu=64\n",
        "123456|FAILED|100|gres/gpu=4\n",
        "123456|FAILED|100|\n",
        "123457|COMPLETED|100|gres/gpu=64\n",
        "",
    ],
)
def test_unverified_scheduler_accounting_retains_full_charge(reservation, monkeypatch, row):
    monkeypatch.setattr(
        production.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=row),
    )
    with pytest.raises(RuntimeError, match="reservation retained"):
        production.reconcile_gpu_allocation(reservation, "submission-test")
    state = json.loads(reservation.read_text())
    assert state["reservations"]["submission-test"]["reserved_gpu_seconds"] == 460800
    assert state["spent_gpu_seconds"] == 0


def test_terminal_scheduler_usage_includes_full_allocation_lifetime(reservation, monkeypatch):
    monkeypatch.setattr(
        production.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="123456|FAILED|145|gres/gpu=64\n"),
    )
    result = production.reconcile_gpu_allocation(reservation, "submission-test")
    assert result["spent_gpu_seconds"] == 145 * 64
    assert result["reserved_gpu_seconds"] == 0
    record = json.loads(reservation.read_text())["allocations"]["submission-test"]
    assert record == {
        "reserved_gpu_seconds": 460800,
        "job_id": "123456",
        "actual_gpu_seconds": 145 * 64,
    }


def test_actual_usage_is_not_silently_clipped_to_reservation(tmp_path):
    path = tmp_path / "budget.json"
    production.reserve_gpu_hours(path, "job", reserved_gpu_seconds=100, cap_gpu_seconds=1000)
    result = production.release_gpu_hours(path, "job", actual_gpu_seconds=101)
    assert result["spent_gpu_seconds"] == 101


def test_unexpected_scheduler_overrun_is_persisted_and_reported(tmp_path):
    path = tmp_path / "budget.json"
    production.reserve_gpu_hours(path, "job", reserved_gpu_seconds=100, cap_gpu_seconds=100)
    with pytest.raises(RuntimeError, match="actual usage was persisted"):
        production.release_gpu_hours(path, "job", actual_gpu_seconds=101)
    state = json.loads(path.read_text())
    assert state["spent_gpu_seconds"] == 101
    assert not state["reservations"]


@pytest.mark.parametrize("failure", ["sdk", "validate", "verify-preflight"])
def test_startup_failures_keep_preallocated_gpu_time(reservation, tmp_path, monkeypatch, failure):
    base = tmp_path / "fake-python"
    base.write_text(
        "#!/bin/bash\n"
        'printf "%s\\n" "$*" >> "$CALLS"\n'
        'if [[ "$*" == *"import copilot"* && "$FAILURE" == sdk ]]; then exit 11; fi\n'
        'if [[ "$*" == *" $FAILURE "* && "$FAILURE" != sdk ]]; then exit 11; fi\n'
    )
    base.chmod(0o700)
    copilot = tmp_path / "copilot"
    copilot.write_text("#!/bin/bash\nexit 0\n")
    copilot.chmod(0o700)
    calls = tmp_path / "calls"
    for name, value in {
        "PROJECT_ROOT": str(production.REPO),
        "SLURM_JOB_ID": "123456",
        "SLURM_JOB_ACCOUNT": "m5216_g",
        "SLURM_JOB_QOS": "gpu_premium",
        "SLURM_NNODES": "16",
        "SLURM_NTASKS": "64",
        "SLURM_ARRAY_TASK_ID": "3",
        "MATSIM_E3_MLIP_RESOURCE_AUTHORIZED": "m5216:premium:768",
        "MATSIM_E3_MLIP_RESERVATION_ID": "submission-test",
        "MATSIM_E3_MLIP_RESOURCE_STATE": str(reservation),
        "MATSIM_E3_MLIP_BASE_PYTHON": str(base),
        "MATSIM_E3_MLIP_WORKER_PYTHON": str(base),
        "MATSIM_E3_MLIP_OUTPUT_ROOT": str(tmp_path),
        "CALLS": str(calls),
        "FAILURE": failure,
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
    }.items():
        monkeypatch.setenv(name, value)
    launcher = production.REPO / "deployments/perlmutter/jobs/job-e3-mlip-production-perlmutter.sh"
    result = subprocess.run(
        ["bash", str(launcher)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode != 0
    assert "budget-release" not in calls.read_text()
    assert json.loads(reservation.read_text())["reservations"]["submission-test"] == {
        "job_id": "123456",
        "reserved_gpu_seconds": 460800,
    }
