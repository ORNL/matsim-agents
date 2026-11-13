"""Durable accounting regressions; no GPU allocations or scientific jobs."""

from __future__ import annotations

import json
import os
import subprocess
import sys
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


def test_legacy_execution_is_not_given_an_invented_zero_duration(tmp_path):
    (tmp_path / "tasks/candidate/attempt-0001").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="legacy execution"):
        production._check_execution_history(tmp_path)


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
def replay(tmp_path, monkeypatch, clock):
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
        "generator_sha256": "old-source",
        "catalog_sha256": "old-source",
        "acquisition_sha256": "old-source",
        "structures": {row["candidate_id"]: row["structure_file_sha256"] for row in tasks},
    }
    production._atomic_json(round_dir / "generation-preflight.json", receipt)
    for name in ("proposal-001.json", "recovery-authorization.json"):
        production._atomic_json(round_dir / "acquisition" / name, {})
    monkeypatch.setattr(production, "_geometry_sha256", lambda path: path.stem)
    durations = [120, 1]
    calls = []

    def materialize(manifest, spec, root, *args):
        calls.append(root)
        clock[0] += durations.pop(0)
        path = root / "cached-tasks.jsonl"
        production._write_jsonl(path, tasks)
        return root / "round-manifest.json", path, {"status": "generation_complete"}

    monkeypatch.setattr(production, "_materialize_round", materialize)
    return SimpleNamespace(
        manifest=manifest_path,
        root=tmp_path,
        round_dir=round_dir,
        calls=calls,
        durations=durations,
    )


def test_exhausted_replay_cannot_retry_its_cached_generation(replay):
    replay.durations[:] = [1701]
    for _ in range(2):
        with pytest.raises(TimeoutError, match="time budget exhausted"):
            production.revalidate_preflight(replay.manifest, replay.root, "lifepo4", 1)
    assert len(replay.calls) == 1
    assert (
        production._time_budget_consumed(
            json.loads((replay.round_dir / "generation-time.json").read_text())
        )
        == 1801
    )


def test_receipt_write_failure_keeps_prior_replay_charge(replay, monkeypatch):
    atomic = production._atomic_json
    receipt_path = replay.round_dir / "generation-preflight.json"

    def fail_receipt(path, value):
        if path == receipt_path and "source_revalidation" in value:
            raise OSError("receipt write failed")
        atomic(path, value)

    monkeypatch.setattr(production, "_atomic_json", fail_receipt)
    with pytest.raises(OSError, match="receipt write failed"):
        production.revalidate_preflight(replay.manifest, replay.root, "lifepo4", 1)
    monkeypatch.setattr(production, "_atomic_json", atomic)
    result = production.revalidate_preflight(replay.manifest, replay.root, "lifepo4", 1)
    assert result["generation_seconds"] == 221
    assert len(replay.calls) == 2


def test_replay_respects_execution_time_already_consumed(replay, clock):
    with production._time_budget_attempt(
        replay.round_dir / "execution-time.json",
        limit_seconds=1700,
    ):
        clock[0] += 1600
    with pytest.raises(TimeoutError, match="time budget exhausted"):
        production.revalidate_preflight(replay.manifest, replay.root, "lifepo4", 1)


def test_preflight_refuses_gpu_allocation_after_combined_budget_exhaustion(replay, clock):
    production.revalidate_preflight(replay.manifest, replay.root, "lifepo4", 1)
    with production._time_budget_attempt(
        replay.round_dir / "execution-time.json",
        limit_seconds=1580,
    ):
        clock[0] += 1579
    assert production._preflight_receipt(replay.round_dir, replay.manifest)
    with production._time_budget_attempt(
        replay.round_dir / "execution-time.json",
        limit_seconds=1580,
    ) as attempt:
        assert production._preflight_receipt(
            replay.round_dir,
            replay.manifest,
            execution_attempt_id=attempt["attempt_id"],
        )
        with pytest.raises(TimeoutError, match="generation plus prior execution"):
            production._preflight_receipt(replay.round_dir, replay.manifest)


@pytest.mark.parametrize("published_proof", [False, True])
def test_legacy_replay_without_time_ledger_requires_reconciliation(replay, published_proof):
    legacy = replay.round_dir / "acquisition/source-revalidation-legacy"
    legacy.mkdir()
    if published_proof:
        production._atomic_json(legacy / "revalidation.json", {"elapsed_seconds": 200})
    with pytest.raises(RuntimeError, match="legacy replay history"):
        production.revalidate_preflight(replay.manifest, replay.root, "lifepo4", 1)
    assert not replay.calls


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
