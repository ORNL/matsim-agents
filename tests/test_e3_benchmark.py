from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from benchmarks.e3 import production


@pytest.mark.parametrize("account", ["m5216", "m5048", "m4828", "amsc001", "amsc004", "dasrepo"])
def test_allocation_checks_explicit_account_and_premium(monkeypatch, account):
    from types import SimpleNamespace

    monkeypatch.setenv("MATSIM_E3_MLIP_ACCOUNT", account)
    fields = f"Account={account}_g QOS=gpu_premium NumNodes=16 TimeLimit=02:00:00"
    monkeypatch.setattr(
        production.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=fields)
    )
    production._validate_benchmark_allocation("123")
    for wrong in [
        fields.replace(f"Account={account}_g", "Account=unauthorized_g"),
        fields.replace("gpu_premium", "gpu_regular"),
    ]:
        monkeypatch.setattr(
            production.subprocess,
            "run",
            lambda *args, output=wrong, **kwargs: SimpleNamespace(stdout=output),
        )
        with pytest.raises(RuntimeError, match="does not match"):
            production._validate_benchmark_allocation("123")


def test_current_manifest_enforces_six_classes_and_bounded_rounds():
    manifest = production.build_campaign_manifest()
    production.validate_campaign_manifest(manifest)

    assert manifest["campaign_id"] == "e3-mlip-production"
    assert len(manifest["rounds"]) == 18
    assert manifest["total_logical_optimizations"] == 1152
    assert manifest["resource_request"]["maximum_scheduled_gpu_hours"] == 768
    assert manifest["resource_request"]["maximum_scheduled_gpu_node_hours"] == 192
    assert manifest["resource_request"]["qos"] == "premium"
    assert manifest["resource_request"]["nodes_per_array_element"] == 16
    assert manifest["execution"]["round_walltime_seconds"] == 1800
    assert manifest["execution"]["allocation_walltime_seconds"] == 7200
    assert manifest["execution"]["hard_optimization_seconds"] == 300
    for material in production.MATERIALS:
        rows = [row for row in manifest["rounds"] if row["material"] == material]
        assert len(rows) == 3
        assert {row["round"] for row in rows} == {1, 2, 3}
        assert {row["strategy"] for row in rows} == {"llm_guided_adaptation"}
        assert {row["seed"] for row in rows} == {20261113}
        assert all(row["main_target"] == 64 for row in rows)
        assert all(row["finite_size_control_target"] == 0 for row in rows)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("campaign_id", "other"),
        ("rounds_per_material", 10),
        ("strategies", ["llm_guided_adaptation", "rule_based_adaptation"]),
        ("total_logical_optimizations", 61_120),
    ],
)
def test_manifest_rejects_other_benchmark_protocols(field, value):
    manifest = production.build_campaign_manifest()
    manifest[field] = value
    with pytest.raises(ValueError):
        production.validate_campaign_manifest(manifest)


def test_supervised_round_kills_stalled_process_group(tmp_path):
    pid_file = tmp_path / "pid"
    command = [
        sys.executable,
        "-c",
        f"import os,time; open({str(pid_file)!r},'w').write(str(os.getpid())); time.sleep(60)",
    ]
    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        production._run_supervised_round(command, seconds=0.5)
    assert time.monotonic() - started < 5
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_file.read_text()), 0)


@pytest.mark.parametrize("seconds", [0, -1, float("nan"), float("inf")])
def test_supervised_round_rejects_invalid_deadline(seconds):
    with pytest.raises(ValueError, match="finite and positive"):
        production._run_supervised_round(["unused"], seconds=seconds)


def test_round_worker_step_uses_remaining_round_walltime(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setenv("SLURM_JOB_ID", "test")
    monkeypatch.setattr(production.time, "monotonic", lambda: 100)
    monkeypatch.setattr(
        production.subprocess, "run", lambda command, **kwargs: calls.append(command)
    )
    production._execute_round_tasks(
        tmp_path / "tasks.jsonl",
        tmp_path / "results",
        64,
        4,
        "worker-python",
        hard_optimization_seconds=300,
        round_deadline=720,
    )
    command = calls[0]
    assert command[command.index("--time") + 1] == "10"
    assert command[command.index("--nodes") + 1] == "16"
    assert command[command.index("--ntasks") + 1] == "64"
    assert command[-2:] == ["--hard-optimization-seconds", "300"]
    with pytest.raises(TimeoutError, match="budget exhausted"):
        production._execute_round_tasks(
            tmp_path / "tasks.jsonl", tmp_path, 64, 4, "worker-python", round_deadline=159
        )
    assert len(calls) == 1


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(production.build_campaign_manifest()))
    output_root = tmp_path / "runs"
    output_root.mkdir()
    production.reserve_gpu_hours(
        output_root / "resource-budget.json",
        "test-job",
        reserved_gpu_seconds=production.GPU_SECONDS_PER_ELEMENT,
        cap_gpu_seconds=production.GPU_HOUR_CAP * 3600,
    )
    monkeypatch.setenv("SLURM_JOB_ID", "test-job")
    monkeypatch.setattr(production, "_validate_benchmark_allocation", lambda job_id: None)
    monkeypatch.setattr(production, "_preflight_receipt", lambda *args: {"generation_seconds": 0})
    return manifest_path, output_root


def test_campaign_supervises_three_rounds_and_writes_report(campaign, monkeypatch):
    manifest_path, output_root = campaign
    commands = []

    def run_round(command, *, seconds=production.ROUND_SECONDS):
        commands.append(command)
        round_index = int(command[command.index("--round-index") + 1])
        stream = (
            output_root / "nb_ta_o/llm_guided_adaptation/seed-20261113" / f"round-{round_index:02d}"
        )
        stream.mkdir(parents=True, exist_ok=True)
        production._atomic_json(
            stream / "round-summary.json",
            {"status": "complete", "task_count": 64, "outcomes": {"converged": 60}},
        )

    monkeypatch.setattr(production, "_run_supervised_round", run_round)
    result = production.run_campaign(manifest_path, output_root, "nb_ta_o", "worker-python")
    assert len(commands) == 3
    assert [cmd[cmd.index("--round-index") + 1] for cmd in commands] == ["1", "2", "3"]
    assert all(cmd[cmd.index("--workers") + 1] == "64" for cmd in commands)
    assert result["status"] == "complete"
    assert result["logical_optimizations_attempted"] == 192
    report = output_root / "nb_ta_o/llm_guided_adaptation/seed-20261113/report.md"
    assert "It does not\nsupport a rule-based comparison" in report.read_text()
    resumed = production.run_campaign(manifest_path, output_root, "nb_ta_o", "worker-python")
    assert resumed["status"] == "complete"
    assert len(commands) == 3


def test_campaign_timeout_is_recorded_as_incomplete(campaign, monkeypatch):
    manifest_path, output_root = campaign
    command = ["test-round"]

    def time_out(*args, **kwargs):
        raise subprocess.TimeoutExpired(command, production.ROUND_SECONDS)

    monkeypatch.setattr(production, "_run_supervised_round", time_out)
    with pytest.raises(subprocess.TimeoutExpired):
        production.run_campaign(manifest_path, output_root, "nb_ta_o", "worker-python")
    stream = output_root / "nb_ta_o/llm_guided_adaptation/seed-20261113"
    summary = json.loads((stream / "campaign-summary.json").read_text())
    assert summary["status"] == "incomplete"
    assert summary["rounds_completed"] == 0
    assert summary["termination_reason"] == "round_time_limit"
    assert "incomplete" in (stream / "report.md").read_text()


def test_campaign_retries_subtract_prior_generation_and_execution(campaign, monkeypatch):
    manifest_path, output_root = campaign
    clock = [1000.0]
    monkeypatch.setattr(production.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(production.time, "time", lambda: clock[0] + 100_000)
    monkeypatch.setattr(production, "_preflight_receipt", lambda *args: {"generation_seconds": 100})
    round_dir = output_root / "nb_ta_o/llm_guided_adaptation/seed-20261113/round-01"
    production._atomic_json(round_dir / "generation-preflight.json", {"generation_seconds": 100})
    with production._time_budget_attempt(round_dir / "execution-time.json", limit_seconds=1700):
        clock[0] += 300
    deadlines = []

    def fail(command, *, seconds):
        deadlines.append(seconds)
        attempt = command[command.index("--round-attempt-id") + 1]
        assert production._execution_allowance(round_dir, attempt) == seconds
        clock[0] += 100
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(production, "_run_supervised_round", fail)
    for _ in range(2):
        with pytest.raises(subprocess.CalledProcessError):
            production.run_campaign(manifest_path, output_root, "nb_ta_o", "worker-python")
    assert deadlines == [1400, 1300]


def test_campaign_generation_exhaustion_stops_without_hypothesis_update(campaign, monkeypatch):
    manifest_path, output_root = campaign
    calls = []

    def run_round(command, **kwargs):
        index = int(command[command.index("--round-index") + 1])
        calls.append(index)
        root = output_root / "phosphorene/llm_guided_adaptation/seed-20261113"
        round_dir = root / f"round-{index:02d}"
        if index < 3:
            production._atomic_json(
                round_dir / "round-summary.json",
                {
                    "status": "complete",
                    "task_count": 64,
                    "outcomes": {"nonconverged": 64},
                },
            )
        else:
            production._atomic_json(
                round_dir / "acquisition/outcome.json",
                {
                    "status": "stopped_generation_exhausted",
                    "new_optimization_evidence": False,
                    "hypotheses_updated": False,
                    "shortfalls": [
                        {
                            "requested": 6,
                            "accepted_unique": 2,
                            "generation_attempts": 180,
                            "rejection_reasons": {
                                "prior_symmetry_equivalent": 143,
                                "symmetry_equivalent": 35,
                            },
                        }
                    ],
                },
            )

    monkeypatch.setattr(production, "_run_supervised_round", run_round)
    summary = production.run_campaign(
        manifest_path,
        output_root,
        "phosphorene",
        "worker-python",
    )
    assert calls == [1, 2, 3]
    assert summary["status"] == "stopped_generation_exhausted"
    assert summary["rounds_completed"] == 2
    assert summary["stopped_round"] == 3
    assert summary["logical_optimizations_attempted"] == 128
    assert summary["outcomes"] == {"nonconverged": 128}
    assert summary["new_optimization_evidence_in_stopped_round"] is False
    assert summary["hypotheses_updated_in_stopped_round"] is False
    stream = output_root / "phosphorene/llm_guided_adaptation/seed-20261113"
    assert not (stream / "round-03/round-summary.json").exists()
    assert "retained without update" in (stream / "report.md").read_text()
    assert json.loads((stream / "round-01/round-summary.json").read_text())["task_count"] == 64


def test_campaign_infrastructure_failure_still_raises(campaign, monkeypatch):
    manifest_path, output_root = campaign

    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, ["infrastructure-failure"])

    monkeypatch.setattr(production, "_run_supervised_round", fail)
    with pytest.raises(subprocess.CalledProcessError):
        production.run_campaign(manifest_path, output_root, "phosphorene", "worker-python")
    stream = output_root / "phosphorene/llm_guided_adaptation/seed-20261113"
    assert (
        json.loads((stream / "campaign-summary.json").read_text())["termination_reason"]
        == "round_failed"
    )


def test_budget_ledger_uses_cfs_compatible_lock(tmp_path, monkeypatch):
    import fcntl

    def unsupported(*args):
        raise OSError(524, "unsupported on CFS")

    monkeypatch.setattr(fcntl, "flock", unsupported)
    ledger = tmp_path / "budget.json"
    production.reserve_gpu_hours(
        ledger, "test", reserved_gpu_seconds=460_800, cap_gpu_seconds=2_764_800
    )
    production.release_gpu_hours(ledger, "test", actual_gpu_seconds=460_800)
    assert json.loads(ledger.read_text())["spent_gpu_seconds"] == 460_800


@pytest.mark.parametrize("preallocated", [False, True])
def test_workflow_wrapper_uses_current_campaign_and_full_reservation(
    campaign,
    monkeypatch,
    preallocated,
):
    from matsim_agents.cli.workflow_catalog import WorkflowConfig
    from matsim_agents.cli.workflows import _check, _command

    manifest_path, output_root = campaign
    ledger = output_root / "resource-budget.json"
    monkeypatch.setenv("SLURM_JOB_ID", "test-job")
    reservation_id = "test-job"
    if preallocated:
        state = json.loads(ledger.read_text())
        reservation_id = "submission-test"
        state["reservations"][reservation_id] = {
            **state["reservations"].pop("test-job"),
            "job_id": "test-job",
        }
        ledger.write_text(json.dumps(state))
        monkeypatch.setenv("MATSIM_E3_MLIP_RESERVATION_ID", reservation_id)
    else:
        monkeypatch.delenv("MATSIM_E3_MLIP_RESERVATION_ID", raising=False)
    config = WorkflowConfig(
        workflow="e3-mlip",
        settings={
            "manifest": manifest_path,
            "output_root": output_root,
            "resource_state": ledger,
            "material": "nb_ta_o",
            "worker_python": Path("worker-python"),
        },
    )
    assert "validated" in " ".join(_check(config))
    command = _command(config, approved=True)
    assert command[2:4] == ["benchmarks.e3.production", "run-campaign"]
    assert "--workers" in command and "64" in command
    state = json.loads(ledger.read_text())
    if preallocated:
        state["reservations"][reservation_id]["job_id"] = "other-job"
        ledger.write_text(json.dumps(state))
        with pytest.raises(ValueError, match="full 16-node/two-hour"):
            _command(config, approved=True)
        state["reservations"][reservation_id]["job_id"] = "test-job"
    state["reservations"][reservation_id]["reserved_gpu_seconds"] = 100
    ledger.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="full 16-node/two-hour"):
        _command(config, approved=True)


def test_perlmutter_launcher_matches_manifest_limits():
    launcher = production.REPO / "deployments/perlmutter/jobs/job-e3-mlip-production-perlmutter.sh"
    text = launcher.read_text()
    for setting in (
        "#SBATCH -N 16",
        "#SBATCH -q premium",
        "#SBATCH -t 02:00:00",
        "#SBATCH --array=0-5%5",
        "benchmarks.e3.production run-campaign",
    ):
        assert setting in text
    helper = (
        production.REPO / "deployments/perlmutter/jobs/submit-e3-mlip-production-perlmutter.sh"
    ).read_text()
    assert "--reserved-gpu-seconds 460800" in helper
    assert "--cap-gpu-seconds 2764800" in helper
    assert helper.index("budget-reserve") < helper.index('SUBMITTED="$(sbatch')
    assert "MATSIM_E3_MLIP_RESERVATION_ID:?" in text
    subprocess.run(["bash", "-n", str(launcher)], check=True)
