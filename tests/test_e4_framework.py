"""Tests for the E4 orchestration benchmark framework (benchmarks/e4/framework)."""

from __future__ import annotations

import copy
import csv
import json
from collections import Counter
from pathlib import Path

import pytest

from benchmarks.e4.framework import cli, portability
from benchmarks.e4.framework.analysis import (
    analyze_runs,
    combine_segments,
    heterogeneous_comparison,
)
from benchmarks.e4.framework.execution import RetryingTaskBackend, _artifact_matches
from benchmarks.e4.framework.records import TaskHistory, read_records, task_histories
from benchmarks.e4.framework.runner import BenchmarkRun
from benchmarks.e4.framework.workload import (
    _validate_chemistry,
    build_workload,
    load_config,
    sha256_file,
)

CONFIGS = Path(__file__).resolve().parents[1] / "benchmarks" / "e4" / "framework" / "configs"
SMOKE = load_config(CONFIGS / "smoke-local.json")


def _run(config, tmp_path, *, experiment="weak", nodes=1, **kwargs):
    run = BenchmarkRun(
        config,
        facility="local",
        experiment=experiment,
        repetition=0,
        output_root=tmp_path,
        nodes=nodes,
        local_device_slots=2,
        **kwargs,
    )
    return run, run.execute()


def _attempts(run_dir: Path) -> list[dict]:
    return [r for r in read_records(run_dir / "attempts.jsonl") if r.get("record") == "attempt"]


# --- workload ---------------------------------------------------------------


def test_workload_is_deterministic():
    a = build_workload(SMOKE, experiment="weak", nodes=2)
    b = build_workload(copy.deepcopy(SMOKE), experiment="weak", nodes=2)
    assert a.workload_hash == b.workload_hash
    assert [t.seconds for t in a.tasks] == [t.seconds for t in b.tasks]
    assert [t.fail_attempts for t in a.tasks] == [t.fail_attempts for t in b.tasks]
    changed = copy.deepcopy(SMOKE)
    changed["seed"] += 1
    assert build_workload(changed, experiment="weak", nodes=2).workload_hash != a.workload_hash


@pytest.mark.parametrize("name", ["e4.1-orchestration.json", "smoke-local.json"])
def test_strong_scaling_uses_identical_task_set(name):
    config = json.loads((CONFIGS / name).read_text())
    points = config["node_points"]["strong"]
    hashes = {build_workload(config, experiment="strong", nodes=n).workload_hash for n in points}
    assert len(hashes) == 1


def test_weak_scaling_keeps_tasks_per_node_constant():
    config = json.loads((CONFIGS / "e4.1-orchestration.json").read_text())
    per_node = sum(spec["tasks_per_node"] for spec in config["task_classes"])
    for nodes in (1, 8, 64, 512):
        workload = build_workload(config, experiment="weak", nodes=nodes)
        assert workload.planned_tasks == per_node * nodes
    small = build_workload(config, experiment="weak", nodes=2).tasks
    large = build_workload(config, experiment="weak", nodes=4).tasks
    assert large[: len(small)] == small  # weak points extend, never reshuffle


def test_load_config_expands_environment(tmp_path, monkeypatch):
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"a": {"b": ["${E4_TEST_DIR}/x"]}}))
    monkeypatch.setenv("E4_TEST_DIR", "/scratch")
    assert load_config(path) == {"a": {"b": ["/scratch/x"]}}
    monkeypatch.delenv("E4_TEST_DIR")
    with pytest.raises(ValueError, match="unset environment variable"):
        load_config(path)


# --- execution, accounting, resume ------------------------------------------


def test_retry_accounting_and_executed_vs_qualified(tmp_path):
    run, summary = _run(SMOKE, tmp_path)
    counts = summary["counts"]
    workload = run.workload
    injected = Counter(t.failure_mode for t in workload.tasks if t.fail_attempts)
    assert counts["planned_tasks"] == workload.planned_tasks
    assert counts["qualified_tasks"] == workload.planned_tasks
    assert counts["failed_tasks"] == 0
    assert counts["retried_tasks"] == sum(injected.values())
    assert counts["failed_attempts"] == sum(injected.values())
    assert counts["total_attempts"] == workload.planned_tasks + sum(injected.values())
    assert counts["failures_by_class"] == {
        "no_result": injected["no-result"],
        "non_finite_output": injected["nonfinite"],
    }
    attempts = _attempts(run.run_dir)
    nonfinite = [a for a in attempts if a["failure_class"] == "non_finite_output"]
    assert nonfinite and all(
        a["outcome"]["executed"] and not a["outcome"]["qualified"] for a in nonfinite
    )
    assert counts["executed_tasks"] >= counts["qualified_tasks"]
    throughput = summary["throughput"]
    makespan = summary["timing"]["application_makespan_seconds"]
    assert throughput["qualified_results_per_second"] == pytest.approx(
        counts["qualified_tasks"] / makespan, rel=1e-3
    )
    assert throughput["qualified_results_per_node_second"] == pytest.approx(
        throughput["qualified_results_per_second"] / summary["nodes"]
    )


def test_exit_failures_are_not_retried(tmp_path):
    config = copy.deepcopy(SMOKE)
    config["task_classes"] = [config["task_classes"][0]]
    config["task_classes"][0]["faults"] = {"fraction": 1.0, "attempts": 1, "mode": "exit"}
    run, summary = _run(config, tmp_path)
    counts = summary["counts"]
    assert counts["failed_tasks"] == counts["planned_tasks"]
    assert counts["qualified_tasks"] == 0
    assert counts["retried_tasks"] == 0
    assert all(not a["retry_scheduled"] and a["terminal"] for a in _attempts(run.run_dir))


def test_cli_run_with_failed_tasks_exits_nonzero(tmp_path):
    config = copy.deepcopy(SMOKE)
    config["task_classes"] = [config["task_classes"][0]]
    config["task_classes"][0]["faults"] = {"fraction": 1.0, "attempts": 1, "mode": "exit"}
    path = tmp_path / "failing.json"
    path.write_text(json.dumps(config))
    argv = ["run", str(path), "--facility", "local", "--experiment", "weak"]
    argv += ["--output-root", str(tmp_path / "out"), "--nodes", "1", "--local-device-slots", "2"]
    assert cli.main(argv) == 2


def test_controlled_artifact_must_exist_and_match_digest(tmp_path):
    task = build_workload(SMOKE, experiment="weak", nodes=1).tasks[0]
    artifact = tmp_path / "artifact.bin"
    assert not _artifact_matches(task, artifact)
    artifact.write_bytes(b"tampered")
    assert not _artifact_matches(task, artifact)


def test_resumed_settings_replay_recorded_recovery(tmp_path):
    backend = RetryingTaskBackend(
        None, {}, None, run_dir=tmp_path, backend="vasp", settings={"nelm": 60}, timeout_sec=10
    )
    assert backend._resumed_settings(TaskHistory()) == ({"nelm": 60}, 10)
    records = [
        {"task_id": "t", "attempt": 1, "settings": {"nelm": 60}, "timeout_sec": 10},
        {
            "task_id": "t",
            "attempt": 2,
            "settings": {"nelm": 120, "algo": "All"},
            "timeout_sec": 10,
            "retry_scheduled": True,
            "failure_class": "scf_not_converged",
        },
    ]
    history = task_histories(records)["t"]
    assert history.last_settings_record["attempt"] == 2
    settings, timeout = backend._resumed_settings(history)
    assert settings == {"nelm": 240, "algo": "All"} and timeout == 10


def test_runs_with_missing_segments_get_no_summed_timing():
    base = {"run_id": "r", "nodes": 1, "executed_tasks": 4, "valid_tasks": 4}
    rows = combine_segments(
        [
            {**base, "segment": 1, "application_makespan_seconds": 10.0},
            {**base, "segment": 3, "application_makespan_seconds": 10.0},
        ]
    )
    assert rows[0]["timing_complete"] is False
    assert rows[0]["application_makespan_seconds"] is None
    assert rows[0]["qualified_results_per_second"] is None
    complete = combine_segments([{**base, "segment": 1, "application_makespan_seconds": 10.0}])
    assert complete[0]["timing_complete"] is True


def test_hydrogen_free_zn_formate_seed_is_rejected():
    seeds = Path(__file__).resolve().parents[1] / "examples" / "paper_cases" / "seeds"
    with pytest.raises(ValueError, match="not a valid Zn"):
        _validate_chemistry(seeds / "zn_formate_alpha.extxyz")
    _validate_chemistry(seeds / "zn_formate_alpha_relaxed.extxyz")


def test_resume_does_not_repeat_completed_work(tmp_path):
    run, _ = _run(SMOKE, tmp_path, nodes=2)
    path = run.run_dir / "attempts.jsonl"
    lines = path.read_text().splitlines(keepends=True)
    kept = lines[: len(lines) // 2]
    path.write_text("".join(kept) + lines[len(kept)][:20])  # torn tail
    done_before = {
        r["task_id"]
        for r in read_records(path)
        if r.get("record") == "attempt" and r.get("terminal") and r["outcome"].get("qualified")
    }
    assert done_before
    resumed = BenchmarkRun(
        SMOKE,
        facility="local",
        experiment="weak",
        repetition=0,
        output_root=tmp_path,
        nodes=2,
        local_device_slots=2,
        resume=True,
    )
    summary = resumed.execute()
    assert summary["counts"]["qualified_tasks"] == run.workload.planned_tasks
    # Device results on disk but missing from the torn log are reconciled, not rerun.
    assert summary["counts"]["skipped_already_complete"] >= len(done_before)
    attempts = _attempts(run.run_dir)
    assert all(a.get("segment") for a in attempts)
    keys = [(a["task_id"], a["attempt"]) for a in attempts]
    assert len(keys) == len(set(keys))
    terminal = Counter(a["task_id"] for a in attempts if a.get("terminal"))
    assert set(terminal.values()) == {1}
    later = [a for a in attempts if a.get("segment") == 2 and a["task_id"] in done_before]
    assert later == []


def test_resume_rejects_changed_identity(tmp_path):
    _run(SMOKE, tmp_path)
    changed = copy.deepcopy(SMOKE)
    changed["seed"] += 1
    with pytest.raises(Exception, match="identity fields changed"):
        BenchmarkRun(
            changed,
            facility="local",
            experiment="weak",
            repetition=0,
            output_root=tmp_path,
            nodes=1,
            local_device_slots=2,
            resume=True,
        )


def test_provenance_files_are_complete(tmp_path):
    run, _ = _run(SMOKE, tmp_path)
    for name in (
        "identity.json",
        "manifest.json",
        "resolved_config.json",
        "placement-01.json",
        "environment-01.json",
        "attempts.jsonl",
        "measurements.jsonl",
        "summary-01.json",
    ):
        assert (run.run_dir / name).is_file(), name
    provenance, errors = portability.stage_provenance(run.run_dir)
    assert errors == []
    assert provenance["workload_hash"] == run.workload.workload_hash
    assert provenance["git_commits"][0]


# --- E4.2 stage level with an injected real-class executor -------------------


class FakeVASP:
    backend_name = "vasp"
    timeout_sec = 60

    def __init__(self, energy_per_atom: float):
        self.energy = energy_per_atom

    def __call__(self, task, launch, *, attempt, work_dir, settings, timeout_sec):
        work_dir.mkdir(parents=True, exist_ok=True)
        (work_dir / "INCAR").write_text(f"input {task.structure_sha256}\n")
        (work_dir / "OUTCAR").write_text(f"energy {self.energy}\n")
        return {
            "return_code": 0,
            "executed": True,
            "converged": True,
            "qualified": True,
            "status": "done",
            "energy_eV": 2 * self.energy,
            "energy_eV_per_atom": self.energy,
            "max_force_eV_per_A": 0.001,
            "input_sha256": {"INCAR": sha256_file(work_dir / "INCAR")},
            "output_sha256": {"OUTCAR": sha256_file(work_dir / "OUTCAR")},
        }


DFT_CONFIG = {
    "workload_id": "e4-test-dft",
    "seed": 1,
    "task_classes": [
        {
            "name": "vasp-si",
            "kind": "dft",
            "pool": "dft",
            "tasks": 2,
            "structures": ["benchmarks/portability/structures/Si.vasp"],
        }
    ],
    "dft_backends": {
        "local": {
            "backend": "vasp",
            "vasp": {"potcar_dir": "potcar", "incar_template": "INCAR"},
        }
    },
    "resources": SMOKE["resources"],
}


def test_stage_level_and_numerical_comparison(tmp_path):
    stages = {}
    for facility, energy in (("a", -5.40), ("b", -5.41)):
        run, summary = _run(
            DFT_CONFIG, tmp_path / facility, experiment="fixed", executors={"dft": FakeVASP(energy)}
        )
        assert summary["counts"]["qualified_tasks"] == 2
        stage = portability.stage_level(run.run_dir)
        assert stage["passed"], stage["errors"]
        assert stage["qualified_by_class"] == {"vasp-si": 2}
        stages[facility] = stage
    rows = portability.numerical_comparison(stages)
    energy = next(r for r in rows if r["metric"] == "energy_eV_per_atom")
    assert energy["spread"] == pytest.approx(0.01)
    assert energy["within_tolerance"]


def test_stage_level_rejects_controlled_only_runs(tmp_path):
    run, _ = _run(SMOKE, tmp_path)
    stage = portability.stage_level(run.run_dir)
    assert not stage["passed"]
    assert "run has no real dft/mlip-relax tasks" in stage["errors"]


def test_stage_level_rejects_executed_but_unqualified(tmp_path):
    class Unqualified(FakeVASP):
        def __call__(self, *args, **kwargs):
            return {**super().__call__(*args, **kwargs), "qualified": False, "status": "failed"}

    run, summary = _run(
        DFT_CONFIG, tmp_path, experiment="fixed", executors={"dft": Unqualified(-5.4)}
    )
    assert summary["counts"]["executed_tasks"] == 2
    assert summary["counts"]["qualified_tasks"] == 0
    assert not portability.stage_level(run.run_dir)["passed"]


# --- analysis, E4.3, jobs, smoke --------------------------------------------


def test_heterogeneous_comparison_uses_homogeneous_baseline():
    base = {"experiment": "fixed", "facility": "aurora", "nodes": 8, "repetition": 0}
    rows = [
        {
            **base,
            "run_id": "mixed",
            "task_classes": ["a", "b"],
            "application_makespan_seconds": 60.0,
        },
        {**base, "run_id": "a", "task_classes": ["a"], "application_makespan_seconds": 40.0},
        {**base, "run_id": "b", "task_classes": ["b"], "application_makespan_seconds": 50.0},
        {
            **base,
            "run_id": "other",
            "nodes": 4,
            "task_classes": ["a"],
            "application_makespan_seconds": 1.0,
        },
    ]
    (row,) = heterogeneous_comparison(rows)
    assert row["homogeneous_serial_makespan_seconds"] == 90.0
    assert row["makespan_ratio_vs_homogeneous_serial"] == pytest.approx(60 / 90)


def test_jobs_estimator_prints_commands(capsys):
    assert (
        cli.main(
            [
                "jobs",
                str(CONFIGS / "e4.1-orchestration.json"),
                "--facility",
                "aurora",
                "--experiment",
                "weak",
                "--nodes",
                "1",
                "8",
                "--repetitions",
                "1",
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert [r["nodes"] for r in report["runs"]] == [1, 8]
    assert report["runs"][1]["command"].startswith("qsub -l select=8 ")
    assert "E4_EXPERIMENT=weak" in report["runs"][1]["command"]
    assert report["total_node_hours_estimate"] > 0


def test_local_smoke_end_to_end(tmp_path, capsys):
    assert cli.main(["smoke", "--output-root", str(tmp_path), "--no-plots"]) == 0
    runs = sorted((tmp_path / "runs").iterdir())
    assert len(runs) == len(SMOKE["node_points"]["weak"]) + len(SMOKE["node_points"]["strong"])
    report = analyze_runs([tmp_path / "runs"], tmp_path / "again", plots=False)
    with (tmp_path / "again" / "scaling.csv").open() as stream:
        scaling = list(csv.DictReader(stream))
    assert len(scaling) == len(runs)
    assert {row["facility"] for row in scaling} == {"local"}
    strong = sorted(
        (row for row in scaling if row["experiment"] == "strong"), key=lambda r: int(r["nodes"])
    )
    # baseline speedup/efficiency
    assert float(strong[0]["speedup"]) == 1.0 and float(strong[0]["efficiency"]) == 1.0
    assert report["outputs"]
