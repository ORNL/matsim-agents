import json
from pathlib import Path

import pytest

from benchmarks.e4.aurora_dft import scaling
from benchmarks.e4.aurora_dft.planning import (
    build_six_class_workload_manifest,
    validate_six_class_workload_manifest,
)
from benchmarks.e4.planning import analyze_scaling

REPO = Path(__file__).resolve().parents[1]
PLANS = REPO / "research/paper/e4/aurora-dft-scaling/priority-scaling"


def _validate_plan_without_machine_artifacts(monkeypatch, path):
    monkeypatch.setattr(scaling, "validate_task", lambda task: None)
    return scaling.validate_manifest(path)


def _measurement(experiment, nodes, repetition, elapsed, *, tasks, valid):
    return {
        "run_id": f"{experiment}-{nodes}-{repetition}",
        "experiment": experiment,
        "facility": "aurora",
        "nodes": nodes,
        "repetition": repetition,
        "workload_hash": "fixed",
        "attempted_tasks": tasks,
        "valid_tasks": valid,
        "failed_tasks": tasks - valid,
        "application_makespan_seconds": elapsed,
        "allocation_seconds": elapsed + 30,
        "source_commit": "abc",
        "science_config_sha256": "science",
        "resource_config_sha256": "resource",
    }


def test_strong_tiers_respect_queue_walltime_and_segment_eight_tasks(monkeypatch):
    manifest = _validate_plan_without_machine_artifacts(monkeypatch, PLANS / "strong-manifest.json")
    tiers = {tier["nodes"]: tier for tier in manifest["tiers"]}
    one_node = tiers[1]
    assert one_node["walltime_cap_hours"] == 96
    assert one_node["node_hours_cap"] == 192
    assert [segment["segment_id"] for segment in one_node["segments"]] == [
        "segment-1",
        "segment-2",
    ]
    assert scaling.dispatch_plan(manifest, 1, "segment-1") == [
        [task_id] for task_id in scaling.STRONG_IDS[:4]
    ]
    assert scaling.dispatch_plan(manifest, 1, "segment-2") == [
        [task_id] for task_id in scaling.STRONG_IDS[4:]
    ]
    for nodes in (2, 4, 8):
        assigned = [
            task_id for wave in scaling.dispatch_plan(manifest, nodes, None) for task_id in wave
        ]
        assert sorted(assigned) == sorted(scaling.STRONG_IDS)
        assert len(set(assigned)) == 8
    assert all(tier["walltime_cap_hours"] <= 168 for tier in tiers.values())


def test_weak_tiers_use_unique_frozen_tasks_sequentially(monkeypatch):
    manifest = _validate_plan_without_machine_artifacts(monkeypatch, PLANS / "weak-manifest.json")
    tiers = {tier["nodes"]: tier for tier in manifest["tiers"]}
    for nodes, tier in tiers.items():
        assigned = [task_id for node in tier["node_assignments"] for task_id in node]
        assert len(assigned) == 2 * nodes
        assert len(set(assigned)) == len(assigned)
        assert tier["walltime_cap_hours"] == 48
        assert tier["node_hours_cap"] == 48 * nodes
    assert len(manifest["available_unique_case_ids"]) == 16
    assert len(manifest["additional_case_ids"]) == 8
    assert scaling.dispatch_plan(manifest, 8, None) == [
        [pair[0] for pair in scaling.WEAK_PAIRS],
        [pair[1] for pair in scaling.WEAK_PAIRS],
    ]


def test_dispatch_requires_separate_authorization_before_pbs(monkeypatch, tmp_path):
    monkeypatch.setattr(scaling, "validate_task", lambda task: None)
    manifest_path = PLANS / "strong-manifest.json"
    with pytest.raises(PermissionError, match="separate E4 scaling authorization"):
        scaling.dispatch(manifest_path, 1, "segment-1", dry_run=False)

    manifest = json.loads(manifest_path.read_text())
    manifest["execution_authorized"] = True
    manifest["authorization"] = {"id": "approved-resource-request"}
    authorized_path = tmp_path / "authorized-test-manifest.json"
    authorized_path.write_text(json.dumps(manifest))
    monkeypatch.setenv("E4_SCALING_AUTHORIZATION", "approved-resource-request")
    monkeypatch.delenv("PBS_JOBID", raising=False)
    monkeypatch.delenv("PBS_NODEFILE", raising=False)
    with pytest.raises(RuntimeError, match="inside its PBS allocation"):
        scaling.dispatch(authorized_path, 1, "segment-1", dry_run=False)


def test_convergence_tags_cover_bcc_and_single_stage_materials():
    assert scaling.is_converged_final_status("converged:N0")
    assert scaling.is_converged_final_status("converged:A17")
    assert not scaling.is_converged_final_status("converged:incomplete")
    assert not scaling.is_converged_final_status("converged:n0")


def test_fcc_and_lifepo4_continuation_schedules_preserve_cumulative_budgets():
    fcc = {
        "calculation_id": "fcc",
        "estimated_task_walltime_hours": 48,
        "launch_settings": {
            "continuation_segments": 2,
            "execution_segment_hours": 24,
            "scheduler_margin_hours_per_segment": 1,
        },
    }
    lifepo4 = {
        "calculation_id": "lifepo4",
        "estimated_task_walltime_hours": 72,
        "launch_settings": {
            "continuation_segments": 3,
            "execution_segment_hours": 24,
            "scheduler_margin_hours_per_segment": 1,
        },
    }
    for task, count, budget in ((fcc, 2, 48), (lifepo4, 3, 72)):
        schedule = scaling.task_continuation_schedule(task)
        assert len(schedule) == count
        assert sum(segment["active_seconds"] for segment in schedule) == budget * 3600
        assert (
            sum(segment["requested_walltime_seconds"] for segment in schedule)
            == (budget + count) * 3600
        )
        assert all(segment["soft_stop_seconds"] == 24 * 3600 for segment in schedule)
        assert [segment["resume"] for segment in schedule] == [False] + [True] * (count - 1)


def test_single_segment_task_retains_existing_shutdown_margin():
    schedule = scaling.task_continuation_schedule(
        {"calculation_id": "other", "estimated_task_walltime_hours": 24}
    )
    assert schedule == [
        {
            "resume": False,
            "active_seconds": 24 * 3600,
            "requested_walltime_seconds": 24 * 3600,
            "soft_stop_seconds": 23 * 3600,
        }
    ]


def test_dispatch_resumes_continuation_from_validated_checkpoint(monkeypatch, tmp_path):
    nodefile = tmp_path / "nodefile"
    nodefile.write_text("node0\n")
    monkeypatch.setattr(scaling, "REPO", tmp_path)
    monkeypatch.setattr(scaling, "validate_manifest_data", lambda manifest: manifest)
    monkeypatch.setenv("PBS_JOBID", "test-job")
    monkeypatch.setenv("PBS_NODEFILE", str(nodefile))
    monkeypatch.setenv("E4_SCALING_AUTHORIZATION", "approved")
    monkeypatch.setenv("E4_SCALING_RUN_ID", "E4-TEST-run")
    seen = []

    def fake_run(args, *, env=None, stdout=None, stderr=None, **kwargs):
        if args[0] == "git":
            return type("Result", (), {"stdout": "test-commit"})()
        assert env is not None
        seen.append(env.copy())
        root = Path(env["E4_OUTPUT_ROOT"])
        root.mkdir(parents=True, exist_ok=True)
        if len(seen) == 1:
            (root / "latest-checkpoint").write_text(
                "A1 Accurate interrupted:stopcar " + "a" * 64 + "\n"
            )
            final, status, code = "Accurate:interrupted:stopcar", 75, 75
        else:
            with (root / "run-record.log").open("a") as record:
                record.write("resume verified tag=A1 prec=Accurate status=interrupted:stopcar\n")
            final, status, code = "converged:A2", 0, 0
        with (root / "run-record.log").open("a") as record:
            record.write(f"EXIT job=test calculation_id=FCC-1 status={status} final={final}\n")
        return type("Result", (), {"returncode": code})()

    monkeypatch.setattr(scaling.subprocess, "run", fake_run)
    task = {
        "calculation_id": "FCC-1",
        "case_dir": "case",
        "source_poscar": "POSCAR",
        "source_poscar_sha256": "hash",
        "composition": {"Fe": 1},
        "structure_id": "structure",
        "protocol_id": "fcc",
        "input_hashes": {},
        "prepared_input_bundle_sha256": "bundle",
        "paw_components": [],
        "potcar_sha256": "potcar",
        "mpi_ranks": 8,
        "omp_threads_per_rank": 8,
        "ediffg_eV_per_A": "-0.02",
        "estimated_task_walltime_hours": 48,
        "launch_settings": {
            "stage_mode": "single",
            "stage_nsw_max": 200,
            "single_precision": "Accurate",
            "single_prefix": "A",
            "continuation_segments": 2,
            "execution_segment_hours": 24,
            "scheduler_margin_hours_per_segment": 1,
        },
    }
    manifest = {
        "execution_authorized": True,
        "authorization": {"id": "approved"},
        "experiment": "strong",
        "material_class": "fcc_hea",
        "run_id_prefix": "E4-TEST",
        "protocol_id": "fcc",
        "tasks": [task],
        "fixed_task_ids": ["FCC-1"],
        "tiers": [{"nodes": 1, "segments": [{"segment_id": "all", "task_ids": ["FCC-1"]}]}],
        "resource_request": {"total_node_hours_cap": 50},
    }
    assert (
        scaling.dispatch(tmp_path / "unused.json", 1, None, False, manifest_override=manifest) == 0
    )
    assert [env["E4_RESUME"] for env in seen] == ["0", "1"]
    assert len({env["E4_OUTPUT_ROOT"] for env in seen}) == 1
    assert [env["E4_SOFT_STOP_SECONDS"] for env in seen] == ["86400", "86400"]
    assert [env["E4_REQUESTED_WALLTIME_SECONDS"] for env in seen] == ["90000", "90000"]
    measurement = json.loads(
        (tmp_path / "runs/e4-scaling/strong/E4-TEST-run/n01/all/measurements.jsonl")
        .read_text()
        .splitlines()[0]
    )
    result = next(
        json.loads(line)
        for line in (tmp_path / "runs/e4-scaling/strong/E4-TEST-run/n01/all/dispatch.jsonl")
        .read_text()
        .splitlines()
        if '"event": "finished"' in line
    )
    assert result["task_status"] == "converged"
    assert result["continuation_segments_used"] == 2
    assert [r["resume"] for r in result["restart_provenance"]] == [False, True]
    assert result["restart_provenance"][1]["checkpoint_validated"] is True
    assert measurement["planned_tasks"] == 1


def test_six_class_production_matrix_covers_all_tiers_without_authorizing():
    manifest = build_six_class_workload_manifest()
    validate_six_class_workload_manifest(manifest, verify_files=False)
    classes = {entry["material_class"]: entry for entry in manifest["material_classes"]}
    assert set(classes) == {
        "bcc_refractory",
        "fcc_hea",
        "lifepo4",
        "phosphorene",
        "zn_formate_mof",
        "nb_ta_o",
    }
    assert manifest["execution_authorized"] is False
    assert manifest["authorization_request"]["additional_jobs_authorized"] is False
    assert manifest["semantics"]["scheduler_submissions"] == 0
    assert manifest["semantics"]["node_tiers"] == [1, 2, 4, 8, 16, 32, 64, 128]
    for name, entry in classes.items():
        assert len(entry["candidates"]) == 256
        assert len({c["source_poscar_sha256"] for c in entry["candidates"]}) == 256
        assert len(entry["strong_scaling"]["fixed_calculation_ids"]) == 128
        assert entry["execution_authorized"] is False and entry["authorization"] is None
        for experiment in ("strong_scaling", "weak_scaling"):
            for tier in entry[experiment]["tiers"]:
                assert tier["inputs_ready"], (name, experiment, tier["nodes"])
                assert tier["queue"] == ("capacity" if tier["nodes"] <= 16 else "reservation")
                assert tier["walltime_cap_hours"] <= 168
        for tier in entry["strong_scaling"]["tiers"]:
            ids = [i for s in tier.get("segments", []) for i in s["calculation_ids"]]
            assert sorted(ids or tier["calculation_ids"]) == sorted(
                entry["strong_scaling"]["fixed_calculation_ids"]
            )
        weak_128 = entry["weak_scaling"]["tiers"][-1]
        assert len(set(weak_128["calculation_ids"])) == 256
        assert all(
            c["allocation"] == "HydraGNN" and c["material_class"] == name
            for c in entry["candidates"]
        )
    assert classes["zn_formate_mof"]["protocol"]["functional"].startswith("PBE-D3(BJ)")
    assert all(
        c["composition"]["H"] == c["composition"]["C"]
        for c in classes["zn_formate_mof"]["candidates"]
    )
    request = manifest["authorization_request"]
    assert request["expected_node_hours"] == pytest.approx(
        request["capacity_queue_expected_node_hours"] + request["reservation_expected_node_hours"]
    )


def test_segmented_one_node_strong_baseline_is_combined_without_queue_wait():
    first = _measurement("strong", 1, 1, 100, tasks=4, valid=4)
    second = _measurement("strong", 1, 1, 120, tasks=4, valid=4)
    for segment, record in (("segment-1", first), ("segment-2", second)):
        record.update(
            {
                "material_class": "bcc_refractory",
                "planned_tasks": 8,
                "segment_group_id": "same-workload-n01",
                "segment_id": segment,
                "segment_count": 2,
                "total_ionic_steps": 20,
                "task_execution_seconds": [record["application_makespan_seconds"]],
            }
        )
    scaled = _measurement("strong", 2, 1, 100, tasks=8, valid=8)
    scaled.update(
        {
            "material_class": "bcc_refractory",
            "planned_tasks": 8,
            "total_ionic_steps": 40,
            "task_execution_seconds": [12.5] * 8,
        }
    )
    rows = analyze_scaling([first, second, scaled])
    assert rows[0]["median_makespan_seconds"] == 220
    assert rows[0]["fully_valid_repetitions"] == 1
    assert rows[1]["speedup"] == 2.2
    assert rows[1]["total_ionic_steps"] == 40
