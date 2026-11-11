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


def test_six_class_manifest_keeps_blocked_inputs_and_authorization_explicit():
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
    assert manifest["semantics"]["scheduler_submissions"] == 0
    assert manifest["semantics"]["node_tiers"] == [1, 2, 4, 8, 16, 32, 64, 128]
    assert manifest["queue_limits"]["maximum_nodes_per_job"] == 16
    assert classes["fcc_hea"]["strong_scaling"]["tiers"][2]["status"] == (
        "input_complete_authorization_required"
    )
    assert classes["fcc_hea"]["strong_scaling"]["tiers"][3]["status"] == (
        "input_complete_authorization_required"
    )
    assert classes["fcc_hea"]["strong_scaling"]["tiers"][4]["status"] == (
        "blocked_fixed_workload_underfills_node_tier"
    )
    for name in ("fcc_hea", "lifepo4", "phosphorene", "zn_formate_mof"):
        weak = classes[name]["weak_scaling"]["tiers"]
        assert weak[3]["status"] == "input_complete_authorization_required"
        assert len(set(weak[3]["calculation_ids"])) == 16
        assert weak[4]["status"] == "blocked_insufficient_unique_inputs"
        assert len(classes[name]["strong_scaling"]["fixed_calculation_ids"]) == 8
    nbtao = classes["nb_ta_o"]
    assert nbtao["strong_scaling"]["tiers"][0]["status"] == (
        "input_complete_authorization_required"
    )
    assert nbtao["weak_scaling"]["tiers"][3]["status"] == "blocked_insufficient_unique_inputs"
    assert len(nbtao["strong_scaling"]["fixed_calculation_ids"]) == 8
    assert classes["zn_formate_mof"]["protocol"]["functional"].startswith("PBE-D3(BJ)")
    assert all(
        candidate["composition"]["H"] == candidate["composition"]["C"]
        for candidate in classes["zn_formate_mof"]["candidates"]
    )
    assert all(
        candidate["input_hashes_validated"] and candidate["execution_ready"] is False
        for name in ("phosphorene", "zn_formate_mof", "nb_ta_o")
        for candidate in classes[name]["candidates"]
    )
    assert all(
        candidate["allocation"] == "HydraGNN" and candidate["material_class"] == name
        for name, entry in classes.items()
        for candidate in entry["candidates"]
    )
    weak_32 = classes["bcc_refractory"]["weak_scaling"]["tiers"][5]
    assert weak_32["required_unique_calculations"] == 64
    assert weak_32["queue_eligible"] is False
    assert weak_32["status"] == "blocked_capacity_queue_node_limit"


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
