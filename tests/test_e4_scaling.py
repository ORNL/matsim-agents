import json
from pathlib import Path

import pytest

from benchmarks.e4.aurora_dft import scaling
from benchmarks.e4.aurora_dft.planning import (
    SIX_CLASS_INPUTS,
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


def test_parse_max_force_accepts_numbers_and_unknown():
    assert scaling.parse_max_force("0.0180257553") == pytest.approx(0.0180257553)
    assert scaling.parse_max_force("unknown") is None
    with pytest.raises(ValueError):
        scaling.parse_max_force("n/a")


def _setup_single_fcc_dispatch(monkeypatch, tmp_path, fake_run, *, continuation_segments=2):
    nodefile = tmp_path / "nodefile"
    nodefile.write_text("node0\n")
    monkeypatch.setattr(scaling, "REPO", tmp_path)
    monkeypatch.setattr(scaling, "validate_manifest_data", lambda manifest: manifest)
    monkeypatch.setenv("PBS_JOBID", "test-job")
    monkeypatch.setenv("PBS_NODEFILE", str(nodefile))
    monkeypatch.setenv("E4_SCALING_AUTHORIZATION", "approved")
    monkeypatch.setenv("E4_SCALING_RUN_ID", "E4-TEST-run")
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
            "continuation_segments": continuation_segments,
            "execution_segment_hours": 24,
            "scheduler_margin_hours_per_segment": 1,
            "nelm_override": 400,
            "nelm_max": 400,
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
    tier_root = tmp_path / "runs/e4-scaling/strong/E4-TEST-run/n01/all"
    return manifest, tier_root


def test_failed_stage_unknown_force_does_not_crash_dispatcher(monkeypatch, tmp_path):
    def fake_run(args, *, env=None, stdout=None, stderr=None, **kwargs):
        if args[0] == "git":
            return type("Result", (), {"stdout": "test-commit"})()
        root = Path(env["E4_OUTPUT_ROOT"])
        root.mkdir(parents=True, exist_ok=True)
        with (root / "run-record.log").open("a") as record:
            record.write(
                "end tag=A31 rc=1 ionic_steps=3 max_force_eV_per_A=unknown "
                "seconds=120 status=failed:exit-1\n"
            )
            record.write("EXIT job=test calculation_id=FCC-1 status=1 final=A31:failed:exit-1\n")
        return type("Result", (), {"returncode": 1})()

    manifest, tier_root = _setup_single_fcc_dispatch(monkeypatch, tmp_path, fake_run)
    assert (
        scaling.dispatch(tmp_path / "unused.json", 1, None, False, manifest_override=manifest) == 1
    )
    result = next(
        json.loads(line)
        for line in (tier_root / "dispatch.jsonl").read_text().splitlines()
        if '"event": "finished"' in line
    )
    assert result["task_status"] == "failed_or_incomplete"
    assert result["final_max_force_eV_per_A"] is None
    assert result["final_status"] == "A31:failed:exit-1"
    assert result["stage_statuses"] == ["failed:exit-1"]
    assert result["continuation_segments_used"] == 1
    measurement = json.loads((tier_root / "measurements.jsonl").read_text().splitlines()[0])
    assert measurement["failed_calculation_ids"] == ["FCC-1"]
    assert measurement["valid_tasks"] == 0
    assert measurement["total_ionic_steps"] == 3


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

    manifest, _ = _setup_single_fcc_dispatch(monkeypatch, tmp_path, fake_run)
    assert (
        scaling.dispatch(tmp_path / "unused.json", 1, None, False, manifest_override=manifest) == 0
    )
    assert [env["E4_RESUME"] for env in seen] == ["0", "1"]
    assert len({env["E4_OUTPUT_ROOT"] for env in seen}) == 1
    assert [env["E4_SOFT_STOP_SECONDS"] for env in seen] == ["86400", "86400"]
    assert [env["E4_REQUESTED_WALLTIME_SECONDS"] for env in seen] == ["90000", "90000"]
    assert [env["E4_NELM_OVERRIDE"] for env in seen] == ["400", "400"]
    assert [env["E4_NELM_MAX"] for env in seen] == ["400", "400"]
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


@pytest.mark.skipif(
    not (REPO / SIX_CLASS_INPUTS).exists(), reason="E4 campaign inputs not committed"
)
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


def test_effective_nelm_reads_start_and_defaults_to_vasp_60():
    assert scaling.effective_nelm("ENCUT = 520\n") == 60
    assert scaling.effective_nelm("NELM = 200\n") == 200
    assert scaling.effective_nelm("ISMEAR = 1; NELM = 170 ! comment\n") == 170
    assert scaling.effective_nelm("# NELM = 999\nNELM = 80\nNELM=90\n") == 90


def test_task_nelm_max_prefers_override_then_frozen_then_start(monkeypatch, tmp_path):
    monkeypatch.setattr(scaling, "REPO", tmp_path)
    (tmp_path / "case").mkdir()
    (tmp_path / "case" / "START").write_text("ALGO = Normal\n")
    task = {"case_dir": "case", "launch_settings": {}}
    assert scaling.task_nelm_max(task) == 60
    assert scaling.task_nelm_max({**task, "launch_settings": {"nelm_max": 200}}) == 200
    assert scaling.task_nelm_max({**task, "launch_settings": {"nelm_override": 300}}) == 300
    both = {**task, "launch_settings": {"nelm_max": 200, "nelm_override": 400}}
    assert scaling.task_nelm_max(both) == 400


def test_task_without_walltime_estimate_is_rejected():
    with pytest.raises(KeyError, match="estimated_task_walltime_hours"):
        scaling.task_continuation_schedule({"calculation_id": "legacy-bcc"})


def test_task_validation_reports_all_missing_required_fields():
    with pytest.raises(ValueError) as error:
        scaling.validate_task({"calculation_id": "incomplete"})
    message = str(error.value)
    assert "incomplete" in message
    assert "case_dir" in message
    assert "estimated_task_walltime_hours" in message


def test_dispatch_dry_run_prints_readable_plan(monkeypatch, capsys):
    monkeypatch.setattr(scaling, "validate_task", lambda task: None)
    assert (
        scaling.main(
            [
                "dispatch",
                "--manifest",
                str(PLANS / "strong-manifest.json"),
                "--nodes",
                "1",
                "--segment-id",
                "segment-1",
                "--dry-run",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "E4 strong dispatch" in output
    assert "wave 1:" in output
    assert "execution_authorized=false" in output


def test_preflight_reports_pbs_blockers(monkeypatch, capsys):
    monkeypatch.setattr(scaling, "validate_task", lambda task: None)
    monkeypatch.delenv("PBS_JOBID", raising=False)
    monkeypatch.delenv("PBS_NODEFILE", raising=False)
    monkeypatch.delenv("E4_SCALING_RUN_ID", raising=False)
    assert (
        scaling.main(
            [
                "preflight",
                "--manifest",
                str(PLANS / "strong-manifest.json"),
                "--nodes",
                "1",
                "--segment-id",
                "segment-1",
            ]
        )
        == 1
    )
    output = capsys.readouterr().out
    assert "BLOCKED: manifest is not execution-authorized" in output
    assert "BLOCKED: PBS_JOBID is not set" in output
    assert "ready=false" in output


def test_main_formats_expected_errors_without_traceback(monkeypatch, capsys):
    def invalid_manifest(path):
        raise ValueError("task FCC-1 is missing source_poscar")

    monkeypatch.setattr(scaling, "validate_manifest", invalid_manifest)
    assert scaling.main(["validate", "missing.json"]) == 2
    captured = capsys.readouterr()
    assert "error: task FCC-1 is missing source_poscar" in captured.err
    assert "Traceback" not in captured.err


def _six_class_stub(authorized: bool, class_authorization):
    names = ("bcc_refractory", "fcc_hea", "lifepo4", "phosphorene", "zn_formate_mof", "nb_ta_o")
    return {
        "execution_authorized": authorized,
        "authorization_request": {"additional_jobs_authorized": authorized},
        "material_classes": [
            {
                "material_class": name,
                "execution_authorized": authorized,
                "authorization": class_authorization,
                "candidates": [],
                "strong_scaling": {"tiers": []},
                "weak_scaling": {"tiers": []},
            }
            for name in names
        ],
    }


def test_six_class_validator_separates_planning_and_execution():
    planning = _six_class_stub(False, None)
    validate_six_class_workload_manifest(planning, verify_files=False)
    authorized = _six_class_stub(True, {"id": "approved"})
    with pytest.raises(ValueError, match="execution_authorized=false"):
        validate_six_class_workload_manifest(authorized, verify_files=False)
    validate_six_class_workload_manifest(authorized, verify_files=False, for_execution=True)
    with pytest.raises(ValueError, match="authorization ID"):
        validate_six_class_workload_manifest(
            _six_class_stub(True, None), verify_files=False, for_execution=True
        )
    with pytest.raises(ValueError, match="explicitly true or false"):
        validate_six_class_workload_manifest(
            {**planning, "execution_authorized": None}, verify_files=False, for_execution=True
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
