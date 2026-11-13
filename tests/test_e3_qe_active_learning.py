"""E3/QE policy and plumbing tests; no real QE, GPU or LLM calls."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.io import write

from benchmarks.e3 import production as e3
from benchmarks.e3.qe_active_learning import E3Sampler, check_resume, main, validate_config
from matsim_agents.active_learning.candidates import Candidate
from matsim_agents.active_learning.config import ALConfig, TrainerConfig
from matsim_agents.active_learning.loop import _audit_force_errors
from matsim_agents.execution.allocation import Allocation

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def qe_config(tmp_path, monkeypatch):
    checkpoint = tmp_path / "initial.model"
    checkpoint.write_bytes(b"synthetic-checkpoint")
    monkeypatch.setattr(
        e3, "E3_MLIP_CHECKPOINT_SHA256", hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    )
    heldout = tmp_path / "heldout.extxyz"
    heldout.touch()
    references = tmp_path / "references.json"
    references.touch()
    for key, value in {
        "PROJ_ROOT": ROOT,
        "MACE_CHECKPOINT": checkpoint,
        "PW_BIN": tmp_path / "pw.x",
        "PSEUDO_DIR": tmp_path,
        "E3_QE_PSEUDOPOTENTIALS": "{Nb: Nb.UPF, Ta: Ta.UPF, O: O.UPF}",
        "E3_QE_ECUTWFC_RY": "80",
        "E3_QE_ECUTRHO_RY": "640",
        "E3_QE_KPTS": "[2,2,2]",
        "E3_QE_OCCUPATIONS": "fixed",
        "E3_QE_SYSTEM": "{input_dft: PBE}",
        "E3_QE_VALIDATION_SET": heldout,
        "ELEMENTAL_REFERENCE_MANIFEST": references,
        "E3_QE_PROMOTION_APPROVED": "false",
        "E3_QE_RUN_DIR": tmp_path / "run",
        "MATSIM_E3_MLIP_STRUCTURES_PER_ROUND": "5000",
    }.items():
        monkeypatch.setenv(key, str(value))
    path = ROOT / "examples/active_learning/e3_qe_mace.yaml"
    return ALConfig.from_yaml(path), path


def test_template_policy_and_placement(qe_config):
    cfg, _ = qe_config
    validate_config(cfg, "nb_ta_o")
    cfg.dft.qe.pseudopotentials["P"] = "P.UPF"
    validate_config(cfg, "phosphorene")
    cfg.dft.qe.pseudopotentials.pop("P")
    with pytest.raises(ValueError, match="pseudopotential mapping lacks"):
        validate_config(cfg, "phosphorene")
    assert cfg.trainer.force_error_trigger_eV_per_A == 0.2
    assert cfg.loop.max_dft_calculations == 48
    assert cfg.acquisition.n_select == 16
    assert not cfg.trainer.promotion_approved
    placement = cfg.resources.resolve(
        Allocation(scheduler="slurm", nodes=tuple(f"node-{i}" for i in range(16)), gpus_per_node=4)
    )
    assert placement.pool("dft").max_concurrent == 16
    assert placement.pool("dft").device_count == 64
    assert placement.pool("mlip").device_count == 1


@pytest.mark.parametrize("threshold", [0, -0.1, float("nan"), float("inf")])
def test_invalid_force_error_tolerance(threshold):
    with pytest.raises(ValueError):
        TrainerConfig(force_error_trigger_eV_per_A=threshold)


def test_check_does_not_create_outputs(qe_config, capsys):
    cfg, path = qe_config
    assert main(["check", "--config", str(path), "--material", "nb_ta_o"]) == 0
    policy = json.loads(capsys.readouterr().out)
    assert policy["max_structures_per_iteration"] == 5000
    assert not cfg.loop.out_dir.exists()


def test_run_requires_approval_and_allocation(qe_config, monkeypatch):
    cfg, path = qe_config
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(RuntimeError, match="approve-compute"):
        main(["run", "--config", str(path), "--material", "nb_ta_o"])
    assert not cfg.loop.out_dir.exists()


@pytest.mark.parametrize(
    "change", ["wrong_model", "missing_trigger", "wrong_budget", "md_acquisition"]
)
def test_invalid_variant_policy(qe_config, change):
    cfg, _ = qe_config
    if change == "wrong_model":
        Path(cfg.mlip.mace.model).write_bytes(b"wrong")
    elif change == "missing_trigger":
        cfg.trainer.force_error_trigger_eV_per_A = None
    elif change == "wrong_budget":
        cfg.loop.max_dft_calculations = 49
    else:
        cfg.acquisition.strategy = "mc_dropout"
    with pytest.raises(ValueError):
        validate_config(cfg, "nb_ta_o")


def test_partial_evidence_is_never_deleted(tmp_path):
    (tmp_path / "e3-qe-policy.json").write_text("{}")
    partial = tmp_path / "iteration_0000"
    partial.mkdir()
    evidence = partial / "qe.out"
    evidence.write_text("retained")
    with pytest.raises(RuntimeError, match="Interrupted iteration retained"):
        check_resume(tmp_path)
    assert evidence.read_text() == "retained"
    (partial / "state.json").write_text(json.dumps({"status": "complete"}))
    check_resume(tmp_path)
    (partial / "generation-stop.json").write_text('{"reason": "capacity"}')
    with pytest.raises(RuntimeError, match="scientifically stopped"):
        check_resume(tmp_path)


def test_refuse_to_adopt_unrelated_results(tmp_path):
    (tmp_path / "dataset.extxyz").touch()
    with pytest.raises(RuntimeError, match="without an E3-QE policy"):
        check_resume(tmp_path)


def audit_example():
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.75]], cell=[5, 5, 5], pbc=True)
    candidate = Candidate("candidate", atoms, "seed.extxyz", 0)
    result = SimpleNamespace(
        backend="qe",
        converged=True,
        work_dir="/dft/candidate",
        final_atoms=atoms.copy(),
        forces_eV_per_A=np.full((2, 3), 0.3),
    )
    return candidate, result


def test_audit_measures_component_mae_not_force_norm():
    candidate, result = audit_example()
    errors = _audit_force_errors([candidate], [result], {"candidate": np.zeros((2, 3))})
    assert errors["candidate"] == pytest.approx(0.3)


@pytest.mark.parametrize("invalid", ["geometry", "species", "cell", "nan", "shape", "missing"])
def test_audit_rejects_incomparable_or_invalid_labels(invalid):
    candidate, result = audit_example()
    if invalid == "geometry":
        result.final_atoms.positions[0, 0] += 0.1
    elif invalid == "species":
        result.final_atoms.numbers[0] = 2
    elif invalid == "cell":
        result.final_atoms.set_cell([6, 6, 6])
    elif invalid == "nan":
        result.forces_eV_per_A[0, 0] = np.nan
    elif invalid == "shape":
        result.forces_eV_per_A = np.zeros(3)
    else:
        result.forces_eV_per_A = None
    with pytest.raises(ValueError):
        _audit_force_errors([candidate], [result], {"candidate": np.zeros((2, 3))})


def test_failed_audit_is_not_evidence_of_accuracy():
    candidate, result = audit_example()
    result.converged = False
    assert _audit_force_errors([candidate], [result], {"candidate": np.zeros((2, 3))}) == {}


def test_qe_print_roundoff_is_normalized_to_labelled_input():
    candidate, result = audit_example()
    result.final_atoms.positions[0, 0] += 2e-6
    result.final_atoms.cell[0, 0] += 2e-6
    _audit_force_errors([candidate], [result], {"candidate": np.zeros((2, 3))})
    np.testing.assert_array_equal(result.final_atoms.positions, candidate.atoms.positions)
    np.testing.assert_array_equal(result.final_atoms.cell, candidate.atoms.cell)
    assert result.final_atoms.info["qe_output_geometry_tolerance_A"] == 1e-5


def test_qe_evidence_reaches_actual_llm_prompt():
    actions = e3._action_space("phosphorene")
    allowed = next(action for action in actions if e3._finite_capacity_group(action) is None)
    history = [
        {
            "evidence_id": "qe-audit-0",
            "qe_audits_by_action": {"action": {"n_audited": 1, "maximum_force_mae_eV_per_A": 0.4}},
            "accuracy_triggered": True,
            "model_promoted": True,
        }
    ]

    class Chat:
        def invoke(self, prompt):
            assert "no DFT labels" not in prompt
            assert "maximum_force_mae_eV_per_A" in prompt
            assert "do not compare raw energies" in prompt
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "allocations": [{"action_id": allowed["action_id"], "count": 2}],
                        "cited_evidence_ids": ["qe-audit-0"],
                        "rationale": "Revisit the audited action.",
                    }
                )
            )

    _, citations, _ = e3._llm_allocation(
        Chat(),
        "phosphorene",
        actions,
        history,
        target=2,
        minimum_by_tier={str(allowed["size_tier"]): 2},
        context={"campaign_variant": "e3-mlip-qe-active-learning"},
    )
    assert citations == ["qe-audit-0"]


def test_geometry_worker_uses_and_records_promoted_checkpoint(tmp_path, monkeypatch):
    from ase.calculators.emt import EMT

    from matsim_agents.backends.mlip import relaxation
    from matsim_agents.backends.mlip.results import RelaxationResult

    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    checkpoint = tmp_path / "promoted.model"
    checkpoint.write_bytes(b"promoted")
    geometry = tmp_path / "input.extxyz"
    write(geometry, Atoms("Cu", positions=[[0, 0, 0]], cell=[3, 3, 3], pbc=True))
    model = {"path": str(checkpoint), "sha256": e3._sha256_file(checkpoint)}
    task = {
        "candidate_id": "candidate",
        "candidate_kind": "unique_configuration",
        "structure_sha256": e3._geometry_sha256(geometry),
        "structure_path": str(geometry),
        "material": "bcc_refractory",
        "active_model": model,
        "campaign_variant": "e3-mlip-qe-active-learning",
    }

    class Cache:
        def get(self, args):
            assert args.mace_model == str(checkpoint)
            assert args.relative_increase_threshold is None
            assert args.max_cell_step_strain is None
            assert args.max_cell_total_strain is None
            assert args.stress_tolerance_eV_per_A3 is None
            return SimpleNamespace(calculator=EMT())

    def relax(args, **kwargs):
        return RelaxationResult(
            structure_path=args.structure_path,
            optimized_structure_path=args.structure_path,
            trajectory_path="trajectory.extxyz",
            log_csv_path="log.csv",
            final_energy_eV=0,
            final_max_force_eV_per_A=0,
            num_steps=1,
            converged=True,
        )

    monkeypatch.setattr(relaxation, "relax_with_mlip", relax)
    result = e3._execute_task(task, tmp_path / "tasks", Cache())
    assert result["status"] == "converged"
    assert result["checkpoint_sha256"] == model["sha256"]
    assert result["mace_model_path"] == model["path"]
    assert result["active_model"] == model
    assert result["model"] != e3.MODEL_NAME
    timeout_task = {**task, "candidate_id": "timeout", "optimizer": e3._hard_optimizer_policy(300)}
    import time

    timeout = e3._timeout_result(timeout_task, tmp_path / "tasks", time.monotonic(), time.time())
    assert timeout["checkpoint_sha256"] == model["sha256"]
    assert timeout["mace_model"] == model["path"]


def test_promoted_checkpoint_verified_and_resume_is_model_specific(tmp_path):
    checkpoint = tmp_path / "promoted.model"
    checkpoint.write_bytes(b"promoted")
    model = {"path": str(checkpoint), "sha256": e3._sha256_file(checkpoint)}
    task = {
        "candidate_id": "c",
        "structure_sha256": "geometry",
        "active_model": model,
        "campaign_variant": "e3-mlip-qe-active-learning",
    }
    assert e3._task_checkpoint(task) == (checkpoint, model["sha256"])
    attempt = tmp_path / "attempt-0001"
    attempt.mkdir()
    result = {
        "terminal": True,
        "structure_sha256": "geometry",
        "optimizer": e3.OPTIMIZER_SETTINGS,
        "active_model": model,
    }
    (attempt / "result.json").write_text(json.dumps(result))
    assert e3._task_result(tmp_path, task) is not None
    assert e3._task_result(tmp_path, {**task, "active_model": {**model, "sha256": "other"}}) is None
    checkpoint.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        e3._task_checkpoint(task)
    with pytest.raises(ValueError, match="separate E3-QE"):
        e3._task_checkpoint({**task, "campaign_variant": "frozen"})


def test_e3_sampler_reuses_generation_workers_and_feeds_qe_to_llm(qe_config, monkeypatch):
    cfg, _ = qe_config
    root = cfg.loop.out_dir
    previous = root / "iteration_0000"
    previous.mkdir(parents=True)
    (previous / "state.json").write_text(
        json.dumps(
            {
                "dft_force_errors_eV_per_A": {"old": 0.4},
                "accuracy_triggered": True,
                "model_promoted": True,
            }
        )
    )
    (previous / "e3-audit-pool.json").write_text(
        json.dumps([{"candidate_id": "old", "action_id": "action-old"}])
    )
    current = root / "iteration_0001"
    current.mkdir()
    round_dir = root / "round-02"
    round_dir.mkdir()
    geometry = root / "geometry.extxyz"
    write(geometry, Atoms("Si", positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True))
    tasks = [
        {"candidate_id": str(i), "structure_path": str(geometry), "action_id": "action"}
        for i in range(5000)
    ]
    monkeypatch.setattr(e3, "_history_for_round", lambda *a: [])
    monkeypatch.setattr(e3, "_chat_model", lambda: object())
    captured = {}

    def materialize(manifest, spec, output, history, chat):
        captured.update(manifest=manifest, history=history, spec=spec)
        return round_dir / "round-manifest.json", round_dir / "tasks.jsonl", {}

    monkeypatch.setattr(e3, "_materialize_round", materialize)
    monkeypatch.setattr(e3, "_read_tasks", lambda *a: tasks)
    monkeypatch.setattr(
        e3, "_execute_round_tasks", lambda *a, **k: captured.update(worker_kwargs=k)
    )
    monkeypatch.setattr(e3, "_round_action_outcomes", lambda *a: [])
    result_path = round_dir / "result.json"
    result_path.write_text(
        json.dumps({"status": "converged", "optimized_structure_path": str(geometry)})
    )
    monkeypatch.setattr(e3, "_task_result", lambda *a: result_path)
    candidates = E3Sampler("nb_ta_o")(cfg, 1, current)
    assert len(candidates) == 5000
    assert captured["spec"]["main_target"] == 5000
    assert (
        captured["history"][0]["qe_audits_by_action"]["action-old"]["maximum_force_mae_eV_per_A"]
        == 0.4
    )
    assert tasks[0]["active_model"]["path"] == cfg.mlip.mace.model
    assert tasks[0]["optimizer"]["relative_increase_threshold"] is None
    assert tasks[0]["optimizer"]["max_cell_step_strain"] is None
    assert tasks[0]["optimizer"]["max_cell_total_strain"] is None
    assert tasks[0]["optimizer"]["stress_tolerance_eV_per_A3"] is None
    assert captured["worker_kwargs"]["hard_optimization_seconds"] == 300
    assert candidates[0].atoms.info["sampling_method"] == "e3_geometry_optimization"
