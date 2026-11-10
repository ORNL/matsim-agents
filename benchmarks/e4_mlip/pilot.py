"""Prepare or explicitly execute a bounded one-/two-tile Aurora qualification."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
from ase.io import read, write

from benchmarks.e4_mlip.analyze import consistency
from benchmarks.e4_mlip.run import (
    Structure,
    TimedCalculator,
    _device,
    numerical_values,
    resource_plan,
    sha256,
    validate_structure,
    verify_model,
    worker_task,
)
from matsim_agents.backends.mlip.pool import MLIPWorkerPool, _worker_calculators
from matsim_agents.backends.mlip.relaxation import RelaxStructureInput
from matsim_agents.execution.allocation import Allocation, discover_allocation

TOLERANCES = {"energy_eV": 1e-6, "force_eV_A": 1e-5, "position_A": 1e-4}


def save_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".pending.json")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class PilotTimeLimit(RuntimeError):
    pass


def validate_checkpoint(model) -> int:
    expected = getattr(model, "_hydragnn_checkpoint_state_keys", None)
    if not expected:
        raise ValueError("Strict HydraGNN checkpoint restoration was not recorded")
    actual = model.state_dict()
    missing = sorted(set(actual) - set(expected))
    unexpected = sorted(set(expected) - set(actual))
    if missing or unexpected:
        raise ValueError(f"Checkpoint state mismatch: missing={missing}, unexpected={unexpected}")
    return len(expected)


def bounded_relaxation(row: dict, args: dict, directory: Path, deadline: float) -> dict:
    from ase.optimize import FIRE

    device = _device()
    atoms = read(row["path"])
    cache = _worker_calculators()
    parsed = RelaxStructureInput.model_validate(
        {**args, "structure_path": row["path"], "output_dir": str(directory)}
    )
    device.synchronize()
    started = time.perf_counter()
    entry = cache.get(parsed)
    device.synchronize()
    load_seconds = time.perf_counter() - started
    matched_tensors = validate_checkpoint(entry.calculator.model)
    timed = TimedCalculator(entry.calculator, device)
    atoms.calc = timed
    optimizer = FIRE(atoms, maxstep=0.01, logfile=str(directory / "fire.log"))
    device.reset_peak_memory_stats()
    record = {}

    def observe():
        energy, forces = numerical_values(
            atoms.get_potential_energy(), atoms.get_forces(), len(atoms)
        )
        record.update(
            structure_id=row["structure_id"],
            phase="relaxation",
            energy_eV=energy,
            forces_eV_A=forces.tolist(),
            positions_A=atoms.positions.tolist(),
            cell_A=atoms.cell.array.tolist(),
            max_force_eV_A=float(np.linalg.norm(forces, axis=1).max()),
            steps=optimizer.nsteps,
            converged=False,
            status="RUNNING",
            synchronized=True,
            device_mask=os.environ["ZE_AFFINITY_MASK"],
            visible_device_count=1,
            peak_allocated_bytes=device.max_memory_allocated(),
            model_load_seconds=load_seconds,
            operation_seconds=time.perf_counter() - started,
            numerical_seconds=timed.seconds,
            force_evaluations=timed.evaluations,
            cold_model=True,
            model_tensors_matched=matched_tensors,
        )
        save_json(directory / "result.json", record)
        write(directory / "latest.extxyz", atoms)
        if time.monotonic() >= deadline:
            raise PilotTimeLimit("Resource cap reached at optimizer boundary")

    optimizer.attach(observe, interval=1)
    try:
        converged = optimizer.run(fmax=0.02, steps=200)
        record.update(status="ok", converged=bool(converged), stop_reason="completed")
    except PilotTimeLimit as exc:
        record.update(status="INCOMPLETE_RESOURCE_CAP", converged=None, stop_reason=str(exc))
    save_json(directory / "result.json", record)
    return record


def load_representatives(path: Path) -> list[Structure]:
    rows = [Structure.model_validate(row) for row in json.loads(path.read_text())]
    if not 1 <= len(rows) <= 3 or len({row.family for row in rows}) != len(rows):
        raise ValueError("Pilot needs exactly one representative per selected family (1 to 3)")
    if len({row.structure_id for row in rows}) != len(rows):
        raise ValueError("Duplicate pilot structure ID")
    for row in rows:
        validate_structure(row)
    return rows


def pilot_worker(payload: dict) -> dict:
    device = _device()
    mask = os.environ["ZE_AFFINITY_MASK"]
    destination = Path(payload["output"]) / f"tile-{mask}"
    destination.mkdir(parents=True, exist_ok=False)
    records = []
    for row in payload["structures"]:
        phases = (
            ("cold-inference", "warm-inference-1", "warm-inference-2")
            if payload["phase_group"] == "inference"
            else ("relaxation",)
        )
        for phase in phases:
            directory = destination / row["structure_id"] / phase
            directory.mkdir(parents=True)
            if time.monotonic() >= payload["deadline"]:
                records.append(
                    {
                        "structure_id": row["structure_id"],
                        "phase": phase,
                        "status": "INCOMPLETE_RESOURCE_CAP",
                        "converged": None,
                    }
                )
                continue
            if phase == "relaxation":
                records.append(
                    bounded_relaxation(row, payload["relax_args"], directory, payload["deadline"])
                )
                continue
            result = worker_task(
                {
                    "structure": row,
                    "args": {
                        **payload["relax_args"],
                        "structure_path": row["path"],
                        "output_dir": str(directory),
                    },
                    "mode": "inference",
                    "result_path": str(directory / "result.json"),
                    "submitted_at": payload["submitted_at"],
                    "warmup": phase == "cold-inference",
                },
                validate_checkpoint=validate_checkpoint,
            )
            result["phase"] = phase
            save_json(directory / "result.json", result)
            records.append(result)
    import torch

    checkpoint = torch.load(
        payload["relax_args"]["checkpoint"], map_location="cpu", weights_only=False
    )
    branch = torch.load(
        payload["relax_args"]["hydragnn_branch_mlp_checkpoint"],
        map_location="cpu",
        weights_only=False,
    )
    built = _worker_calculators().get(
        RelaxStructureInput.model_validate(
            {**payload["relax_args"], "structure_path": payload["structures"][0]["path"]}
        )
    )
    matched_tensors = validate_checkpoint(built.calculator.model)
    association = {
        key: str(value)
        for key, value in branch.items()
        if key != "mlp_state_dict" and isinstance(value, (str, int, float, bool))
    }
    return {
        "mask": mask,
        "pid": os.getpid(),
        "device_total_memory_bytes": device.get_device_properties(0).total_memory,
        "records": records,
        "checkpoint_keys": sorted(checkpoint),
        "branch_checkpoint_keys": sorted(branch),
        "branch_metadata": association,
        "model_state_keys_match": True,
        "model_tensors_matched": matched_tensors,
        "branch_output_dimension": next(
            layer.out_features
            for layer in reversed(list(built.calculator.mlp.modules()))
            if isinstance(layer, torch.nn.Linear)
        ),
        "model_branches": built.num_branches,
    }


def qualification(stages: dict, tolerances: dict) -> list[str]:
    failures = []
    if set(stages) != {"1", "2"} or len(stages["1"]) != 1 or len(stages["2"]) != 2:
        return ["Missing one-/two-worker results"]
    workers = stages["2"]
    if len({worker["mask"] for worker in workers}) != 2:
        failures.append("Two workers share a device mask")
    if len({worker["pid"] for worker in workers}) != 2:
        failures.append("Two workers share a process")
    reference = {(row["structure_id"], row["phase"]): row for row in stages["1"][0]["records"]}
    expected = set(reference)
    if not expected:
        return failures + ["No pilot records"]
    phases = {"cold-inference", "warm-inference-1", "warm-inference-2", "relaxation"}
    identifiers = {key[0] for key in expected}
    if expected != {(identifier, phase) for identifier in identifiers for phase in phases}:
        return failures + ["Incomplete pilot phases"]
    for stage in stages.values():
        for worker in stage:
            if worker["device_total_memory_bytes"] <= 0:
                failures.append(f"tile {worker['mask']}: device memory capacity unverified")
            keys = {(row["structure_id"], row["phase"]) for row in worker["records"]}
            if keys != expected or len(keys) != len(worker["records"]):
                failures.append(f"tile {worker['mask']}: missing or duplicate measurements")
                continue
            for row in worker["records"]:
                prefix = f"tile {worker['mask']} {row['structure_id']} {row['phase']}"
                try:
                    numerical_values(row["energy_eV"], row["forces_eV_A"], len(row["positions_A"]))
                except ValueError as exc:
                    failures.append(f"{prefix}: {exc}")
                    continue
                if (
                    row["status"] != "ok"
                    or row["visible_device_count"] != 1
                    or row["device_mask"] != worker["mask"]
                    or not row["synchronized"]
                    or row["force_evaluations"] < 1
                ):
                    failures.append(f"{prefix}: placement or numerical execution invalid")
                if row["peak_allocated_bytes"] > 0.8 * worker["device_total_memory_bytes"]:
                    failures.append(f"{prefix}: less than 20% allocated-memory headroom")
                if row["phase"] != "cold-inference" and row["cold_model"]:
                    failures.append(f"{prefix}: unexpected checkpoint reload")
                baseline_phase = (
                    "relaxation" if row["phase"] == "relaxation" else "warm-inference-1"
                )
                baseline = reference[(row["structure_id"], baseline_phase)]
                failures.extend(
                    f"{prefix}: {failure}"
                    for failure in consistency(
                        {"results": [baseline]}, {"results": [row]}, tolerances
                    )
                )
                if row["phase"] == "relaxation" and (
                    not row["converged"] or row["max_force_eV_A"] > 0.02
                ):
                    failures.append(f"{prefix}: bounded relaxation not converged")
    return failures


def execute(rows: list[Structure], metadata: dict, output: Path) -> int:
    allocation = discover_allocation()
    if allocation.scheduler != "pbs" or allocation.node_count != 1:
        raise ValueError("Pilot execution requires one existing PBS node")
    if os.environ.get("PBS_ACCOUNT") not in {None, "HydraGNN"}:
        raise ValueError("Pilot allocation must be HydraGNN")
    root = Path(metadata["hydragnn_root"]).resolve()
    frozen = json.loads((Path(__file__).parent / "frozen-source.json").read_text())
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != frozen["commit"] or revision != metadata["hydragnn_revision"]:
        raise ValueError("HydraGNN revision differs from frozen source")
    if subprocess.check_output(["git", "-C", str(root), "diff", "HEAD", "--"], text=True):
        raise ValueError("Tracked frozen HydraGNN source has modifications")
    spec = importlib.util.find_spec("hydragnn")
    if spec is None or spec.origin is None or not Path(spec.origin).resolve().is_relative_to(root):
        raise ValueError("Imported HydraGNN must originate from frozen source")
    stages, errors = {}, []
    relax_args = {
        "mlip_backend": "hydragnn",
        "logdir": str(Path(metadata["config"]["path"]).parent),
        "checkpoint": metadata["checkpoint"]["path"],
        "hydragnn_branch_mlp_checkpoint": metadata["branch_checkpoint"]["path"],
        "mlp_device": "xpu",
        "precision": "fp64",
        "mlp_precision": "fp64",
        "optimizer": "FIRE",
        "fmax": 0.02,
        "maxiter": 200,
        "maxstep": 0.01,
        "relax_cell": False,
    }
    started = time.perf_counter()
    deadline = time.monotonic() + 720
    identity = {
        "source_revision": revision,
        "hydragnn_import": spec.origin,
        "checkpoint": metadata["checkpoint"],
        "branch_checkpoint": metadata["branch_checkpoint"],
        "config": metadata["config"],
        "pbs_job_id": os.environ["PBS_JOBID"],
        "scientific_accuracy_validated": False,
        "production_submission_authorized": False,
        "parameters": relax_args,
        "tolerances": TOLERANCES,
        "inference_priority": True,
        "pool_lifetime_note": (
            "Inference pools close before optional relaxation pools; warm inference reuses "
            "each model. Relaxation phases deliberately start fresh workers."
        ),
    }
    save_json(output / "runtime-identity.json", identity)
    for phase_group, devices in (
        ("inference", 1),
        ("inference", 2),
        ("relaxation", 1),
        ("relaxation", 2),
    ):
        if errors or time.monotonic() >= deadline:
            break
        directory = output / f"{phase_group}-devices-{devices}"
        directory.mkdir()
        placement = resource_plan(devices).resolve(allocation)
        placement.write(directory)
        with MLIPWorkerPool.from_placement(placement) as pool:
            outcomes = pool.on_each_worker(
                pilot_worker,
                {
                    "structures": [row.model_dump(mode="json") for row in rows],
                    "relax_args": relax_args,
                    "output": str(directory),
                    "submitted_at": time.perf_counter(),
                    "phase_group": phase_group,
                    "deadline": deadline,
                },
            )
        stages[f"{phase_group}-{devices}"] = [item.value for item in outcomes if item.ok]
        errors.extend(str(item.error) for item in outcomes if not item.ok)
        save_json(
            output / "pilot-progress.json",
            {
                **identity,
                "stages": stages,
                "errors": errors,
                "elapsed_seconds": time.perf_counter() - started,
            },
        )
        if errors:
            break
    failures = list(errors)
    differences = []
    baseline = stages.get("inference-1", [])
    two_workers = stages.get("inference-2", [])
    if len(baseline) != 1 or len(two_workers) != 2:
        failures.append("Incomplete one-/two-worker inference")
    elif (
        len({worker["mask"] for worker in two_workers}) != 2
        or len({worker["pid"] for worker in two_workers}) != 2
    ):
        failures.append("Two-worker isolation failed")
    for key, workers in stages.items():
        for worker in workers:
            if worker["branch_output_dimension"] != worker["model_branches"]:
                failures.append(f"{key}: branch MLP output dimension mismatch")
            for row in worker["records"]:
                if row["status"] != "ok":
                    if key.startswith("inference"):
                        failures.append(f"{key}: incomplete inference")
                    continue
                if row["peak_allocated_bytes"] > 0.8 * worker["device_total_memory_bytes"]:
                    failures.append(f"{key}: memory headroom below 20%")
                if row["visible_device_count"] != 1 or row["device_mask"] != worker["mask"]:
                    failures.append(f"{key}: invalid tile visibility")
                if (
                    key.startswith("inference")
                    and row["phase"].startswith("warm")
                    and row["cold_model"]
                ):
                    failures.append(f"{key}: unexpected warm model reload")
                if key.startswith("inference") and baseline:
                    reference = next(
                        item
                        for item in baseline[0]["records"]
                        if item["structure_id"] == row["structure_id"]
                        and item["phase"] == "warm-inference-1"
                    )
                    differences.append(
                        {
                            "stage": key,
                            "mask": worker["mask"],
                            "phase": row["phase"],
                            "structure_id": row["structure_id"],
                            "energy_absolute_difference_raw": abs(
                                row["energy_eV"] - reference["energy_eV"]
                            ),
                            "force_component_max_difference_raw": float(
                                np.max(
                                    np.abs(
                                        np.asarray(row["forces_eV_A"])
                                        - np.asarray(reference["forces_eV_A"])
                                    )
                                )
                            ),
                        }
                    )
    units_verified = metadata.get("energy_convention_verified", False)
    numerical_failures = []
    if units_verified:
        for difference in differences:
            if (
                difference["energy_absolute_difference_raw"] > TOLERANCES["energy_eV"]
                or difference["force_component_max_difference_raw"] > TOLERANCES["force_eV_A"]
            ):
                numerical_failures.append(difference)
    else:
        units_verified = False
    report = {
        **identity,
        "status": "FAILED" if failures else "TECHNICALLY_QUALIFIED",
        "failures": failures,
        "stages": stages,
        "elapsed_seconds": time.perf_counter() - started,
        "numerical_differences": differences,
        "numerical_failures": numerical_failures,
        "numerical_status": ("FAILED" if numerical_failures else "PASSED_INFERENCE")
        if units_verified
        else "DEFERRED_UNVERIFIED_UNITS",
        "output_units_verified": units_verified,
        "output_label_note": (
            "Legacy *_eV and *_eV_A fields retain raw calculator values; labels are not "
            "unit verification. Absolute energies are not verified native DFT totals."
        ),
        "relaxation_status_note": (
            "INCOMPLETE_RESOURCE_CAP means uncompleted, not numerical nonconvergence. "
            "All relaxation is technical optimizer exercise, not scientific validation."
        ),
    }
    (output / "pilot-results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    rows = load_representatives(args.manifest)
    metadata = json.loads(args.metadata.read_text())
    blockers = verify_model({**metadata, "numerical_tolerances_approved": True}, rows)
    supported = set(metadata.get("pilot_supported_families", []))
    if any(row.family not in supported for row in rows):
        blockers.append("Pilot material families require checkpoint-specific coverage evidence")
    systems_only = metadata.get("pilot_scope") == "approved_controlled_systems_only"
    if systems_only:
        # verify_model above still validates all artifact hashes before scope gating.
        blockers = [
            blocker
            for blocker in blockers
            if blocker == "Optional backend has not been qualified for this benchmark"
        ]
    args.output.mkdir(parents=True, exist_ok=False)
    preview = {
        "status": "NO-GO" if blockers else "PREPARED",
        "execution_requested": args.execute,
        "systems_only": systems_only,
        "units_verified": metadata.get("energy_convention_verified", False),
        "blockers": blockers,
        "manifest_sha256": sha256(args.manifest),
        "model_metadata_sha256": sha256(args.metadata),
        "tolerances": TOLERANCES,
        "tolerances_status": "Provisional reproducibility criteria; not DFT accuracy thresholds",
        "nodes": 1,
        "walltime": "00:15:00",
        "maximum_node_hours": 0.25,
        "devices_sequential": [1, 2],
        "placement": [
            resource_plan(count)
            .resolve(Allocation("preview", ("pilot-node",)))
            .model_dump(mode="json")
            for count in (1, 2)
        ],
    }
    (args.output / "pilot-preflight.json").write_text(json.dumps(preview, indent=2) + "\n")
    if not args.execute:
        print(
            json.dumps(
                {key: value for key, value in preview.items() if key != "placement"}, indent=2
            )
        )
        return 0
    if blockers:
        raise ValueError(f"Pilot execution blocked: {blockers}")
    return execute(rows, metadata, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
