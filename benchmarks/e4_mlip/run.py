"""Validate and run E4-B using the existing MLIP pool and calculator cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
from ase.io import read
from pydantic import BaseModel, ConfigDict, Field

from matsim_agents.backends.mlip.pool import MLIPWorkerPool, _worker_calculators
from matsim_agents.backends.mlip.relaxation import RelaxStructureInput, relax_with_mlip
from matsim_agents.execution.allocation import Allocation, discover_allocation
from matsim_agents.execution.placement import ResourcePlan

DEVICES = (1, 2, 4, 8, 12)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class Structure(BaseModel):
    model_config = ConfigDict(extra="forbid")
    structure_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
    family: Literal["bcc_refractory", "fcc_hea", "lifepo4"]
    path: Path
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    geometry_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    atom_count: int = Field(gt=0)
    species: list[str]
    source: str


def geometry_hash(atoms) -> str:
    payload = {
        "numbers": atoms.numbers.tolist(),
        "positions": np.round(atoms.positions, 10).tolist(),
        "cell": np.round(atoms.cell.array, 10).tolist(),
        "pbc": atoms.pbc.tolist(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def validate_structure(row: Structure):
    if sha256(row.path) != row.sha256:
        raise ValueError(f"{row.structure_id}: input hash mismatch")
    atoms = read(row.path)
    if (
        len(atoms) != row.atom_count
        or sorted(set(atoms.get_chemical_symbols())) != row.species
        or not np.isfinite(atoms.positions).all()
        or not np.isfinite(atoms.cell.array).all()
        or not atoms.pbc.all()
        or atoms.get_volume() <= 0
        or geometry_hash(atoms) != row.geometry_sha256
    ):
        raise ValueError(f"{row.structure_id}: invalid structure or metadata")
    expected = 112 if row.family == "lifepo4" else 128
    if len(atoms) != expected:
        raise ValueError(f"{row.structure_id}: expected {expected} atoms")
    return atoms


def load_inventory(path: Path) -> list[Structure]:
    rows = [Structure.model_validate(row) for row in json.loads(path.read_text())]
    if len(rows) != 120:
        raise ValueError(
            f"Primary fixed workload requires exactly 120 structures, found {len(rows)}"
        )
    for field in ("structure_id", "sha256", "geometry_sha256"):
        if len({getattr(row, field) for row in rows}) != len(rows):
            raise ValueError(f"Duplicate {field} in inventory")
    for row in rows:
        validate_structure(row)
    return rows


def resource_plan(devices: int) -> ResourcePlan:
    if devices not in DEVICES:
        raise ValueError(f"Supported logical device counts: {DEVICES}")
    return ResourcePlan.model_validate(
        {"preset": "aurora", "pools": {"mlip": {"nodes": 1, "devices": devices}}}
    )


def verify_model(metadata: dict[str, Any], rows: list[Structure]) -> list[str]:
    blockers = []
    for key in ("checkpoint", "branch_checkpoint", "config"):
        if key not in metadata:
            raise ValueError(
                f"Frozen metadata is incomplete: missing {key}; upload and re-audit weights"
            )
        entry = metadata[key]
        if sha256(Path(entry["path"])) != entry["sha256"]:
            raise ValueError(f"Model {key} hash mismatch")
    if metadata.get("backend") != "hydragnn":
        blockers.append("Optional backend has not been qualified for this benchmark")
    if not metadata.get("species_verified") or not metadata.get("species_evidence"):
        blockers.append("Checkpoint species support is not verified")
    selected = {symbol for row in rows for symbol in row.species}
    unsupported = selected - set(metadata.get("supported_species", []))
    if unsupported:
        blockers.append(f"Unsupported or unverified species: {sorted(unsupported)}")
    if metadata.get("prediction_targets") != ["energy", "forces"]:
        blockers.append("Energy/force capability is not declared")
    if not metadata.get("numerical_tolerances_approved"):
        blockers.append("Numerical consistency tolerances require approval before execution")
    return blockers


def _device():
    import torch

    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        raise RuntimeError("XPU is required; CPU fallback is forbidden")
    if torch.xpu.device_count() != 1:
        raise RuntimeError("A worker must see exactly one assigned XPU tile")
    torch.xpu.set_device(0)
    return torch.xpu


def numerical_values(energy, forces, count: int) -> tuple[float, np.ndarray]:
    energy = float(energy)
    forces = np.asarray(forces, dtype=float)
    if not np.isfinite(energy) or forces.shape != (count, 3) or not np.isfinite(forces).all():
        raise ValueError("Invalid energy/force output")
    return energy, forces


class TimedCalculator:
    """Delegate ASE calculations, timing synchronized numerical evaluations only."""

    def __init__(self, inner, device):
        self.inner = inner
        self.device = device
        self.seconds = 0.0
        self.evaluations = 0

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def get_property(self, name, atoms=None, allow_calculation=True):
        needs_calculation = name not in self.inner.results or (
            atoms is not None and bool(self.inner.check_state(atoms))
        )
        if needs_calculation:
            self.device.synchronize()
            start = time.perf_counter()
            result = self.inner.get_property(name, atoms, allow_calculation)
            self.device.synchronize()
            self.seconds += time.perf_counter() - start
            self.evaluations += 1
            return result
        return self.inner.get_property(name, atoms, allow_calculation)

    def get_potential_energy(self, atoms=None, force_consistent=False):
        name = "free_energy" if force_consistent else "energy"
        return self.get_property(name, atoms)

    def get_forces(self, atoms=None):
        return self.get_property("forces", atoms)


class PreparedCache:
    def __init__(self, entry):
        self.entry = entry

    def get(self, args):
        return self.entry


def worker_task(payload: dict[str, Any]) -> dict[str, Any]:
    start = time.perf_counter()
    device = _device()
    row = Structure.model_validate(payload["structure"])
    parse_start = time.perf_counter()
    atoms = validate_structure(row)
    parse_seconds = time.perf_counter() - parse_start
    args = RelaxStructureInput.model_validate(payload["args"])
    cache = _worker_calculators()
    before = cache.loads
    device.synchronize()
    load_start = time.perf_counter()
    built = cache.get(args)
    device.synchronize()
    load_seconds = time.perf_counter() - load_start
    if "energy" not in built.calculator.implemented_properties or (
        "forces" not in built.calculator.implemented_properties
    ):
        raise ValueError("Calculator must implement energy and forces")
    model = getattr(built.calculator, "model", None)
    if model is None or any(parameter.device.type != "xpu" for parameter in model.parameters()):
        raise RuntimeError("Calculator model residency on XPU could not be verified")
    mlp = getattr(built.calculator, "mlp", None)
    if mlp is not None and any(parameter.device.type != "xpu" for parameter in mlp.parameters()):
        raise RuntimeError("Branch weighting model is not on XPU")
    timed = TimedCalculator(built.calculator, device)
    measured_entry = built._replace(calculator=timed)
    device.reset_peak_memory_stats()
    compute_start = time.perf_counter()
    if payload["mode"] == "inference":
        original = atoms.positions.copy()
        atoms.calc = timed
        energy, forces = numerical_values(
            atoms.get_potential_energy(), atoms.get_forces(), len(atoms)
        )
        if not np.array_equal(atoms.positions, original):
            raise ValueError("Inference changed atomic coordinates")
        steps = 0
        converged = None
        geometry = atoms.positions.tolist()
    else:
        result = relax_with_mlip(args, calculators=PreparedCache(measured_entry))
        atoms = read(result.optimized_structure_path)
        atoms.calc = timed
        energy, forces = numerical_values(
            atoms.get_potential_energy(), atoms.get_forces(), len(atoms)
        )
        steps, converged = result.num_steps, result.converged
        geometry = atoms.positions.tolist()
    device.synchronize()
    operation_seconds = time.perf_counter() - compute_start
    output = {
        "structure_id": row.structure_id,
        "status": "ok",
        "energy_eV": energy,
        "forces_eV_A": forces.tolist(),
        "max_force_eV_A": float(np.linalg.norm(forces, axis=1).max()),
        "positions_A": geometry,
        "cell_A": atoms.cell.array.tolist(),
        "converged": converged,
        "steps": steps,
        "parse_seconds": parse_seconds,
        "model_load_seconds": load_seconds if cache.loads != before else 0.0,
        "cache_loads": cache.loads,
        "cold_model": cache.loads != before,
        "numerical_seconds": timed.seconds,
        "operation_seconds": operation_seconds,
        "non_numerical_operation_seconds": max(0.0, operation_seconds - timed.seconds),
        "force_evaluations": timed.evaluations,
        "peak_allocated_bytes": device.max_memory_allocated(),
        "pid": os.getpid(),
        "device_mask": os.environ.get("ZE_AFFINITY_MASK"),
        "visible_device_count": device.device_count(),
        "synchronized": True,
        "worker_seconds": time.perf_counter() - start,
        "worker_start": start,
        "worker_end": time.perf_counter(),
        "startup_observed_seconds": max(0.0, start - payload["submitted_at"])
        if payload.get("warmup")
        else None,
    }
    write_start = time.perf_counter()
    Path(payload["result_path"]).write_text(json.dumps(output, allow_nan=False) + "\n")
    output["result_write_seconds"] = time.perf_counter() - write_start
    return output


def execute(args, rows, metadata, output: Path) -> int:
    import importlib.util

    source_root = Path(metadata["hydragnn_root"]).resolve()
    actual_revision = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_revision != metadata["hydragnn_revision"]:
        raise ValueError("HydraGNN source revision changed")
    module = importlib.util.find_spec("hydragnn")
    if (
        module is None
        or module.origin is None
        or not Path(module.origin).resolve().is_relative_to(source_root)
    ):
        raise ValueError("PYTHONPATH must select the frozen HydraGNN checkout")
    allocation = discover_allocation()
    if allocation.scheduler != "pbs" or allocation.node_count != 1:
        raise ValueError("Execution requires one existing PBS node; no local/GPU fallback")
    placement = resource_plan(args.devices).resolve(allocation)
    pool = MLIPWorkerPool.from_placement(placement)
    if len(pool.slots) != args.devices:
        raise ValueError("Resolved worker count differs from requested device count")
    relax_args = {
        "mlip_backend": args.backend,
        "logdir": str(Path(metadata["config"]["path"]).parent),
        "checkpoint": metadata["checkpoint"]["path"],
        "hydragnn_branch_mlp_checkpoint": metadata["branch_checkpoint"]["path"],
        "mlp_device": "xpu",
        "precision": args.precision,
        "mlp_precision": args.precision,
        "optimizer": "FIRE",
        "fmax": args.fmax,
        "maxiter": args.max_steps,
        "relax_cell": False,
    }
    output.mkdir(parents=True, exist_ok=False)
    placement.write(output)
    work_start = time.perf_counter()
    warmup_payload = {
        "structure": rows[0].model_dump(mode="json"),
        "args": {**relax_args, "structure_path": str(rows[0].path)},
        "mode": "inference",
        # Each worker writes its own file using its mask; no shared output paths.
        "result_path": str(output / "warmup-placeholder"),
        "warmup": True,
        "submitted_at": time.perf_counter(),
    }
    try:
        cold_start = time.perf_counter()
        warm = pool.on_each_worker(warmup_task, warmup_payload)
        cold_seconds = time.perf_counter() - cold_start
        if any(not item.ok for item in warm):
            errors = [str(item.error) for item in warm if not item.ok]
            (output / "failed-initialization.json").write_text(json.dumps(errors))
            raise RuntimeError(f"Worker initialization failed: {errors}")
        payloads = []
        for row in rows:
            directory = output / row.structure_id
            directory.mkdir()
            payloads.append(
                {
                    "structure": row.model_dump(mode="json"),
                    "args": {
                        **relax_args,
                        "structure_path": str(row.path),
                        "output_dir": str(directory),
                    },
                    "mode": args.mode,
                    "result_path": str(directory / "result.json"),
                    "submitted_at": time.perf_counter(),
                }
            )
        steady_start = time.perf_counter()
        outcomes = pool.map(worker_task, payloads)
        steady_seconds = time.perf_counter() - steady_start
    finally:
        pool.close()
    results = [
        item.value
        if item.ok
        else {"structure_id": row.structure_id, "status": "failed", "error": str(item.error)}
        for row, item in zip(rows, outcomes, strict=True)
    ]
    valid = sum(row["status"] == "ok" for row in results)
    converged = sum(row.get("converged") is True for row in results)
    # A worker replacement is cold again: it invalidates a steady-state scaling point.
    restarts = sum(row.get("cold_model", False) for row in results)
    report = {
        "schema_version": 1,
        "backend": args.backend,
        "mode": args.mode,
        "devices": args.devices,
        "repetition": args.repetition,
        "precision": args.precision,
        "fmax": args.fmax,
        "max_steps": args.max_steps,
        "optimizer": "FIRE",
        "relax_cell": False,
        "manifest_sha256": sha256(args.manifest),
        "model_metadata_sha256": sha256(args.metadata),
        "pbs_job_id": os.environ.get("PBS_JOBID"),
        "host": socket.gethostname(),
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "python": sys.executable,
        "cold_phase_seconds": cold_seconds,
        "warmups": [item.value for item in warm],
        "steady_seconds": steady_seconds,
        "total_seconds": time.perf_counter() - work_start,
        "attempted": len(rows),
        "valid": valid,
        "converged": converged,
        "worker_restart_count": restarts,
        "measurement_valid": valid == len(rows) and restarts == 0,
        "throughput": (valid if args.mode == "inference" else converged) / steady_seconds,
        "results": results,
        "accelerator_utilization": None,
        "utilization_note": "Not measured; attach an approved facility telemetry collector",
    }
    report["per_worker"] = [
        {
            "pid": pid,
            "device_mask": next(row["device_mask"] for row in results if row.get("pid") == pid),
            "tasks": sum(row.get("pid") == pid for row in results),
            "idle_and_dispatch_seconds": max(
                0.0,
                steady_seconds
                - sum(row["worker_seconds"] for row in results if row.get("pid") == pid),
            ),
        }
        for pid in sorted({row["pid"] for row in results if row["status"] == "ok"})
    ]
    (output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return 0 if report["measurement_valid"] else 1


def warmup_task(payload):
    payload = dict(payload)
    destination = Path(payload["result_path"]).parent
    device = os.environ["ZE_AFFINITY_MASK"]
    directory = destination / f"warmup-{device}"
    directory.mkdir(exist_ok=False)
    payload["result_path"] = str(directory / "result.json")
    return worker_task(payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--backend", choices=["hydragnn"], default="hydragnn")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--devices", type=int, choices=DEVICES, required=True)
    parser.add_argument("--repetition", type=int, choices=[1, 2, 3], required=True)
    parser.add_argument("--mode", choices=["inference", "relaxation"], required=True)
    parser.add_argument("--precision", choices=["fp64"], default="fp64")
    parser.add_argument("--fmax", type=float, default=0.02)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not np.isfinite(args.fmax) or args.fmax <= 0 or args.max_steps < 1:
        parser.error("Force threshold and maximum steps must be positive")
    metadata = json.loads(args.metadata.read_text())
    if (
        args.checkpoint is not None
        and args.checkpoint.resolve() != Path(metadata["checkpoint"]["path"]).resolve()
    ):
        parser.error("Checkpoint override must match the frozen metadata")
    rows = load_inventory(args.manifest)
    blockers = verify_model(metadata, rows)
    preview = resource_plan(args.devices).resolve(Allocation("preview", (socket.gethostname(),)))
    report = {
        "status": "NO-GO" if blockers else "READY_FOR_APPROVED_PILOT",
        "blockers": blockers,
        "workload_count": len(rows),
        "structure_ids": [row.structure_id for row in rows],
        "placement": preview.model_dump(mode="json"),
        "gpu_initialized": False,
    }
    if not args.execute:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "preflight.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return 2 if blockers else 0
    if blockers:
        parser.error("; ".join(blockers))
    return execute(args, rows, metadata, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
