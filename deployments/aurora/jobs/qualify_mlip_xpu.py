"""Qualify checkpoint-backed MLIP energy/forces without allowing CPU fallback."""

from __future__ import annotations

import argparse
import json
import math
import platform
import traceback
from pathlib import Path


def validate_predictions(energy: float, forces: list[list[float]], atom_count: int) -> None:
    if not math.isfinite(energy):
        raise ValueError("Nonfinite model energy")
    if len(forces) != atom_count or any(len(row) != 3 for row in forces):
        raise ValueError("Unexpected model force shape")
    if any(not math.isfinite(value) for row in forces for value in row):
        raise ValueError("Nonfinite model forces")


def qualify(args) -> dict:
    import torch
    from ase.build import bulk

    from matsim_agents.active_learning.calculator import (
        _find_torch_module,
        build_hydragnn_calculator,
        build_mace_calculator,
        build_uma_calculator,
    )
    from matsim_agents.active_learning.config import HydraGNNConfig, MACEConfig, UMAConfig

    if not torch.xpu.is_available():
        raise RuntimeError("No XPU visible; qualification must run under mpiexec on a compute node")
    torch.xpu.set_device(0)
    probe = torch.tensor([2.0, 3.0], device="xpu")
    if float((probe * probe).sum().cpu()) != 13.0:
        raise RuntimeError("XPU arithmetic probe failed")
    if args.backend == "hydragnn":
        calculator = build_hydragnn_calculator(HydraGNNConfig(
            logdir=args.hydragnn_logdir,
            checkpoint=args.hydragnn_checkpoint,
            mlp_checkpoint=args.hydragnn_mlp,
            mlp_device="xpu",
            precision="fp32",
        ))
    elif args.backend == "uma":
        calculator = build_uma_calculator(UMAConfig(
            model_name=args.uma_model, device="xpu", task_name="omat"
        ))
    else:
        calculator = build_mace_calculator(MACEConfig(
            model=args.mace_model,
            family="checkpoint" if Path(args.mace_model).is_file() else "mace_mp",
            device="xpu", precision="fp32",
        ))
    roots = [calculator, getattr(calculator, "predictor", None)]
    roots.extend(getattr(calculator, "models", []) or [])
    roots.append(getattr(calculator, "mlp", None))
    modules = {}
    for root in roots:
        model = _find_torch_module(root)
        if model is not None:
            modules[id(model)] = model
    devices = {
        str(parameter.device) for model in modules.values() for parameter in model.parameters()
    }
    if not devices or any(not device.startswith("xpu") for device in devices):
        raise RuntimeError(f"Model is not exclusively resident on XPU: {sorted(devices)}")
    atoms = bulk("Si", "diamond", a=5.43)
    atoms.positions[1, 0] += 0.02
    atoms.calc = calculator
    energy = float(atoms.get_potential_energy())
    forces = atoms.get_forces().tolist()
    torch.xpu.synchronize()
    validate_predictions(energy, forces, len(atoms))
    return {
        "status": "PASSED", "backend": args.backend, "host": platform.node(),
        "torch": torch.__version__, "torch_path": torch.__file__,
        "xpu_name": torch.xpu.get_device_name(0), "model_devices": sorted(devices),
        "energy_eV": energy, "forces_eV_per_A": forces,
        "qualification": "two-atom checkpoint inference smoke test; not scientific accuracy",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("hydragnn", "uma", "mace"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hydragnn-logdir", type=Path)
    parser.add_argument("--hydragnn-checkpoint")
    parser.add_argument("--hydragnn-mlp", type=Path)
    parser.add_argument("--uma-model", default="uma-s-1p1")
    parser.add_argument("--mace-model", default="small")
    args = parser.parse_args()
    try:
        result = qualify(args)
    except Exception as error:
        traceback.print_exc()
        result = {
            "status": "FAILED", "backend": args.backend,
            "host": platform.node(), "error": f"{type(error).__name__}: {error}",
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return 0 if result["status"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())