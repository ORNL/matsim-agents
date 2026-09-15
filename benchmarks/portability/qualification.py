"""Real numerical qualification for the cross-facility benchmark.

The deterministic contract suite proves orchestration semantics.  This module
is deliberately separate: every configuration supplied here is executed by
the production relaxation workflow with its real MLIP or DFT backend.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from matsim_agents.workflows.relaxation import ScientificRelaxationConfig, run_relaxation


def _load_mapping(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a mapping in {path}")
    return value


def _run_case(
    config_path: Path,
    *,
    structure: Path,
    output: Path,
    runner=run_relaxation,
) -> dict[str, Any]:
    raw = _load_mapping(config_path)
    raw["structure_path"] = str(structure.resolve())
    raw["output_root"] = str(output.resolve())
    config = ScientificRelaxationConfig.model_validate(raw)
    result = runner(config)
    stages = [stage.model_dump(mode="json") for stage in result.stages]
    final = result.stages[-1] if result.stages else None
    return {
        "config": str(config_path.resolve()),
        "mode": config.mode.value,
        "status": result.status.value,
        "run_id": result.run_id,
        "run_directory": result.run_directory,
        "final_structure_path": result.final_structure_path,
        "energy_eV": None if final is None else final.energy_eV,
        "max_force_eV_per_A": None if final is None else final.max_force_eV_per_A,
        "converged": bool(final and final.converged),
        "stages": stages,
    }


def execute_compute_qualification(
    *,
    structure: Path,
    output: Path,
    relaxation_configs: list[Path],
    runner=run_relaxation,
) -> dict[str, Any]:
    """Execute real relaxation cases and return their comparable metrics.

    At least one pure MLIP case and one pure QE case are required.  Additional
    MLIP variants and warm-start configurations may be supplied with repeated
    ``--relaxation-config`` arguments.
    """

    if not relaxation_configs:
        raise ValueError("compute qualification requires relaxation configurations")
    output.mkdir(parents=True, exist_ok=True)
    cases = [
        _run_case(
            path,
            structure=structure,
            output=output / f"case-{index:02d}",
            runner=runner,
        )
        for index, path in enumerate(relaxation_configs)
    ]
    modes = {case["mode"] for case in cases}
    has_qe = any(
        case["mode"] == "dft" and case["stages"] and case["stages"][-1]["backend"] == "qe"
        for case in cases
    )
    if "mlip" not in modes:
        raise ValueError("compute qualification requires at least one mode=mlip case")
    if not has_qe:
        raise ValueError("compute qualification requires at least one mode=dft QE case")
    passed = all(case["converged"] and case["energy_eV"] is not None for case in cases)
    return {
        "schema_version": 1,
        "qualification": "compute",
        "status": "passed" if passed else "failed",
        "structure_count": 1,
        "cases": cases,
    }


def write_scientific_summary(path: Path, payload: dict[str, Any]) -> None:
    """Persist the mandatory machine-comparable scientific summary."""

    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


__all__ = ["execute_compute_qualification", "write_scientific_summary"]
