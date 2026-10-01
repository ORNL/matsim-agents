"""Score campaign structures in a backend-specific Python environment."""

from __future__ import annotations

import json
import sys

import numpy as np
from ase.io import read as ase_read

from matsim_agents.active_learning.calculator import make_mlip_calculator
from matsim_agents.active_learning.config import ALConfig
from matsim_agents.campaign.surrogate_hull import evaluate_surrogate_hull


def main() -> int:
    request = json.load(sys.stdin)
    cfg = ALConfig.from_yaml(request["config"])
    calculator = make_mlip_calculator(cfg.mlip)
    labels = []
    for structure_path in request["structure_paths"]:
        atoms = ase_read(structure_path)
        atoms.calc = calculator
        forces = np.asarray(atoms.get_forces(), dtype=float)
        energy = float(atoms.get_potential_energy())
        labels.append(
            {
                "structure_path": structure_path,
                "energy_eV": energy,
                "energy_per_atom_eV": energy / len(atoms),
                "max_force_eV_per_A": float(np.linalg.norm(forces, axis=1).max()),
            }
        )
    ranking = [
        label["structure_path"]
        for label in sorted(labels, key=lambda item: item["energy_per_atom_eV"])
    ]
    surrogate_hull = None
    if request.get("reference_manifest"):
        from pathlib import Path

        surrogate_hull = evaluate_surrogate_hull(
            Path(request["reference_manifest"]),
            request["formula"],
            labels,
            calculator,
            model_identifier=request.get("model_identifier", cfg.mlip.backend),
            unary_max_steps=int(request.get("unary_max_steps", 200)),
            unary_fmax_eV_per_A=float(request.get("unary_fmax_eV_per_A", 0.02)),
            unary_maxstep_A=float(request.get("unary_maxstep_A", 0.01)),
            minimum_unique_unary=int(request.get("minimum_unique_unary", 2)),
        )
    print(json.dumps({"labels": labels, "ranking": ranking, "surrogate_hull": surrogate_hull}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
