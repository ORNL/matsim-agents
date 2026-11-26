"""Re-export of a fine-tuned MLIP for LAMMPS ML-IAP, gated by parity and MD stability.

The bridge evaluates the Python calculator inside LAMMPS, so no model conversion is
involved: "export" means building the calculator from the fine-tuned checkpoint,
proving that LAMMPS reproduces it, and recording the checkpoint bytes it was proven
for (:func:`~matsim_agents.backends.external.lammps.mliap.parity.write_export_record`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ase.build import bulk

from matsim_agents.backends.external.lammps.mliap.parity import (
    md_stability_check,
    parity_check,
    write_export_record,
)


def checkpoint_file(mlip_cfg: Any) -> Path:
    """The single file whose bytes identify the model of ``mlip_cfg``."""
    kind = type(mlip_cfg).__name__
    if kind == "UMAConfig":
        path = mlip_cfg.local_checkpoint()
    elif kind == "MACEConfig":
        path = Path(str(mlip_cfg.model))
    elif kind == "HydraGNNConfig":
        logdir = Path(mlip_cfg.logdir)
        if mlip_cfg.checkpoint:
            path = Path(mlip_cfg.checkpoint)
            path = path if path.is_absolute() else logdir / path
        else:
            candidates = sorted(logdir.glob("*.pk"))
            path = candidates[-1] if candidates else None
    else:
        raise ValueError(f"no LAMMPS export for {kind}")
    if path is None or not Path(path).is_file():
        raise FileNotFoundError(f"{kind}: no checkpoint file to export (got {path})")
    return Path(path)


def export_checkpoint_for_lammps(
    mlip_cfg: Any,
    checkpoint: str | Path,
    out_dir: str | Path,
    *,
    species: list[str],
    md_steps: int = 20,
    md_temperature_K: float = 300.0,
    md_timestep_fs: float = 1.0,
    calculator: Any = None,
) -> dict[str, Any]:
    """Validate ``checkpoint`` under LAMMPS and write ``<out_dir>/lammps_export.json``.

    Runs the ASE-vs-LAMMPS parity check and an NVE stability check on a rattled fcc
    cell of the first species, so the checks are software-level (they say the bridge
    reproduces the calculator, not that the model is accurate). Returns the record,
    with ``approved`` true only when both pass. ``calculator`` is for callers that
    already hold one.
    """
    import numpy as np
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary

    candidate = mlip_cfg.with_checkpoint(checkpoint)
    calc = calculator if calculator is not None else candidate.build_calculator()
    element = species[0]
    atoms = bulk(element, "fcc", a=_lattice_guess(element), cubic=True).repeat((3, 3, 3))
    atoms.rattle(0.05, seed=7)
    parity = parity_check(atoms, calc, species)
    md = None
    if md_steps > 0:
        md_atoms = atoms.copy()
        MaxwellBoltzmannDistribution(
            md_atoms, temperature_K=md_temperature_K, rng=np.random.default_rng(11)
        )
        Stationary(md_atoms)
        md = md_stability_check(md_atoms, calc, species, steps=md_steps, timestep_fs=md_timestep_fs)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    record = write_export_record(
        out / "lammps_export.json",
        checkpoint=checkpoint_file(candidate),
        species=species,
        short_range=None,
        parity=parity,
    )
    record["md"] = md.__dict__ if md is not None else None
    record["approved"] = bool(parity.passed and (md is None or md.passed))
    (out / "lammps_export.json").write_text(json.dumps(record, indent=2, default=float) + "\n")
    return record


def _lattice_guess(element: str) -> float:
    """fcc lattice constant (A) from the covalent radius; only a plausible test geometry."""
    from ase.data import atomic_numbers, covalent_radii

    return float(covalent_radii[atomic_numbers[element]] * 2 * 2**0.5)


__all__ = ["checkpoint_file", "export_checkpoint_for_lammps"]
