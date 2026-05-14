"""HydraGNN-driven ASE MD sampler that produces candidate structures for AL.

We run short MD trajectories from each user-provided seed structure using the
current HydraGNN MLFF as the force engine, then sub-sample frames at
``sample_every`` steps. Frames where the model predicts unphysically large
forces or atoms drift implausibly far from their starting positions are
filtered out — these usually indicate the model is exploring well outside its
training distribution and would crash VASP too.

This module is intentionally agnostic to the calculator implementation: it
calls ``calculator.calculate(atoms)`` via ASE and reads ``calc.results``.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read as ase_read
from ase.io import write as ase_write

from matsim_agents.active_learning.config import MDConfig

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Public dataclass                                                            #
# --------------------------------------------------------------------------- #


@dataclass
class Candidate:
    """One candidate frame for potential VASP labelling."""

    candidate_id: str
    atoms: Atoms
    seed_path: str
    md_step: int
    predicted_energy_eV: float | None = None
    predicted_max_force_eV_per_A: float | None = None


# --------------------------------------------------------------------------- #
# Sampler                                                                     #
# --------------------------------------------------------------------------- #


def _read_structure(path: str | Path) -> Atoms:
    """Best-effort read of POSCAR/CIF/XYZ etc."""
    return ase_read(str(path))


def _max_displacement(start: np.ndarray, current: np.ndarray) -> float:
    """Max per-atom displacement between two configurations (Å, no MIC)."""
    return float(np.max(np.linalg.norm(current - start, axis=1)))


def sample_md_candidates(
    md_cfg: MDConfig,
    calculator,
    out_dir: str | Path,
    seed_paths: list | None = None,
) -> list[Candidate]:
    """Run MD from each seed structure and collect filtered snapshots.

    The MD driver uses ASE's ``Langevin`` (or ``NVTBerendsen``/``VelocityVerlet``)
    in-process. Trajectories are written to ``out_dir/traj_<seed_stem>.extxyz``
    for inspection.

    Parameters
    ----------
    md_cfg
        MD configuration.
    calculator
        Any ASE-compatible calculator (typically a ``FusedHydraGNNCalculator``).
    out_dir
        Where to write per-seed trajectory files.
    seed_paths
        Resolved list of seed structure files (the AL loop computes this
        from ``md_cfg.seed_source`` once per iteration and passes it in).
    """
    from ase import units
    from ase.md.langevin import Langevin
    from ase.md.nvtberendsen import NVTBerendsen
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from ase.md.verlet import VelocityVerlet

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates: list[Candidate] = []

    if seed_paths is None:
        raise ValueError(
            "sample_md_candidates requires a resolved seed_paths list. "
            "The AL loop computes this from md_cfg.seed_source."
        )

    for seed_path in seed_paths:
        seed_path = str(seed_path)
        atoms = _read_structure(seed_path)
        atoms.calc = calculator

        # Initialise velocities from a Maxwell-Boltzmann distribution.
        MaxwellBoltzmannDistribution(atoms, temperature_K=md_cfg.temperature_K)

        timestep = md_cfg.timestep_fs * units.fs
        if md_cfg.thermostat == "langevin":
            dyn = Langevin(
                atoms,
                timestep=timestep,
                temperature_K=md_cfg.temperature_K,
                friction=md_cfg.friction_inv_ps / units.fs * 1e-3,  # ps^-1 -> ASE units
            )
        elif md_cfg.thermostat == "nvt-berendsen":
            dyn = NVTBerendsen(
                atoms, timestep=timestep, temperature_K=md_cfg.temperature_K, taut=100 * units.fs
            )
        else:
            dyn = VelocityVerlet(atoms, timestep=timestep)

        seed_stem = Path(seed_path).stem
        traj_path = out_dir / f"traj_{seed_stem}.extxyz"
        # Truncate the trajectory file so resumed iterations don't accumulate.
        if traj_path.exists():
            traj_path.unlink()

        start_pos = atoms.get_positions().copy()
        accepted = 0
        skipped_force = 0
        skipped_drift = 0

        for step in range(md_cfg.n_steps):
            try:
                dyn.run(1)
            except Exception as exc:  # noqa: BLE001 — calculator may diverge
                log.warning(
                    "MD step %d for seed %s crashed (%s); ending trajectory.",
                    step,
                    seed_stem,
                    exc,
                )
                break

            if (step + 1) % md_cfg.sample_every != 0:
                continue

            forces = atoms.get_forces()
            fmax = float(np.max(np.linalg.norm(forces, axis=1)))
            if fmax > md_cfg.max_force_threshold_eV_per_A:
                skipped_force += 1
                continue
            if _max_displacement(start_pos, atoms.get_positions()) > md_cfg.max_displacement_A:
                skipped_drift += 1
                continue

            energy = None
            with contextlib.suppress(Exception):
                energy = float(atoms.get_potential_energy())

            cand_atoms = atoms.copy()
            cand_atoms.calc = None  # detach calculator before serialising
            cand = Candidate(
                candidate_id=f"{seed_stem}_step{step + 1:06d}",
                atoms=cand_atoms,
                seed_path=seed_path,
                md_step=step + 1,
                predicted_energy_eV=energy,
                predicted_max_force_eV_per_A=fmax,
            )
            candidates.append(cand)
            ase_write(str(traj_path), cand_atoms, append=True, format="extxyz")
            accepted += 1

        log.info(
            "MD seed=%s: accepted=%d, skipped_force=%d, skipped_drift=%d",
            seed_stem,
            accepted,
            skipped_force,
            skipped_drift,
        )

    return candidates


def iter_candidates_from_extxyz(path: str | Path, seed_path: str = "") -> Iterator[Candidate]:
    """Helper: re-load a previously written candidate trajectory."""
    for i, atoms in enumerate(ase_read(str(path), index=":")):
        yield Candidate(
            candidate_id=f"{Path(path).stem}_frame{i:06d}",
            atoms=atoms,
            seed_path=seed_path or str(path),
            md_step=i,
        )
