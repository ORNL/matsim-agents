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
from matsim_agents.backends.mlip.config import MLIPConfig
from matsim_agents.backends.mlip.pool import MLIPWorkerPool

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


def seed_rngs(random_seed: int | None, count: int) -> list[np.random.Generator | None]:
    """One independent random stream per seed, derived from ``md.random_seed``.

    Each seed's trajectory depends only on ``random_seed`` and the seed's
    position in the list, so seeds can run in any order or concurrently and
    still reproduce. ``None`` keeps sampling stochastic.
    """

    if random_seed is None:
        return [None] * count
    return [
        np.random.default_rng(child) for child in np.random.SeedSequence(random_seed).spawn(count)
    ]


def _sample_seed(
    md_cfg: MDConfig,
    calculator,
    seed_path: str,
    out_dir: Path,
    rng: np.random.Generator | None,
) -> list[Candidate]:
    """Run MD from one seed and return its accepted snapshots in step order."""
    from ase import units
    from ase.md.langevin import Langevin
    from ase.md.nvtberendsen import NVTBerendsen
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from ase.md.verlet import VelocityVerlet

    atoms = _read_structure(seed_path)
    atoms.calc = calculator

    if rng is not None:
        MaxwellBoltzmannDistribution(atoms, temperature_K=md_cfg.temperature_K, rng=rng)
    else:
        MaxwellBoltzmannDistribution(atoms, temperature_K=md_cfg.temperature_K)

    timestep = md_cfg.timestep_fs * units.fs
    if md_cfg.thermostat == "langevin":
        dyn = Langevin(
            atoms,
            timestep=timestep,
            temperature_K=md_cfg.temperature_K,
            friction=md_cfg.friction_inv_ps / units.fs * 1e-3,  # ps^-1 -> ASE units
            rng=rng,
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

    candidates: list[Candidate] = []
    start_pos = atoms.get_positions().copy()
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
        candidates.append(
            Candidate(
                candidate_id=f"{seed_stem}_step{step + 1:06d}",
                atoms=cand_atoms,
                seed_path=seed_path,
                md_step=step + 1,
                predicted_energy_eV=energy,
                predicted_max_force_eV_per_A=fmax,
            )
        )
        ase_write(str(traj_path), cand_atoms, append=True, format="extxyz")

    log.info(
        "MD seed=%s: accepted=%d, skipped_force=%d, skipped_drift=%d",
        seed_stem,
        len(candidates),
        skipped_force,
        skipped_drift,
    )
    return candidates


# Calculators loaded by an MD worker process, reused across its seeds; across AL
# iterations too while the worker pool stays up (when no other pool uses the mlip
# devices). A different model (a promotion) replaces the loaded one.
_WORKER_CALCULATORS: dict[tuple[str, bool], object] = {}


@dataclass(frozen=True)
class _SeedTask:
    md_json: str
    mlip_json: str
    enable_mc_dropout: bool
    seed_path: str
    out_dir: str
    rng: np.random.Generator | None


def _sample_seed_task(task: _SeedTask) -> list[Candidate]:
    """MLIP worker entry point: MD from one seed with the worker's own calculator."""
    from matsim_agents.backends.mlip.calculator import make_mlip_calculator
    from matsim_agents.backends.mlip.config import MLIPConfig
    from matsim_agents.backends.mlip.pool import prepare_worker_device, release_device_memory

    prepare_worker_device()
    key = (task.mlip_json, task.enable_mc_dropout)
    calculator = _WORKER_CALCULATORS.get(key)
    if calculator is None:
        if _WORKER_CALCULATORS:
            _WORKER_CALCULATORS.clear()
            release_device_memory()  # free the previous model before loading the next
        calculator = make_mlip_calculator(
            MLIPConfig.model_validate_json(task.mlip_json),
            enable_mc_dropout=task.enable_mc_dropout,
        )
        _WORKER_CALCULATORS[key] = calculator
    md_cfg = MDConfig.model_validate_json(task.md_json)
    return _sample_seed(md_cfg, calculator, task.seed_path, Path(task.out_dir), task.rng)


def sample_md_candidates(
    md_cfg: MDConfig,
    calculator,
    out_dir: str | Path,
    seed_paths: list | None = None,
    *,
    mlip_pool: MLIPWorkerPool | None = None,
    mlip_cfg: MLIPConfig | None = None,
    enable_mc_dropout: bool = False,
) -> list[Candidate]:
    """Run MD from each seed structure and collect filtered snapshots.

    The MD driver uses ASE's ``Langevin`` (or ``NVTBerendsen``/``VelocityVerlet``).
    Trajectories are written to ``out_dir/traj_<seed_stem>.extxyz`` for
    inspection. Every seed draws from its own random stream (:func:`seed_rngs`),
    so the candidate pool is the same whether seeds run one after another or
    concurrently.

    Parameters
    ----------
    md_cfg
        MD configuration.
    calculator
        ASE calculator used when seeds run in this process.
    out_dir
        Where to write per-seed trajectory files.
    seed_paths
        Resolved list of seed structure files (the AL loop computes this
        from ``md_cfg.seed_source`` once per iteration and passes it in).
    mlip_pool, mlip_cfg, enable_mc_dropout
        With a pool that has worker processes, seeds run concurrently, one per
        device; each worker builds its own calculator from *mlip_cfg*.
        Candidates are returned in seed order either way.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if seed_paths is None:
        raise ValueError(
            "sample_md_candidates requires a resolved seed_paths list. "
            "The AL loop computes this from md_cfg.seed_source."
        )
    seeds = [str(path) for path in seed_paths]
    rngs = seed_rngs(md_cfg.random_seed, len(seeds))

    if mlip_pool is None or not mlip_pool.uses_workers or len(seeds) < 2:
        candidates: list[Candidate] = []
        for seed_path, rng in zip(seeds, rngs, strict=True):
            candidates.extend(_sample_seed(md_cfg, calculator, seed_path, out_dir, rng))
        return candidates

    if mlip_cfg is None:
        raise ValueError("concurrent MD needs mlip_cfg so each worker can build its calculator")
    tasks = [
        _SeedTask(
            md_json=md_cfg.model_dump_json(),
            mlip_json=mlip_cfg.model_dump_json(),
            enable_mc_dropout=enable_mc_dropout,
            seed_path=seed_path,
            out_dir=str(out_dir),
            rng=rng,
        )
        for seed_path, rng in zip(seeds, rngs, strict=True)
    ]
    candidates = []
    for seed_path, outcome in zip(seeds, mlip_pool.map(_sample_seed_task, tasks), strict=True):
        if outcome.error is not None:
            raise RuntimeError(
                f"MD from seed {seed_path} failed: {outcome.error}"
            ) from outcome.error
        candidates.extend(outcome.value)
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
