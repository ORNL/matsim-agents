"""Append labelled VASP results to the AL dataset and (optionally) retrain HydraGNN.

The dataset is stored as an extended-XYZ file (one frame per labelled
structure) so it is human-inspectable, ASE-readable, and easy to slurp into
HydraGNN's training pipeline.

Retraining is delegated to a user-supplied training script (typically one of
``HydraGNN/examples/.../train.py``) so we don't hard-code any particular
HydraGNN training config here. The trainer just shells out to the
training-launcher bash script (which itself handles ``srun`` + module setup).
"""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ase import Atoms
from ase.io import write as ase_write

from matsim_agents.active_learning.config import HydraGNNConfig, TrainerConfig
from matsim_agents.active_learning.dft_backend import DFTResult

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Dataset I/O                                                                 #
# --------------------------------------------------------------------------- #


@dataclass
class LabelledFrame:
    """One DFT-labelled structure, ready to be appended to the dataset."""

    atoms: Atoms
    energy_eV: float
    forces_eV_per_A: object  # numpy.ndarray
    stress_eV_per_A3: object | None  # numpy.ndarray | None
    source_work_dir: str
    iteration: int
    backend: str  # "vasp" | "qe" — never mix backends in a single training set


def dft_results_to_frames(results: Iterable[DFTResult], iteration: int) -> list[LabelledFrame]:
    """Filter to converged jobs and convert into ``LabelledFrame``s."""
    frames: list[LabelledFrame] = []
    for r in results:
        if not r.converged or r.final_atoms is None or r.energy_eV is None:
            continue
        frames.append(
            LabelledFrame(
                atoms=r.final_atoms,
                energy_eV=r.energy_eV,
                forces_eV_per_A=r.forces_eV_per_A,
                stress_eV_per_A3=r.stress_eV_per_A3,
                source_work_dir=r.work_dir,
                iteration=iteration,
                backend=r.backend,
            )
        )
    return frames


# Back-compat alias for the previous VASP-only public name.
vasp_results_to_frames = dft_results_to_frames


def append_frames_to_extxyz(frames: list[LabelledFrame], path: str | Path) -> int:
    """Append labelled frames to an extxyz dataset. Returns the count appended."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    for fr in frames:
        atoms = fr.atoms.copy()
        atoms.info["energy"] = fr.energy_eV
        atoms.info["al_iteration"] = fr.iteration
        atoms.info["source_work_dir"] = fr.source_work_dir
        atoms.info["dft_backend"] = fr.backend
        # Forces -> per-atom array
        if fr.forces_eV_per_A is not None:
            atoms.arrays["forces"] = fr.forces_eV_per_A
        if fr.stress_eV_per_A3 is not None:
            atoms.info["stress"] = fr.stress_eV_per_A3
        ase_write(str(path), atoms, append=True, format="extxyz")
        n += 1
    return n


# --------------------------------------------------------------------------- #
# Retrain hook                                                                #
# --------------------------------------------------------------------------- #


def retrain_hydragnn(
    trainer_cfg: TrainerConfig,
    hydragnn_cfg: HydraGNNConfig,
    dataset_path: str | Path,
    iteration: int,
    out_logdir: str | Path,
) -> Path:
    """Spawn the HydraGNN training script as a child process.

    Returns the path to the new logdir (which becomes the ``HydraGNNConfig.logdir``
    for the next iteration). If ``trainer_cfg.enabled`` is False, returns the
    existing logdir unchanged so the loop can keep running with the same model.
    """
    out_logdir = Path(out_logdir)
    if not trainer_cfg.enabled:
        log.info("trainer.enabled=False; skipping retrain at iteration %d", iteration)
        return Path(hydragnn_cfg.logdir)

    out_logdir.mkdir(parents=True, exist_ok=True)

    # The convention we follow: the launcher script accepts these positional
    # args: <train_script> <dataset_path> <out_logdir> <resume_logdir> <epochs>
    # and is responsible for srun/module setup. If no launcher is given we
    # fall back to a plain `python train_script ...` invocation in-process.
    if trainer_cfg.train_launcher is not None:
        argv = [
            "bash",
            str(trainer_cfg.train_launcher),
            str(trainer_cfg.train_script),
            str(dataset_path),
            str(out_logdir),
            str(hydragnn_cfg.logdir),  # resume from current logdir
            str(trainer_cfg.epochs_per_iter),
            str(trainer_cfg.nodes_for_train),
            str(trainer_cfg.ranks_per_node),
        ]
    else:
        log.warning(
            "trainer.train_launcher not set; running training in-process "
            "(no srun, no module swap). For multi-node training set train_launcher."
        )
        argv = [
            "python",
            str(trainer_cfg.train_script),
            "--dataset", str(dataset_path),
            "--logdir", str(out_logdir),
            "--resume_from", str(hydragnn_cfg.logdir),
            "--epochs", str(trainer_cfg.epochs_per_iter),
        ]

    log.info("Launching retrain: %s", " ".join(argv))
    log_path = out_logdir / "train.log"
    with open(log_path, "w") as f:
        proc = subprocess.run(argv, stdout=f, stderr=subprocess.STDOUT, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"HydraGNN retrain failed with exit {proc.returncode}; see {log_path}"
        )

    return out_logdir
