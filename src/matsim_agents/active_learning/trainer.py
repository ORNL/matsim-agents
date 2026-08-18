"""Append labelled DFT results to the AL dataset and optionally retrain the MLIP.

The dataset is stored as an extended-XYZ file (one frame per labelled
structure) so it is human-inspectable, ASE-readable, and easy to slurp into
HydraGNN's training pipeline.

Retraining is delegated to a user-supplied training script so we don't hard-code
any particular backend training config here. The trainer just shells out to a
training-launcher bash script (which itself handles ``srun`` + module setup), or
falls back to invoking the script directly with Python for single-process use.
"""

from __future__ import annotations

import logging
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from ase import Atoms
from ase.io import write as ase_write

from matsim_agents.active_learning.config import (
    HydraGNNConfig,
    MACEConfig,
    TrainerConfig,
    UMAConfig,
)
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
            "--dataset",
            str(dataset_path),
            "--logdir",
            str(out_logdir),
            "--resume_from",
            str(hydragnn_cfg.logdir),
            "--epochs",
            str(trainer_cfg.epochs_per_iter),
        ]

    log.info("Launching retrain: %s", " ".join(argv))
    log_path = out_logdir / "train.log"
    with open(log_path, "w") as f:
        proc = subprocess.run(argv, stdout=f, stderr=subprocess.STDOUT, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"HydraGNN retrain failed with exit {proc.returncode}; see {log_path}")

    return out_logdir


def retrain_uma(
    trainer_cfg: TrainerConfig,
    uma_cfg: UMAConfig,
    dataset_path: str | Path,
    iteration: int,
    out_model_dir: str | Path,
) -> Path:
    """Spawn a user-supplied UMA fine-tuning script as a child process.

    Returns the path to the fine-tuned model directory/checkpoint, which becomes
    ``UMAConfig.model_name`` for the next AL iteration. The launcher convention is
    analogous to the HydraGNN hook, but passes UMA-specific inputs:

    ``<train_script> <dataset_path> <out_model_dir> <base_model> <task_name>``
    ``<epochs> <nodes> <ranks>``.

    The launcher is responsible for translating the extxyz dataset into the
    exact FairChem/UMA fine-tuning command used at the deployment site.
    """
    out_model_dir = Path(out_model_dir)
    if not trainer_cfg.enabled:
        log.info("trainer.enabled=False; skipping UMA fine-tune at iteration %d", iteration)
        return Path(str(uma_cfg.model_name))

    out_model_dir.mkdir(parents=True, exist_ok=True)

    if trainer_cfg.train_launcher is not None:
        argv = [
            "bash",
            str(trainer_cfg.train_launcher),
            str(trainer_cfg.train_script),
            str(dataset_path),
            str(out_model_dir),
            str(uma_cfg.model_name),
            str(uma_cfg.task_name),
            str(trainer_cfg.epochs_per_iter),
            str(trainer_cfg.nodes_for_train),
            str(trainer_cfg.ranks_per_node),
        ]
    else:
        log.warning(
            "trainer.train_launcher not set; running UMA fine-tune in-process "
            "(no srun, no module swap). For multi-node training set train_launcher."
        )
        argv = [
            "python",
            str(trainer_cfg.train_script),
            "--dataset",
            str(dataset_path),
            "--output-dir",
            str(out_model_dir),
            "--base-model",
            str(uma_cfg.model_name),
            "--task-name",
            str(uma_cfg.task_name),
            "--epochs",
            str(trainer_cfg.epochs_per_iter),
        ]

    log.info("Launching UMA fine-tune: %s", " ".join(argv))
    log_path = out_model_dir / "train.log"
    with open(log_path, "w") as f:
        proc = subprocess.run(argv, stdout=f, stderr=subprocess.STDOUT, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"UMA fine-tune failed with exit {proc.returncode}; see {log_path}")

    return out_model_dir


def retrain_mace(
    trainer_cfg: TrainerConfig,
    mace_cfg: MACEConfig,
    dataset_path: str | Path,
    iteration: int,
    out_model_dir: str | Path,
) -> Path:
    """Spawn a MACE fine-tuning job as a child process.

    Returns the path to the fine-tuned ``mace_finetuned.model`` checkpoint, which
    becomes the next iteration's base model (``MACEConfig.family='checkpoint'``,
    ``MACEConfig.model=<path>``). All MACE versions are supported -- the base
    variant is taken from ``mace_cfg.family`` / ``mace_cfg.model``.

    The launcher convention mirrors the HydraGNN/UMA hooks:

    ``<train_script> <dataset_path> <out_model_dir> <family> <base_model>``
    ``<epochs> <nodes> <ranks>``.

    With no launcher, the built-in ``finetune_mace`` CLI is invoked in-process.
    """
    out_model_dir = Path(out_model_dir)
    if not trainer_cfg.enabled:
        log.info("trainer.enabled=False; skipping MACE fine-tune at iteration %d", iteration)
        # Point downstream at the existing base model (checkpoint path or variant).
        return Path(str(mace_cfg.model))

    out_model_dir.mkdir(parents=True, exist_ok=True)

    if trainer_cfg.train_launcher is not None:
        argv = [
            "bash",
            str(trainer_cfg.train_launcher),
            str(trainer_cfg.train_script),
            str(dataset_path),
            str(out_model_dir),
            str(mace_cfg.family),
            str(mace_cfg.model),
            str(trainer_cfg.epochs_per_iter),
            str(trainer_cfg.nodes_for_train),
            str(trainer_cfg.ranks_per_node),
        ]
    else:
        log.warning(
            "trainer.train_launcher not set; running MACE fine-tune in-process "
            "(no srun, no module swap). For multi-node training set train_launcher."
        )
        import sys

        argv = [
            sys.executable,
            "-m",
            "matsim_agents.active_learning.finetune_mace",
            "--dataset",
            str(dataset_path),
            "--output-dir",
            str(out_model_dir),
            "--family",
            str(mace_cfg.family),
            "--base-model",
            str(mace_cfg.model),
            "--epochs",
            str(trainer_cfg.epochs_per_iter),
        ]

    log.info("Launching MACE fine-tune: %s", " ".join(argv))
    log_path = out_model_dir / "train.log"
    with open(log_path, "w") as f:
        proc = subprocess.run(argv, stdout=f, stderr=subprocess.STDOUT, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"MACE fine-tune failed with exit {proc.returncode}; see {log_path}")

    # The in-process CLI writes mace_finetuned.model; a launcher should do the same.
    finetuned = out_model_dir / "mace_finetuned.model"
    return finetuned if finetuned.is_file() else out_model_dir
