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

import json
import logging
import socket
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from functools import singledispatch
from pathlib import Path

from ase import Atoms
from ase.io import read as ase_read
from ase.io import write as ase_write

from matsim_agents.backends.dft.base import DFTResult
from matsim_agents.backends.dft.config import DFTConfig
from matsim_agents.backends.mlip.config import (
    HydraGNNConfig,
    MACEConfig,
    MLIPBackendConfig,
    UMAConfig,
)
from matsim_agents.execution.placement import ProcessLaunch
from matsim_agents.workflows.active_learning.config import TrainerConfig

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


def append_frames_to_extxyz(frames: list[LabelledFrame], path: str | Path) -> int:
    """Append frames while enforcing one DFT energy reference per dataset."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    incoming_backends = {frame.backend for frame in frames}
    if len(incoming_backends) > 1:
        raise ValueError(
            "Refusing to mix DFT energy references in one dataset: "
            f"incoming frames use {sorted(incoming_backends)}"
        )
    if path.exists() and path.stat().st_size and incoming_backends:
        first = ase_read(str(path), index=0, format="extxyz")
        existing_backend = first.info.get("dft_backend")
        incoming_backend = next(iter(incoming_backends))
        if not existing_backend:
            raise ValueError(
                f"Existing dataset {path} has no dft_backend provenance; refusing an unsafe append"
            )
        if existing_backend != incoming_backend:
            raise ValueError(
                f"Refusing to append {incoming_backend!r} energies to {path}; "
                f"the existing dataset uses {existing_backend!r}. Apply an "
                "explicit energy-reference transformation into a new dataset."
            )

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


def _check_fine_tune_launch(mlip_cfg: MLIPBackendConfig, launch: ProcessLaunch) -> None:
    """Reject fine-tunes the built-in recipes cannot run, before any expensive setup."""
    mlip_cfg.check_fine_tunable()
    # The built-in fine-tuners are single-process; extra ranks would race on outputs.
    if launch.total_ranks != 1:
        raise ValueError(
            f"Built-in fine-tuners run as one process ({mlip_cfg.fine_tuner}); set "
            f"resources.pools.training to 1 node and 1 device (it resolved to "
            f"{launch.node_count} node(s) x {launch.ranks_per_node} rank(s))"
        )


def _fine_tune_argv(
    trainer_cfg: TrainerConfig,
    mlip_cfg: MLIPBackendConfig,
    launch: ProcessLaunch,
    *,
    dataset: Path,
    out_dir: Path,
    references: Path,
    seed: int,
) -> list[str]:
    """Command that runs the backend's fine-tuner, directly or through the site launcher.

    A launcher receives ``<backend> <nodes> <ranks_per_node> -m <module> <args...>``:
    it prepares the backend's environment on the site and runs ``python`` with the
    remaining arguments.
    """
    _check_fine_tune_launch(mlip_cfg, launch)
    python_args = [
        "-m",
        mlip_cfg.fine_tuner_module,
        *mlip_cfg.fine_tune_args(
            dataset=dataset,
            out_dir=out_dir,
            epochs=trainer_cfg.epochs_per_iter,
            references=references,
            finetune_mode=trainer_cfg.finetune_mode,
            seed=seed,
        ),
    ]
    if trainer_cfg.train_launcher is None:
        log.warning(
            "trainer.train_launcher not set; running the %s fine-tune in-process "
            "(no srun, no module swap). For multi-node training set train_launcher.",
            mlip_cfg.backend,
        )
        return [sys.executable, *python_args]
    return [
        "bash",
        str(trainer_cfg.train_launcher),
        mlip_cfg.backend,
        str(launch.node_count),
        str(launch.ranks_per_node),
        *python_args,
    ]


def _require_direct_placement(trainer_cfg: TrainerConfig, launch: ProcessLaunch, what: str) -> None:
    """Without a launcher, training is one Python process on this node; the plan must agree."""

    if trainer_cfg.train_launcher is not None:
        return
    if launch.total_ranks != 1:
        raise ValueError(
            f"{what} without trainer.train_launcher runs one local process, but "
            f"resources.pools.training resolved to {launch.node_count} node(s) x "
            f"{launch.ranks_per_node} rank(s); set train_launcher or use 1 node and 1 device"
        )
    local = socket.gethostname().split(".", 1)[0]
    host = launch.hosts[0].split(".", 1)[0]
    if host != local:
        raise ValueError(
            f"{what} without trainer.train_launcher runs on this node ({local}), but "
            f"resources.pools.training places it on {launch.hosts[0]}; set train_launcher "
            "or place the training pool on the driver's node"
        )


def retrain_hydragnn(
    trainer_cfg: TrainerConfig,
    hydragnn_cfg: HydraGNNConfig,
    dataset_path: str | Path,
    iteration: int,
    out_logdir: str | Path,
    *,
    seed: int,
    launch: ProcessLaunch,
    dft_config: DFTConfig | None = None,
    dft_launch: ProcessLaunch | None = None,
    reference_root: Path | None = None,
) -> Path:
    """Spawn a formation-aware HydraGNN fine-tuner as a child process.

    Returns the path to the new logdir (which becomes the ``HydraGNNConfig.logdir``
    for the next iteration). If ``trainer_cfg.enabled`` is False, returns the
    existing logdir unchanged so the loop can keep running with the same model.
    """
    out_logdir = Path(out_logdir)
    if not trainer_cfg.enabled:
        log.info("trainer.enabled=False; skipping retrain at iteration %d", iteration)
        return Path(hydragnn_cfg.logdir)

    _check_fine_tune_launch(hydragnn_cfg, launch)
    _require_direct_placement(trainer_cfg, launch, "HydraGNN fine-tuning")

    from matsim_agents.core.hashing import sha256_file
    from matsim_agents.workflows.active_learning.training_references import (
        resolve_training_references,
    )

    reference_manifest = resolve_training_references(
        trainer_cfg,
        dataset_path,
        dft_config=dft_config,
        dft_launch=dft_launch,
        reference_root=reference_root or out_logdir.parent / "elemental-training-reference",
    )
    dataset_sha256 = sha256_file(Path(dataset_path))
    out_logdir.mkdir(parents=True, exist_ok=True)

    argv = _fine_tune_argv(
        trainer_cfg,
        hydragnn_cfg,
        launch,
        dataset=Path(dataset_path),
        out_dir=out_logdir,
        references=reference_manifest,
        seed=seed,
    )

    log.info("Launching retrain: %s", " ".join(argv))
    log_path = out_logdir / "train.log"
    with open(log_path, "w") as f:
        proc = subprocess.run(
            argv,
            stdout=f,
            stderr=subprocess.STDOUT,
            check=False,
            # Without a launcher the trainer runs directly on this node.
            env=launch.environment(local=trainer_cfg.train_launcher is None),
        )
    if proc.returncode != 0:
        raise RuntimeError(f"HydraGNN retrain failed with exit {proc.returncode}; see {log_path}")

    if not (out_logdir / "energy-convention.json").is_file():
        raise RuntimeError(f"HydraGNN trainer produced no energy-convention.json; see {log_path}")
    if not (out_logdir / "config.json").is_file():
        raise RuntimeError(f"HydraGNN trainer produced no config.json; see {log_path}")
    from ase.calculators.calculator import Calculator

    from matsim_agents.backends.mlip.hydragnn_energy import restore_hydragnn_total_energy

    restore_hydragnn_total_energy(Calculator(), out_logdir)
    metadata = json.loads((out_logdir / "energy-convention.json").read_text())
    if (
        metadata.get("source_dataset_sha256") != dataset_sha256
        or sha256_file(Path(dataset_path)) != dataset_sha256
    ):
        raise ValueError("HydraGNN trainer output does not match the requested training dataset")
    if metadata.get("source_elemental_manifest_sha256") != sha256_file(reference_manifest):
        raise ValueError(
            "HydraGNN trainer output does not match the requested elemental reference manifest"
        )
    return out_logdir


def _references_for_training(
    trainer_cfg: TrainerConfig,
    dataset_path: str | Path,
    out_dir: Path,
    *,
    dft_config: DFTConfig | None,
    dft_launch: ProcessLaunch | None,
    reference_root: Path | None,
) -> Path:
    from matsim_agents.workflows.active_learning.training_references import (
        resolve_training_references,
    )

    return resolve_training_references(
        trainer_cfg,
        dataset_path,
        dft_config=dft_config,
        dft_launch=dft_launch,
        reference_root=reference_root or out_dir.parent / "elemental-training-reference",
    )


def _run_trainer(argv: list[str], log_path: Path, launch: ProcessLaunch, local: bool, what: str):
    log.info("Launching %s: %s", what, " ".join(argv))
    with open(log_path, "w") as f:
        proc = subprocess.run(
            argv,
            stdout=f,
            stderr=subprocess.STDOUT,
            check=False,
            # Without a launcher the trainer runs directly on this node.
            env=launch.environment(local=local),
        )
    if proc.returncode != 0:
        raise RuntimeError(f"{what} failed with exit {proc.returncode}; see {log_path}")


def _check_energy_alignment(out_dir: Path, dataset_path: Path, dataset_sha256: str, manifest: Path):
    """The fine-tune aligned exactly the requested dataset with the requested references."""
    from matsim_agents.core.hashing import sha256_file
    from matsim_agents.data.energy_alignment import ENERGY_ALIGNMENT_FILE, EnergyAlignment

    path = out_dir / ENERGY_ALIGNMENT_FILE
    if not path.is_file():
        raise RuntimeError(
            f"fine-tune produced no {ENERGY_ALIGNMENT_FILE}; see {out_dir / 'train.log'}"
        )
    alignment = EnergyAlignment.model_validate_json(path.read_text())
    if (
        alignment.source_dataset_sha256 != dataset_sha256
        or sha256_file(dataset_path) != dataset_sha256
    ):
        raise ValueError("fine-tuned model does not match the requested training dataset")
    if alignment.manifest_sha256 != sha256_file(manifest):
        raise ValueError("fine-tuned model does not match the requested elemental references")


def retrain_uma(
    trainer_cfg: TrainerConfig,
    uma_cfg: UMAConfig,
    dataset_path: str | Path,
    iteration: int,
    out_model_dir: str | Path,
    *,
    seed: int,
    launch: ProcessLaunch,
    dft_config: DFTConfig | None = None,
    dft_launch: ProcessLaunch | None = None,
    reference_root: Path | None = None,
) -> Path:
    """Fine-tune ``uma_cfg``'s model on the dataset as a child process.

    Returns the fine-tuned ``inference_ckpt.pt``, which becomes
    ``UMAConfig.model_name`` when the model is promoted. The labels are aligned
    to the base model's energy scale with the training references (see
    :mod:`matsim_agents.backends.mlip.training.finetune_uma`).
    """
    from matsim_agents.core.hashing import sha256_file

    out_model_dir = Path(out_model_dir)
    if not trainer_cfg.enabled:
        log.info("trainer.enabled=False; skipping UMA fine-tune at iteration %d", iteration)
        return Path(str(uma_cfg.model_name))

    _check_fine_tune_launch(uma_cfg, launch)
    _require_direct_placement(trainer_cfg, launch, "UMA fine-tuning")
    manifest = _references_for_training(
        trainer_cfg,
        dataset_path,
        out_model_dir,
        dft_config=dft_config,
        dft_launch=dft_launch,
        reference_root=reference_root,
    )
    dataset_sha256 = sha256_file(Path(dataset_path))
    out_model_dir.mkdir(parents=True, exist_ok=True)

    argv = _fine_tune_argv(
        trainer_cfg,
        uma_cfg,
        launch,
        dataset=Path(dataset_path),
        out_dir=out_model_dir,
        references=manifest,
        seed=seed,
    )

    log_path = out_model_dir / "train.log"
    _run_trainer(argv, log_path, launch, trainer_cfg.train_launcher is None, "UMA fine-tune")

    try:
        checkpoint = uma_cfg.with_checkpoint(out_model_dir).local_checkpoint()
    except ValueError:
        checkpoint = None
    if checkpoint is None:
        raise RuntimeError(
            "UMA fine-tune exited successfully but did not produce an inference_ckpt.pt "
            f"under {out_model_dir}; see {log_path}"
        )
    _check_energy_alignment(out_model_dir, Path(dataset_path), dataset_sha256, manifest)
    return checkpoint


def retrain_mace(
    trainer_cfg: TrainerConfig,
    mace_cfg: MACEConfig,
    dataset_path: str | Path,
    iteration: int,
    out_model_dir: str | Path,
    *,
    seed: int,
    launch: ProcessLaunch,
    dft_config: DFTConfig | None = None,
    dft_launch: ProcessLaunch | None = None,
    reference_root: Path | None = None,
) -> Path:
    """Fine-tune ``mace_cfg``'s model on the dataset as a child process.

    Returns the fine-tuned ``mace_finetuned.model``, which becomes the model
    (``MACEConfig.family='checkpoint'``) when it is promoted. The labels are
    aligned to the base model's energy scale with the training references (see
    :mod:`matsim_agents.backends.mlip.training.finetune_mace`).
    """
    from matsim_agents.core.hashing import sha256_file

    out_model_dir = Path(out_model_dir)
    if not trainer_cfg.enabled:
        log.info("trainer.enabled=False; skipping MACE fine-tune at iteration %d", iteration)
        # Point downstream at the existing base model (checkpoint path or variant).
        return Path(str(mace_cfg.model))

    _check_fine_tune_launch(mace_cfg, launch)
    _require_direct_placement(trainer_cfg, launch, "MACE fine-tuning")
    manifest = _references_for_training(
        trainer_cfg,
        dataset_path,
        out_model_dir,
        dft_config=dft_config,
        dft_launch=dft_launch,
        reference_root=reference_root,
    )
    dataset_sha256 = sha256_file(Path(dataset_path))
    out_model_dir.mkdir(parents=True, exist_ok=True)

    argv = _fine_tune_argv(
        trainer_cfg,
        mace_cfg,
        launch,
        dataset=Path(dataset_path),
        out_dir=out_model_dir,
        references=manifest,
        seed=seed,
    )

    log_path = out_model_dir / "train.log"
    _run_trainer(argv, log_path, launch, trainer_cfg.train_launcher is None, "MACE fine-tune")
    finetuned = out_model_dir / "mace_finetuned.model"
    if not finetuned.is_file():
        raise RuntimeError(
            f"MACE fine-tune exited successfully but did not write {finetuned}; see {log_path}"
        )
    _check_energy_alignment(out_model_dir, Path(dataset_path), dataset_sha256, manifest)
    return finetuned


@singledispatch
def retrain(mlip_cfg: MLIPBackendConfig, trainer_cfg: TrainerConfig, *, out_dir: Path, **kwargs):
    """Fine-tune ``mlip_cfg``'s model with its backend's recipe; return the trained model.

    ``kwargs`` are the backend-independent inputs (dataset, iteration, launches).
    """
    raise NotImplementedError(f"no fine-tuning registered for {type(mlip_cfg).__name__}")


@retrain.register
def _(mlip_cfg: HydraGNNConfig, trainer_cfg: TrainerConfig, *, out_dir: Path, **kwargs) -> Path:
    return retrain_hydragnn(trainer_cfg, mlip_cfg, out_logdir=out_dir, **kwargs)


@retrain.register
def _(mlip_cfg: UMAConfig, trainer_cfg: TrainerConfig, *, out_dir: Path, **kwargs) -> Path:
    return retrain_uma(trainer_cfg, mlip_cfg, out_model_dir=out_dir, **kwargs)


@retrain.register
def _(mlip_cfg: MACEConfig, trainer_cfg: TrainerConfig, *, out_dir: Path, **kwargs) -> Path:
    return retrain_mace(trainer_cfg, mlip_cfg, out_model_dir=out_dir, **kwargs)
