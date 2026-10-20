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
from pathlib import Path

from ase import Atoms
from ase.io import read as ase_read
from ase.io import write as ase_write

from matsim_agents.active_learning.config import TrainerConfig
from matsim_agents.backends.dft.base import DFTResult
from matsim_agents.backends.dft.config import DFTConfig
from matsim_agents.backends.mlip.config import HydraGNNConfig, MACEConfig, UMAConfig
from matsim_agents.execution.placement import ProcessLaunch

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


HYDRAGNN_FINETUNERS = frozenset({"finetune_hydragnn", "finetune_hydragnn_newhead"})


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

    train_script = Path(str(trainer_cfg.train_script))
    script_name = train_script.stem
    if script_name not in HYDRAGNN_FINETUNERS:
        raise ValueError(
            f"trainer.train_script must be one of the built-in HydraGNN fine-tuners "
            f"({', '.join(sorted(HYDRAGNN_FINETUNERS))}); got {train_script}"
        )
    if launch.total_ranks != 1:
        raise ValueError(
            "Built-in HydraGNN fine-tuners run as one process; set resources.pools.training "
            f"to 1 node and 1 device (it resolved to {launch.node_count} node(s) x "
            f"{launch.ranks_per_node} rank(s))"
        )
    _require_direct_placement(trainer_cfg, launch, "HydraGNN fine-tuning")

    from matsim_agents.active_learning.training_references import resolve_training_references
    from matsim_agents.datasets.governance import sha256_file

    reference_manifest = resolve_training_references(
        trainer_cfg,
        dataset_path,
        dft_config=dft_config,
        dft_launch=dft_launch,
        reference_root=reference_root or out_logdir.parent / "elemental-training-reference",
    )
    dataset_sha256 = sha256_file(Path(dataset_path))
    out_logdir.mkdir(parents=True, exist_ok=True)

    # Launchers receive script, dataset, output, resume, epochs, nodes, ranks,
    # references, checkpoint, branch MLP, ft_repo, charge, spin and HydraGNN root.
    # Without a launcher, invoke the script directly with Python.
    if trainer_cfg.train_launcher is not None:
        argv = [
            "bash",
            str(trainer_cfg.train_launcher),
            str(train_script),
            str(dataset_path),
            str(out_logdir),
            str(hydragnn_cfg.logdir),  # resume from current logdir
            str(trainer_cfg.epochs_per_iter),
            str(launch.node_count),
            str(launch.ranks_per_node),
            str(reference_manifest),
            hydragnn_cfg.checkpoint or "",
            str(hydragnn_cfg.hydragnn_branch_mlp_checkpoint or ""),
            str(hydragnn_cfg.ft_repo or ""),
            str(hydragnn_cfg.charge),
            str(hydragnn_cfg.spin),
            str(hydragnn_cfg.hydragnn_root or ""),
        ]
    else:
        log.warning(
            "trainer.train_launcher not set; running training in-process "
            "(no srun, no module swap). For multi-node training set train_launcher."
        )
        argv = [
            sys.executable,
            str(train_script),
            "--dataset",
            str(dataset_path),
            "--output-dir",
            str(out_logdir),
            "--gfm-logdir",
            str(hydragnn_cfg.logdir),
            "--epochs",
            str(trainer_cfg.epochs_per_iter),
            "--elemental-reference-manifest",
            str(reference_manifest),
        ]
        if script_name == "finetune_hydragnn":
            if hydragnn_cfg.hydragnn_branch_mlp_checkpoint is None:
                raise ValueError("Routed HydraGNN training requires a branch MLP checkpoint")
            argv.extend(["--branch-mlp", str(hydragnn_cfg.hydragnn_branch_mlp_checkpoint)])
        if hydragnn_cfg.checkpoint is not None:
            argv.extend(["--gfm-checkpoint", hydragnn_cfg.checkpoint])
        argv.extend(["--charge", str(hydragnn_cfg.charge), "--spin", str(hydragnn_cfg.spin)])
        if hydragnn_cfg.hydragnn_root is not None:
            argv.extend(["--hydragnn-root", str(hydragnn_cfg.hydragnn_root)])
        if script_name == "finetune_hydragnn_newhead" and hydragnn_cfg.ft_repo is not None:
            argv.extend(["--ft-repo", str(hydragnn_cfg.ft_repo)])

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
    from matsim_agents.active_learning.training_references import resolve_training_references

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
    from matsim_agents.datasets.energy_alignment import ENERGY_ALIGNMENT_FILE, EnergyAlignment
    from matsim_agents.datasets.governance import sha256_file

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
    launch: ProcessLaunch,
    dft_config: DFTConfig | None = None,
    dft_launch: ProcessLaunch | None = None,
    reference_root: Path | None = None,
) -> Path:
    """Fine-tune ``uma_cfg``'s model on the dataset as a child process.

    Returns the fine-tuned ``inference_ckpt.pt``, which becomes
    ``UMAConfig.model_name`` when the model is promoted. The labels are aligned
    to the base model's energy scale with the training references (see
    :mod:`matsim_agents.training.finetune_uma`). A launcher receives
    ``<train_script> <dataset_path> <out_model_dir> <base_model> <task_name>``
    ``<epochs> <nodes> <ranks> <reference_manifest> <finetune_mode>`` and must
    run the fine-tune with ``--output-dir <out_model_dir>``.
    """
    from matsim_agents.datasets.governance import sha256_file

    out_model_dir = Path(out_model_dir)
    if not trainer_cfg.enabled:
        log.info("trainer.enabled=False; skipping UMA fine-tune at iteration %d", iteration)
        return Path(str(uma_cfg.model_name))

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
            str(launch.node_count),
            str(launch.ranks_per_node),
            str(manifest),
            trainer_cfg.finetune_mode,
        ]
    else:
        log.warning(
            "trainer.train_launcher not set; running UMA fine-tune in-process "
            "(no srun, no module swap). For multi-node training set train_launcher."
        )
        argv = [
            sys.executable,
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
            "--elemental-reference-manifest",
            str(manifest),
        ]
        if trainer_cfg.finetune_mode == "lora":
            argv.append("--lora")

    log_path = out_model_dir / "train.log"
    _run_trainer(argv, log_path, launch, trainer_cfg.train_launcher is None, "UMA fine-tune")

    from matsim_agents.backends.mlip.calculator import _resolve_finetuned_uma_checkpoint

    checkpoint = _resolve_finetuned_uma_checkpoint(str(out_model_dir))
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
    launch: ProcessLaunch,
    dft_config: DFTConfig | None = None,
    dft_launch: ProcessLaunch | None = None,
    reference_root: Path | None = None,
) -> Path:
    """Fine-tune ``mace_cfg``'s model on the dataset as a child process.

    Returns the fine-tuned ``mace_finetuned.model``, which becomes the model
    (``MACEConfig.family='checkpoint'``) when it is promoted. The labels are
    aligned to the base model's energy scale with the training references (see
    :mod:`matsim_agents.training.finetune_mace`). A launcher receives
    ``<train_script> <dataset_path> <out_model_dir> <family> <base_model>``
    ``<epochs> <nodes> <ranks> <reference_manifest> <finetune_mode>`` and must
    write ``mace_finetuned.model`` to ``<out_model_dir>``. With no launcher, the
    built-in ``finetune_mace`` CLI runs in-process.
    """
    from matsim_agents.datasets.governance import sha256_file

    out_model_dir = Path(out_model_dir)
    if not trainer_cfg.enabled:
        log.info("trainer.enabled=False; skipping MACE fine-tune at iteration %d", iteration)
        # Point downstream at the existing base model (checkpoint path or variant).
        return Path(str(mace_cfg.model))

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
            str(launch.node_count),
            str(launch.ranks_per_node),
            str(manifest),
            trainer_cfg.finetune_mode,
        ]
    else:
        log.warning(
            "trainer.train_launcher not set; running MACE fine-tune in-process "
            "(no srun, no module swap). For multi-node training set train_launcher."
        )
        argv = [
            sys.executable,
            "-m",
            "matsim_agents.training.finetune_mace",
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
            "--elemental-reference-manifest",
            str(manifest),
        ]
        if trainer_cfg.finetune_mode == "lora":
            argv.append("--lora")

    log_path = out_model_dir / "train.log"
    _run_trainer(argv, log_path, launch, trainer_cfg.train_launcher is None, "MACE fine-tune")
    finetuned = out_model_dir / "mace_finetuned.model"
    if not finetuned.is_file():
        raise RuntimeError(
            f"MACE fine-tune exited successfully but did not write {finetuned}; see {log_path}"
        )
    _check_energy_alignment(out_model_dir, Path(dataset_path), dataset_sha256, manifest)
    return finetuned
