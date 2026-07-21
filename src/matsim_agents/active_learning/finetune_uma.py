"""Fine-tune a UMA (fairchem) foundation model on an AL-collected DFT dataset.

This wraps fairchem-core 2.21.0's fine-tuning workflow so the active-learning
pipeline can turn an extended-XYZ dataset (as written by
:func:`matsim_agents.active_learning.trainer.append_frames_to_extxyz`) into a
fine-tuned ``inference_ckpt.pt`` that
:func:`matsim_agents.active_learning.calculator.build_uma_calculator` can load.

The heavy lifting is delegated to fairchem's own scripts:

* ``fairchem.core.scripts.create_finetune_dataset`` converts ASE atoms into the
  ``aselmdb`` format and computes the normaliser / linear element references,
* ``fairchem.core.scripts.create_uma_finetune_dataset`` fills the Hydra config
  templates shipped in the fairchem *source* tree (the PyPI wheel does not ship
  ``configs/``), and
* the ``fairchem`` CLI (``fairchem -c <config>``) runs the actual training.

Because ``create_uma_finetune_dataset`` reads its templates from a *relative*
``configs/uma/finetune`` path, a checkout of the matching fairchem source tree
must be provided via ``--fairchem-src`` (or the ``FAIRCHEM_SRC`` env var). The
generated Hydra config is self-contained (the ``fairchem`` CLI initialises its
config dir from the config file's own directory), so training itself does not
need the source tree on ``sys.path``/CWD.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read as ase_read
from ase.io import write as ase_write

from matsim_agents.active_learning.cost import CostReport, GpuMemorySampler, track_cost

log = logging.getLogger(__name__)

# UMA fine-tuning tasks understood by create_uma_finetune_dataset.
_UMA_TASKS = {"omat", "omol", "oc20", "odac", "omc"}
_REGRESSION_TASKS = {"e", "ef", "efs"}


# --------------------------------------------------------------------------- #
# Dataset preparation                                                         #
# --------------------------------------------------------------------------- #


def _reference_energy(atoms: Atoms) -> float | None:
    if "energy" in atoms.info:
        return float(atoms.info["energy"])
    try:
        return float(atoms.get_potential_energy())
    except Exception:  # noqa: BLE001
        return None


def _reference_forces(atoms: Atoms) -> np.ndarray | None:
    if "forces" in atoms.arrays:
        return np.asarray(atoms.arrays["forces"], dtype=float)
    try:
        return np.asarray(atoms.get_forces(), dtype=float)
    except Exception:  # noqa: BLE001
        return None


def _reference_stress(atoms: Atoms) -> np.ndarray | None:
    if "stress" in atoms.info:
        return np.asarray(atoms.info["stress"], dtype=float)
    try:
        return np.asarray(atoms.get_stress(), dtype=float)
    except Exception:  # noqa: BLE001
        return None


def _with_singlepoint(atoms: Atoms, *, need_stress: bool) -> Atoms | None:
    """Attach a SinglePointCalculator so fairchem's converter finds results.

    fairchem's ``write_ase_db`` asserts that ``atoms.calc.results`` contains
    ``energy`` and ``forces`` (and ``stress`` for the ``efs`` task). Our extxyz
    frames store these under ``info``/``arrays``, so re-attach them explicitly.
    """
    e = _reference_energy(atoms)
    f = _reference_forces(atoms)
    if e is None or f is None:
        return None
    out = atoms.copy()
    results: dict[str, object] = {"energy": e, "forces": f}
    if need_stress:
        s = _reference_stress(atoms)
        if s is None:
            return None
        results["stress"] = s
    out.calc = SinglePointCalculator(out, **results)
    return out


def _auto_regression_tasks(frames: list[Atoms]) -> str:
    """Pick the richest regression task supported by *all* frames."""
    have_stress = all(_reference_stress(a) is not None for a in frames)
    have_forces = all(_reference_forces(a) is not None for a in frames)
    if have_stress:
        return "efs"
    if have_forces:
        return "ef"
    return "e"


def _split_train_val(
    frames: list[Atoms], val_fraction: float, seed: int
) -> tuple[list[Atoms], list[Atoms]]:
    n = len(frames)
    if n < 2:
        raise ValueError(f"Need >=2 frames to fine-tune, got {n}.")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = max(1, int(round(val_fraction * n)))
    n_val = min(n_val, n - 1)  # keep at least one train frame
    val_idx = set(perm[:n_val].tolist())
    train = [frames[i] for i in range(n) if i not in val_idx]
    val = [frames[i] for i in range(n) if i in val_idx]
    return train, val


# --------------------------------------------------------------------------- #
# Config generation                                                           #
# --------------------------------------------------------------------------- #


def _resolve_fairchem_src(fairchem_src: str | Path | None) -> Path:
    candidate = fairchem_src or os.environ.get("FAIRCHEM_SRC")
    if not candidate:
        raise ValueError(
            "A fairchem source checkout is required (its configs/uma/finetune "
            "templates are not shipped in the PyPI wheel). Pass --fairchem-src "
            "or set FAIRCHEM_SRC to the repo root."
        )
    root = Path(candidate).expanduser().resolve()
    template_dir = root / "configs" / "uma" / "finetune"
    if not template_dir.is_dir():
        raise FileNotFoundError(
            f"fairchem templates not found under {template_dir}. Point "
            "--fairchem-src at a fairchem source checkout matching the installed "
            "fairchem-core version."
        )
    return root


def _prepare_dataset_and_config(
    *,
    train_dir: Path,
    val_dir: Path,
    artifacts_dir: Path,
    task_name: str,
    regression_tasks: str,
    base_model: str,
    fairchem_root: Path,
    num_workers: int,
) -> Path:
    """Convert ASE dirs to aselmdb and emit the fine-tune Hydra config.

    Returns the path to the generated ``uma_sm_finetune_template.yaml``.
    """
    from fairchem.core.scripts import create_uma_finetune_dataset as cufd
    from fairchem.core.scripts.create_finetune_dataset import (
        compute_normalizer_and_linear_reference,
        launch_processing,
    )

    # create_uma_finetune_dataset reads templates from a CWD-relative path; point
    # it at the absolute template dir in the provided source checkout instead.
    cufd.TEMPLATE_DIR = fairchem_root / "configs" / "uma" / "finetune"

    if artifacts_dir.exists():
        raise FileExistsError(f"artifacts dir must not already exist: {artifacts_dir}")

    train_out = artifacts_dir / "train"
    val_out = artifacts_dir / "val"
    launch_processing(str(train_dir), train_out, num_workers)
    force_rms, linref_coeff = compute_normalizer_and_linear_reference(train_out, num_workers)
    if regression_tasks == "e":
        force_rms = 1.0
    launch_processing(str(val_dir), val_out, num_workers)

    cufd.create_yaml(
        train_path=str(train_out),
        val_path=str(val_out),
        force_rms=float(force_rms),
        linref_coeff=linref_coeff,
        output_dir=artifacts_dir,
        dataset_name=task_name,
        regression_tasks=regression_tasks,
        base_model_name=base_model,
    )
    return artifacts_dir / cufd.UMA_SM_FINETUNE_YAML


def _patch_finetune_config(
    config_path: Path,
    *,
    run_dir: Path,
    run_name: str,
    epochs: int,
    batch_size: int,
    lr: float,
    max_neighbors: int,
    ranks_per_node: int,
    device_type: str,
    weight_decay: float | None = None,
    warmup_epochs: float | None = None,
    lr_min_factor: float | None = None,
) -> None:
    """Override the generated Hydra config for a controlled single-node run."""
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    job = cfg.setdefault("job", {})
    job["run_dir"] = str(run_dir)
    job["run_name"] = run_name
    job["device_type"] = device_type
    job["debug"] = True
    # Disable the WandB logger (offline HPC nodes have no network / no wandb).
    job["logger"] = None
    scheduler = job.setdefault("scheduler", {})
    scheduler["mode"] = "LOCAL"
    scheduler["num_nodes"] = 1
    scheduler["ranks_per_node"] = int(ranks_per_node)

    cfg["epochs"] = int(epochs)
    cfg["steps"] = None
    cfg["batch_size"] = int(batch_size)
    cfg["lr"] = float(lr)
    cfg["max_neighbors"] = int(max_neighbors)
    if weight_decay is not None:
        cfg["weight_decay"] = float(weight_decay)

    # The cosine LR schedule's warmup fraction and floor are hard-coded in the
    # template's runner block (not top-level interpolation), so patch them in
    # place when a gentler foundation-model recipe asks for it.
    if warmup_epochs is not None or lr_min_factor is not None:
        try:
            sched = cfg["runner"]["train_eval_unit"]["cosine_lr_scheduler_fn"]
            if warmup_epochs is not None:
                sched["warmup_epochs"] = float(warmup_epochs)
            if lr_min_factor is not None:
                sched["lr_min_factor"] = float(lr_min_factor)
        except (KeyError, TypeError):
            log.warning("cosine_lr_scheduler_fn not found in config; skipping warmup patch")

    with open(config_path, "w") as fh:
        yaml.safe_dump(cfg, fh, default_flow_style=False, sort_keys=False)


def _find_inference_checkpoint(run_dir: Path) -> Path:
    """Locate the final fine-tuned checkpoint under a fairchem run dir."""
    matches = sorted(
        run_dir.glob("**/checkpoints/final/inference_ckpt.pt"),
        key=lambda p: p.stat().st_mtime,
    )
    if not matches:
        raise FileNotFoundError(
            f"No inference_ckpt.pt produced under {run_dir}. Check the training log."
        )
    return matches[-1]


# --------------------------------------------------------------------------- #
# Public entry point                                                          #
# --------------------------------------------------------------------------- #


def finetune_uma(
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    base_model: str = "uma-s-1p1",
    task_name: str = "omat",
    epochs: int = 1,
    regression_tasks: str | None = None,
    val_fraction: float = 0.1,
    batch_size: int = 2,
    lr: float = 2e-5,
    weight_decay: float = 1e-2,
    warmup_epochs: float | None = 0.1,
    lr_min_factor: float | None = 0.01,
    max_neighbors: int = 300,
    ranks_per_node: int = 1,
    num_workers: int = 8,
    seed: int = 0,
    fairchem_src: str | Path | None = None,
    work_dir: str | Path | None = None,
    device: str | None = None,
    run: bool = True,
) -> Path:
    """Fine-tune ``base_model`` on ``dataset_path`` and return the checkpoint path.

    When ``run`` is False the dataset and Hydra config are prepared but training
    is skipped (returns the config path) -- useful for dry-run validation.
    """
    if task_name not in _UMA_TASKS:
        raise ValueError(f"task_name must be one of {sorted(_UMA_TASKS)}, got {task_name!r}")

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    fairchem_root = _resolve_fairchem_src(fairchem_src)

    raw = ase_read(str(dataset_path), index=":")
    frames = [raw] if isinstance(raw, Atoms) else list(raw)
    if not frames:
        raise ValueError(f"No frames read from {dataset_path}")

    reg = regression_tasks or _auto_regression_tasks(frames)
    if reg not in _REGRESSION_TASKS:
        raise ValueError(f"regression_tasks must be one of {sorted(_REGRESSION_TASKS)}, got {reg!r}")
    need_stress = reg == "efs"

    prepared = [_with_singlepoint(a, need_stress=need_stress) for a in frames]
    usable = [a for a in prepared if a is not None]
    if len(usable) < 2:
        raise ValueError(
            f"Only {len(usable)}/{len(frames)} frames carry the labels required for "
            f"regression_tasks={reg!r}; need >=2."
        )

    train_frames, val_frames = _split_train_val(usable, val_fraction, seed)

    ase_in = output_dir / "ase_input"
    train_dir = ase_in / "train"
    val_dir = ase_in / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    ase_write(str(train_dir / "train.extxyz"), train_frames, format="extxyz")
    ase_write(str(val_dir / "val.extxyz"), val_frames, format="extxyz")
    log.info(
        "UMA fine-tune: %d train / %d val frames (regression=%s, task=%s)",
        len(train_frames),
        len(val_frames),
        reg,
        task_name,
    )

    # fairchem's dataset builder writes LMDB (``aselmdb``) files, whose mmap +
    # file-locking are NOT supported on the CFS/GPFS parallel filesystem
    # ("lmdb.Error: ... Function not implemented"). Run all fairchem dataset +
    # training IO in a node-local scratch dir (``$TMPDIR``/``/tmp`` by default,
    # overridable via ``MATSIM_UMA_WORKDIR``), then copy the small inference
    # checkpoint back to the CFS output dir so it persists past the job.
    scratch_base = Path(work_dir or os.environ.get("MATSIM_UMA_WORKDIR") or tempfile.gettempdir())
    scratch_base.mkdir(parents=True, exist_ok=True)
    work_root = Path(tempfile.mkdtemp(prefix="matsim_uma_ft_", dir=str(scratch_base)))
    artifacts_dir = work_root / "ft"
    run_dir = work_root / "train_runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        config_path = _prepare_dataset_and_config(
            train_dir=train_dir,
            val_dir=val_dir,
            artifacts_dir=artifacts_dir,
            task_name=task_name,
            regression_tasks=reg,
            base_model=base_model,
            fairchem_root=fairchem_root,
            num_workers=num_workers,
        )

        device_type = device or ("CUDA" if _cuda_available() else "CPU")
        _patch_finetune_config(
            config_path,
            run_dir=run_dir,
            run_name="uma_finetune",
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            max_neighbors=max_neighbors,
            ranks_per_node=ranks_per_node,
            device_type=device_type,
            weight_decay=weight_decay,
            warmup_epochs=warmup_epochs,
            lr_min_factor=lr_min_factor,
        )

        if not run:
            log.info("run=False; prepared config at %s (training skipped)", config_path)
            return config_path

        num_gpus = int(ranks_per_node) if device_type == "CUDA" else 0
        report = CostReport(
            model_backend="uma",
            dataset_label=Path(dataset_path).parent.name or Path(dataset_path).stem,
            base_model=base_model,
            dataset_path=str(Path(dataset_path).resolve()),
            n_train_frames=len(train_frames),
            n_val_frames=len(val_frames),
            epochs=int(epochs),
            num_gpus=num_gpus,
            device=device_type,
            extra={"task_name": task_name, "regression_tasks": reg, "protocol": "gentle-finetune",
                   "lr": float(lr), "weight_decay": float(weight_decay)},
        )
        # UMA trains in a subprocess -> sample GPU memory via nvidia-smi, but only
        # on a real GPU run (nvidia-smi reports node-total usage, which is bogus on
        # a shared CPU login node).
        sampler = GpuMemorySampler(enabled=device_type == "CUDA")
        with sampler, track_cost(report):
            _run_fairchem(config_path)
        if sampler.peak_gb:
            report.peak_gpu_mem_gb = round(sampler.peak_gb, 3)
        report.write(output_dir / "cost.json")

        local_ckpt = _find_inference_checkpoint(run_dir)
        # Persist the (self-contained) inference checkpoint on CFS; the training
        # DCP shards stay in node-local scratch and are discarded with it.
        dest_ckpt = output_dir / "inference_ckpt.pt"
        shutil.copy2(local_ckpt, dest_ckpt)
        log.info(
            "UMA fine-tune complete -> %s (%.1fs, %.4f GPU-h)",
            dest_ckpt,
            report.wall_time_s,
            report.gpu_hours,
        )
        return dest_ckpt
    finally:
        # Node-local scratch is ephemeral, but clean up eagerly so co-scheduled
        # ``shared``-QOS jobs on the same node don't accumulate stale LMDB dirs.
        if run:
            shutil.rmtree(work_root, ignore_errors=True)


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


def _run_fairchem(config_path: Path) -> None:
    """Invoke the fairchem training CLI on the generated config.

    The training entry point is the ``fairchem`` *console script*, which calls
    ``fairchem.core._cli:main``. The ``_cli`` module has NO ``if __name__ ==
    "__main__"`` guard, so ``python -m fairchem.core._cli`` merely imports it and
    exits 0 without training. We therefore prefer the console script that ships
    next to the active interpreter (the venv ``bin/`` dir, which is not
    necessarily on ``PATH`` when the interpreter is invoked by absolute path),
    then fall back to ``PATH``, and finally to importing and calling ``main()``
    explicitly in a fresh interpreter.
    """
    candidate = Path(sys.executable).with_name("fairchem")
    fairchem_bin = str(candidate) if candidate.is_file() else shutil.which("fairchem")
    if fairchem_bin:
        argv = [fairchem_bin, "-c", str(config_path)]
    else:  # module has no __main__ guard -> call main() explicitly
        argv = [
            sys.executable,
            "-c",
            "import sys; from fairchem.core._cli import main; sys.exit(main())",
            "-c",
            str(config_path),
        ]
    log.info("Launching fairchem: %s", " ".join(argv))
    proc = subprocess.run(argv, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"fairchem fine-tune failed with exit {proc.returncode}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", required=True, help="AL-collected extxyz dataset.")
    parser.add_argument("--output-dir", required=True, help="Work/output directory.")
    parser.add_argument("--base-model", default="uma-s-1p1", help="Base UMA model name.")
    parser.add_argument(
        "--task-name",
        default="omat",
        choices=sorted(_UMA_TASKS),
        help="UMA task (omat: inorganic bulk, omol: molecules/MOFs, ...).",
    )
    parser.add_argument("--epochs", type=int, default=1, help="Number of fine-tune epochs.")
    parser.add_argument(
        "--regression-tasks",
        default=None,
        choices=sorted(_REGRESSION_TASKS),
        help="Targets: e / ef / efs. Default: auto-detect from dataset labels.",
    )
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Fine-tune learning rate. Default 2e-5 is a gentle foundation-model "
        "recipe that avoids catastrophic forgetting on small AL datasets.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-2,
        help="AdamW weight decay (regularises toward the base model).",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=float,
        default=0.1,
        help="Cosine-LR warmup length in epochs.",
    )
    parser.add_argument("--max-neighbors", type=int, default=300)
    parser.add_argument("--ranks-per-node", type=int, default=1, help="GPUs to train on.")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--fairchem-src",
        default=None,
        help="fairchem source checkout root (or set FAIRCHEM_SRC).",
    )
    parser.add_argument("--device", default=None, choices=["CUDA", "CPU"])
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare dataset + config but do not launch training.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    ckpt = finetune_uma(
        args.dataset,
        args.output_dir,
        base_model=args.base_model,
        task_name=args.task_name,
        epochs=args.epochs,
        regression_tasks=args.regression_tasks,
        val_fraction=args.val_fraction,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        max_neighbors=args.max_neighbors,
        ranks_per_node=args.ranks_per_node,
        num_workers=args.num_workers,
        seed=args.seed,
        fairchem_src=args.fairchem_src,
        device=args.device,
        run=not args.dry_run,
    )
    print(ckpt)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
