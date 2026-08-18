"""Fine-tune a MACE foundation model on an AL-collected DFT dataset.

This turns an extended-XYZ dataset (as written by
:func:`matsim_agents.active_learning.trainer.append_frames_to_extxyz`) into a
fine-tuned ``mace_finetuned.model`` that
:func:`matsim_agents.active_learning.calculator.build_mace_calculator` can load
via ``family="checkpoint"``.

All MACE versions are fine-tunable behind the same entry point -- the base model
is selected exactly as for inference (see
:class:`matsim_agents.active_learning.config.MACEConfig`):

* ``family="mace_mp"`` with ``base_model in {small, medium, large}`` (or a
  release tag / URL) -- Materials Project inorganic foundation models;
* ``family="mace_off"`` with the same size variants -- organic-molecule models;
* ``family="checkpoint"`` with ``base_model`` a local ``.model`` path -- continue
  fine-tuning an already-adapted checkpoint.

A curated set of variants is also exposed through :data:`MACE_MODELS`, so a
launcher can iterate "all versions" by model id.

Method (matches allaffa/HydraGNN_GFM_FineTuning4Materials @ ``uma-sota-comparison``)
------------------------------------------------------------------------------------
Fine-tuning is delegated to MACE's own ``mace_run_train`` entry point (run as a
subprocess), using the exact reference recipe so the results are reproducible
against that benchmark pipeline::

    mace_run_train --foundation_model <ckpt> --loss ef --E0s average
      --default_dtype float64 --lr <lr> --max_num_epochs <N> --batch_size 4
      --ema --ema_decay 0.995 --amsgrad --clip_grad 10.0 --weight_decay <wd>
      [--lora True --lora_rank <r> --lora_alpha <a>]

Defaults follow the reference: naive full fine-tune uses ``lr=1e-3`` for 50
epochs; LoRA (parameter-efficient, mitigates catastrophic forgetting) uses
``lr=5e-3`` for 10 epochs with ``rank=4, alpha=1.0``.

The produced ``.model`` file is a pickled MACE module; it is copied to
``mace_finetuned.model`` and reloads through the ordinary
``MACECalculator(model_paths=[...])`` path used by ``family="checkpoint"``.
"""

from __future__ import annotations

import argparse
import logging
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from matsim_agents.active_learning.cost import CostReport, GpuMemorySampler, track_cost
from matsim_agents.active_learning.finetune_uma import (
    _load_labeled_frames,
    _reference_energy,
    _reference_forces,
)

log = logging.getLogger(__name__)

# Foundation families MACE ships; "checkpoint" continues from a local .model.
_MACE_FAMILIES = {"mace_mp", "mace_off", "checkpoint"}

# Reference recipe defaults (run_mace_finetune.py): naive vs. LoRA fine-tune.
_NAIVE_DEFAULTS = {"lr": 1e-3, "epochs": 50}
_LORA_DEFAULTS = {"lr": 5e-3, "epochs": 10, "rank": 4, "alpha": 1.0}

_DTYPE_MAP = {"fp32": "float32", "fp64": "float64", None: "float64"}

# Curated MACE foundation variants (mirrors the reference utils/mace_calculator.py
# MACE_MODELS registry). A launcher can iterate these ids to fine-tune "all
# versions". Organic-checkpoint variants that require a local .model file
# (e.g. MACE-OFF polar/omol checkpoints) are reached via family="checkpoint"
# with an explicit path rather than being hard-coded here.
MACE_MODELS: dict[str, dict[str, str]] = {
    "mace_mp_small": {"family": "mace_mp", "model": "small", "label": "MACE-MP-0 small"},
    "mace_mp_medium": {"family": "mace_mp", "model": "medium", "label": "MACE-MP-0 medium"},
    "mace_mp_large": {"family": "mace_mp", "model": "large", "label": "MACE-MP-0 large"},
    "mace_off_small": {"family": "mace_off", "model": "small", "label": "MACE-OFF23 small"},
    "mace_off_medium": {"family": "mace_off", "model": "medium", "label": "MACE-OFF23 medium"},
    "mace_off_large": {"family": "mace_off", "model": "large", "label": "MACE-OFF23 large"},
}


# --------------------------------------------------------------------------- #
# Foundation-model + dataset preparation                                      #
# --------------------------------------------------------------------------- #


def _resolve_foundation_model_file(
    family: str,
    base_model: str,
    *,
    precision: str | None,
    dispersion: bool,
    work_dir: Path,
) -> str:
    """Return a local ``.model`` path for ``mace_run_train --foundation_model``.

    A local checkpoint path is returned as-is. For ``mace_mp`` / ``mace_off``
    families the foundation model is loaded through the shared calculator (which
    downloads/caches it), then its module is serialised to ``foundation.model``
    so ``mace_run_train`` can consume it uniformly across families/machines.
    """
    import os

    import torch

    if os.path.isfile(base_model):
        return str(Path(base_model).resolve())

    from matsim_agents.active_learning.calculator import build_mace_calculator
    from matsim_agents.active_learning.config import MACEConfig

    mcfg = MACEConfig(
        family=family,  # type: ignore[arg-type]
        model=base_model,
        device="cpu",
        precision=precision,  # type: ignore[arg-type]
        dispersion=dispersion,
    )
    calc = build_mace_calculator(mcfg, enable_mc_dropout=False)
    models = getattr(calc, "models", None)
    module = models[0] if models else getattr(calc, "model", None)
    if module is None:
        raise RuntimeError("Could not obtain the MACE foundation module to seed fine-tuning.")
    foundation = Path(work_dir) / "foundation.model"
    torch.save(module.to("cpu"), str(foundation))
    log.info("Seeded foundation model for %s:%s -> %s", family, base_model, foundation)
    return str(foundation)


def _write_xyz_splits(
    frames: list,
    work_dir: Path,
    *,
    seed: int,
    val_fraction: float = 0.1,
) -> tuple[Path, Path, int, int]:
    """Write ``train.xyz`` / ``val.xyz`` with ``REF_energy`` / ``REF_forces`` keys.

    ``mace_run_train`` reads labels from the extended-XYZ ``info``/``arrays``
    fields named by ``--energy_key`` / ``--forces_key``, so we materialise those
    keys explicitly. Returns ``(train_xyz, val_xyz, n_train, n_val)``.
    """
    from ase.io import write as ase_write

    rng = np.random.default_rng(seed)
    n = len(frames)
    order = rng.permutation(n)
    n_val = max(1, int(round(val_fraction * n)))
    n_val = min(n_val, n - 1)
    val_idx = set(order[:n_val].tolist())

    train_atoms, val_atoms = [], []
    for i, a in enumerate(frames):
        e = _reference_energy(a)
        f = _reference_forces(a)
        a2 = a.copy()
        a2.calc = None  # never serialise a live calculator into the training XYZ
        a2.info["REF_energy"] = float(e)
        a2.arrays["REF_forces"] = np.asarray(f, dtype=float)
        a2.info.setdefault("charge", 0)
        a2.info.setdefault("spin", 1)
        (val_atoms if i in val_idx else train_atoms).append(a2)

    train_xyz = Path(work_dir) / "train.xyz"
    val_xyz = Path(work_dir) / "val.xyz"
    ase_write(str(train_xyz), train_atoms, format="extxyz")
    ase_write(str(val_xyz), val_atoms, format="extxyz")
    return train_xyz, val_xyz, len(train_atoms), len(val_atoms)


# --------------------------------------------------------------------------- #
# mace_run_train driver                                                       #
# --------------------------------------------------------------------------- #


def _mace_run_train_cmd() -> list[str]:
    """Locate the ``mace_run_train`` console entry point (or fall back to -m)."""
    exe = Path(sys.executable).parent / "mace_run_train"
    if exe.exists():
        return [str(exe)]
    return [sys.executable, "-m", "mace.cli.run_train"]


def _run_mace_train(
    *,
    foundation: str,
    train_xyz: Path,
    val_xyz: Path,
    run_name: str,
    work_dir: Path,
    model_dir: Path,
    device: str,
    dtype: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    lora: bool,
    lora_rank: int,
    lora_alpha: float,
) -> None:
    """Invoke ``mace_run_train`` with the reference fine-tuning recipe."""
    cmd = [
        *_mace_run_train_cmd(),
        "--name",
        run_name,
        "--foundation_model",
        foundation,
        "--train_file",
        str(train_xyz),
        "--valid_file",
        str(val_xyz),
        "--energy_key",
        "REF_energy",
        "--forces_key",
        "REF_forces",
        "--loss",
        "ef",
        "--E0s",
        "average",
        "--lr",
        str(lr),
        "--max_num_epochs",
        str(epochs),
        "--batch_size",
        str(batch_size),
        "--valid_batch_size",
        str(batch_size),
        "--device",
        device,
        "--default_dtype",
        dtype,
        "--work_dir",
        str(work_dir),
        "--model_dir",
        str(model_dir),
        "--checkpoints_dir",
        str(work_dir / "checkpoints"),
        "--results_dir",
        str(work_dir / "results"),
        "--compute_stress",
        "False",
        "--log_level",
        "WARNING",
    ]
    if lora:
        cmd += ["--lora", "True", "--lora_rank", str(lora_rank), "--lora_alpha", str(lora_alpha)]
    cmd += [
        "--ema",
        "--ema_decay",
        "0.995",
        "--amsgrad",
        "--clip_grad",
        "10.0",
        "--weight_decay",
        str(weight_decay),
    ]
    log.info("Running mace_run_train: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _select_trained_model(model_dir: Path, run_name: str) -> Path:
    """Pick the fine-tuned ``.model`` file that ``mace_run_train`` produced."""
    exact = model_dir / f"{run_name}.model"
    if exact.exists():
        return exact
    candidates = sorted(model_dir.glob("*.model"))
    if not candidates:
        raise RuntimeError(f"mace_run_train produced no .model file in {model_dir}")
    # Prefer the plain (non-compiled) checkpoint for portable reloading.
    non_compiled = [c for c in candidates if "compiled" not in c.name]
    return (non_compiled or candidates)[0]


# --------------------------------------------------------------------------- #
# Public entry point                                                          #
# --------------------------------------------------------------------------- #


def finetune_mace(
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    family: str = "mace_mp",
    base_model: str = "medium",
    model_id: str | None = None,
    precision: str | None = "fp64",
    dispersion: bool = False,
    epochs: int | None = None,
    batch_size: int = 4,
    lr: float | None = None,
    lora: bool = False,
    lora_rank: int | None = None,
    lora_alpha: float | None = None,
    weight_decay: float = 0.0,
    seed: int = 0,
    device: str | None = None,
    run: bool = True,
    # Accepted for parity with the other fine-tune entry points. mace_run_train
    # uses its own energy/force weighting and (for LoRA) parameter-efficient
    # freezing, so these are recorded but not passed through.
    force_weight: float = 10.0,
    freeze_backbone: bool = False,
    ranks_per_node: int = 1,
    **_legacy,
) -> Path:
    """Fine-tune a MACE foundation model and return the checkpoint path.

    ``family`` + ``base_model`` (or a :data:`MACE_MODELS` ``model_id``) select the
    MACE version. Fine-tuning is delegated to ``mace_run_train`` with the
    reference recipe; pass ``lora=True`` for the parameter-efficient variant.

    The returned ``mace_finetuned.model`` reloads through
    :func:`matsim_agents.active_learning.calculator.build_mace_calculator`
    (``family="checkpoint"``). When ``run`` is False the dataset is loaded and
    validated but no training/checkpoint write occurs; the intended checkpoint
    path is still returned.
    """
    import torch

    if model_id is not None:
        if model_id not in MACE_MODELS:
            raise ValueError(f"model_id must be one of {sorted(MACE_MODELS)}, got {model_id!r}")
        family = MACE_MODELS[model_id]["family"]
        base_model = MACE_MODELS[model_id]["model"]

    if family not in _MACE_FAMILIES:
        raise ValueError(f"family must be one of {sorted(_MACE_FAMILIES)}, got {family!r}")

    _recipe = _LORA_DEFAULTS if lora else _NAIVE_DEFAULTS
    epochs = int(epochs if epochs is not None else _recipe["epochs"])
    lr = float(lr if lr is not None else _recipe["lr"])
    lora_rank = int(lora_rank if lora_rank is not None else _LORA_DEFAULTS["rank"])
    lora_alpha = float(lora_alpha if lora_alpha is not None else _LORA_DEFAULTS["alpha"])
    dtype = _DTYPE_MAP[precision]

    if freeze_backbone and not lora:
        log.warning(
            "freeze_backbone=True is ignored by the mace_run_train path; use lora=True "
            "for parameter-efficient fine-tuning."
        )

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dest_ckpt = output_dir / "mace_finetuned.model"

    dev = (device or ("cuda" if torch.cuda.is_available() else "cpu")).lower()
    if dev.startswith("cuda") and not torch.cuda.is_available():
        log.warning("CUDA requested but unavailable; falling back to CPU.")
        dev = "cpu"
    mace_device = "cuda" if dev.startswith("cuda") else "cpu"

    train_atoms = _load_labeled_frames(dataset_path)
    log.info(
        "MACE fine-tune: %d frames (family=%s, base=%s, lora=%s, device=%s)",
        len(train_atoms),
        family,
        base_model,
        lora,
        mace_device,
    )

    if not run:
        log.info("run=False; loaded %d frames (training skipped)", len(train_atoms))
        return dest_ckpt

    work_dir = output_dir / "mace_run_train"
    model_dir = work_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    foundation = _resolve_foundation_model_file(
        family, base_model, precision=precision, dispersion=dispersion, work_dir=work_dir
    )
    train_xyz, val_xyz, n_train, n_val = _write_xyz_splits(train_atoms, work_dir, seed=seed)
    run_name = re.sub(r"[^0-9A-Za-z_]+", "_", f"mace_ft_{family}_{Path(base_model).name}")
    if lora:
        run_name += "_lora"

    report = CostReport(
        model_backend="mace",
        dataset_label=Path(dataset_path).parent.name or Path(dataset_path).stem,
        base_model=f"{family}:{base_model}",
        dataset_path=str(Path(dataset_path).resolve()),
        n_train_frames=n_train,
        n_val_frames=n_val,
        epochs=int(epochs),
        num_gpus=1 if mace_device == "cuda" else 0,
        device=mace_device.upper(),
        extra={
            "family": family,
            "base_model": base_model,
            "model_id": model_id,
            "precision": precision or "fp64",
            "protocol": "mace_run_train-lora" if lora else "mace_run_train",
            "lr": float(lr),
            "batch_size": int(batch_size),
            "weight_decay": float(weight_decay),
            "lora": bool(lora),
            "lora_rank": int(lora_rank) if lora else None,
            "lora_alpha": float(lora_alpha) if lora else None,
            "ema_decay": 0.995,
            "clip_grad": 10.0,
            "amsgrad": True,
            "loss": "ef",
            "e0s": "average",
        },
    )
    sampler = GpuMemorySampler(enabled=mace_device == "cuda")
    with sampler, track_cost(report):
        _run_mace_train(
            foundation=foundation,
            train_xyz=train_xyz,
            val_xyz=val_xyz,
            run_name=run_name,
            work_dir=work_dir,
            model_dir=model_dir,
            device=mace_device,
            dtype=dtype,
            epochs=int(epochs),
            batch_size=int(batch_size),
            lr=float(lr),
            weight_decay=float(weight_decay),
            lora=bool(lora),
            lora_rank=int(lora_rank),
            lora_alpha=float(lora_alpha),
        )
    if sampler.peak_gb:
        report.peak_gpu_mem_gb = round(sampler.peak_gb, 3)
    report.write(output_dir / "cost.json")

    produced = _select_trained_model(model_dir, run_name)
    # The .model is a pickled MACE module; MACECalculator reloads it with
    # map_location=<device>, so a straight copy is portable across GPUs/machines.
    shutil.copy(str(produced), str(dest_ckpt))
    log.info(
        "MACE fine-tune complete -> %s (%.1fs, %.4f GPU-h)",
        dest_ckpt,
        report.wall_time_s,
        report.gpu_hours,
    )
    return dest_ckpt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", required=True, help="AL-collected extxyz dataset.")
    parser.add_argument("--output-dir", required=True, help="Work/output directory.")
    parser.add_argument(
        "--model-id",
        default=None,
        choices=sorted(MACE_MODELS),
        help="Curated MACE variant id (overrides --family/--base-model).",
    )
    parser.add_argument(
        "--family",
        default="mace_mp",
        choices=sorted(_MACE_FAMILIES),
        help="MACE family: mace_mp (inorganic), mace_off (organic), or checkpoint (local .model).",
    )
    parser.add_argument(
        "--base-model",
        default="medium",
        help="Size/variant (small|medium|large, or a release tag/URL), or a .model path for family=checkpoint.",
    )
    parser.add_argument(
        "--precision", default="fp64", choices=["fp32", "fp64"], help="Training dtype."
    )
    parser.add_argument(
        "--dispersion", action="store_true", help="Seed with DFT-D3 dispersion (mace_mp/mace_off)."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Fine-tune epochs (default: 10 for LoRA, 50 for naive).",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="mace_run_train batch size.")
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (default: 5e-3 for LoRA, 1e-3 for naive).",
    )
    parser.add_argument(
        "--lora",
        action="store_true",
        help="Parameter-efficient LoRA fine-tune (native mace_run_train --lora).",
    )
    parser.add_argument("--lora-rank", type=int, default=None, help="LoRA rank (default: 4).")
    parser.add_argument("--lora-alpha", type=float, default=None, help="LoRA alpha (default: 1.0).")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None, choices=["CUDA", "CPU", "cuda", "cpu"])
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load the dataset but do not train / write a checkpoint.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    ckpt = finetune_mace(
        args.dataset,
        args.output_dir,
        family=args.family,
        base_model=args.base_model,
        model_id=args.model_id,
        precision=args.precision,
        dispersion=args.dispersion,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lora=args.lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
        run=not args.dry_run,
    )
    print(ckpt)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
