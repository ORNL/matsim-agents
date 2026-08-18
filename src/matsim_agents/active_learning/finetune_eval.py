"""Endpoints fine-tune + evaluation campaign for one (backend, dataset) pair.

Produces the two learning-curve endpoints the paper reports -- *before*
(baseline foundation model, no fine-tuning) and *after* (fine-tuned on all the
AL-corrected data) -- on a fixed held-out test set, for a single backend on a
single dataset. Emits, under ``output_dir``:

* ``split/train.extxyz`` + ``split/test_set.extxyz`` -- deterministic 80/20
  split (seeded, so both backends score the *same* held-out frames);
* ``ft/`` -- the fine-tuned model (logdir/checkpoint) and its ``cost.json``;
* ``eval/iter0.json`` (+ ``iter0_parity.npz``) -- baseline "before";
* ``eval/iter<N>.json`` (+ ``iter<N>_parity.npz``) -- fine-tuned "after", where
  ``N`` is the number of distinct AL iterations in the dataset (so the x-axis
  is meaningful and matches the ``iter(\\d+)`` plot regex).

Backend/venv coupling (IMPORTANT): HydraGNN needs the *hydragnn* venv and UMA
needs the *fairchem* venv, so this runner handles ONE backend per invocation.
The Perlmutter launcher submits one job per (backend, dataset) with the correct
interpreter/venv.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read as ase_read
from ase.io import write as ase_write

from matsim_agents.active_learning.config import (
    HydraGNNConfig,
    MACEConfig,
    MCDropoutInjectionConfig,
    MLIPConfig,
    UMAConfig,
)
from matsim_agents.active_learning.evaluate import (
    _subsample,
    evaluate_frames,
)

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Dataset split                                                               #
# --------------------------------------------------------------------------- #


def _num_al_iterations(frames: list[Atoms]) -> int:
    iters = {int(a.info["al_iteration"]) for a in frames if "al_iteration" in a.info}
    return len(iters) if iters else 1


def _split_dataset(
    frames: list[Atoms], test_fraction: float, seed: int
) -> tuple[list[Atoms], list[Atoms]]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(frames))
    rng.shuffle(idx)
    n_test = max(1, int(round(len(frames) * test_fraction)))
    test_idx = set(idx[:n_test].tolist())
    train = [f for i, f in enumerate(frames) if i not in test_idx]
    test = [f for i, f in enumerate(frames) if i in test_idx]
    return train, test


# --------------------------------------------------------------------------- #
# MLIP config builders                                                        #
# --------------------------------------------------------------------------- #


def _hydragnn_cfg(logdir: Path, checkpoint: str, branch_mlp: Path, device: str) -> MLIPConfig:
    return MLIPConfig(
        backend="hydragnn",
        hydragnn=HydraGNNConfig(
            logdir=logdir,
            checkpoint=checkpoint,
            hydragnn_branch_mlp_checkpoint=branch_mlp,
            mlp_device="cuda" if device.startswith("cuda") else "cpu",
        ),
    )


def _newhead_cfg(
    logdir: Path, checkpoint: str, ft_repo: str | Path | None, device: str
) -> MLIPConfig:
    """Eval config for a single-branch 'new head' fine-tuned model (no branch-MLP)."""
    return MLIPConfig(
        backend="hydragnn",
        hydragnn=HydraGNNConfig(
            logdir=logdir,
            checkpoint=checkpoint,
            newhead_ft_config=logdir / "newhead.json",
            ft_repo=Path(ft_repo).expanduser().resolve() if ft_repo else None,
            mlp_device="cuda" if device.startswith("cuda") else "cpu",
        ),
    )


def _newhead_cfg(
    logdir: Path, checkpoint: str, ft_repo: str | Path | None, device: str
) -> MLIPConfig:
    """Eval config for a single-branch 'new head' fine-tuned model (no branch-MLP)."""
    return MLIPConfig(
        backend="hydragnn",
        hydragnn=HydraGNNConfig(
            logdir=logdir,
            checkpoint=checkpoint,
            newhead_ft_config=logdir / "newhead.json",
            ft_repo=Path(ft_repo).expanduser().resolve() if ft_repo else None,
            mlp_device="cuda" if device.startswith("cuda") else "cpu",
        ),
    )


def _newhead_cfg(
    logdir: Path, checkpoint: str, ft_repo: str | Path | None, device: str
) -> MLIPConfig:
    """Eval config for a single-branch 'new head' fine-tuned model (no branch-MLP)."""
    return MLIPConfig(
        backend="hydragnn",
        hydragnn=HydraGNNConfig(
            logdir=logdir,
            checkpoint=checkpoint,
            newhead_ft_config=logdir / "newhead.json",
            ft_repo=Path(ft_repo).expanduser().resolve() if ft_repo else None,
            mlp_device="cuda" if device.startswith("cuda") else "cpu",
        ),
    )


def _hydragnn_newhead_cfg(
    logdir: Path, checkpoint: str, device: str, ft_repo: Path | None = None
) -> MLIPConfig:
    """Eval config for a single-head (drop-all-heads) fine-tune model."""
    return MLIPConfig(
        backend="hydragnn",
        hydragnn=HydraGNNConfig(
            logdir=logdir,
            checkpoint=checkpoint,
            newhead_ft_config=logdir / "newhead.json",
            ft_repo=ft_repo,
            mlp_device="cuda" if device.startswith("cuda") else "cpu",
        ),
    )


def _uma_cfg(model_name: str, task_name: str, device: str) -> MLIPConfig:
    return MLIPConfig(
        backend="uma",
        uma=UMAConfig(
            model_name=model_name,
            task_name=task_name,  # type: ignore[arg-type]
            device="cuda" if device.startswith("cuda") else "cpu",
            dropout=MCDropoutInjectionConfig(enabled=False),  # deterministic eval
        ),
    )


def _mace_cfg(family: str, model: str, device: str, *, precision: str = "fp64") -> MLIPConfig:
    return MLIPConfig(
        backend="mace",
        mace=MACEConfig(
            family=family,  # type: ignore[arg-type]
            model=model,
            device="cuda" if device.startswith("cuda") else "cpu",
            precision=precision,  # type: ignore[arg-type]
            dropout=MCDropoutInjectionConfig(enabled=False),  # deterministic eval
        ),
    )


# --------------------------------------------------------------------------- #
# Eval helper                                                                 #
# --------------------------------------------------------------------------- #


def _run_eval(
    mlip_cfg: MLIPConfig,
    frames: list[Atoms],
    *,
    iteration: int,
    model_path: str,
    test_label: str,
    eval_dir: Path,
    max_parity_points: int,
    ref_frames: list[Atoms] | None = None,
    out_suffix: str = "",
) -> dict:
    metrics, parity = evaluate_frames(
        mlip_cfg,
        frames,
        iteration=iteration,
        model_path=model_path,
        test_set_label=test_label,
        ref_frames=ref_frames,
    )
    eval_dir.mkdir(parents=True, exist_ok=True)
    out_json = eval_dir / f"iter{iteration}{out_suffix}.json"
    out_json.write_text(json.dumps(asdict(metrics), indent=2))
    np.savez_compressed(
        eval_dir / f"iter{iteration}{out_suffix}_parity.npz",
        **_subsample(parity, max_parity_points),
    )
    log.info(
        "iter%d %s: E_MAE=%.4f (shift %.4f) eV/atom  F_MAE=%.4f eV/A",
        iteration,
        mlip_cfg.backend,
        metrics.energy_mae_eV_per_atom,
        metrics.energy_mae_eV_per_atom_shifted,
        metrics.force_mae_eV_per_A,
    )
    return asdict(metrics)


# --------------------------------------------------------------------------- #
# Campaign                                                                    #
# --------------------------------------------------------------------------- #


def run_campaign(
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    backend: str,
    # UMA
    uma_base_model: str = "uma-s-1p1",
    uma_task_name: str = "omat",
    # UMA fine-tune recipe (custom conservative-head loop; see finetune_uma).
    # Matches allaffa/HydraGNN_GFM_FineTuning4Materials @ uma-sota-comparison:
    # Adam lr=1e-4, force-weighted MSE loss, no weight decay.
    uma_epochs: int = 20,
    uma_lr: float = 1e-4,
    uma_force_weight: float = 10.0,
    uma_weight_decay: float = 0.0,
    uma_freeze_backbone: bool = False,
    uma_lora: bool = False,
    uma_lora_r: int = 8,
    uma_lora_alpha: float = 16.0,
    # MACE (all versions: family in {mace_mp, mace_off, checkpoint}; model in
    # {small, medium, large}/tag/URL/path, or a MACE_MODELS id). Fine-tuning is
    # delegated to mace_run_train (reference recipe), with optional native LoRA.
    mace_family: str = "mace_mp",
    mace_model: str = "medium",
    mace_model_id: str | None = None,
    mace_precision: str = "fp64",
    mace_dispersion: bool = False,
    mace_epochs: int | None = None,
    mace_lr: float | None = None,
    mace_force_weight: float = 10.0,
    mace_weight_decay: float = 0.0,
    mace_freeze_backbone: bool = False,
    mace_lora: bool = False,
    mace_lora_rank: int | None = None,
    mace_lora_alpha: float | None = None,
    # HydraGNN
    gfm_logdir: str | Path | None = None,
    branch_mlp_path: str | Path | None = None,
    gfm_checkpoint: str | None = None,
    weight_threshold: float = 0.1,
    unfreeze_backbone: bool = False,
    hydragnn_strategy: str = "routed",
    hydragnn_head: str | int | None = None,
    ft_repo: str | Path | None = None,
    hydragnn_root: str | Path | None = None,
    # shared FT knobs
    epochs: int = 20,
    lr: float | None = None,
    batch_size: int = 4,
    test_fraction: float = 0.2,
    seed: int = 0,
    device: str | None = None,
    max_parity_points: int = 20000,
    eval_only: bool = False,
) -> Path:
    """Run before/after fine-tune + eval for one backend on one dataset.

    When ``eval_only`` is True, the training step is skipped: the existing
    deterministic train/test split and the already-fine-tuned checkpoint under
    ``ft/`` are reused, and both endpoints are *re-scored* with the per-element
    energy reference fit on the TRAIN partition (leakage-free), writing
    ``eval/iter*_trainref.json``. The original ``iter*.json`` files are left
    untouched.
    """
    # Apply the GPU-visibility policy *before* any torch/HydraGNN import happens
    # downstream, so an explicit CPU request is actually honoured.
    _enforce_device_visibility(device)

    output_dir = Path(output_dir).expanduser().resolve()
    (output_dir / "split").mkdir(parents=True, exist_ok=True)
    eval_dir = output_dir / "eval"
    ft_dir = output_dir / "ft"

    dev = device or _default_device()

    raw = ase_read(str(dataset_path), index=":")
    frames = [raw] if isinstance(raw, Atoms) else list(raw)
    if len(frames) < 4:
        raise ValueError(f"Need >=4 frames for a split; got {len(frames)}.")

    train_frames, test_frames = _split_dataset(frames, test_fraction, seed)
    n_iters = _num_al_iterations(frames)
    train_path = output_dir / "split" / "train.extxyz"
    test_path = output_dir / "split" / "test_set.extxyz"
    if eval_only and train_path.exists() and test_path.exists():
        # Reuse the exact frames scored originally (avoid clobbering the split).
        train_frames = list(ase_read(str(train_path), index=":"))
        test_frames = list(ase_read(str(test_path), index=":"))
    else:
        ase_write(str(train_path), train_frames, format="extxyz")
        ase_write(str(test_path), test_frames, format="extxyz")
    log.info(
        "%s campaign: %d train / %d test frames; N_al_iterations=%d",
        backend,
        len(train_frames),
        len(test_frames),
        n_iters,
    )

    # Leakage-free re-score mode: fit the per-element reference on TRAIN and
    # write to a distinct ``_trainref`` suffix so originals are preserved.
    _ref = train_frames if eval_only else None
    _suf = "_trainref" if eval_only else ""

    if backend == "uma":
        base_cfg = _uma_cfg(uma_base_model, uma_task_name, dev)
        _run_eval(
            base_cfg,
            test_frames,
            iteration=0,
            model_path=uma_base_model,
            test_label=str(test_path),
            eval_dir=eval_dir,
            max_parity_points=max_parity_points,
            ref_frames=_ref,
            out_suffix=_suf,
        )
        if eval_only:
            ckpt = ft_dir / "inference_ckpt.pt"
            if not ckpt.exists():
                raise FileNotFoundError(f"eval-only: missing UMA checkpoint {ckpt}")
        else:
            from matsim_agents.active_learning.finetune_uma import finetune_uma

            ckpt = finetune_uma(
                train_path,
                ft_dir,
                base_model=uma_base_model,
                task_name=uma_task_name,
                epochs=uma_epochs,
                batch_size=batch_size,
                lr=lr if lr is not None else uma_lr,
                force_weight=uma_force_weight,
                freeze_backbone=uma_freeze_backbone,
                weight_decay=uma_weight_decay,
                device="CUDA" if dev.startswith("cuda") else "CPU",
                seed=seed,
                lora=uma_lora,
                lora_r=uma_lora_r,
                lora_alpha=uma_lora_alpha,
            )
        after_cfg = _uma_cfg(str(ckpt), uma_task_name, dev)
        after_model_path = str(ckpt)

    elif backend == "mace":
        # A MACE_MODELS id (if given) selects the family/variant for the
        # baseline eval too, so "before" and "after" use the same foundation.
        if mace_model_id is not None:
            from matsim_agents.active_learning.finetune_mace import MACE_MODELS

            if mace_model_id not in MACE_MODELS:
                raise ValueError(
                    f"mace_model_id must be one of {sorted(MACE_MODELS)}, got {mace_model_id!r}"
                )
            mace_family = MACE_MODELS[mace_model_id]["family"]
            mace_model = MACE_MODELS[mace_model_id]["model"]
        base_cfg = _mace_cfg(mace_family, mace_model, dev, precision=mace_precision)
        _run_eval(
            base_cfg,
            test_frames,
            iteration=0,
            model_path=f"{mace_family}:{mace_model}",
            test_label=str(test_path),
            eval_dir=eval_dir,
            max_parity_points=max_parity_points,
            ref_frames=_ref,
            out_suffix=_suf,
        )
        if eval_only:
            ckpt = ft_dir / "mace_finetuned.model"
            if not ckpt.exists():
                raise FileNotFoundError(f"eval-only: missing MACE checkpoint {ckpt}")
        else:
            from matsim_agents.active_learning.finetune_mace import finetune_mace

            ckpt = finetune_mace(
                train_path,
                ft_dir,
                family=mace_family,
                base_model=mace_model,
                model_id=mace_model_id,
                precision=mace_precision,
                dispersion=mace_dispersion,
                epochs=mace_epochs,
                batch_size=batch_size,
                lr=lr if lr is not None else mace_lr,
                lora=mace_lora,
                lora_rank=mace_lora_rank,
                lora_alpha=mace_lora_alpha,
                force_weight=mace_force_weight,
                freeze_backbone=mace_freeze_backbone,
                weight_decay=mace_weight_decay,
                device="CUDA" if dev.startswith("cuda") else "CPU",
                seed=seed,
            )
        # Fine-tuned checkpoint reloads via family="checkpoint".
        after_cfg = _mace_cfg("checkpoint", str(ckpt), dev, precision=mace_precision)
        after_model_path = str(ckpt)

    elif backend == "hydragnn":
        if gfm_logdir is None or branch_mlp_path is None:
            raise ValueError("hydragnn backend requires gfm_logdir and branch_mlp_path.")
        gfm_logdir = Path(gfm_logdir).expanduser().resolve()
        branch_mlp_path = Path(branch_mlp_path).expanduser().resolve()
        base_ckpt = gfm_checkpoint or _newest_pk(gfm_logdir).name
        base_cfg = _hydragnn_cfg(gfm_logdir, base_ckpt, branch_mlp_path, dev)
        _run_eval(
            base_cfg,
            test_frames,
            iteration=0,
            model_path=str(gfm_logdir),
            test_label=str(test_path),
            eval_dir=eval_dir,
            max_parity_points=max_parity_points,
            ref_frames=_ref,
            out_suffix=_suf,
        )
        from matsim_agents.active_learning.finetune_hydragnn import finetune_hydragnn

        if hydragnn_strategy == "routed":
            if eval_only:
                ft_logdir = ft_dir
                if not (ft_logdir / "ft_model.pk").exists():
                    raise FileNotFoundError(
                        f"eval-only: missing HydraGNN checkpoint {ft_logdir / 'ft_model.pk'}"
                    )
            else:
                ft_logdir = finetune_hydragnn(
                    train_path,
                    ft_dir,
                    gfm_logdir=gfm_logdir,
                    branch_mlp_path=branch_mlp_path,
                    gfm_checkpoint=base_ckpt,
                    weight_threshold=weight_threshold,
                    epochs=epochs,
                    # 1e-4 (not 1e-3): a full-backbone step at 1e-3 collapses the
                    # net into a constant predictor; force_weight stays ~94 (config).
                    lr=lr if lr is not None else 1e-4,
                    batch_size=batch_size,
                    seed=seed,
                    unfreeze_backbone=unfreeze_backbone,
                    hydragnn_root=hydragnn_root,
                    device=dev,
                )
            after_cfg = _hydragnn_cfg(ft_logdir, "ft_model.pk", branch_mlp_path, dev)
        else:
            from matsim_agents.active_learning.finetune_hydragnn_newhead import (
                finetune_hydragnn_newhead,
            )

            ft_repo_p = Path(ft_repo).expanduser().resolve() if ft_repo else None
            if eval_only:
                ft_logdir = ft_dir
                if not (ft_logdir / "ft_model.pk").exists():
                    raise FileNotFoundError(
                        f"eval-only: missing HydraGNN checkpoint {ft_logdir / 'ft_model.pk'}"
                    )
            else:
                ft_logdir = finetune_hydragnn_newhead(
                    train_path,
                    ft_dir,
                    gfm_logdir=gfm_logdir,
                    strategy=hydragnn_strategy,
                    head=hydragnn_head,
                    gfm_checkpoint=base_ckpt,
                    epochs=epochs,
                    # The new-head loss is normalised (forces/energy scaled to O(1))
                    # with LR warmup+cosine and gradient clipping, so a 1e-3 step no
                    # longer collapses the head into a constant predictor; this is
                    # what actually moves the metrics off the zero-shot baseline.
                    lr=lr if lr is not None else 1e-3,
                    batch_size=batch_size,
                    seed=seed,
                    hydragnn_root=hydragnn_root,
                    ft_repo=ft_repo_p,
                    device=dev,
                )
            after_cfg = _hydragnn_newhead_cfg(ft_logdir, "ft_model.pk", dev, ft_repo=ft_repo_p)
        after_model_path = str(ft_logdir)

    else:
        raise ValueError(f"Unknown backend {backend!r} (expected 'hydragnn', 'uma', or 'mace').")

    _run_eval(
        after_cfg,
        test_frames,
        iteration=n_iters,
        model_path=after_model_path,
        test_label=str(test_path),
        eval_dir=eval_dir,
        max_parity_points=max_parity_points,
        ref_frames=_ref,
        out_suffix=_suf,
    )
    log.info("Campaign complete -> %s (eval: iter0 vs iter%d)", output_dir, n_iters)
    return output_dir


def _default_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"


def _enforce_device_visibility(device: str | None) -> None:
    """Apply the same GPU-visibility policy the Slurm job scripts rely on.

    On compute nodes, Slurm binds one GPU per task (``--gpu-bind=closest``) and
    the code just calls ``torch.cuda.is_available()``. When we explicitly ask
    for CPU (e.g. login-node validation), neither ``build_hydragnn_calculator``
    nor HydraGNN's internal ``get_device()`` honour that request -- they grab a
    (contended) GPU regardless. Hiding all devices via ``CUDA_VISIBLE_DEVICES``
    is the standard visibility lever (the same one Slurm uses), so set it here
    *before* torch is imported. CUDA requests are left to Slurm/torch untouched.
    """
    if device is None:
        return
    if str(device).strip().lower().startswith("cpu"):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""


def _newest_pk(logdir: Path) -> Path:
    cands = sorted(logdir.glob("*.pk"), key=lambda p: p.stat().st_mtime)
    if not cands:
        raise FileNotFoundError(f"No .pk checkpoint in {logdir}")
    return cands[-1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--backend", required=True, choices=["hydragnn", "uma", "mace"])
    parser.add_argument("--dataset", required=True, help="AL-collected extxyz dataset.")
    parser.add_argument("--output-dir", required=True, help="Campaign output directory.")
    # UMA
    parser.add_argument("--uma-base-model", default="uma-s-1p1")
    parser.add_argument(
        "--uma-task-name", default="omat", choices=["omat", "omol", "oc20", "odac", "omc"]
    )
    parser.add_argument("--uma-epochs", type=int, default=20, help="UMA fine-tune epochs.")
    parser.add_argument(
        "--uma-lr",
        type=float,
        default=1e-4,
        help="UMA fine-tune Adam LR (reference recipe: 1e-4).",
    )
    parser.add_argument(
        "--uma-force-weight",
        type=float,
        default=10.0,
        help="Weight on the force-MSE term relative to energy-MSE.",
    )
    parser.add_argument("--uma-freeze-backbone", action="store_true")
    parser.add_argument("--uma-weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--uma-lora",
        action="store_true",
        help="LoRA fine-tune of UMA backbone scalar linears (preserves equivariance).",
    )
    parser.add_argument("--uma-lora-r", type=int, default=8, help="UMA LoRA rank (default: 8).")
    parser.add_argument(
        "--uma-lora-alpha", type=float, default=16.0, help="UMA LoRA alpha (default: 16.0)."
    )
    # MACE (all versions)
    parser.add_argument(
        "--mace-model-id",
        default=None,
        help=(
            "Curated MACE variant id (see finetune_mace.MACE_MODELS); "
            "overrides --mace-family/--mace-model."
        ),
    )
    parser.add_argument(
        "--mace-family",
        default="mace_mp",
        choices=["mace_mp", "mace_off", "checkpoint"],
        help="MACE family: mace_mp (inorganic), mace_off (organic), or checkpoint (local .model).",
    )
    parser.add_argument(
        "--mace-model",
        default="medium",
        help=(
            "MACE size/variant (small|medium|large, or tag/URL), "
            "or a .model path for family=checkpoint."
        ),
    )
    parser.add_argument("--mace-precision", default="fp64", choices=["fp32", "fp64"])
    parser.add_argument("--mace-dispersion", action="store_true")
    parser.add_argument(
        "--mace-epochs",
        type=int,
        default=None,
        help="MACE fine-tune epochs (default: 10 for LoRA, 50 for naive).",
    )
    parser.add_argument(
        "--mace-lr",
        type=float,
        default=None,
        help="MACE fine-tune LR (default: 5e-3 for LoRA, 1e-3 for naive).",
    )
    parser.add_argument(
        "--mace-force-weight",
        type=float,
        default=10.0,
        help="Recorded for parity; mace_run_train uses its own energy/force weighting.",
    )
    parser.add_argument("--mace-freeze-backbone", action="store_true")
    parser.add_argument("--mace-weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--mace-lora",
        action="store_true",
        help="Parameter-efficient LoRA fine-tune (native mace_run_train --lora).",
    )
    parser.add_argument(
        "--mace-lora-rank", type=int, default=None, help="MACE LoRA rank (default: 4)."
    )
    parser.add_argument(
        "--mace-lora-alpha", type=float, default=None, help="MACE LoRA alpha (default: 1.0)."
    )
    # HydraGNN
    parser.add_argument(
        "--gfm-logdir", default=None, help="HydraGNN GFM logdir (config.json + .pk)."
    )
    parser.add_argument("--branch-mlp", default=None, help="mlp_branch_weights.pt path.")
    parser.add_argument(
        "--gfm-checkpoint", default=None, help="GFM .pk filename (default: newest)."
    )
    parser.add_argument("--weight-threshold", type=float, default=0.1)
    parser.add_argument("--unfreeze-backbone", action="store_true")
    parser.add_argument(
        "--hydragnn-strategy",
        default="routed",
        choices=["routed", "unfrozen", "frozen", "scratch"],
        help="HydraGNN fine-tune method: 'routed' (adapt routed heads) or a "
        "drop-all-heads + new-head strategy (unfrozen/frozen/scratch).",
    )
    parser.add_argument(
        "--hydragnn-head",
        default=None,
        help="Fine-tune a PRETRAINED HydraGNN head instead of a fresh random one: "
        "a branch index 0..15 or dataset name (e.g. MPTrj). Requires "
        "--hydragnn-strategy unfrozen|frozen.",
    )
    parser.add_argument(
        "--ft-repo", default=None, help="ORNL HydraGNN_GFM_FineTuning4Materials path."
    )
    parser.add_argument("--hydragnn-root", default=None)
    # shared
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-parity-points", type=int, default=20000)
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training: reuse the existing split + fine-tuned checkpoint and "
        "re-score both endpoints with the per-element reference fit on the TRAIN "
        "partition (leakage-free), writing eval/iter*_trainref.json.",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    # Set GPU visibility from the requested device before torch is imported.
    _enforce_device_visibility(args.device)

    run_campaign(
        args.dataset,
        args.output_dir,
        backend=args.backend,
        uma_base_model=args.uma_base_model,
        uma_task_name=args.uma_task_name,
        uma_epochs=args.uma_epochs,
        uma_lr=args.uma_lr,
        uma_force_weight=args.uma_force_weight,
        uma_freeze_backbone=args.uma_freeze_backbone,
        uma_weight_decay=args.uma_weight_decay,
        uma_lora=args.uma_lora,
        uma_lora_r=args.uma_lora_r,
        uma_lora_alpha=args.uma_lora_alpha,
        mace_family=args.mace_family,
        mace_model=args.mace_model,
        mace_model_id=args.mace_model_id,
        mace_precision=args.mace_precision,
        mace_dispersion=args.mace_dispersion,
        mace_epochs=args.mace_epochs,
        mace_lr=args.mace_lr,
        mace_force_weight=args.mace_force_weight,
        mace_freeze_backbone=args.mace_freeze_backbone,
        mace_weight_decay=args.mace_weight_decay,
        mace_lora=args.mace_lora,
        mace_lora_rank=args.mace_lora_rank,
        mace_lora_alpha=args.mace_lora_alpha,
        gfm_logdir=args.gfm_logdir,
        branch_mlp_path=args.branch_mlp,
        gfm_checkpoint=args.gfm_checkpoint,
        weight_threshold=args.weight_threshold,
        unfreeze_backbone=args.unfreeze_backbone,
        hydragnn_strategy=args.hydragnn_strategy,
        hydragnn_head=args.hydragnn_head,
        ft_repo=args.ft_repo,
        hydragnn_root=args.hydragnn_root,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        test_fraction=args.test_fraction,
        seed=args.seed,
        device=args.device,
        max_parity_points=args.max_parity_points,
        eval_only=args.eval_only,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
