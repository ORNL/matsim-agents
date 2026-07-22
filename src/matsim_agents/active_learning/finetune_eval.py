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


def _hydragnn_cfg(
    logdir: Path, checkpoint: str, branch_mlp: Path, device: str
) -> MLIPConfig:
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
) -> dict:
    metrics, parity = evaluate_frames(
        mlip_cfg,
        frames,
        iteration=iteration,
        model_path=model_path,
        test_set_label=test_label,
    )
    eval_dir.mkdir(parents=True, exist_ok=True)
    out_json = eval_dir / f"iter{iteration}.json"
    out_json.write_text(json.dumps(asdict(metrics), indent=2))
    np.savez_compressed(
        eval_dir / f"iter{iteration}_parity.npz", **_subsample(parity, max_parity_points)
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
    # HydraGNN
    gfm_logdir: str | Path | None = None,
    branch_mlp_path: str | Path | None = None,
    gfm_checkpoint: str | None = None,
    weight_threshold: float = 0.1,
    unfreeze_backbone: bool = False,
    hydragnn_strategy: str = "routed",
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
) -> Path:
    """Run before/after fine-tune + eval for one backend on one dataset."""
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
    ase_write(str(train_path), train_frames, format="extxyz")
    ase_write(str(test_path), test_frames, format="extxyz")
    log.info(
        "%s campaign: %d train / %d test frames; N_al_iterations=%d",
        backend,
        len(train_frames),
        len(test_frames),
        n_iters,
    )

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
        )
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
        )
        after_cfg = _uma_cfg(str(ckpt), uma_task_name, dev)
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
        )
        from matsim_agents.active_learning.finetune_hydragnn import finetune_hydragnn

        if hydragnn_strategy == "routed":
            ft_logdir = finetune_hydragnn(
                train_path,
                ft_dir,
                gfm_logdir=gfm_logdir,
                branch_mlp_path=branch_mlp_path,
                gfm_checkpoint=base_ckpt,
                weight_threshold=weight_threshold,
                epochs=epochs,
                lr=lr if lr is not None else 1e-3,
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
            ft_logdir = finetune_hydragnn_newhead(
                train_path,
                ft_dir,
                gfm_logdir=gfm_logdir,
                strategy=hydragnn_strategy,
                gfm_checkpoint=base_ckpt,
                epochs=epochs,
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
        raise ValueError(f"Unknown backend {backend!r} (expected 'hydragnn' or 'uma').")

    _run_eval(
        after_cfg,
        test_frames,
        iteration=n_iters,
        model_path=after_model_path,
        test_label=str(test_path),
        eval_dir=eval_dir,
        max_parity_points=max_parity_points,
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
    parser.add_argument("--backend", required=True, choices=["hydragnn", "uma"])
    parser.add_argument("--dataset", required=True, help="AL-collected extxyz dataset.")
    parser.add_argument("--output-dir", required=True, help="Campaign output directory.")
    # UMA
    parser.add_argument("--uma-base-model", default="uma-s-1p1")
    parser.add_argument(
        "--uma-task-name", default="omat", choices=["omat", "omol", "oc20", "odac", "omc"]
    )
    parser.add_argument(
        "--uma-epochs", type=int, default=20, help="UMA fine-tune epochs."
    )
    parser.add_argument(
        "--uma-lr", type=float, default=1e-4,
        help="UMA fine-tune Adam LR (reference recipe: 1e-4).",
    )
    parser.add_argument(
        "--uma-force-weight", type=float, default=10.0,
        help="Weight on the force-MSE term relative to energy-MSE.",
    )
    parser.add_argument("--uma-freeze-backbone", action="store_true")
    parser.add_argument("--uma-weight-decay", type=float, default=0.0)
    # HydraGNN
    parser.add_argument("--gfm-logdir", default=None, help="HydraGNN GFM logdir (config.json + .pk).")
    parser.add_argument("--branch-mlp", default=None, help="mlp_branch_weights.pt path.")
    parser.add_argument("--gfm-checkpoint", default=None, help="GFM .pk filename (default: newest).")
    parser.add_argument("--weight-threshold", type=float, default=0.1)
    parser.add_argument("--unfreeze-backbone", action="store_true")
    parser.add_argument(
        "--hydragnn-strategy",
        default="routed",
        choices=["routed", "unfrozen", "frozen", "scratch"],
        help="HydraGNN fine-tune method: 'routed' (adapt routed heads) or a "
        "drop-all-heads + new-head strategy (unfrozen/frozen/scratch).",
    )
    parser.add_argument("--ft-repo", default=None, help="ORNL HydraGNN_GFM_FineTuning4Materials path.")
    parser.add_argument("--hydragnn-root", default=None)
    # shared
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-parity-points", type=int, default=20000)
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
        gfm_logdir=args.gfm_logdir,
        branch_mlp_path=args.branch_mlp,
        gfm_checkpoint=args.gfm_checkpoint,
        weight_threshold=args.weight_threshold,
        unfreeze_backbone=args.unfreeze_backbone,
        hydragnn_strategy=args.hydragnn_strategy,
        ft_repo=args.ft_repo,
        hydragnn_root=args.hydragnn_root,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        test_fraction=args.test_fraction,
        seed=args.seed,
        device=args.device,
        max_parity_points=args.max_parity_points,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
