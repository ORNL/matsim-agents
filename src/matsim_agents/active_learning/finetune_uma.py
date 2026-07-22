"""Fine-tune a UMA (fairchem) foundation model on an AL-collected DFT dataset.

This turns an extended-XYZ dataset (as written by
:func:`matsim_agents.active_learning.trainer.append_frames_to_extxyz`) into a
fine-tuned ``inference_ckpt.pt`` that
:func:`matsim_agents.active_learning.calculator.build_uma_calculator` can load
via ``load_predict_unit``.

Approach (matches allaffa/HydraGNN_GFM_FineTuning4Materials @ ``uma-sota-comparison``)
------------------------------------------------------------------------------------
UMA exposes a *conservative* energy-and-force head: forces are obtained as the
autograd gradient of the predicted energy w.r.t. atomic positions *inside* the
forward pass. We therefore fine-tune with a small custom PyTorch loop that runs
the model directly, rather than fairchem's ``create_yaml`` / normalizer /
element-reference config pipeline (which trains the *direct*-force head and
re-fits an output normalizer -- rescaling the heads and inflating forces on the
small AL datasets).

Concretely:

* ``pu = load_predict_unit(ckpt, device, inference_settings="default")`` -- the
  ``"default"`` settings avoid ``torch.compile`` so the autograd graph is clean
  for the double-backprop the conservative head needs.
* ``model = pu.model.module`` is the trainable ``HydraModel``.
* ``model.train()`` is *required* so the force head sets ``create_graph=True``;
  otherwise the energy graph is freed by the internal force autograd and the
  outer ``loss.backward()`` fails.
* Loss per sample is ``E_MSE + force_weight * F_MSE`` (``force_weight=10`` by
  default), optimised with Adam (``lr=1e-4``). No normalizer / element-reference
  re-fitting is performed -- the model's native heads are used as-is.

The fine-tuned weights are written back into a
:class:`~fairchem.core.units.mlip_unit.api.inference.MLIPInferenceCheckpoint`
(reusing the base checkpoint's ``model_config`` / ``tasks_config``), so the
resulting ``.pt`` reloads through the ordinary ``load_predict_unit`` path.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read as ase_read

from matsim_agents.active_learning.cost import CostReport, GpuMemorySampler, track_cost

log = logging.getLogger(__name__)

# UMA task heads: omat (inorganic bulk), omol (molecules/MOFs), oc20, odac, omc.
_UMA_TASKS = {"omat", "omol", "oc20", "odac", "omc"}


# --------------------------------------------------------------------------- #
# Dataset helpers                                                             #
# --------------------------------------------------------------------------- #


def _reference_energy(atoms: Atoms) -> float | None:
    """Ground-truth (DFT) total energy from ``info`` or an attached calculator."""
    for key in ("REF_energy", "energy"):
        if key in atoms.info:
            return float(atoms.info[key])
    try:
        return float(atoms.get_potential_energy())
    except Exception:  # noqa: BLE001
        return None


def _reference_forces(atoms: Atoms) -> np.ndarray | None:
    """Ground-truth (DFT) forces from ``arrays`` or an attached calculator."""
    for key in ("REF_forces", "forces"):
        if key in atoms.arrays:
            return np.asarray(atoms.arrays[key], dtype=float)
    try:
        return np.asarray(atoms.get_forces(), dtype=float)
    except Exception:  # noqa: BLE001
        return None


def _load_labeled_frames(dataset_path: str | Path) -> list[Atoms]:
    """Read frames that carry both an energy and a forces label."""
    raw = ase_read(str(dataset_path), index=":")
    frames = [raw] if isinstance(raw, Atoms) else list(raw)
    usable: list[Atoms] = []
    for a in frames:
        if _reference_energy(a) is not None and _reference_forces(a) is not None:
            usable.append(a)
    if len(usable) < 2:
        raise ValueError(
            f"Only {len(usable)}/{len(frames)} frames in {dataset_path} carry both "
            "energy and forces labels; need >=2 to fine-tune."
        )
    return usable


# --------------------------------------------------------------------------- #
# UMA model loading + differentiable forward pass                             #
# --------------------------------------------------------------------------- #


def _resolve_base_checkpoint(base_model: str) -> str:
    """Resolve a UMA model *name* (or local path) to a checkpoint file path."""
    import os

    if os.path.isfile(base_model):
        return base_model
    from fairchem.core.calculate.pretrained_mlip import (
        pretrained_checkpoint_path_from_name,
    )

    return str(pretrained_checkpoint_path_from_name(base_model))


def load_trainable_uma(base_model: str, task_name: str, device: str):
    """Load a UMA checkpoint exposing a trainable ``HydraModel``.

    Returns ``(predict_unit, calculator, model)`` where ``model`` is
    ``predict_unit.model.module`` -- the module whose parameters are updated
    in-place by the training loop and read back at inference time.
    """
    from fairchem.core import FAIRChemCalculator, pretrained_mlip

    ckpt = _resolve_base_checkpoint(base_model)
    # 'default' inference settings avoid torch.compile so autograd graphs are
    # clean for the double-backprop of the conservative force head.
    pu = pretrained_mlip.load_predict_unit(
        ckpt, device=device, inference_settings="default"
    )
    calc = FAIRChemCalculator(pu, task_name=task_name)
    model = pu.model.module  # trainable HydraModel
    return pu, calc, model


def uma_energy_forces(model, calc, atoms: Atoms, task_name: str):
    """Differentiable forward pass returning ``(energy[1], forces[N, 3])`` tensors.

    ``atoms`` must have ``info["charge"]`` / ``info["spin"]`` for the ``omol``
    head (defaults to 0 / 1 if missing).
    """
    import torch

    atoms.info.setdefault("charge", 0)
    atoms.info.setdefault("spin", 1)

    param = next(model.parameters())
    data = calc.a2g(atoms).to(param.device)
    # Match the model's working dtype on all floating tensors.
    for key, val in data:
        if torch.is_tensor(val) and val.is_floating_point():
            data[key] = val.to(param.dtype)

    out = model(data)
    energy = out[f"{task_name}_energy"]["energy"]
    forces = out[f"{task_name}_forces"]["forces"]
    return energy, forces


# --------------------------------------------------------------------------- #
# Training loop                                                               #
# --------------------------------------------------------------------------- #


def _train_uma(
    model,
    calc,
    train_atoms: list[Atoms],
    *,
    task_name: str,
    epochs: int,
    lr: float,
    batch_size: int,
    force_weight: float,
    freeze_backbone: bool,
    weight_decay: float,
    seed: int,
) -> list[float]:
    """Custom PyTorch fine-tune loop on UMA's conservative energy/force head.

    Loss per sample is ``E_MSE + force_weight * F_MSE``. Gradients are
    accumulated over ``batch_size`` samples before each optimiser step.
    Returns the mean per-sample loss for each epoch.
    """
    import torch

    param = next(model.parameters())
    device, dtype = param.device, param.dtype

    if freeze_backbone and hasattr(model, "backbone"):
        for p in model.backbone.parameters():
            p.requires_grad_(False)
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("No trainable parameters after freezing; check freeze_backbone.")

    opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    # model.train() -> conservative force head uses create_graph=True (double backprop).
    model.train()

    # Precompute targets once.
    targets: list[tuple[torch.Tensor, torch.Tensor]] = []
    for atoms in train_atoms:
        e = _reference_energy(atoms)
        f = _reference_forces(atoms)
        targets.append(
            (
                torch.as_tensor(e, dtype=dtype, device=device),
                torch.as_tensor(f, dtype=dtype, device=device),
            )
        )

    rng = np.random.default_rng(seed)
    n = len(train_atoms)
    history: list[float] = []

    for epoch in range(epochs):
        order = rng.permutation(n)
        opt.zero_grad()
        epoch_loss = 0.0
        pending = 0
        for idx in order:
            atoms = train_atoms[int(idx)]
            e_true, f_true = targets[int(idx)]
            e_pred, f_pred = uma_energy_forces(model, calc, atoms, task_name)
            e_loss = (e_pred.squeeze() - e_true) ** 2
            f_loss = (f_pred - f_true).pow(2).mean()
            loss = e_loss + force_weight * f_loss
            (loss / batch_size).backward()
            epoch_loss += float(loss.detach())
            pending += 1
            if pending == batch_size:
                opt.step()
                opt.zero_grad()
                pending = 0
        if pending > 0:  # flush the final partial batch
            opt.step()
            opt.zero_grad()
        mean_loss = epoch_loss / n
        history.append(mean_loss)
        log.info("UMA fine-tune epoch %d/%d: mean loss=%.6f", epoch + 1, epochs, mean_loss)

    model.eval()
    return history


# --------------------------------------------------------------------------- #
# Checkpoint persistence                                                      #
# --------------------------------------------------------------------------- #


def _save_finetuned_checkpoint(pu, base_model: str, out_ckpt: Path) -> None:
    """Write the fine-tuned weights into a reloadable ``MLIPInferenceCheckpoint``.

    ``load_predict_unit`` reconstructs the model as an ``AveragedModel`` and
    loads ``ema_state_dict`` for inference, so we persist the trained
    ``AveragedModel`` state dict (``module.*`` + ``n_averaged``) as the EMA
    weights (and the bare module state dict as the non-EMA weights), reusing the
    base checkpoint's ``model_config`` / ``tasks_config``.
    """
    import torch

    base_ckpt_path = _resolve_base_checkpoint(base_model)
    ckpt = torch.load(base_ckpt_path, map_location="cpu", weights_only=False)

    ema_sd = {k: v.detach().cpu().clone() for k, v in pu.model.state_dict().items()}
    model_sd = {k: v.detach().cpu().clone() for k, v in pu.model.module.state_dict().items()}
    ckpt.ema_state_dict = ema_sd
    ckpt.model_state_dict = model_sd

    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out_ckpt)
    log.info("Saved fine-tuned UMA checkpoint -> %s", out_ckpt)


# --------------------------------------------------------------------------- #
# Public entry point                                                          #
# --------------------------------------------------------------------------- #


def finetune_uma(
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    base_model: str = "uma-s-1p1",
    task_name: str = "omat",
    epochs: int = 20,
    batch_size: int = 4,
    lr: float = 1e-4,
    force_weight: float = 10.0,
    freeze_backbone: bool = False,
    weight_decay: float = 0.0,
    seed: int = 0,
    device: str | None = None,
    run: bool = True,
    # Accepted for backward compatibility with the previous fairchem-config
    # based implementation; ignored by the custom training loop.
    ranks_per_node: int = 1,
    **_legacy,
) -> Path:
    """Fine-tune ``base_model`` on ``dataset_path`` and return the checkpoint path.

    The returned ``inference_ckpt.pt`` reloads through
    :func:`fairchem.core.units.mlip_unit.load_predict_unit` (and hence
    :func:`matsim_agents.active_learning.calculator.build_uma_calculator`).

    When ``run`` is False the model and dataset are loaded/validated but no
    training or checkpoint write occurs (dry-run); the intended checkpoint path
    is still returned.
    """
    import torch

    if task_name not in _UMA_TASKS:
        raise ValueError(f"task_name must be one of {sorted(_UMA_TASKS)}, got {task_name!r}")

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dest_ckpt = output_dir / "inference_ckpt.pt"

    # Normalise device string ("CUDA"/"CPU" from callers -> torch form).
    dev = (device or ("cuda" if torch.cuda.is_available() else "cpu")).lower()
    if dev.startswith("cuda") and not torch.cuda.is_available():
        log.warning("CUDA requested but unavailable; falling back to CPU.")
        dev = "cpu"

    train_atoms = _load_labeled_frames(dataset_path)
    log.info(
        "UMA fine-tune: %d train frames (task=%s, base=%s, device=%s)",
        len(train_atoms),
        task_name,
        base_model,
        dev,
    )

    pu, calc, model = load_trainable_uma(base_model, task_name, dev)

    if not run:
        log.info("run=False; loaded model + %d frames (training skipped)", len(train_atoms))
        return dest_ckpt

    report = CostReport(
        model_backend="uma",
        dataset_label=Path(dataset_path).parent.name or Path(dataset_path).stem,
        base_model=base_model,
        dataset_path=str(Path(dataset_path).resolve()),
        n_train_frames=len(train_atoms),
        n_val_frames=0,
        epochs=int(epochs),
        num_gpus=1 if dev.startswith("cuda") else 0,
        device=dev.upper(),
        extra={
            "task_name": task_name,
            "protocol": "conservative-head-finetune",
            "lr": float(lr),
            "force_weight": float(force_weight),
            "batch_size": int(batch_size),
            "freeze_backbone": bool(freeze_backbone),
            "weight_decay": float(weight_decay),
        },
    )
    sampler = GpuMemorySampler(enabled=dev.startswith("cuda"))
    with sampler, track_cost(report):
        history = _train_uma(
            model,
            calc,
            train_atoms,
            task_name=task_name,
            epochs=int(epochs),
            lr=float(lr),
            batch_size=int(batch_size),
            force_weight=float(force_weight),
            freeze_backbone=bool(freeze_backbone),
            weight_decay=float(weight_decay),
            seed=int(seed),
        )
    if sampler.peak_gb:
        report.peak_gpu_mem_gb = round(sampler.peak_gb, 3)
    if history:
        report.extra["final_train_loss"] = round(history[-1], 6)
    report.write(output_dir / "cost.json")

    _save_finetuned_checkpoint(pu, base_model, dest_ckpt)
    log.info(
        "UMA fine-tune complete -> %s (%.1fs, %.4f GPU-h)",
        dest_ckpt,
        report.wall_time_s,
        report.gpu_hours,
    )
    return dest_ckpt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", required=True, help="AL-collected extxyz dataset.")
    parser.add_argument("--output-dir", required=True, help="Work/output directory.")
    parser.add_argument("--base-model", default="uma-s-1p1", help="Base UMA model name or path.")
    parser.add_argument(
        "--task-name",
        default="omat",
        choices=sorted(_UMA_TASKS),
        help="UMA task (omat: inorganic bulk, omol: molecules/MOFs, ...).",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of fine-tune epochs.")
    parser.add_argument("--batch-size", type=int, default=4, help="Gradient-accumulation batch size.")
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Adam learning rate (reference UMA fine-tune recipe: 1e-4).",
    )
    parser.add_argument(
        "--force-weight",
        type=float,
        default=10.0,
        help="Weight on the force MSE term relative to the energy MSE term.",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze the backbone and train only the output head.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="Adam weight decay (default 0, matching the reference recipe).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None, choices=["CUDA", "CPU", "cuda", "cpu"])
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load model + dataset but do not train / write a checkpoint.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    ckpt = finetune_uma(
        args.dataset,
        args.output_dir,
        base_model=args.base_model,
        task_name=args.task_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        force_weight=args.force_weight,
        freeze_backbone=args.freeze_backbone,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
        run=not args.dry_run,
    )
    print(ckpt)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
