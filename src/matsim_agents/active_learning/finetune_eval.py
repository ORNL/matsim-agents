"""Fine-tune and evaluate one MLIP backend on one dataset, with explicit data roles.

The dataset is split by a declared strategy (:mod:`matsim_agents.datasets.splits`)
into ``train``, optional ``validation``, and ``test`` partitions, written once
with a content-addressed manifest that every backend reuses. The run then:

1. fine-tunes the foundation model on ``train`` (``validation`` serves early
   stopping when present);
2. evaluates the foundation ("baseline") and the fine-tuned model on
   ``validation`` and selects one with a declared rule (``SelectionRule``), or
   reports the fine-tuned model without selection when there is no validation
   partition;
3. writes and freezes that decision in ``selection.json``;
4. only then evaluates both models on ``test`` and on every named test domain.

Outputs under ``output_dir`` (or one directory per repetition):

* ``split/`` -- partitions and ``split-manifest.json`` (or the reused split);
* ``ft/`` -- the fine-tuned model and its ``cost.json``;
* ``eval/iter0.json`` / ``eval/iter<N>.json`` (+ ``_parity.npz``) -- baseline and
  fine-tuned on ``test``, ``N`` being the number of AL iterations in the dataset;
* ``eval/validation/`` and ``eval/domains/<name>/`` -- the other evaluations;
* ``eval/report.json`` -- metric definitions, split summary and limitations, the
  selection, per-group summaries with group-bootstrap intervals, and paired
  fine-tuned minus baseline differences;
* ``repetitions.json`` -- spread over split and training seeds, when repeated.

Backend/venv coupling (IMPORTANT): HydraGNN needs the *hydragnn* venv and UMA
needs the *fairchem* venv, so this runner handles ONE backend per invocation.
The Perlmutter launcher submits one job per (backend, dataset) with the correct
interpreter/venv; later backends reuse the first run's split (``split_manifest``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from ase import Atoms
from ase.io import read as ase_read
from pydantic import BaseModel, Field, model_validator

from matsim_agents.active_learning.eval_report import (
    METRIC_DEFINITIONS,
    group_report,
    paired_difference,
    summarize_repetitions,
)
from matsim_agents.active_learning.evaluate import (
    _subsample,
    assess_promotion,
    evaluate_frames,
)
from matsim_agents.backends.mlip.config import (
    HydraGNNConfig,
    MACEConfig,
    MCDropoutInjectionConfig,
    MLIPConfig,
    UMAConfig,
)
from matsim_agents.datasets.splits import (
    SplitConfig,
    SplitManifest,
    TestDomainConfig,
    load_split,
    make_split,
)

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Dataset split                                                               #
# --------------------------------------------------------------------------- #


def _num_al_iterations(frames: list[Atoms]) -> int:
    iters = {int(a.info["al_iteration"]) for a in frames if "al_iteration" in a.info}
    return len(iters) if iters else 1


class SelectionRule(BaseModel):
    """When the fine-tuned model is selected over the foundation model, on validation.

    The fine-tuned model must meet the absolute limits, not be worse in either
    formation-energy or force MAE, and improve one of them by
    ``min_relative_improvement``; otherwise the foundation model is selected.
    """

    min_evaluated_frames: int = Field(1, ge=1)
    max_formation_energy_mae_eV_per_atom: float = Field(float("inf"), gt=0)
    max_force_mae_eV_per_A: float = Field(float("inf"), gt=0)
    min_relative_improvement: float = Field(0.0, ge=0, le=1)

    def as_promotion_rule(self) -> SimpleNamespace:
        return SimpleNamespace(
            promotion_min_evaluated_frames=self.min_evaluated_frames,
            promotion_max_energy_mae_eV_per_atom=self.max_formation_energy_mae_eV_per_atom,
            promotion_max_force_mae_eV_per_A=self.max_force_mae_eV_per_A,
            promotion_min_relative_improvement=self.min_relative_improvement,
        )


class Repetitions(BaseModel):
    split_seeds: list[int] = Field(
        default_factory=list, description="Splits to repeat with; empty uses split.seed."
    )
    training_seeds: list[int] = Field(
        default_factory=list, description="Fine-tuning seeds per split; empty uses one."
    )


class EvaluationConfig(BaseModel):
    """What the evaluation tests, and how the data are divided for it."""

    objective: str = Field(min_length=1, description="The scientific question being tested.")
    split: SplitConfig | None = None
    split_manifest: Path | None = Field(
        None, description="Reuse an existing split (e.g. made for another backend)."
    )
    test_domains: list[TestDomainConfig] = Field(default_factory=list)
    repetitions: Repetitions = Field(default_factory=Repetitions)
    selection: SelectionRule = Field(default_factory=SelectionRule)

    @model_validator(mode="after")
    def _one_split_source(self) -> EvaluationConfig:
        if (self.split is None) == (self.split_manifest is None):
            raise ValueError("give exactly one of split and split_manifest")
        if self.split_manifest is not None and (self.test_domains or self.repetitions.split_seeds):
            raise ValueError(
                "a reused split_manifest already fixes its test domains and split; "
                "test_domains and repetitions.split_seeds need a split configuration"
            )
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> EvaluationConfig:
        import yaml

        data = yaml.safe_load(Path(path).read_text())
        return cls.model_validate(data.get("evaluation", data))


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


def _evaluate(
    mlip_cfg: MLIPConfig,
    frames: list[Atoms],
    group_ids: list[str],
    *,
    iteration: int,
    model_path: str,
    label: str,
    out_json: Path,
    max_parity_points: int,
    reference_manifest: Path,
    seed: int,
) -> dict:
    """Evaluate one model on one partition; write its metrics and parity data."""
    metrics, parity = evaluate_frames(
        mlip_cfg,
        frames,
        iteration=iteration,
        model_path=model_path,
        test_set_label=label,
        elemental_reference_manifest=reference_manifest,
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(asdict(metrics), indent=2))
    np.savez_compressed(
        out_json.with_name(out_json.stem + "_parity.npz"),
        **_subsample(parity, max_parity_points),
    )
    log.info(
        "%s %s on %s: raw_E_MAE=%.4f (formation %.4f) eV/atom  F_MAE=%.4f eV/A",
        out_json.stem,
        mlip_cfg.backend,
        Path(label).name,
        metrics.energy_mae_eV_per_atom,
        metrics.formation_energy_mae_eV_per_atom,
        metrics.force_mae_eV_per_A,
    )
    return {
        "metrics": asdict(metrics),
        "groups": group_report(parity, group_ids, seed=seed),
    }


def _dataset_method_check(dataset_path: Path, reference_manifest: Path):
    """Verify the dataset's DFT method against the references; return the checker."""
    from matsim_agents.datasets.energy_references import (
        load_elemental_reference_manifest,
        validate_dataset_reference_method,
    )

    frames = list(ase_read(str(dataset_path), index=":"))
    if len(frames) < 4:
        raise ValueError(f"Need >=4 frames for a split; got {len(frames)}.")
    manifest, _, _ = load_elemental_reference_manifest(
        reference_manifest,
        required_elements={symbol for frame in frames for symbol in frame.get_chemical_symbols()},
    )

    def validate_method(path: Path) -> None:
        validate_dataset_reference_method(
            path,
            reference_backend=manifest["backend"],
            reference_method_signature=manifest["method_signature"],
            reference_base_method_signature=manifest.get("base_method_signature"),
            reference_element_method_signatures=manifest.get("element_method_signatures"),
            require_sidecar=True,
        )

    validate_method(dataset_path)
    return frames, validate_method


def _prepare_split(
    dataset_path: Path,
    split_dir: Path,
    evaluation: EvaluationConfig,
    split_seed: int,
    *,
    eval_only: bool,
    validate_method,
) -> SplitManifest:
    """Make, or load and verify, the split one repetition uses."""
    from matsim_agents.datasets.governance import sha256_file

    if evaluation.split_manifest is not None:
        manifest = load_split(evaluation.split_manifest)
    elif eval_only:
        if not (split_dir / "split-manifest.json").is_file():
            raise FileNotFoundError(
                f"eval-only reuses the run's split, but {split_dir} has no split-manifest.json; "
                "evaluate runs made before split manifests by splitting again (see "
                "docs/mlip-evaluation.md)"
            )
        manifest = load_split(split_dir)
    else:
        assert evaluation.split is not None
        manifest = make_split(
            dataset_path,
            evaluation.split.model_copy(update={"seed": split_seed}),
            split_dir,
            objective=evaluation.objective,
            test_domains=evaluation.test_domains,
        )
    if manifest.source["sha256"] != sha256_file(dataset_path):
        raise ValueError(f"split {manifest.source['path']} was made from another dataset")
    for membership in manifest.roles.values():
        validate_method(Path(membership.path))
    for domain in manifest.test_domains:
        validate_method(Path(domain.dataset))
    log.info(
        "%s split (%s): %s",
        manifest.strategy,
        manifest.objective,
        ", ".join(
            f"{role} {m.n_frames} frames / {m.n_groups} groups"
            for role, m in manifest.roles.items()
        ),
    )
    return manifest


def _require_checkpoint(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"eval-only: missing {label} checkpoint {path}")
    return path


def _training_device(dev: str) -> str:
    return "CUDA" if dev.startswith("cuda") else "CPU"


@dataclass
class _Models:
    """The foundation ("baseline") and fine-tuned models of one run."""

    baseline: MLIPConfig
    baseline_path: str
    finetuned: MLIPConfig
    finetuned_path: str


def _uma_endpoints(
    train_path: Path,
    validation_path: Path | None,
    ft_dir: Path,
    dev: str,
    *,
    eval_only: bool,
    base_model: str,
    task_name: str,
    finetune_kwargs: dict[str, Any],
) -> _Models:
    """Fine-tune the UMA foundation model (or reuse the fine-tune)."""
    if eval_only:
        ckpt = _require_checkpoint(ft_dir / "inference_ckpt.pt", "UMA")
    else:
        from matsim_agents.training.finetune_uma import finetune_uma

        ckpt = finetune_uma(
            train_path,
            ft_dir,
            base_model=base_model,
            task_name=task_name,
            device=_training_device(dev),
            validation_dataset=validation_path,
            **finetune_kwargs,
        )
    return _Models(
        _uma_cfg(base_model, task_name, dev),
        base_model,
        _uma_cfg(str(ckpt), task_name, dev),
        str(ckpt),
    )


def _mace_endpoints(
    train_path: Path,
    validation_path: Path | None,
    ft_dir: Path,
    dev: str,
    *,
    eval_only: bool,
    family: str,
    model: str,
    model_id: str | None,
    precision: str,
    finetune_kwargs: dict[str, Any],
) -> _Models:
    """Fine-tune the MACE foundation model (or reuse the fine-tune)."""
    # A MACE_MODELS id (if given) selects the family/variant for the
    # baseline eval too, so "before" and "after" use the same foundation.
    if model_id is not None:
        from matsim_agents.training.finetune_mace import MACE_MODELS

        if model_id not in MACE_MODELS:
            raise ValueError(
                f"mace_model_id must be one of {sorted(MACE_MODELS)}, got {model_id!r}"
            )
        family = MACE_MODELS[model_id]["family"]
        model = MACE_MODELS[model_id]["model"]
    if eval_only:
        ckpt = _require_checkpoint(ft_dir / "mace_finetuned.model", "MACE")
    else:
        from matsim_agents.training.finetune_mace import finetune_mace

        ckpt = finetune_mace(
            train_path,
            ft_dir,
            family=family,
            base_model=model,
            model_id=model_id,
            precision=precision,
            device=_training_device(dev),
            validation_dataset=validation_path,
            **finetune_kwargs,
        )
    # Fine-tuned checkpoint reloads via family="checkpoint".
    return _Models(
        _mace_cfg(family, model, dev, precision=precision),
        f"{family}:{model}",
        _mace_cfg("checkpoint", str(ckpt), dev, precision=precision),
        str(ckpt),
    )


def _hydragnn_endpoints(
    train_path: Path,
    validation_path: Path | None,
    ft_dir: Path,
    dev: str,
    *,
    eval_only: bool,
    reference_manifest: Path,
    gfm_logdir: str | Path | None,
    branch_mlp_path: str | Path | None,
    gfm_checkpoint: str | None,
    strategy: str,
    head: str | int | None,
    ft_repo: str | Path | None,
    hydragnn_root: str | Path | None,
    weight_threshold: float,
    unfreeze_backbone: bool,
    epochs: int,
    lr: float | None,
    batch_size: int,
    seed: int,
) -> _Models:
    """Run the routed or single-head HydraGNN fine-tune (or reuse it)."""
    if gfm_logdir is None or branch_mlp_path is None:
        raise ValueError("hydragnn backend requires gfm_logdir and branch_mlp_path.")
    gfm_logdir = Path(gfm_logdir).expanduser().resolve()
    branch_mlp_path = Path(branch_mlp_path).expanduser().resolve()
    base_ckpt = gfm_checkpoint or _newest_pk(gfm_logdir).name
    baseline = _hydragnn_cfg(gfm_logdir, base_ckpt, branch_mlp_path, dev)
    from matsim_agents.training.finetune_hydragnn import finetune_hydragnn

    if strategy == "routed":
        if eval_only:
            ft_logdir = ft_dir
            _require_checkpoint(ft_logdir / "ft_model.pk", "HydraGNN")
        else:
            ft_logdir = finetune_hydragnn(
                train_path,
                ft_dir,
                elemental_reference_manifest=reference_manifest,
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
                validation_dataset=validation_path,
            )
        return _Models(
            baseline,
            str(gfm_logdir),
            _hydragnn_cfg(ft_logdir, "ft_model.pk", branch_mlp_path, dev),
            str(ft_logdir),
        )

    from matsim_agents.training.finetune_hydragnn_newhead import finetune_hydragnn_newhead

    ft_repo_p = Path(ft_repo).expanduser().resolve() if ft_repo else None
    if eval_only:
        ft_logdir = ft_dir
        _require_checkpoint(ft_logdir / "ft_model.pk", "HydraGNN")
    else:
        ft_logdir = finetune_hydragnn_newhead(
            train_path,
            ft_dir,
            elemental_reference_manifest=reference_manifest,
            gfm_logdir=gfm_logdir,
            strategy=strategy,
            head=head,
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
            validation_dataset=validation_path,
        )
    return _Models(
        baseline,
        str(gfm_logdir),
        _hydragnn_newhead_cfg(ft_logdir, "ft_model.pk", dev, ft_repo=ft_repo_p),
        str(ft_logdir),
    )


# --------------------------------------------------------------------------- #
# Campaign                                                                    #
# --------------------------------------------------------------------------- #


def run_finetune_endpoints(
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    backend: str,
    elemental_reference_manifest: str | Path,
    evaluation: EvaluationConfig,
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
    # MACE (all installed foundation families; model is an alias, URL, or path).
    # {small, medium, large}/tag/URL/path, or a MACE_MODELS id). Fine-tuning is
    # delegated to mace_run_train (reference recipe), with optional native LoRA.
    mace_family: str = "mace_mp",
    mace_model: str = "medium",
    mace_model_id: str | None = None,
    mace_precision: str = "fp64",
    mace_epochs: int | None = None,
    mace_lr: float | None = None,
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
    seed: int = 0,
    device: str | None = None,
    max_parity_points: int = 20000,
    eval_only: bool = False,
) -> Path:
    """Fine-tune one backend on one dataset and evaluate it with explicit data roles.

    ``evaluation`` declares the objective, the split, test domains, repetitions,
    and the selection rule (module docstring). There is no default split: a
    random-frame split only tests interpolation, so it must be chosen explicitly
    (``examples/evaluation/random-frame-interpolation.yaml``). ``seed`` is the
    fine-tuning seed unless ``evaluation.repetitions`` lists training seeds. With
    ``eval_only`` the existing split and fine-tuned checkpoint are re-scored and
    evaluations are written with a ``_formation`` suffix, leaving the original
    ones untouched.
    """
    # Apply the GPU-visibility policy *before* any torch/HydraGNN import happens
    # downstream, so an explicit CPU request is actually honoured.
    _enforce_device_visibility(device)

    dataset_path = Path(dataset_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    dev = device or _default_device()
    reference_manifest = Path(elemental_reference_manifest)
    frames, validate_method = _dataset_method_check(dataset_path, reference_manifest)
    n_iters = _num_al_iterations(frames)

    def fine_tune(train: Path, validation: Path | None, ft_dir: Path, run_seed: int) -> _Models:
        if backend == "uma":
            return _uma_endpoints(
                train,
                validation,
                ft_dir,
                dev,
                eval_only=eval_only,
                base_model=uma_base_model,
                task_name=uma_task_name,
                finetune_kwargs={
                    "elemental_reference_manifest": reference_manifest,
                    "epochs": uma_epochs,
                    "batch_size": batch_size,
                    "lr": lr if lr is not None else uma_lr,
                    "force_weight": uma_force_weight,
                    "freeze_backbone": uma_freeze_backbone,
                    "weight_decay": uma_weight_decay,
                    "seed": run_seed,
                    "lora": uma_lora,
                    "lora_r": uma_lora_r,
                    "lora_alpha": uma_lora_alpha,
                },
            )
        if backend == "mace":
            return _mace_endpoints(
                train,
                validation,
                ft_dir,
                dev,
                eval_only=eval_only,
                family=mace_family,
                model=mace_model,
                model_id=mace_model_id,
                precision=mace_precision,
                finetune_kwargs={
                    "elemental_reference_manifest": reference_manifest,
                    "epochs": mace_epochs,
                    "batch_size": batch_size,
                    "lr": lr if lr is not None else mace_lr,
                    "lora": mace_lora,
                    "lora_rank": mace_lora_rank,
                    "lora_alpha": mace_lora_alpha,
                    "freeze_backbone": mace_freeze_backbone,
                    "weight_decay": mace_weight_decay,
                    "seed": run_seed,
                },
            )
        if backend == "hydragnn":
            return _hydragnn_endpoints(
                train,
                validation,
                ft_dir,
                dev,
                eval_only=eval_only,
                reference_manifest=reference_manifest,
                gfm_logdir=gfm_logdir,
                branch_mlp_path=branch_mlp_path,
                gfm_checkpoint=gfm_checkpoint,
                strategy=hydragnn_strategy,
                head=hydragnn_head,
                ft_repo=ft_repo,
                hydragnn_root=hydragnn_root,
                weight_threshold=weight_threshold,
                unfreeze_backbone=unfreeze_backbone,
                epochs=epochs,
                lr=lr,
                batch_size=batch_size,
                seed=run_seed,
            )
        raise ValueError(f"Unknown backend {backend!r} (expected 'hydragnn', 'uma', or 'mace').")

    split_seeds = evaluation.repetitions.split_seeds or [
        evaluation.split.seed if evaluation.split is not None else 0
    ]
    training_seeds = evaluation.repetitions.training_seeds or [seed]
    summaries = []
    test_groups = []
    for split_seed in split_seeds:
        split_root = output_dir / (f"split-seed-{split_seed}" if len(split_seeds) > 1 else "")
        manifest = _prepare_split(
            dataset_path,
            split_root / "split",
            evaluation,
            split_seed,
            eval_only=eval_only,
            validate_method=validate_method,
        )
        for training_seed in training_seeds:
            run_dir = split_root / (
                f"train-seed-{training_seed}" if len(training_seeds) > 1 else ""
            )
            summary = _run_repetition(
                manifest,
                run_dir,
                fine_tune,
                backend=backend,
                training_seed=training_seed,
                selection=evaluation.selection,
                reference_manifest=reference_manifest,
                n_iters=n_iters,
                max_parity_points=max_parity_points,
                eval_only=eval_only,
            )
            summaries.append({**summary, "split_seed": split_seed})
            test_groups.append(manifest.roles["test"].n_groups if "test" in manifest.roles else 0)
    if len(summaries) > 1:
        (output_dir / "repetitions.json").write_text(
            json.dumps(summarize_repetitions(summaries, n_test_groups=test_groups), indent=2)
        )
    log.info("Fine-tune evaluation complete -> %s", output_dir)
    return output_dir


def _role_frames(manifest: SplitManifest, role: str) -> tuple[list[Atoms], list[str], Path]:
    membership = manifest.roles[role]
    return (
        list(ase_read(membership.path, index=":")),
        membership.frame_group_ids,
        Path(membership.path),
    )


def _artifact_digest(path: str) -> str | None:
    """SHA-256 of a model file, or of the files directly in a model directory."""
    target = Path(path)
    if target.is_file():
        from matsim_agents.datasets.governance import sha256_file

        return sha256_file(target)
    if target.is_dir():
        from matsim_agents.datasets.governance import sha256_file

        digest = hashlib.sha256()
        for item in sorted(p for p in target.iterdir() if p.is_file()):
            digest.update(f"{item.name}:{sha256_file(item)}\n".encode())
        return digest.hexdigest()
    return None  # a foundation model named by its release


def _select(
    models: _Models,
    manifest: SplitManifest,
    run_dir: Path,
    *,
    rule: SelectionRule,
    evaluate,
    suffix: str,
) -> tuple[dict, dict]:
    """Choose between the models on validation only, and freeze the choice."""
    evaluations: dict = {}
    if "validation" in manifest.roles:
        frames, groups, path = _role_frames(manifest, "validation")
        for name, cfg, model_path in (
            ("baseline", models.baseline, models.baseline_path),
            ("finetuned", models.finetuned, models.finetuned_path),
        ):
            evaluations[name] = evaluate(
                cfg,
                frames,
                groups,
                model_path=model_path,
                label=str(path),
                out_json=run_dir / "eval" / "validation" / f"{name}{suffix}.json",
                iteration=0,
            )
        decision = assess_promotion(
            _metrics(evaluations["finetuned"]),
            _metrics(evaluations["baseline"]),
            rule.as_promotion_rule(),
        )
        selected = "finetuned" if decision.approved else "baseline"
        basis = {
            "rule": rule.model_dump(),
            "satisfied": decision.approved,
            "reasons": decision.reasons,
        }
    else:
        selected = "finetuned"
        basis = {
            "rule": None,
            "satisfied": None,
            "reasons": [
                "no validation partition: the fine-tuned model is reported without selection"
            ],
        }
    selection = {
        "selected": selected,
        "decided_on": "validation" if "validation" in manifest.roles else None,
        **basis,
        "split_manifest_sha256": manifest.content_sha256,
        "models": {
            "baseline": {
                "path": models.baseline_path,
                "sha256": _artifact_digest(models.baseline_path),
            },
            "finetuned": {
                "path": models.finetuned_path,
                "sha256": _artifact_digest(models.finetuned_path),
            },
        },
        "frozen_at_utc": datetime.now(UTC).isoformat(),
    }
    path = run_dir / f"selection{suffix}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(selection, indent=2))
    return selection, evaluations


def _metrics(evaluation: dict):
    from matsim_agents.active_learning.evaluate import EvalMetrics

    return EvalMetrics(**evaluation["metrics"])


def _run_repetition(
    manifest: SplitManifest,
    run_dir: Path,
    fine_tune,
    *,
    backend: str,
    training_seed: int,
    selection: SelectionRule,
    reference_manifest: Path,
    n_iters: int,
    max_parity_points: int,
    eval_only: bool,
) -> dict:
    """Fine-tune, select on validation, freeze, then evaluate test and test domains."""
    suffix = "_formation" if eval_only else ""

    def evaluate(cfg, frames, groups, **kwargs):
        return _evaluate(
            cfg,
            frames,
            groups,
            max_parity_points=max_parity_points,
            reference_manifest=reference_manifest,
            seed=training_seed,
            **kwargs,
        )

    validation_path = (
        Path(manifest.roles["validation"].path) if "validation" in manifest.roles else None
    )
    models = fine_tune(
        Path(manifest.roles["train"].path), validation_path, run_dir / "ft", training_seed
    )
    chosen, validation = _select(
        models, manifest, run_dir, rule=selection, evaluate=evaluate, suffix=suffix
    )
    # The selection is frozen on disk before any test frame is evaluated.
    evaluations: dict = {"validation": validation} if validation else {}
    targets = []
    if "test" in manifest.roles:
        frames, groups, path = _role_frames(manifest, "test")
        targets.append(
            (
                "test",
                frames,
                groups,
                path,
                run_dir / "eval" / f"iter0{suffix}.json",
                run_dir / "eval" / f"iter{n_iters}{suffix}.json",
            )
        )
    for domain in manifest.test_domains:
        directory = run_dir / "eval" / "domains" / domain.name
        targets.append(
            (
                f"domain:{domain.name}",
                list(ase_read(domain.dataset, index=":")),
                domain.frame_group_ids,
                Path(domain.dataset),
                directory / f"baseline{suffix}.json",
                directory / f"finetuned{suffix}.json",
            )
        )
    for name, frames, groups, path, baseline_json, finetuned_json in targets:
        evaluations[name] = {
            "baseline": evaluate(
                models.baseline,
                frames,
                groups,
                model_path=models.baseline_path,
                label=str(path),
                out_json=baseline_json,
                iteration=0,
            ),
            "finetuned": evaluate(
                models.finetuned,
                frames,
                groups,
                model_path=models.finetuned_path,
                label=str(path),
                out_json=finetuned_json,
                iteration=n_iters,
            ),
        }
    report = _report(manifest, backend, chosen, evaluations, run_dir)
    (run_dir / "eval" / f"report{suffix}.json").write_text(json.dumps(report, indent=2))
    return {
        "training_seed": training_seed,
        "selected": chosen["selected"],
        "selected_metrics": {
            name: {
                metric: models_[chosen["selected"]]["metrics"].get(metric)
                for metric in METRIC_DEFINITIONS
            }
            for name, models_ in evaluations.items()
            if name != "validation"
        },
    }


def _report(
    manifest: SplitManifest, backend: str, selection: dict, evaluations: dict, run_dir: Path
) -> dict:
    cost = run_dir / "ft" / "cost.json"
    return {
        "backend": backend,
        "metric_definitions": METRIC_DEFINITIONS,
        "split": {
            "strategy": manifest.strategy,
            "objective": manifest.objective,
            "manifest_sha256": manifest.content_sha256,
            "roles": {
                role: {"n_frames": m.n_frames, "n_groups": m.n_groups}
                for role, m in manifest.roles.items()
            },
            "test_domains": {
                d.name: {"membership": d.membership, "n_frames": d.n_frames, "n_groups": d.n_groups}
                for d in manifest.test_domains
            },
            "limitations": manifest.limitations,
        },
        "selection": selection,
        "evaluations": {
            name: {
                model: {
                    "metrics": {k: result["metrics"].get(k) for k in METRIC_DEFINITIONS},
                    "groups": result["groups"],
                }
                for model, result in models.items()
            }
            for name, models in evaluations.items()
        },
        "paired_finetuned_minus_baseline": {
            name: paired_difference(models["finetuned"]["groups"], models["baseline"]["groups"])
            for name, models in evaluations.items()
        },
        "fine_tuning_cost": json.loads(cost.read_text()) if cost.is_file() else None,
    }


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
    parser.add_argument("--elemental-reference-manifest", required=True, type=Path)
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
        choices=[
            "mace_mp",
            "mace_off",
            "mace_omol",
            "mace_polar",
            "mace_anicc",
            "checkpoint",
        ],
        help="MACE foundation-loader family or checkpoint (local .model).",
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
    parser.add_argument(
        "--evaluation-config",
        type=Path,
        required=True,
        help="YAML evaluation config (objective, split, test domains, repetitions, "
        "selection); see docs/mlip-evaluation.md and examples/evaluation/.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Fine-tuning seed.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-parity-points", type=int, default=20000)
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training: reuse the existing split + fine-tuned checkpoint and "
        "re-score both endpoints using fixed elemental references, "
        "writing eval/iter*_formation.json.",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    # Set GPU visibility from the requested device before torch is imported.
    _enforce_device_visibility(args.device)

    run_finetune_endpoints(
        args.dataset,
        args.output_dir,
        backend=args.backend,
        elemental_reference_manifest=args.elemental_reference_manifest,
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
        mace_epochs=args.mace_epochs,
        mace_lr=args.mace_lr,
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
        seed=args.seed,
        device=args.device,
        max_parity_points=args.max_parity_points,
        eval_only=args.eval_only,
        evaluation=EvaluationConfig.from_yaml(args.evaluation_config),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
