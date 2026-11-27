"""Fine-tune and evaluate one MLIP backend on one dataset, with explicit data roles.

The dataset is split by a declared strategy (:mod:`matsim_agents.data.splits`)
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

The foundation model is an ``mlip`` block (``--mlip-config``); the evaluation
YAML's optional ``fine_tune`` block holds that backend's fine-tuning
hyperparameters. Each backend needs its own environment (HydraGNN ``.venv``, UMA
``.venv-uma``, MACE ``.venv-mace``), so one invocation evaluates one model; later
models reuse the first run's split (``split_manifest``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated, Any, Literal

import numpy as np
from ase import Atoms
from ase.io import read as ase_read
from pydantic import BaseModel, ConfigDict, Field, model_validator

from matsim_agents.backends.mlip.arguments import add_mlip_arguments, mlip_config_from_args
from matsim_agents.backends.mlip.config import (
    HydraGNNConfig,
    MLIPBackendConfig,
    MLIPConfig,
    parse_mlip_config,
)
from matsim_agents.core.seeding import stream_seed
from matsim_agents.data.splits import (
    SplitConfig,
    SplitManifest,
    TestDomainConfig,
    load_split,
    make_split,
)
from matsim_agents.workflows.active_learning.eval_report import (
    METRIC_DEFINITIONS,
    group_report,
    paired_difference,
    summarize_repetitions,
)
from matsim_agents.workflows.active_learning.evaluate import (
    _subsample,
    assess_promotion,
    evaluate_frames,
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
    fine_tune: FineTuneRecipe | None = Field(
        None,
        description=(
            "Fine-tuning hyperparameters for the selected backend (backend: hydragnn | uma | "
            "mace); omitted means that backend's defaults."
        ),
    )

    def recipe_for(self, mlip: MLIPBackendConfig) -> _FineTuneRecipe:
        """This evaluation's fine-tune recipe, which must match the selected backend."""
        if self.fine_tune is None:
            return _DEFAULT_RECIPES[mlip.backend]()
        if self.fine_tune.backend != mlip.backend:
            raise ValueError(
                f"fine_tune.backend is {self.fine_tune.backend!r} but the selected MLIP is "
                f"{mlip.backend!r}"
            )
        return self.fine_tune

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
# Fine-tuning recipes (one per backend)                                       #
# --------------------------------------------------------------------------- #


def _require_checkpoint(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"eval-only: missing {label} checkpoint {path}")
    return path


def _training_device(device: str) -> str:
    return "CUDA" if device == "cuda" else "CPU"


@dataclass
class _Models:
    """The foundation ("baseline") and fine-tuned models of one run."""

    baseline: MLIPBackendConfig
    baseline_path: str
    finetuned: MLIPBackendConfig
    finetuned_path: str


class _FineTuneRecipe(BaseModel, ABC):
    """Hyperparameters of one backend's fine-tune; the model comes from the mlip block."""

    model_config = ConfigDict(extra="forbid")

    backend: str
    batch_size: int = Field(4, ge=1)

    @abstractmethod
    def endpoints(
        self,
        mlip: Any,
        train: Path,
        validation: Path | None,
        ft_dir: Path,
        *,
        eval_only: bool,
        reference_manifest: Path,
        seed: int,
    ) -> _Models:
        """Fine-tune ``mlip`` on ``train`` (or reuse the fine-tune) and return both models."""


class HydraGNNFineTune(_FineTuneRecipe):
    """HydraGNN fine-tune of the single branch-7 (OMat24) head."""

    backend: Literal["hydragnn"] = "hydragnn"
    epochs: int = Field(20, ge=1)
    lr: float = Field(1e-3, gt=0)
    strategy: Literal["unfrozen", "frozen"] = "unfrozen"
    head: Literal["OMat24"] = "OMat24"

    @model_validator(mode="before")
    @classmethod
    def _only_branch_7(cls, data: Any) -> Any:
        if isinstance(data, dict):
            from matsim_agents.backends.mlip.config import resolve_hydragnn_inference_head

            if forbidden := sorted({"weight_threshold", "unfreeze_backbone"} & set(data)):
                raise ValueError(
                    f"fine_tune {forbidden} belong to the forbidden routed HydraGNN fine-tune; "
                    "matsim-agents uses only branch-7 (OMat24)"
                )
            resolve_hydragnn_inference_head(data.get("head"))
            data = {**data, "head": "OMat24"}
        return data

    def _check_recipe(self, mlip: HydraGNNConfig) -> None:
        if mlip.checkpoint is None:
            raise ValueError("HydraGNN fine-tuning needs mlip.checkpoint (the foundation weights)")

    def endpoints(self, mlip, train, validation, ft_dir, *, eval_only, reference_manifest, seed):
        self._check_recipe(mlip)
        common = {
            "elemental_reference_manifest": reference_manifest,
            "gfm_logdir": mlip.logdir,
            "gfm_checkpoint": mlip.checkpoint,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "seed": seed,
            "charge": mlip.charge,
            "spin": mlip.spin,
            "hydragnn_root": mlip.hydragnn_root,
            "device": mlip.device,
            "validation_dataset": validation,
        }
        if eval_only:
            _require_checkpoint(ft_dir / "ft_model.pk", "HydraGNN")
            ft_logdir = ft_dir
        else:
            from matsim_agents.backends.mlip.training.finetune_hydragnn_newhead import (
                finetune_hydragnn_newhead,
            )

            # The new-head loss is normalised (O(1) forces/energy) with LR
            # warmup+cosine and gradient clipping, so 1e-3 moves the metrics off
            # the zero-shot baseline without collapsing the head.
            ft_logdir = finetune_hydragnn_newhead(
                train,
                ft_dir,
                strategy=self.strategy,
                head=self.head,
                ft_repo=mlip.ft_repo,
                lr=self.lr,
                **common,
            )
        update: dict[str, Any] = {
            "logdir": Path(ft_logdir),
            "checkpoint": "ft_model.pk",
            "newhead_ft_config": Path(ft_logdir) / "newhead.json",
        }
        return _Models(mlip, str(mlip.logdir), mlip.model_copy(update=update), str(ft_logdir))


class UMAFineTune(_FineTuneRecipe):
    """UMA fine-tune (custom conservative-head loop; see finetune_uma).

    Defaults match allaffa/HydraGNN_GFM_FineTuning4Materials @ uma-sota-comparison:
    Adam lr=1e-4, force-weighted MSE loss, no weight decay.
    """

    backend: Literal["uma"] = "uma"
    epochs: int = Field(20, ge=1)
    lr: float = Field(1e-4, gt=0)
    force_weight: float = Field(10.0, ge=0, description="Force-MSE weight relative to energy.")
    weight_decay: float = Field(0.0, ge=0)
    freeze_backbone: bool = False
    lora: bool = Field(False, description="LoRA on backbone scalar linears (keeps equivariance).")
    lora_r: int = Field(8, ge=1)
    lora_alpha: float = Field(16.0, gt=0)

    def endpoints(self, mlip, train, validation, ft_dir, *, eval_only, reference_manifest, seed):
        if eval_only:
            checkpoint = _require_checkpoint(ft_dir / "inference_ckpt.pt", "UMA")
        else:
            from matsim_agents.backends.mlip.training.finetune_uma import finetune_uma

            checkpoint = finetune_uma(
                train,
                ft_dir,
                elemental_reference_manifest=reference_manifest,
                base_model=mlip.model_name,
                task_name=mlip.task_name,
                device=_training_device(mlip.device),
                validation_dataset=validation,
                epochs=self.epochs,
                batch_size=self.batch_size,
                lr=self.lr,
                force_weight=self.force_weight,
                freeze_backbone=self.freeze_backbone,
                weight_decay=self.weight_decay,
                seed=seed,
                lora=self.lora,
                lora_r=self.lora_r,
                lora_alpha=self.lora_alpha,
            )
        return _Models(mlip, mlip.model_name, mlip.with_checkpoint(checkpoint), str(checkpoint))


class MACEFineTune(_FineTuneRecipe):
    """MACE fine-tune via the reference mace_run_train recipe, with optional native LoRA."""

    backend: Literal["mace"] = "mace"
    epochs: int | None = Field(None, ge=1, description="Default: 10 for LoRA, 50 for naive.")
    lr: float | None = Field(None, gt=0, description="Default: 5e-3 for LoRA, 1e-3 for naive.")
    weight_decay: float = Field(0.0, ge=0)
    freeze_backbone: bool = False
    lora: bool = False
    lora_rank: int | None = Field(None, ge=1, description="Default: 4.")
    lora_alpha: float | None = Field(None, gt=0, description="Default: 1.0.")

    def endpoints(self, mlip, train, validation, ft_dir, *, eval_only, reference_manifest, seed):
        if eval_only:
            checkpoint = _require_checkpoint(ft_dir / "mace_finetuned.model", "MACE")
        else:
            from matsim_agents.backends.mlip.training.finetune_mace import finetune_mace

            checkpoint = finetune_mace(
                train,
                ft_dir,
                elemental_reference_manifest=reference_manifest,
                family=mlip.family,
                base_model=mlip.model,
                precision=mlip.precision or "fp64",
                device=_training_device(mlip.device),
                validation_dataset=validation,
                epochs=self.epochs,
                batch_size=self.batch_size,
                lr=self.lr,
                lora=self.lora,
                lora_rank=self.lora_rank,
                lora_alpha=self.lora_alpha,
                freeze_backbone=self.freeze_backbone,
                weight_decay=self.weight_decay,
                seed=seed,
            )
        return _Models(
            mlip, f"{mlip.family}:{mlip.model}", mlip.with_checkpoint(checkpoint), str(checkpoint)
        )


FineTuneRecipe = Annotated[
    HydraGNNFineTune | UMAFineTune | MACEFineTune, Field(discriminator="backend")
]
_DEFAULT_RECIPES: dict[str, type[_FineTuneRecipe]] = {
    "hydragnn": HydraGNNFineTune,
    "uma": UMAFineTune,
    "mace": MACEFineTune,
}
EvaluationConfig.model_rebuild()


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
        **_subsample(parity, max_parity_points, seed=stream_seed(seed, "parity-plot")),
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
    from matsim_agents.core.energy_references import (
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
    from matsim_agents.core.hashing import sha256_file

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


# --------------------------------------------------------------------------- #
# Campaign                                                                    #
# --------------------------------------------------------------------------- #


def run_finetune_endpoints(
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    mlip: MLIPConfig | dict,
    elemental_reference_manifest: str | Path,
    evaluation: EvaluationConfig,
    seed: int,
    max_parity_points: int = 20000,
    eval_only: bool = False,
) -> Path:
    """Fine-tune the selected MLIP on one dataset and evaluate it with explicit data roles.

    ``mlip`` is the foundation model (an ``mlip`` block); ``evaluation`` declares
    the objective, the split, test domains, repetitions, the selection rule, and
    the backend's fine-tuning recipe (module docstring). There is no default split:
    a random-frame split only tests interpolation, so it must be chosen explicitly
    (``examples/evaluation/random-frame-interpolation.yaml``). ``seed`` is the
    fine-tuning seed unless ``evaluation.repetitions`` lists training seeds. With
    ``eval_only`` the existing split and fine-tuned checkpoint are re-scored and
    evaluations are written with a ``_formation`` suffix, leaving the original
    ones untouched.
    """
    mlip = parse_mlip_config(mlip)
    recipe = evaluation.recipe_for(mlip)
    if not eval_only:
        mlip.check_fine_tunable()
    # Apply the GPU-visibility policy *before* any torch/HydraGNN import happens
    # downstream, so an explicit CPU request is actually honoured.
    _enforce_device_visibility(mlip.device)

    dataset_path = Path(dataset_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    reference_manifest = Path(elemental_reference_manifest)
    frames, validate_method = _dataset_method_check(dataset_path, reference_manifest)
    n_iters = _num_al_iterations(frames)

    def fine_tune(train: Path, validation: Path | None, ft_dir: Path, run_seed: int) -> _Models:
        return recipe.endpoints(
            mlip,
            train,
            validation,
            ft_dir,
            eval_only=eval_only,
            reference_manifest=reference_manifest,
            seed=run_seed,
        )

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
                mlip=mlip,
                recipe=recipe,
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
        from matsim_agents.core.hashing import sha256_file

        return sha256_file(target)
    if target.is_dir():
        from matsim_agents.core.hashing import sha256_file

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
    from matsim_agents.workflows.active_learning.evaluate import EvalMetrics

    return EvalMetrics(**evaluation["metrics"])


def _run_repetition(
    manifest: SplitManifest,
    run_dir: Path,
    fine_tune,
    *,
    mlip: MLIPBackendConfig,
    recipe: _FineTuneRecipe,
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
            seed=stream_seed(training_seed, "bootstrap"),
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
    report = _report(
        manifest,
        mlip,
        recipe,
        chosen,
        evaluations,
        run_dir,
        seed=stream_seed(training_seed, "bootstrap"),
    )
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
    manifest: SplitManifest,
    mlip: MLIPBackendConfig,
    recipe: _FineTuneRecipe,
    selection: dict,
    evaluations: dict,
    run_dir: Path,
    *,
    seed: int,
) -> dict:
    cost = run_dir / "ft" / "cost.json"
    return {
        "backend": mlip.backend,
        "model": mlip.model_identifier(),
        "fine_tune": recipe.model_dump(mode="json"),
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
            name: paired_difference(
                models["finetuned"]["groups"], models["baseline"]["groups"], seed=seed
            )
            for name, models in evaluations.items()
        },
        "fine_tuning_cost": json.loads(cost.read_text()) if cost.is_file() else None,
    }


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_mlip_arguments(parser)
    parser.add_argument("--dataset", required=True, help="AL-collected extxyz dataset.")
    parser.add_argument("--elemental-reference-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, help="Campaign output directory.")
    parser.add_argument(
        "--evaluation-config",
        type=Path,
        required=True,
        help="YAML evaluation config (objective, split, test domains, repetitions, "
        "selection, fine_tune recipe); see docs/mlip-evaluation.md and examples/evaluation/.",
    )
    parser.add_argument("--seed", type=int, required=True, help="Fine-tuning seed.")
    parser.add_argument("--max-parity-points", type=int, default=20000)
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training: reuse the existing split + fine-tuned checkpoint and "
        "re-score both endpoints using fixed elemental references, "
        "writing eval/iter*_formation.json.",
    )
    args = parser.parse_args(argv)
    mlip = mlip_config_from_args(parser, args)
    if mlip is None:
        parser.error("--mlip-config is required (the foundation model to fine-tune)")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    # Set GPU visibility from the requested device before torch is imported.
    _enforce_device_visibility(mlip.device)

    run_finetune_endpoints(
        args.dataset,
        args.output_dir,
        mlip=mlip,
        elemental_reference_manifest=args.elemental_reference_manifest,
        evaluation=EvaluationConfig.from_yaml(args.evaluation_config),
        seed=args.seed,
        max_parity_points=args.max_parity_points,
        eval_only=args.eval_only,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
