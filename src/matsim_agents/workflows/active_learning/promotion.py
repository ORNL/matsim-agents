"""Candidate-versus-incumbent comparison, promotion and benchmark scoring."""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

from matsim_agents.core.energy_references import (
    load_elemental_reference_manifest,
    validate_dataset_reference_method,
)
from matsim_agents.data.governance import (
    structure_identity,
)
from matsim_agents.workflows.active_learning.config import (
    ALConfig,
)
from matsim_agents.workflows.active_learning.eval_report import METRIC_DEFINITIONS
from matsim_agents.workflows.active_learning.evaluate import (
    _apply_model_override,
    evaluate_frames,
    evaluate_promotion_candidate,
)
from matsim_agents.workflows.active_learning.iteration_state import IterationState
from matsim_agents.workflows.active_learning.surrogate import _SurrogateModels

log = logging.getLogger(__name__)


def _validate_comparison_inputs(cfg: ALConfig) -> bool:
    """Return whether candidate comparison is enabled, validating its inputs up front."""
    comparison_enabled = cfg.trainer.enabled and (
        cfg.trainer.compare_after_training or cfg.trainer.promote_model
    )
    if comparison_enabled:
        if cfg.trainer.validation_reference_set is None:
            raise ValueError(
                "model comparison requires a DFT-labelled elemental reference manifest"
            )
        reference_manifest, _, _ = load_elemental_reference_manifest(
            cfg.trainer.validation_reference_set, required_elements=set()
        )
        if cfg.trainer.validation_set is not None:
            validate_dataset_reference_method(
                cfg.trainer.validation_set,
                reference_backend=reference_manifest["backend"],
                reference_method_signature=reference_manifest["method_signature"],
                reference_base_method_signature=reference_manifest.get("base_method_signature"),
                reference_element_method_signatures=reference_manifest.get(
                    "element_method_signatures"
                ),
                require_sidecar=True,
            )
    return comparison_enabled


def _compare_and_promote(
    cfg: ALConfig,
    state: IterationState,
    i: int,
    dataset_path: Path,
    validation_dataset_path: Path,
    comparison_enabled: bool,
    incumbent_calculator: Any = None,
) -> None:
    """Compare the candidate against the incumbent and promote it when approved.

    ``incumbent_calculator`` is the incumbent model when it is still loaded.
    """
    compare_candidate = comparison_enabled and state.candidate_model_path is not None
    if compare_candidate:
        try:
            evaluation_cfg = cfg.model_copy(deep=True)
            if cfg.trainer.validation_fraction > 0:
                evaluation_cfg.trainer.validation_set = validation_dataset_path
                evaluation_cfg.trainer.validation_fraction = 0.0
            decision = evaluate_promotion_candidate(
                evaluation_cfg,
                state.candidate_model_path,
                iteration=i,
                training_set=dataset_path,
                incumbent_calculator=incumbent_calculator,
            )
            state.model_comparison = asdict(decision)
        except Exception as exc:  # noqa: BLE001
            state.model_comparison = {
                "approved": False,
                "reasons": [f"model comparison failed: {exc}"],
                "candidate_metrics": {},
                "incumbent_metrics": {},
            }
            log.exception("Iteration %d candidate model comparison failed", i)
        if cfg.trainer.promote_model:
            state.promotion_validation = state.model_comparison
    if (
        cfg.trainer.promote_model
        and state.promotion_validation is not None
        and state.promotion_validation["approved"]
    ):
        state.new_logdir = state.candidate_model_path
        state.model_promoted = True
        _apply_model_override(cfg, state.candidate_model_path)
        _export_for_lammps(cfg, state)


def _export_for_lammps(cfg: ALConfig, state: IterationState) -> None:
    """Opt-in: validate the promoted model under the LAMMPS ML-IAP bridge and record the result."""
    species = list(cfg.trainer.lammps_export_species)
    if not species or state.candidate_model_path is None:
        return
    try:
        from matsim_agents.backends.external.lammps.mliap.export import (
            export_checkpoint_for_lammps,
        )

        state.lammps_export = export_checkpoint_for_lammps(
            cfg.mlip,
            state.candidate_model_path,
            Path(state.candidate_model_path)
            if Path(state.candidate_model_path).is_dir()
            else Path(state.candidate_model_path).parent,
            species=species,
            seed=cfg.random_seed,
        )
    except Exception as exc:  # noqa: BLE001 - recorded, never blocks the promotion
        log.exception("LAMMPS export validation failed for %s", state.candidate_model_path)
        state.lammps_export = {"approved": False, "error": str(exc)}


def _score_benchmarks(
    cfg: ALConfig,
    state: IterationState,
    models: _SurrogateModels,
    dataset_path: Path,
    validation_dataset_path: Path,
) -> None:
    """Score the model in use on the frozen benchmark sets, for learning curves.

    Runs after the iteration's training and promotion decisions and feeds none
    of them; frames that were also acquired as labels are counted and logged.
    """
    if not cfg.loop.benchmark_sets:
        return
    from ase.io import read as ase_read

    labelled = {
        structure_identity(frame)
        for path in (dataset_path, validation_dataset_path)
        if path.exists()
        for frame in ase_read(path, index=":")
    }
    for benchmark in cfg.loop.benchmark_sets:
        frames = list(ase_read(benchmark.dataset, index=":"))
        overlap = sum(structure_identity(frame) in labelled for frame in frames)
        if overlap:
            log.warning(
                "Benchmark %s: %d frame(s) were also acquired as labels", benchmark.name, overlap
            )
        metrics, _ = evaluate_frames(
            cfg.mlip,
            frames,
            iteration=state.iteration,
            model_path=cfg.mlip.model_reference,
            test_set_label=str(benchmark.dataset),
            elemental_reference_manifest=cfg.benchmark_reference_manifest(),
            calculator=models.incumbent_calculator(cfg.mlip),
        )
        values = asdict(metrics)
        state.benchmark[benchmark.name] = {
            "model": cfg.mlip.model_reference,
            "n_dft_labels_total": state.n_training_frames_total + state.n_validation_frames_total,
            "frames_also_labelled": overlap,
            "metrics": {key: values[key] for key in METRIC_DEFINITIONS},
        }
