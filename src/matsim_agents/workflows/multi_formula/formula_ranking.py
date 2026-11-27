"""Independent DFT ranking of MLIP candidates, kept apart from the training dataset."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ase.io import read as ase_read

from matsim_agents.core.composition import parse_composition
from matsim_agents.core.contracts import EvidenceLevel
from matsim_agents.core.hashing import sha256_file
from matsim_agents.core.relaxation import RelaxationResult
from matsim_agents.core.seeding import stream_seed
from matsim_agents.science.thermodynamics.stability import (
    RankingMode,
    score_stability,
)
from matsim_agents.workflows.active_learning.config import ALConfig
from matsim_agents.workflows.candidate_registry import (
    select_dft_refinement_candidates,
    structure_content_hash,
)
from matsim_agents.workflows.exploration.phase import (
    PhaseExplorationWorkflowResult,
)
from matsim_agents.workflows.multi_formula.formula_al import _model_identifier
from matsim_agents.workflows.multi_formula.formula_config import CampaignFormulaExecutionConfig
from matsim_agents.workflows.multi_formula.formula_references import (
    _converged_dft_relaxation,
    _dft_relaxation_config,
    _generate_reference_energies,
)
from matsim_agents.workflows.multi_formula.formula_stages import _StageTracker
from matsim_agents.workflows.relaxation import (
    ScientificRelaxationConfig,
    ScientificRelaxationResult,
)


def _relaxation_energy_per_atom(relaxation: RelaxationResult) -> float:
    return relaxation.final_energy_eV / len(ase_read(relaxation.optimized_structure_path))


def _run_independent_dft_ranking(
    formula: str,
    formula_root: Path,
    config: CampaignFormulaExecutionConfig,
    *,
    al_cfg: ALConfig,
    effective_al_cfg: ALConfig,
    result: PhaseExplorationWorkflowResult,
    exploration: Any,
    execution_evidence: dict[str, Any],
    tracker: _StageTracker,
    relaxation_runner: Callable[[ScientificRelaxationConfig], ScientificRelaxationResult],
) -> PhaseExplorationWorkflowResult:
    """Rank MLIP candidates with DFT, kept independent of the AL training dataset."""
    refinement = config.dft_refinement
    if refinement is None:
        return result
    if not config.phase_policy.dft_approved:
        tracker.set(
            "independent_dft_ranking",
            "blocked",
            reason="explicit DFT approval was not provided",
        )
        raise PermissionError("campaign DFT refinement requires explicit DFT approval")
    tracker.set(
        "independent_dft_ranking",
        "in_progress",
        model_identifier=_model_identifier(effective_al_cfg),
        training_data=execution_evidence.get("training_data"),
    )
    dft_progress = {"attempts": 0, "elapsed_seconds": 0.0}

    def tracked_dft_relaxation_runner(
        dft_config: ScientificRelaxationConfig,
    ) -> ScientificRelaxationResult:
        dft_progress["attempts"] += 1
        started = time.monotonic()
        try:
            return relaxation_runner(dft_config)
        finally:
            dft_progress["elapsed_seconds"] += time.monotonic() - started
            tracker.set(
                "independent_dft_ranking",
                "in_progress",
                dft_attempts=dft_progress["attempts"],
                dft_elapsed_seconds=dft_progress["elapsed_seconds"],
                dft_node_hours=(dft_progress["elapsed_seconds"] * config.compute_nodes / 3600.0),
            )

    candidates = [relaxation for relaxation in exploration.relaxations if relaxation.converged]
    if refinement.candidate_acquisition.enabled:
        selected, candidate_scores = select_dft_refinement_candidates(
            exploration,
            max_candidates=refinement.max_candidates,
            policy=refinement.candidate_acquisition,
            uncertainty_by_candidate=(result.active_learning_result or {}).get(
                "candidate_uncertainty", {}
            ),
            random_seed=stream_seed(effective_al_cfg.random_seed, "dft-refinement"),
        )
    else:
        candidates.sort(key=_relaxation_energy_per_atom)
        selected = candidates[: refinement.max_candidates]
        candidate_scores = {}
    if not selected:
        tracker.set(
            "independent_dft_ranking",
            "failed",
            failure_reason="DFT refinement requires at least one converged MLIP relaxation",
        )
        raise RuntimeError("DFT refinement requires at least one converged MLIP relaxation")

    training_data = execution_evidence.get("training_data")
    training_data_hash_before = (
        sha256_file(training_data["path"])
        if training_data is not None and Path(training_data["path"]).is_file()
        else None
    )
    if training_data is not None and training_data_hash_before is None:
        raise FileNotFoundError(
            f"active-learning training dataset is missing: {training_data['path']}"
        )
    candidate_inputs = [
        {
            "structure_path": candidate.optimized_structure_path,
            "structure_sha256": structure_content_hash(candidate.optimized_structure_path),
        }
        for candidate in selected
    ]
    reference_inputs = [
        {
            "structure_path": str(spec.structure_path),
            "structure_sha256": structure_content_hash(spec.structure_path),
            "phase_id": spec.phase_id,
            "source": spec.source,
        }
        for spec in refinement.reference_phases
    ]
    references, reference_calculations, reference_seconds = _generate_reference_energies(
        al_cfg,
        refinement,
        formula_root / "dft_refinement",
        tracked_dft_relaxation_runner,
    )
    composition = parse_composition(formula)
    assert composition is not None
    missing = set(composition.elements) - set(references.elemental_energies_eV_per_atom)
    if missing:
        raise ValueError(f"reference-energy set lacks elemental references for {sorted(missing)}")

    refined: list[RelaxationResult] = []
    refinement_seconds = 0.0
    for index, candidate in enumerate(selected):
        started = time.monotonic()
        relaxation_result = tracked_dft_relaxation_runner(
            _dft_relaxation_config(
                candidate.optimized_structure_path,
                formula_root / "dft_refinement" / "candidates" / f"candidate-{index:03d}",
                al_cfg,
                refinement,
            )
        )
        refinement_seconds += time.monotonic() - started
        refined.append(_converged_dft_relaxation(candidate, relaxation_result))
    training_data_hash_after = (
        sha256_file(training_data["path"])
        if training_data is not None and Path(training_data["path"]).is_file()
        else None
    )
    if training_data_hash_before != training_data_hash_after:
        raise RuntimeError("independent DFT ranking modified the active-learning training dataset")
    training_dataset_unchanged = (
        training_data_hash_before == training_data_hash_after if training_data is not None else None
    )
    exploration.stability = score_stability(
        formula,
        refined,
        force_tol_eV_per_A=refinement.force_tolerance_eV_per_A,
        degeneracy_tol_eV_per_atom=float(
            config.exploration_kwargs.get("degeneracy_tol_eV_per_atom", 0.01)
        ),
        candidates=exploration.phase_candidates,
        ranking_mode=RankingMode.CONVEX_HULL,
        reference_energies=references,
        method_signature=refinement.method_signature,
        evidence_level=EvidenceLevel.CONVERGED_DFT,
    )
    evidence = result.active_learning_result or {}
    evidence["n_dft_calculations"] = (
        int(evidence.get("n_dft_calculations", 0)) + len(refined) + reference_calculations
    )
    evidence["node_hours"] = (
        float(evidence.get("node_hours", 0.0))
        + (refinement_seconds + reference_seconds) * config.compute_nodes / 3600.0
    )
    evidence["dft_refinement"] = {
        "backend": al_cfg.dft.backend,
        "method_signature": refinement.method_signature,
        "reference_set_id": references.identifier,
        "reference_energy_set": references.model_dump(mode="json"),
        "reference_calculations": reference_calculations,
        "candidate_calculations": len(refined),
        "ranking_mode": RankingMode.CONVEX_HULL,
        "candidate_acquisition": {
            key: value.model_dump(mode="json") for key, value in candidate_scores.items()
        },
    }
    evidence["independent_dft_ranking"] = {
        "model_identifier": _model_identifier(effective_al_cfg),
        "training_dataset_path": training_data.get("path") if training_data else None,
        "training_dataset_sha256_before": training_data_hash_before,
        "training_dataset_sha256_after": training_data_hash_after,
        "training_dataset_unchanged": training_dataset_unchanged,
        "independence_contract": (
            "final DFT refinement is stored separately and is not appended to the "
            "active-learning training dataset"
        ),
        "candidate_inputs": candidate_inputs,
        "reference_inputs": reference_inputs,
        "candidate_results": [
            {
                "source_structure_path": candidate.optimized_structure_path,
                "dft_structure_path": dft_result.optimized_structure_path,
                "dft_structure_sha256": structure_content_hash(dft_result.optimized_structure_path),
                "total_energy_eV": dft_result.final_energy_eV,
                "residual_force_eV_per_A": dft_result.final_max_force_eV_per_A,
            }
            for candidate, dft_result in zip(selected, refined, strict=True)
        ],
        "reference_set_id": references.identifier,
        "method_signature": refinement.method_signature,
        "ranking_report": (
            exploration.stability.model_dump(mode="json")
            if exploration.stability is not None
            else None
        ),
    }
    result.active_learning_result = evidence
    tracker.set(
        "independent_dft_ranking",
        "completed",
        **evidence["independent_dft_ranking"],
        reference_calculations=reference_calculations,
        candidate_calculations=len(refined),
    )
    return result
