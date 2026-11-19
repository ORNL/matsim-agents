"""High-level wrapper: composition -> seed generation -> relaxation -> stability.

This module ties the discovery pieces together so that an agent (or a user
in the chat REPL) can dispatch a substantial atomistic exploration with a
single call.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel, Field

from matsim_agents.backends.mlip.config import MLIPConfig, parse_mlip_config
from matsim_agents.backends.mlip.pool import MLIPWorkerPool
from matsim_agents.backends.mlip.relaxation import RelaxStructureInput, relax_with_mlip
from matsim_agents.backends.mlip.results import RelaxationResult
from matsim_agents.discovery.composition import Composition, parse_composition
from matsim_agents.discovery.seeds import (
    PhaseCandidate,
    SeedGenerationAction,
    generate_seed_action,
    generate_seeds,
)
from matsim_agents.discovery.stability import StabilityReport, score_stability

_RANKING_FORCE_TOL_EV_PER_A = 0.05
log = logging.getLogger(__name__)


class CompositionExplorationResult(BaseModel):
    """Aggregated output of :func:`explore_composition`."""

    composition: Composition
    phase_candidates: list[PhaseCandidate]
    relaxations: list[RelaxationResult] = Field(default_factory=list)
    stability: StabilityReport | None = None
    failures: list[str] = Field(default_factory=list)
    callback_failures: list[str] = Field(default_factory=list)
    relaxation_budget_exhausted: bool = False
    unrelaxed_candidate_paths: list[str] = Field(default_factory=list)
    ranking_failure: str | None = None
    outcome_class: Literal[
        "generation_failure",
        "relaxation_non_convergence",
        "ranking_failure",
        "seed_only",
        "usable_minimum",
    ] = "generation_failure"

    @property
    def candidate_counts(self) -> dict[str, int]:
        backend_errors = len(self.failures)
        nonconverged = sum(not result.converged for result in self.relaxations)
        return {
            "generated": len(self.phase_candidates),
            "attempted": len(self.relaxations) + backend_errors,
            "completed": len(self.relaxations),
            "converged": sum(result.converged for result in self.relaxations),
            "failed": backend_errors + nonconverged,
            "backend_errors": backend_errors,
            "nonconverged": nonconverged,
        }


def explore_composition(
    composition: str | Composition,
    *,
    mlip: MLIPConfig,
    output_dir: str,
    optimizer: str = "FIRE",
    maxiter: int = 200,
    maxstep: float = 1e-2,
    fmax: float = 0.02,
    relax_cell: bool = False,
    degeneracy_tol_eV_per_atom: float = 0.01,
    relative_increase_threshold: float | None = 0.2,
    n_random: int = 50,
    random_seed: int | None = None,
    generation_action: SeedGenerationAction | None = None,
    max_relaxations: int | None = None,
    on_relaxation_attempt: Callable[[], None] | None = None,
    on_phase_start: Callable[[PhaseCandidate], None] | None = None,
    on_phase_done: Callable[[PhaseCandidate, RelaxationResult], None] | None = None,
    relax_fn: Callable[[RelaxStructureInput], RelaxationResult] | None = None,
    mlip_pool: MLIPWorkerPool | None = None,
) -> CompositionExplorationResult:
    """Enumerate seeds for a composition, relax each, and score stability.

    Parameters
    ----------
    composition:
        Either a formula string ("Li2MnO3") or a parsed :class:`Composition`.
    mlip:
        MLIP backend and its model settings, forwarded to :class:`RelaxStructureInput`.
    output_dir:
        Where seed structures, optimized structures, trajectories, and
        per-step logs are written.
    n_random:
        Number of random pyXtal structures to draw as supplementary
        seeds for novelty / characterization (in addition to every
        applicable AFLOW prototype decoration). ``0`` disables.
    random_seed:
        Seed of the pyXtal search; required when ``n_random`` is positive.
    generation_action:
        Optional strategy-selected generator action. When supplied, its AFLOW,
        pyXtal, space-group, and formula-unit controls replace ``n_random`` and
        ``random_seed``. Every generated candidate is relaxed unless an explicit
        ``max_relaxations`` budget is set.
    max_relaxations:
        Cap on candidate relaxation attempts, including failures. Generation
        is unchanged; unrelaxed candidates are recorded explicitly.
    on_phase_start, on_phase_done:
        Optional callbacks for live progress reporting (e.g. in the chat REPL).
        Completion callback errors are logged and recorded separately from
        relaxation failures; successful relaxation results remain eligible.
    relax_fn:
        Override the relaxation backend (used by tests / stub mode).
    mlip_pool:
        Relax candidates concurrently on this pool (one worker per device of the
        ``mlip`` pool). Start callbacks run for every candidate before work is
        submitted; results and completion callbacks follow candidate order.
    """
    if relax_fn is not None and mlip_pool is not None:
        raise ValueError("pass either relax_fn or mlip_pool, not both")
    mlip = parse_mlip_config(mlip)
    if max_relaxations is not None and max_relaxations < 0:
        raise ValueError("max_relaxations must be non-negative")
    if generation_action is not None and (n_random != 50 or random_seed is not None):
        raise ValueError(
            "generation_action cannot be combined with n_random or random_seed; "
            "set those controls on the action"
        )

    if isinstance(composition, str):
        parsed = parse_composition(composition)
        if parsed is None:
            raise ValueError(f"Could not parse chemical composition: {composition!r}")
        composition = parsed

    if generation_action is not None:
        if generation_action.formula != composition.formula:
            raise ValueError(
                f"generation_action formula {generation_action.formula!r} does not match "
                f"requested composition {composition.formula!r}"
            )
        seeds_dir = os.path.join(
            output_dir, composition.formula, "seeds", generation_action.action_id
        )
        relax_dir = os.path.join(
            output_dir, composition.formula, "relaxed", generation_action.action_id
        )
    else:
        seeds_dir = os.path.join(output_dir, composition.formula, "seeds")
        relax_dir = os.path.join(output_dir, composition.formula, "relaxed")
    os.makedirs(relax_dir, exist_ok=True)

    if generation_action is None:
        candidates = generate_seeds(
            composition,
            seeds_dir,
            n_random=n_random,
            random_seed=random_seed,
        )
    else:
        candidates = generate_seed_action(generation_action, seeds_dir)
    relaxations: list[RelaxationResult] = []
    failures: list[str] = []
    callback_failures: list[str] = []
    unrelaxed_candidate_paths: list[str] = []

    def relax_input(cand: PhaseCandidate) -> RelaxStructureInput:
        return RelaxStructureInput(
            structure_path=cand.structure_path,
            mlip=mlip,
            optimizer=optimizer,
            maxiter=maxiter,
            maxstep=maxstep,
            fmax=fmax,
            relax_cell=relax_cell,
            relative_increase_threshold=relative_increase_threshold,
            output_dir=relax_dir,
        )

    def record_failure(cand: PhaseCandidate, exc: BaseException) -> None:
        tag = "seed"
        if cand.prototype_id:
            tag = cand.prototype_id
        elif cand.source == "random" and cand.space_group is not None:
            tag = f"pyxtal_sg{int(cand.space_group):03d}"
        failures.append(f"{tag}: {exc!s}")

    def record_success(cand: PhaseCandidate, result: RelaxationResult) -> None:
        relaxations.append(result)
        if on_phase_done is not None:
            try:
                on_phase_done(cand, result)
            except Exception as exc:
                callback_failures.append(f"{cand.structure_path}: {exc!s}")
                log.exception("Completion callback failed for %s", cand.structure_path)

    attempted = candidates
    if max_relaxations is not None and len(candidates) > max_relaxations:
        attempted = candidates[:max_relaxations]
        unrelaxed_candidate_paths = [item.structure_path for item in candidates[max_relaxations:]]
        log.warning(
            "Candidate relaxation budget exhausted for %s; %d seeds remain unrelaxed",
            composition.formula,
            len(unrelaxed_candidate_paths),
        )

    if mlip_pool is not None:
        inputs = []
        for cand in attempted:
            if on_phase_start is not None:
                on_phase_start(cand)
            if on_relaxation_attempt is not None:
                on_relaxation_attempt()
            inputs.append(relax_input(cand))
        for cand, outcome in zip(attempted, mlip_pool.relax(inputs), strict=True):
            if outcome.error is not None:
                record_failure(cand, outcome.error)
            else:
                record_success(cand, outcome.value)
    else:
        relax = relax_fn or relax_with_mlip
        for cand in attempted:
            if on_phase_start is not None:
                on_phase_start(cand)
            if on_relaxation_attempt is not None:
                on_relaxation_attempt()
            try:
                result = relax(relax_input(cand))
            except Exception as exc:  # pragma: no cover - depends on HydraGNN env
                record_failure(cand, exc)
                continue
            record_success(cand, result)

    report: StabilityReport | None = None
    ranking_failure: str | None = None
    if relaxations:
        try:
            report = score_stability(
                composition.formula,
                relaxations,
                force_tol_eV_per_A=_RANKING_FORCE_TOL_EV_PER_A,
                degeneracy_tol_eV_per_atom=degeneracy_tol_eV_per_atom,
                candidates=candidates,
            )
        except ValueError as exc:
            ranking_failure = str(exc)

    if not candidates:
        outcome_class = "generation_failure"
    elif ranking_failure is not None and any(
        result.converged and result.final_max_force_eV_per_A <= _RANKING_FORCE_TOL_EV_PER_A
        for result in relaxations
    ):
        outcome_class = "ranking_failure"
    elif report is None:
        outcome_class = "relaxation_non_convergence"
    else:
        outcome_class = "usable_minimum"

    return CompositionExplorationResult(
        composition=composition,
        phase_candidates=candidates,
        relaxations=relaxations,
        stability=report,
        failures=failures,
        callback_failures=callback_failures,
        relaxation_budget_exhausted=bool(unrelaxed_candidate_paths),
        unrelaxed_candidate_paths=unrelaxed_candidate_paths,
        ranking_failure=ranking_failure,
        outcome_class=outcome_class,
    )
