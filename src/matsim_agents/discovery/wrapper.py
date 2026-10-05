"""High-level wrapper: composition -> seed generation -> relaxation -> stability.

This module ties the discovery pieces together so that an agent (or a user
in the chat REPL) can dispatch a substantial atomistic exploration with a
single call.
"""

from __future__ import annotations

import logging
import os
from typing import Callable, Literal

from pydantic import BaseModel, Field

from matsim_agents.discovery.composition import Composition, parse_composition
from matsim_agents.discovery.seeds import PhaseCandidate, generate_seeds
from matsim_agents.discovery.stability import StabilityReport, score_stability
from matsim_agents.orchestration.state import RelaxationResult
from matsim_agents.backends.mlip.relaxation import RelaxStructureInput, _run as _run_relaxation


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
        return {
            "generated": len(self.phase_candidates),
            "attempted": len(self.relaxations) + len(self.failures),
            "completed": len(self.relaxations),
            "converged": sum(result.converged for result in self.relaxations),
            "failed": len(self.failures),
        }


def explore_composition(
    composition: str | Composition,
    logdir: str | None = None,
    hydragnn_branch_mlp_checkpoint: str | None = None,
    hydragnn_inference_head: str | int | None = None,
    *,
    output_dir: str,
    mlip_backend: str = "hydragnn",
    uma_model_name: str = "uma-s-1p1",
    uma_task: str = "omat",
    mace_family: str = "mace_mp",
    mace_model: str = "medium",
    mace_dispersion: bool = False,
    checkpoint: str | None = None,
    optimizer: str = "FIRE",
    maxiter: int = 200,
    maxstep: float = 1e-2,
    fmax: float = 0.02,
    degeneracy_tol_eV_per_atom: float = 0.01,
    relative_increase_threshold: float = 0.05,
    mlp_device: str = "cuda",
    precision: str | None = None,
    mlp_precision: str | None = None,
    n_random: int = 50,
    random_seed: int = 0,
    on_phase_start: Callable[[PhaseCandidate], None] | None = None,
    on_phase_done: Callable[[PhaseCandidate, RelaxationResult], None] | None = None,
    relax_fn: Callable[[RelaxStructureInput], RelaxationResult] | None = None,
) -> CompositionExplorationResult:
    """Enumerate seeds for a composition, relax each, and score stability.

    Parameters
    ----------
    composition:
        Either a formula string ("Li2MnO3") or a parsed :class:`Composition`.
    logdir, hydragnn_branch_mlp_checkpoint, checkpoint, ...:
        Forwarded to :class:`RelaxStructureInput`.
    output_dir:
        Where seed structures, optimized structures, trajectories, and
        per-step logs are written.
    n_random:
        Number of random pyXtal structures to draw as supplementary
        seeds for novelty / characterization (in addition to every
        applicable AFLOW prototype decoration). ``0`` disables.
    random_seed:
        Seed for the pyXtal RNG (reproducibility).
    on_phase_start, on_phase_done:
        Optional callbacks for live progress reporting (e.g. in the chat REPL).
        Completion callback errors are logged and recorded separately from
        relaxation failures; successful relaxation results remain eligible.
    relax_fn:
        Override the relaxation backend (used by tests / stub mode).
    """
    if isinstance(composition, str):
        parsed = parse_composition(composition)
        if parsed is None:
            raise ValueError(f"Could not parse chemical composition: {composition!r}")
        composition = parsed

    seeds_dir = os.path.join(output_dir, composition.formula, "seeds")
    relax_dir = os.path.join(output_dir, composition.formula, "relaxed")
    os.makedirs(relax_dir, exist_ok=True)

    candidates = generate_seeds(
        composition,
        seeds_dir,
        n_random=n_random,
        random_seed=random_seed,
    )
    relax = relax_fn or _run_relaxation

    relaxations: list[RelaxationResult] = []
    failures: list[str] = []
    callback_failures: list[str] = []

    for cand in candidates:
        if on_phase_start is not None:
            on_phase_start(cand)
        try:
            result = relax(
                RelaxStructureInput(
                    structure_path=cand.structure_path,
                    mlip_backend=mlip_backend,
                    logdir=logdir,
                    hydragnn_branch_mlp_checkpoint=hydragnn_branch_mlp_checkpoint,
                    hydragnn_inference_head=hydragnn_inference_head,
                    checkpoint=checkpoint,
                    uma_model_name=uma_model_name,
                    uma_task=uma_task,
                    mace_family=mace_family,
                    mace_model=mace_model,
                    mace_dispersion=mace_dispersion,
                    optimizer=optimizer,
                    maxiter=maxiter,
                    maxstep=maxstep,
                    fmax=fmax,
                    relative_increase_threshold=relative_increase_threshold,
                    precision=precision,
                    mlp_precision=mlp_precision,
                    mlp_device=mlp_device,
                    output_dir=relax_dir,
                )
            )
            relaxations.append(result)
        except Exception as exc:  # pragma: no cover - depends on HydraGNN env
            tag = "seed"
            if cand.prototype_id:
                tag = cand.prototype_id
            elif cand.source == "random" and cand.space_group is not None:
                tag = f"pyxtal_sg{int(cand.space_group):03d}"
            failures.append(f"{tag}: {exc!s}")
            continue
        if on_phase_done is not None:
            try:
                on_phase_done(cand, result)
            except Exception as exc:
                callback_failures.append(f"{cand.structure_path}: {exc!s}")
                log.exception("Completion callback failed for %s", cand.structure_path)

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
        ranking_failure=ranking_failure,
        outcome_class=outcome_class,
    )
