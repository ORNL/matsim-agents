"""High-level wrapper: composition -> phase enumeration -> relaxation -> stability.

This module ties the discovery pieces together so that an agent (or a user
in the chat REPL) can dispatch a substantial atomistic exploration with a
single call.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Callable

from pydantic import BaseModel, Field

from matsim_agents.discovery.composition import Composition, parse_composition
from matsim_agents.discovery.phase_explorer import PhaseCandidate, enumerate_phases
from matsim_agents.discovery.stability import StabilityReport, score_stability
from matsim_agents.state import RelaxationResult
from matsim_agents.tools.relaxation import RelaxStructureInput, _run as _run_relaxation


class CompositionExplorationResult(BaseModel):
    """Aggregated output of :func:`explore_composition`."""

    composition: Composition
    phase_candidates: list[PhaseCandidate]
    relaxations: list[RelaxationResult] = Field(default_factory=list)
    stability: StabilityReport | None = None
    failures: list[str] = Field(default_factory=list)


def explore_composition(
    composition: str | Composition,
    logdir: str,
    mlp_checkpoint: str,
    *,
    output_dir: str,
    checkpoint: str | None = None,
    optimizer: str = "FIRE",
    maxiter: int = 200,
    maxstep: float = 1e-2,
    fmax: float = 0.02,
    mlp_device: str = "cuda",
    precision: str | None = None,
    mlp_precision: str | None = None,
    supercell: tuple[int, int, int] | None = None,
    min_atoms: int = 32,
    include_2d: bool = False,
    num_layers: int = 1,
    vacuum: float = 15.0,
    interlayer: float | None = None,
    n_orderings: int = 1,
    lattice_scales: Sequence[float] | None = None,
    ordering_seed: int = 0,
    on_phase_start: Callable[[PhaseCandidate], None] | None = None,
    on_phase_done: Callable[[PhaseCandidate, RelaxationResult], None] | None = None,
    relax_fn: Callable[[RelaxStructureInput], RelaxationResult] | None = None,
) -> CompositionExplorationResult:
    """Enumerate phases for a composition, relax each, and score stability.

    Parameters
    ----------
    composition:
        Either a formula string ("Li2MnO3") or a parsed :class:`Composition`.
    logdir, mlp_checkpoint, checkpoint, ...:
        Forwarded to :class:`RelaxStructureInput`.
    output_dir:
        Where seed structures, optimized structures, trajectories, and
        per-step logs are written.
    on_phase_start, on_phase_done:
        Optional callbacks for live progress reporting (e.g. in the chat REPL).
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

    candidates = enumerate_phases(
        composition,
        seeds_dir,
        supercell=supercell,
        min_atoms=min_atoms,
        include_2d=include_2d,
        num_layers=num_layers,
        vacuum=vacuum,
        interlayer=interlayer,
        n_orderings=n_orderings,
        lattice_scales=lattice_scales,
        ordering_seed=ordering_seed,
    )
    relax = relax_fn or _run_relaxation

    relaxations: list[RelaxationResult] = []
    failures: list[str] = []

    for cand in candidates:
        if on_phase_start is not None:
            on_phase_start(cand)
        try:
            result = relax(RelaxStructureInput(
                structure_path=cand.structure_path,
                logdir=logdir,
                mlp_checkpoint=mlp_checkpoint,
                checkpoint=checkpoint,
                optimizer=optimizer,
                maxiter=maxiter,
                maxstep=maxstep,
                fmax=fmax,
                precision=precision,
                mlp_precision=mlp_precision,
                mlp_device=mlp_device,
                output_dir=relax_dir,
            ))
            relaxations.append(result)
            if on_phase_done is not None:
                on_phase_done(cand, result)
        except Exception as exc:  # pragma: no cover - depends on HydraGNN env
            failures.append(f"{cand.phase}: {exc!s}")

    report: StabilityReport | None = None
    if relaxations:
        report = score_stability(composition.formula, relaxations)

    return CompositionExplorationResult(
        composition=composition,
        phase_candidates=candidates,
        relaxations=relaxations,
        stability=report,
        failures=failures,
    )
