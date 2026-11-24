"""Plan, execute (when authorized) and rank competing collinear magnetic states (M6).

The ranking covers only the states that were evaluated under one identical numerical method.
It never claims a global magnetic ground state.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from matsim_agents.discovery.magnetic_results import (
    ClassificationTolerances,
    MagneticResult,
    classify_moments,
    spin_reversed_equivalent,
)
from matsim_agents.discovery.magnetism import MagneticConfiguration

NOT_GLOBAL_GROUND_STATE = (
    "lowest energy among the evaluated states only; unsampled magnetic arrangements, "
    "noncollinear order and other structures were not considered"
)


def method_fingerprint(method: dict[str, Any]) -> str:
    """Hash of every numerical setting that makes energies comparable (functional, cutoffs,
    k-mesh, smearing, pseudopotentials, code version, ...)."""
    payload = json.dumps(method, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _reversed_equal(a: MagneticConfiguration, b: MagneticConfiguration) -> bool:
    if a.spin_polarized != b.spin_polarized or set(a.atom_ids) != set(b.atom_ids):
        return False
    if not a.spin_polarized:
        return True
    ma = dict(zip(a.atom_ids, a.initial_moments_muB or (), strict=True))
    mb = dict(zip(b.atom_ids, b.initial_moments_muB or (), strict=True))
    ids = sorted(ma)
    x = np.array([ma[i] for i in ids])
    y = np.array([mb[i] for i in ids])
    return bool(np.allclose(x, -y, atol=1e-9))


@dataclass(frozen=True)
class StateRequest:
    config: MagneticConfiguration
    method_hash: str

    @property
    def key(self) -> str:
        return f"{self.config.structure_id}:{self.config.fingerprint()}:{self.method_hash}"


@dataclass(frozen=True)
class CompetitionPlan:
    requests: tuple[StateRequest, ...]
    skipped: tuple[tuple[str, str], ...]  # (config_id, reason)


def plan_competition(
    configs: Iterable[MagneticConfiguration],
    method: dict[str, Any],
    *,
    already_done: Iterable[str] = (),
) -> CompetitionPlan:
    """Select the calculations to run: no repeated fingerprint, no global spin-reversal twin,
    and nothing already evaluated for the same structure, state and method."""
    method_hash = method_fingerprint(method)
    done = set(already_done)
    kept: list[StateRequest] = []
    skipped: list[tuple[str, str]] = []
    for config in configs:
        request = StateRequest(config, method_hash)
        if request.key in done:
            skipped.append((config.config_id, "already evaluated with this method"))
        elif any(request.key == r.key for r in kept):
            skipped.append((config.config_id, "duplicate fingerprint"))
        elif any(
            r.config.structure_id == config.structure_id and _reversed_equal(r.config, config)
            for r in kept
        ):
            skipped.append((config.config_id, "global spin reversal of a requested state"))
        else:
            kept.append(request)
    return CompetitionPlan(tuple(kept), tuple(skipped))


@dataclass(frozen=True)
class StateEvaluation:
    config: MagneticConfiguration
    method_hash: str
    energy_eV: float | None
    n_atoms: int
    result: MagneticResult | None
    composition: str = ""
    notes: tuple[str, ...] = field(default=())

    @property
    def converged_fingerprint(self) -> str | None:
        if self.result is None or not self.result.local_moments_muB:
            return None
        return self.config.with_converged(self.result.local_moments_muB).fingerprint()


def run_pending(
    plan: CompetitionPlan,
    runner: Callable[[StateRequest], StateEvaluation],
    *,
    authorized: bool,
) -> list[StateEvaluation]:
    """Execute ``runner`` per request only when the caller explicitly authorizes it."""
    if not authorized:
        raise PermissionError(
            "running DFT calculations requires explicit authorization (authorized=True); "
            "use plan_competition() to prepare inputs only"
        )
    return [runner(request) for request in plan.requests]


@dataclass(frozen=True)
class RankedState:
    config_id: str
    classification: str
    energy_eV: float
    relative_energy_meV_per_atom: float
    within_tie_tolerance: bool
    duplicate_of: str | None = None


@dataclass(frozen=True)
class CompetitionReport:
    method_hash: str | None
    ranking: tuple[RankedState, ...]
    excluded: tuple[tuple[str, str], ...]
    missing_evidence: tuple[str, ...]
    statement: str = NOT_GLOBAL_GROUND_STATE


def rank_states(
    evaluations: Sequence[StateEvaluation],
    *,
    tie_tolerance_meV_per_atom: float = 1.0,
    tol: ClassificationTolerances | None = None,
) -> CompetitionReport:
    """Rank converged states compared under one method; report what could not be compared."""
    excluded: list[tuple[str, str]] = []
    missing: list[str] = []
    if not evaluations:
        return CompetitionReport(None, (), (), ("no evaluations supplied",))

    methods = {e.method_hash for e in evaluations}
    reference = evaluations[0].method_hash
    if len(methods) > 1:
        missing.append(
            f"{len(methods)} different numerical methods supplied; only the method of the "
            "first evaluation is ranked"
        )
    comparable = []
    for e in evaluations:
        cid = e.config.config_id
        if e.method_hash != reference:
            excluded.append((cid, "different numerical method"))
        elif e.energy_eV is None or e.result is None or not e.result.converged:
            excluded.append((cid, "not converged or no energy"))
            missing.append(f"{cid}: converged energy missing")
        else:
            comparable.append(e)

    sizes = {(e.n_atoms, e.composition) for e in comparable}
    if len(sizes) > 1:
        missing.append("states differ in size or composition; absolute energies are not comparable")
        return CompetitionReport(reference, (), tuple(excluded), tuple(missing))

    unique: list[StateEvaluation] = []
    twin_of: dict[str, str] = {}
    for e in sorted(comparable, key=lambda x: x.energy_eV):
        match = next(
            (u for u in unique if spin_reversed_equivalent(u.result, e.result)),
            None,
        )
        if match is None:
            unique.append(e)
        else:
            twin_of[e.config.config_id] = match.config.config_id
    for cid, original in twin_of.items():
        excluded.append((cid, f"converged state duplicates {original} (spin reversal allowed)"))

    ranking: list[RankedState] = []
    if unique:
        e0 = unique[0].energy_eV
        for e in unique:
            delta = (e.energy_eV - e0) * 1000.0 / e.n_atoms
            ranking.append(
                RankedState(
                    config_id=e.config.config_id,
                    classification=classify_moments(e.result, tol),
                    energy_eV=float(e.energy_eV),
                    relative_energy_meV_per_atom=float(delta),
                    within_tie_tolerance=delta <= tie_tolerance_meV_per_atom,
                )
            )
        tied = [r.config_id for r in ranking if r.within_tie_tolerance]
        if len(tied) > 1:
            missing.append(
                f"states {tied} lie within {tie_tolerance_meV_per_atom} meV/atom; "
                "the order is not resolved"
            )
    collapsed = [r.config_id for r in ranking if r.classification == "nonmagnetic"]
    if collapsed and len(collapsed) < len(ranking):
        missing.append(f"states {collapsed} collapsed to nonmagnetic during the calculation")
    return CompetitionReport(reference, tuple(ranking), tuple(excluded), tuple(missing))
