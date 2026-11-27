"""Build forward and reverse cluster reactions from barriers and prefactor evidence.

The barrier and, where used, the vibrational modes are inputs. This module only combines
them. Every prefactor has a declared basis:

* ``user_supplied``: a constant attempt frequency with a citation (experiment or calculation).
* ``harmonic``: temperature-dependent prefactor and vibrational free-energy correction from
  supplied minimum and saddle modes (see :mod:`harmonic_tst`).
* ``documented_fallback``: an assumed constant that is labelled as such; the result is never
  ``qualified``.

The reverse barrier follows from the forward barrier and the reaction energy unless an
independent reverse barrier is given. ``check_equilibrium_detailed_balance`` tests
``ln(k_f/k_r) = -(G_target - G_source)/kT`` only for closed pairs between states of equal
inventory. Reactions that exchange with a reservoir, change the vacancy/interstitial count, or
carry sources and sinks are excluded, because irradiation-driven terms do not obey detailed
balance. A passing check says the rates are mutually consistent with the supplied free
energies; it does not validate the barriers.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from matsim_agents.science.kinetics.cluster_dynamics import (
    ClusterError,
    ClusterNetwork,
    ClusterReaction,
    ClusterState,
    RateQualification,
)
from matsim_agents.science.kinetics.harmonic_tst import harmonic_tst_rate, reverse_barrier_eV
from matsim_agents.science.kinetics.rates import KB_EV_PER_K, RateError

PrefactorKind = Literal["user_supplied", "harmonic", "documented_fallback"]


class PrefactorBasis(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: PrefactorKind
    attempt_frequency_hz: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    source_freqs_hz: tuple[float, ...] = ()
    saddle_freqs_hz: tuple[float, ...] = ()
    target_freqs_hz: tuple[float, ...] = ()
    citation: str | None = None

    @model_validator(mode="after")
    def _consistent(self) -> PrefactorBasis:
        if self.kind == "harmonic":
            if not (self.source_freqs_hz and self.saddle_freqs_hz and self.target_freqs_hz):
                raise ValueError("harmonic needs source, saddle and target modes")
            if self.attempt_frequency_hz is not None:
                raise ValueError("harmonic prefactors are computed, not supplied")
        else:
            if self.attempt_frequency_hz is None:
                raise ValueError(f"{self.kind} needs attempt_frequency_hz")
            if not (self.citation and self.citation.strip()):
                raise ValueError(f"{self.kind} needs a citation or stated basis")
        return self


class ConstructedReaction(BaseModel):
    model_config = ConfigDict(frozen=True)

    reaction: ClusterReaction
    barrier_eV: float
    log_rate_sigma: float | None
    prefactor_kind: PrefactorKind
    assumptions: tuple[str, ...]


class PairReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    forward: ConstructedReaction
    reverse: ConstructedReaction
    reverse_barrier_source: Literal["from_reaction_energy", "supplied"]
    temperature_K: float
    qualification: RateQualification


def _qualification(evidence_level: RateQualification, basis: PrefactorBasis) -> RateQualification:
    if basis.kind == "documented_fallback" and evidence_level == "qualified":
        return "exploratory"
    return evidence_level


def _construct(
    network: ClusterNetwork,
    mechanism: str,
    source: ClusterState,
    target: ClusterState,
    barrier_eV: float,
    basis: PrefactorBasis,
    modes: tuple[Sequence[float], Sequence[float]] | None,
    evidence_ref: str,
    qualification: RateQualification,
    multiplicity: int,
    barrier_sigma_eV: float | None,
) -> ConstructedReaction:
    notes: list[str] = []
    sigma = None
    if basis.kind == "harmonic":
        assert modes is not None
        try:
            h = harmonic_tst_rate(
                barrier_eV,
                modes[0],
                modes[1],
                network.temperature_K,
                multiplicity=multiplicity,
                barrier_sigma_eV=barrier_sigma_eV,
            )
        except RateError as exc:
            raise ClusterError(str(exc)) from exc
        reaction = ClusterReaction(
            mechanism=mechanism,
            source_key=source.key,
            target_key=target.key,
            evidence_ref=evidence_ref,
            qualification=qualification,
            rate_per_s=h.rate_per_s,
            table_temperature_K=network.temperature_K,
            multiplicity=multiplicity,
        )
        notes.extend(h.assumptions)
        sigma = h.log_rate_sigma
    else:
        assert basis.attempt_frequency_hz is not None
        reaction = ClusterReaction(
            mechanism=mechanism,
            source_key=source.key,
            target_key=target.key,
            evidence_ref=evidence_ref,
            qualification=qualification,
            barrier_eV=barrier_eV,
            attempt_frequency_hz=basis.attempt_frequency_hz,
            multiplicity=multiplicity,
        )
        notes.append(
            f"constant attempt frequency {basis.attempt_frequency_hz:g} Hz: {basis.citation}"
        )
        if basis.kind == "documented_fallback":
            notes.append("assumed prefactor, not computed; result is exploratory")
        if barrier_sigma_eV is not None:
            sigma = barrier_sigma_eV / (KB_EV_PER_K * network.temperature_K)
    if barrier_sigma_eV is None:
        notes.append("barrier uncertainty not supplied; it is unknown, not zero")
    return ConstructedReaction(
        reaction=reaction,
        barrier_eV=barrier_eV,
        log_rate_sigma=sigma,
        prefactor_kind=basis.kind,
        assumptions=tuple(notes),
    )


def add_reaction_pair(
    network: ClusterNetwork,
    source: ClusterState,
    target: ClusterState,
    *,
    forward_mechanism: str,
    reverse_mechanism: str,
    forward_barrier_eV: float,
    basis: PrefactorBasis,
    evidence_ref: str,
    evidence_level: RateQualification,
    reaction_energy_eV: float | None = None,
    reverse_barrier_eV_supplied: float | None = None,
    multiplicity: int = 1,
    reverse_multiplicity: int | None = None,
    barrier_sigma_eV: float | None = None,
) -> PairReport:
    """Add both directions of a closed event; the states are added if missing.

    Give ``reaction_energy_eV`` (``E_target - E_source``) to derive the reverse barrier, or an
    independent ``reverse_barrier_eV_supplied``. A pair that needs neither is rejected.
    """
    if source.inventory != target.inventory:
        raise ClusterError(
            "a reversible pair needs equal inventories; use add_reaction for open events"
        )
    if reverse_barrier_eV_supplied is None and reaction_energy_eV is None:
        raise ClusterError("give a reaction energy or an independent reverse barrier")
    if reverse_barrier_eV_supplied is not None:
        reverse_barrier, origin = reverse_barrier_eV_supplied, "supplied"
    else:
        assert reaction_energy_eV is not None
        try:
            reverse_barrier = reverse_barrier_eV(forward_barrier_eV, reaction_energy_eV)
        except RateError as exc:
            raise ClusterError(str(exc)) from exc
        origin = "from_reaction_energy"
    level = _qualification(evidence_level, basis)
    network.add_state(source)
    network.add_state(target)
    forward = _construct(
        network,
        forward_mechanism,
        source,
        target,
        forward_barrier_eV,
        basis,
        (basis.source_freqs_hz, basis.saddle_freqs_hz),
        f"{evidence_ref}:forward",
        level,
        multiplicity,
        barrier_sigma_eV,
    )
    reverse = _construct(
        network,
        reverse_mechanism,
        target,
        source,
        reverse_barrier,
        basis,
        (basis.target_freqs_hz, basis.saddle_freqs_hz),
        f"{evidence_ref}:reverse",
        level,
        reverse_multiplicity or multiplicity,
        barrier_sigma_eV,
    )
    network.add_reaction(forward.reaction)
    network.add_reaction(reverse.reaction)
    return PairReport(
        forward=forward,
        reverse=reverse,
        reverse_barrier_source=origin,
        temperature_K=network.temperature_K,
        qualification=level,
    )


class DetailedBalanceReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    checked_pairs: tuple[str, ...]
    violations: tuple[str, ...]
    max_abs_log_error: float
    excluded_driven_terms: tuple[str, ...]
    excluded_reactions: tuple[str, ...]
    unpaired: tuple[str, ...]
    zero_rate_pairs: tuple[str, ...]
    scope: str

    @property
    def consistent(self) -> bool:
        return not self.violations


def check_equilibrium_detailed_balance(
    network: ClusterNetwork,
    free_energies_eV: Mapping[str, float],
    *,
    tolerance: float = 1e-6,
) -> DetailedBalanceReport:
    """Pairwise ``ln(k_f/k_r)`` against the supplied state free energies, closed pairs only."""
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ClusterError("tolerance must be finite and positive")
    kt = KB_EV_PER_K * network.temperature_K
    reactions = {(r.source_key, r.target_key): r for r in network.reactions}
    checked: list[str] = []
    bad: list[str] = []
    excluded: list[str] = []
    unpaired: list[str] = []
    zero: list[str] = []
    worst = 0.0
    for (a, b), fwd in sorted(reactions.items()):
        sa, sb = network.state(a), network.state(b)
        if fwd.open_terms or sa.inventory != sb.inventory:
            excluded.append(fwd.reaction_id)
            continue
        rev = reactions.get((b, a))
        if rev is None:
            unpaired.append(fwd.reaction_id)
            continue
        if a > b:
            continue
        for key in (a, b):
            if key not in free_energies_eV or not math.isfinite(free_energies_eV[key]):
                raise ClusterError(f"missing free energy for state {key!r}")
        kf, kr = network.reaction_rate(fwd), network.reaction_rate(rev)
        label = f"{fwd.reaction_id} <-> {rev.mechanism}"
        if kf == 0 or kr == 0:
            zero.append(label)
            continue
        expected = -(free_energies_eV[b] - free_energies_eV[a]) / kt
        error = abs(math.log(kf / kr) - expected)
        worst = max(worst, error)
        checked.append(label)
        if error > tolerance:
            bad.append(f"{label}: |ln(kf/kr) - expected| = {error:.3g}")
    driven = tuple(
        [f"source {s.state_key} ({s.reservoir})" for s in network.sources]
        + [f"sink {s.state_key} ({s.sink})" for s in network.sinks]
    )
    return DetailedBalanceReport(
        checked_pairs=tuple(checked),
        violations=tuple(bad),
        max_abs_log_error=worst,
        excluded_driven_terms=driven,
        excluded_reactions=tuple(excluded),
        unpaired=tuple(unpaired),
        zero_rate_pairs=tuple(zero),
        scope=(
            "closed pairs of equal inventory at the network temperature; sources, sinks and "
            "reservoir-exchanging events are driven and are not tested"
        ),
    )


__all__ = [
    "ConstructedReaction",
    "DetailedBalanceReport",
    "PairReport",
    "PrefactorBasis",
    "add_reaction_pair",
    "check_equilibrium_detailed_balance",
]
