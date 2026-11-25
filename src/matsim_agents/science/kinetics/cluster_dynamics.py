"""Linear cluster-dynamics bridge on states ``(n_vacancies, n_helium, n_hydrogen, environment)``.

A population vector ``N`` over cluster states obeys the master equation
``dN/dt = A N + g`` where ``A`` collects unimolecular reactions (and first-order sinks) and
``g`` collects explicit source terms. This is the transition-network <-> cluster-dynamics
bridge: it consumes rates, it never creates them.

Rules enforced here:

* A reaction carries either a table rate (with the temperature it was tabulated at and an
  evidence reference) or barrier evidence with an explicit attempt frequency. A reaction with
  neither is rejected; there is no default rate, barrier or prefactor.
* Closed reactions conserve the inventory ``(V, He, H)``. A reaction that exchanges gas or
  vacancies with a reservoir must carry explicit ``OpenSystemTerm`` records that equal the
  change. Sources and sinks name their reservoir and basis.
* Environment labels are part of the state identity. Closed reactions between different
  environments represent rearrangements (for example ``trap_mutation``).
* Bubble pressure, bubble morphology, bubble bursting and mean-field bimolecular coupling are
  flagged future work and are rejected, not approximated.

Uncertainty is propagated by Monte Carlo over barrier, table-rate and source errors whose
widths the caller supplies. Model-form error and missing channels are not propagated and are
reported as such.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator
from scipy.linalg import expm

from matsim_agents.core.seeding import numpy_rng
from matsim_agents.science.kinetics.gas_mechanisms import (
    GasMechanismError,
    OpenSystemTerm,
    check_event_inventory,
    get_gas_mechanism,
)
from matsim_agents.science.kinetics.graph import TransitionNetwork
from matsim_agents.science.kinetics.rates import RateError, event_rate, htst_rate
from matsim_agents.science.kinetics.uncertainty import _quantile

FUTURE_WORK: dict[str, str] = {
    "bubble_pressure": "equation of state and overpressure of gas-filled cavities",
    "bubble_morphology": "cavity shape, faceting and surface-energy anisotropy",
    "bubble_bursting": "cavity rupture and gas release at surfaces or boundaries",
    "mean_field_bimolecular_coupling": "concentration-dependent association rates between "
    "evolving cluster populations",
}

RateQualification = Literal["synthetic", "exploratory", "qualified"]


class ClusterError(ValueError):
    """The requested reaction, source or solve is not justified by supplied evidence."""


class FutureWorkError(ClusterError):
    """The request needs a model that is flagged as future work."""


def reject_future_work(name: str) -> None:
    if name in FUTURE_WORK:
        raise FutureWorkError(f"{name} is future work: {FUTURE_WORK[name]}")


def capabilities() -> dict[str, str]:
    """Implementation status of each feature, for reports."""
    status = {
        "linear_master_equation": "implemented",
        "closed_and_open_inventory_accounting": "implemented",
        "transition_network_bridge": "implemented",
        "uncertainty_propagation": "implemented",
    }
    status.update({name: "FUTURE_WORK" for name in FUTURE_WORK})
    return status


class ClusterState(BaseModel):
    model_config = ConfigDict(frozen=True)

    n_vacancies: StrictInt = Field(ge=0)
    n_helium: StrictInt = Field(ge=0)
    n_hydrogen: StrictInt = Field(ge=0)
    environment: str = Field(min_length=1)

    @property
    def key(self) -> str:
        return f"V{self.n_vacancies}He{self.n_helium}H{self.n_hydrogen}@{self.environment}"

    @property
    def inventory(self) -> dict[str, int]:
        raw = {"V": self.n_vacancies, "He": self.n_helium, "H": self.n_hydrogen}
        return {k: v for k, v in raw.items() if v}


class ClusterReaction(BaseModel):
    """One unimolecular event with its rate evidence."""

    model_config = ConfigDict(frozen=True)

    mechanism: str = Field(min_length=1)
    source_key: str = Field(min_length=1)
    target_key: str = Field(min_length=1)
    evidence_ref: str = Field(min_length=1)
    qualification: RateQualification
    rate_per_s: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    table_temperature_K: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    barrier_eV: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    attempt_frequency_hz: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    multiplicity: StrictInt = Field(default=1, ge=1)
    open_terms: tuple[OpenSystemTerm, ...] = ()

    @property
    def reaction_id(self) -> str:
        return f"{self.mechanism}:{self.source_key}->{self.target_key}"

    @property
    def barrier_based(self) -> bool:
        return self.barrier_eV is not None

    @model_validator(mode="after")
    def _one_rate_basis(self) -> ClusterReaction:
        table = self.rate_per_s is not None
        barrier = self.barrier_eV is not None
        if table == barrier:
            raise ValueError("give either a table rate or barrier evidence, not both or neither")
        if table and self.table_temperature_K is None:
            raise ValueError("a table rate needs the temperature it was tabulated at")
        if barrier and self.attempt_frequency_hz is None:
            raise ValueError("barrier evidence needs an explicit attempt frequency")
        if self.source_key == self.target_key:
            raise ValueError("a reaction must change the cluster state")
        return self


class SourceTerm(BaseModel):
    """Explicit generation of one cluster state from a named reservoir."""

    model_config = ConfigDict(frozen=True)

    state_key: str = Field(min_length=1)
    rate_per_s: float = Field(gt=0, allow_inf_nan=False)
    reservoir: str = Field(min_length=1)
    basis: str = Field(min_length=1)


class SinkTerm(BaseModel):
    """Explicit first-order removal of one cluster state to a named sink."""

    model_config = ConfigDict(frozen=True)

    state_key: str = Field(min_length=1)
    rate_per_s: float = Field(gt=0, allow_inf_nan=False)
    sink: str = Field(min_length=1)
    basis: str = Field(min_length=1)


class ClusterUncertainty(BaseModel):
    model_config = ConfigDict(frozen=True)

    n_samples: int
    mean: float
    std: float
    p05: float
    p95: float
    barrier_sigma_eV: float
    table_log10_sigma: float
    source_log10_sigma: float
    propagated: tuple[str, ...]
    not_propagated: tuple[str, ...]


class ClusterNetwork:
    def __init__(self, temperature_K: float, population_unit: str = "clusters") -> None:
        if not math.isfinite(temperature_K) or temperature_K <= 0:
            raise ClusterError("temperature must be finite and positive")
        self.temperature_K = temperature_K
        self.population_unit = population_unit
        self._states: dict[str, ClusterState] = {}
        self._reactions: dict[str, ClusterReaction] = {}
        self._sources: list[SourceTerm] = []
        self._sinks: list[SinkTerm] = []

    @property
    def state_keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._states))

    @property
    def reactions(self) -> tuple[ClusterReaction, ...]:
        return tuple(self._reactions[k] for k in sorted(self._reactions))

    @property
    def sources(self) -> tuple[SourceTerm, ...]:
        return tuple(self._sources)

    @property
    def sinks(self) -> tuple[SinkTerm, ...]:
        return tuple(self._sinks)

    def state(self, key: str) -> ClusterState:
        try:
            return self._states[key]
        except KeyError:
            raise ClusterError(f"unknown cluster state {key!r}") from None

    # -- construction ------------------------------------------------------- #

    def add_state(self, state: ClusterState) -> str:
        self._states.setdefault(state.key, state)
        return state.key

    def add_reaction(self, reaction: ClusterReaction) -> None:
        reject_future_work(reaction.mechanism)
        try:
            get_gas_mechanism(reaction.mechanism)
        except GasMechanismError as exc:
            raise ClusterError(str(exc)) from exc
        source = self.state(reaction.source_key)
        target = self.state(reaction.target_key)
        try:
            check_event_inventory(
                reaction.mechanism, source.inventory, target.inventory, reaction.open_terms
            )
        except GasMechanismError as exc:
            raise ClusterError(str(exc)) from exc
        if reaction.reaction_id in self._reactions:
            raise ClusterError(f"duplicate reaction {reaction.reaction_id}")
        if (
            reaction.rate_per_s is not None
            and reaction.table_temperature_K is not None
            and not math.isclose(reaction.table_temperature_K, self.temperature_K, rel_tol=1e-9)
        ):
            raise ClusterError(
                f"table rate is for {reaction.table_temperature_K} K but the network is at "
                f"{self.temperature_K} K; supply barrier evidence or a rate at this temperature"
            )
        self._reactions[reaction.reaction_id] = reaction

    def add_source(self, term: SourceTerm) -> None:
        self.state(term.state_key)
        self._sources.append(term)

    def add_sink(self, term: SinkTerm) -> None:
        self.state(term.state_key)
        self._sinks.append(term)

    # -- equations ---------------------------------------------------------- #

    def reaction_rate(self, reaction: ClusterReaction, barrier_shift_eV: float = 0.0) -> float:
        if reaction.rate_per_s is not None:
            return reaction.rate_per_s
        barrier = max(0.0, (reaction.barrier_eV or 0.0) + barrier_shift_eV)
        assert reaction.attempt_frequency_hz is not None
        return htst_rate(
            barrier, self.temperature_K, reaction.attempt_frequency_hz, reaction.multiplicity
        )

    def _generator(
        self,
        rates: Mapping[str, float] | None = None,
        sink_rates: Sequence[float] | None = None,
        source_rates: Sequence[float] | None = None,
    ) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
        keys = self.state_keys
        index = {k: i for i, k in enumerate(keys)}
        matrix = np.zeros((len(keys), len(keys)))
        gen = np.zeros(len(keys))
        for r in self.reactions:
            k = rates[r.reaction_id] if rates is not None else self.reaction_rate(r)
            matrix[index[r.source_key], index[r.source_key]] -= k
            matrix[index[r.target_key], index[r.source_key]] += k
        for n, s in enumerate(self._sinks):
            matrix[index[s.state_key], index[s.state_key]] -= (
                sink_rates[n] if sink_rates is not None else s.rate_per_s
            )
        for n, t in enumerate(self._sources):
            gen[index[t.state_key]] += source_rates[n] if source_rates is not None else t.rate_per_s
        return keys, matrix, gen

    def _vector(self, populations: Mapping[str, float], keys: Sequence[str]) -> np.ndarray:
        unknown = set(populations) - set(keys)
        if unknown:
            raise ClusterError(f"populations for unknown states: {sorted(unknown)}")
        if any(v < 0 or not math.isfinite(v) for v in populations.values()):
            raise ClusterError("populations must be finite and nonnegative")
        return np.array([populations.get(k, 0.0) for k in keys])

    def _evolve(
        self,
        populations: Mapping[str, float],
        t_s: float,
        rates: Mapping[str, float] | None = None,
        sink_rates: Sequence[float] | None = None,
        source_rates: Sequence[float] | None = None,
    ) -> dict[str, float]:
        if not math.isfinite(t_s) or t_s < 0:
            raise ClusterError("time must be finite and nonnegative")
        keys, matrix, gen = self._generator(rates, sink_rates, source_rates)
        n = len(keys)
        augmented = np.zeros((n + 1, n + 1))
        augmented[:n, :n] = matrix
        augmented[:n, n] = gen
        x0 = np.append(self._vector(populations, keys), 1.0)
        x = expm(augmented * t_s) @ x0
        return {k: 0.0 if abs(v) < 1e-12 else float(v) for k, v in zip(keys, x[:n], strict=True)}

    def evolve(self, populations: Mapping[str, float], t_s: float) -> dict[str, float]:
        """Exact solution of ``dN/dt = A N + g`` by matrix exponential."""
        return self._evolve(populations, t_s)

    def steady_state(self) -> dict[str, float]:
        """``N = -A^-1 g``; raises when the network has no sink for the sourced populations."""
        keys, matrix, gen = self._generator()
        if not gen.any():
            return dict.fromkeys(keys, 0.0)
        try:
            if np.linalg.cond(matrix) > 1e12:
                raise np.linalg.LinAlgError("singular")
            solution = np.linalg.solve(matrix, -gen)
        except np.linalg.LinAlgError:
            raise ClusterError("no steady state: sourced populations have no sink path") from None
        return dict(zip(keys, (float(v) for v in solution), strict=True))

    # -- inventory ---------------------------------------------------------- #

    def inventory(self, populations: Mapping[str, float]) -> dict[str, float]:
        total: Counter = Counter()
        for key, n in populations.items():
            for species, count in self.state(key).inventory.items():
                total[species] += n * count
        return dict(sorted(total.items()))

    def inventory_balance(self, populations: Mapping[str, float]) -> dict[str, dict[str, float]]:
        """Rate of change of each species, split by cause; closed reactions contribute zero."""
        parts: dict[str, Counter] = {
            "open_reactions": Counter(),
            "sources": Counter(),
            "sinks": Counter(),
        }
        for r in self.reactions:
            if not r.open_terms:
                continue
            flux = self.reaction_rate(r) * populations.get(r.source_key, 0.0)
            for term in r.open_terms:
                parts["open_reactions"][term.species] += flux * term.delta
        for t in self._sources:
            for species, count in self.state(t.state_key).inventory.items():
                parts["sources"][species] += t.rate_per_s * count
        for s in self._sinks:
            for species, count in self.state(s.state_key).inventory.items():
                parts["sinks"][species] -= s.rate_per_s * populations.get(s.state_key, 0.0) * count
        out = {name: dict(sorted(c.items())) for name, c in parts.items()}
        total: Counter = Counter()
        for c in parts.values():
            total.update(c)
        out["total"] = dict(sorted(total.items()))
        return out

    # -- uncertainty -------------------------------------------------------- #

    def propagate_uncertainty(
        self,
        populations: Mapping[str, float],
        t_s: float,
        observable: Callable[[Mapping[str, float]], float],
        *,
        barrier_sigma_eV: float,
        table_log10_sigma: float,
        source_log10_sigma: float,
        n_samples: int,
        seed: int,
    ) -> ClusterUncertainty:
        """Monte Carlo over rate and source errors.

        Draw order per sample ``j`` from ``numpy_rng(seed, "cluster-uncertainty", j)``:
        reactions sorted by id, then sinks, then sources. Barrier-based reactions shift the
        barrier by ``sigma * z`` eV; table rates and sinks scale by ``10**(sigma * z)``.
        """
        if n_samples < 2:
            raise ClusterError("n_samples must be at least 2")
        if min(barrier_sigma_eV, table_log10_sigma, source_log10_sigma) < 0:
            raise ClusterError("sigmas must be nonnegative")
        values: list[float] = []
        for j in range(n_samples):
            rng = numpy_rng(seed, "cluster-uncertainty", j)
            rates = {}
            for r in self.reactions:
                z = float(rng.standard_normal())
                if r.barrier_based:
                    rates[r.reaction_id] = self.reaction_rate(r, barrier_sigma_eV * z)
                else:
                    rates[r.reaction_id] = self.reaction_rate(r) * 10.0 ** (table_log10_sigma * z)
            sinks = [
                s.rate_per_s * 10.0 ** (table_log10_sigma * float(rng.standard_normal()))
                for s in self._sinks
            ]
            sources = [
                t.rate_per_s * 10.0 ** (source_log10_sigma * float(rng.standard_normal()))
                for t in self._sources
            ]
            values.append(float(observable(self._evolve(populations, t_s, rates, sinks, sources))))
        mean = math.fsum(values) / n_samples
        var = math.fsum((v - mean) ** 2 for v in values) / (n_samples - 1)
        return ClusterUncertainty(
            n_samples=n_samples,
            mean=mean,
            std=math.sqrt(var),
            p05=_quantile(values, 0.05),
            p95=_quantile(values, 0.95),
            barrier_sigma_eV=barrier_sigma_eV,
            table_log10_sigma=table_log10_sigma,
            source_log10_sigma=source_log10_sigma,
            propagated=("barrier errors", "table-rate and sink errors", "source-strength errors"),
            not_propagated=(
                "model-form error of the linear master equation",
                "missing channels and states",
                "temperature error",
                "attempt-frequency error of barrier-based reactions",
            ),
        )


# --------------------------------------------------------------------------- #
# Transition-network bridge                                                   #
# --------------------------------------------------------------------------- #


class BridgeReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    reactions_added: tuple[str, ...]
    intra_state_events: tuple[str, ...]
    notes: tuple[str, ...]


def import_transition_network(
    network: ClusterNetwork,
    transitions: TransitionNetwork,
    state_map: Mapping[str, ClusterState],
    *,
    require: Literal["qualified", "exploratory"] = "qualified",
    fallback_attempt_frequency_hz: float | None = None,
) -> BridgeReport:
    """Add every event of a transition network as a cluster reaction.

    ``state_map`` assigns each transition-network state to a cluster state; an unmapped
    endpoint is an error. Events whose endpoints map to the same cluster state change no
    population and are reported, not added. An event without usable barrier evidence raises
    ``ClusterError`` because its rate would have to be invented.
    """
    added: list[str] = []
    intra: list[str] = []
    notes: list[str] = []
    for event in transitions.events:
        for sid in (event.source_id, event.target_id):
            if sid not in state_map:
                raise ClusterError(f"transition state {sid!r} has no cluster-state mapping")
        source, target = state_map[event.source_id], state_map[event.target_id]
        try:
            rate = event_rate(
                event,
                network.temperature_K,
                require=require,
                fallback_attempt_frequency_hz=fallback_attempt_frequency_hz,
            )
        except RateError as exc:
            raise ClusterError(f"cannot import {event.transition_id}: {exc}") from exc
        if source.key == target.key:
            intra.append(event.transition_id)
            continue
        assert event.evidence is not None and rate is not None
        nu = event.attempt_frequency_hz or fallback_attempt_frequency_hz
        qualification: RateQualification = (
            "synthetic"
            if event.evidence.synthetic
            else "qualified"
            if event.qualification == "qualified"
            else "exploratory"
        )
        network.add_state(source)
        network.add_state(target)
        network.add_reaction(
            ClusterReaction(
                mechanism=event.mechanism,
                source_key=source.key,
                target_key=target.key,
                evidence_ref=event.transition_id,
                qualification=qualification,
                barrier_eV=event.evidence.forward_barrier_eV,
                attempt_frequency_hz=nu,
                multiplicity=event.multiplicity,
            )
        )
        added.append(event.transition_id)
        notes.extend(rate.assumptions)
    return BridgeReport(
        reactions_added=tuple(added),
        intra_state_events=tuple(intra),
        notes=tuple(dict.fromkeys(notes)),
    )


__all__ = [
    "FUTURE_WORK",
    "BridgeReport",
    "ClusterError",
    "ClusterNetwork",
    "ClusterReaction",
    "ClusterState",
    "ClusterUncertainty",
    "FutureWorkError",
    "SinkTerm",
    "SourceTerm",
    "capabilities",
    "import_transition_network",
    "reject_future_work",
]
