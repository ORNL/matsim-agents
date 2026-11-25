"""Analytical benchmarks for the kinetic model.

The KMC engine is run on tiny synthetic networks whose exact answers are known, and the
estimates are compared with those answers using a statistical tolerance:

* single channel, rate ``k``: first-passage time is exponential, mean ``1/k``; the check uses
  the sample standard error and a ``z`` multiple (default 3);
* two competing channels ``k1``, ``k2``: absorption probability into channel 1 is
  ``k1 / (k1 + k2)``; the check asks that the Wilson interval contains it;
* HTST rate against a hand-computed Arrhenius value.

Passing shows the *engine* reproduces known maths. It says nothing about attempt frequencies,
network completeness or any real material, so only the ``analytical-benchmarks`` criterion is
supported. Every network here is synthetic and labelled so.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from matsim_agents.science.kinetics.contracts import (
    BarrierEvidence,
    DefectState,
    NetworkProvenance,
    TransitionEvent,
)
from matsim_agents.science.kinetics.ensemble import run_ensemble
from matsim_agents.science.kinetics.graph import TransitionNetwork, state_id, transition_id
from matsim_agents.science.kinetics.rates import KB_EV_PER_K, htst_rate

Verdict = Literal["pass", "fail"]
DEFAULT_Z = 3.0
TEMPERATURE_K = 300.0
_PROV = NetworkProvenance(
    network_id="kinetic-benchmark", parent_id="p", parent_ordering_id="o1", synthetic=True
)


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    expected: float
    observed: float
    low: float
    high: float
    n_trajectories: int
    verdict: Verdict
    synthetic: bool = True


@dataclass(frozen=True)
class KineticBenchmarkReport:
    seed: int
    results: tuple[BenchmarkResult, ...]
    verdict: Verdict
    supports: tuple[str, ...]
    caveats: tuple[str, ...]


def _state(label: str, **over: object) -> DefectState:
    base: dict[str, object] = {
        "state_id": state_id("p", "o1", None, label),
        "parent_id": "p",
        "parent_ordering_id": "o1",
        "n_atoms": 2,
        "atom_ids": (0, 1),
        "species_counts": {"W": 2},
        "provenance": {"origin": "kinetic-benchmark"},
    }
    base.update(over)
    return DefectState(**base)  # type: ignore[arg-type]


def _add_edge(
    net: TransitionNetwork, src: DefectState, dst: DefectState, rate_hz: float, atom: int
):
    """Zero barrier with attempt frequency ``rate_hz`` makes the HTST rate exactly ``rate_hz``."""
    evidence = BarrierEvidence(
        method_id="synthetic-benchmark",
        neb_state="unconverged",
        forward_barrier_eV=0.0,
        reverse_barrier_eV=0.0,
        n_images=6,
        synthetic=True,
    )
    net.add_transition(
        TransitionEvent(
            transition_id=transition_id(src.state_id, dst.state_id, "vacancy_hop", (atom,)),
            source_id=src.state_id,
            target_id=dst.state_id,
            mechanism="vacancy_hop",
            moving_atom_ids=(atom,),
            provenance={"origin": "kinetic-benchmark"},
            evidence=evidence,
            qualification="unqualified",
            attempt_frequency_hz=rate_hz,
        )
    )


def decay_network(rate_hz: float) -> tuple[TransitionNetwork, str]:
    """One channel ``a -> done`` at ``rate_hz``."""
    net = TransitionNetwork(_PROV)
    a = _state("a")
    done = _state("done", absorbing=True, absorbing_kind="recombined")
    net.add_state(a)
    net.add_state(done)
    _add_edge(net, a, done, rate_hz, 0)
    return net, a.state_id


def branching_network(k1_hz: float, k2_hz: float) -> tuple[TransitionNetwork, str]:
    """Two competing channels into absorbing kinds ``recombined`` (channel 1) and ``sink``."""
    net = TransitionNetwork(_PROV)
    a = _state("a")
    one = _state("one", absorbing=True, absorbing_kind="recombined")
    two = _state("two", absorbing=True, absorbing_kind="sink")
    for s in (a, one, two):
        net.add_state(s)
    _add_edge(net, a, one, k1_hz, 0)
    _add_edge(net, a, two, k2_hz, 1)
    return net, a.state_id


def chain_network(n_states: int, rate_hz: float = 1.0e3) -> tuple[TransitionNetwork, str]:
    """``n_states`` states in a line, forward hops only, the last absorbing (synthetic)."""
    if n_states < 2:
        raise ValueError("n_states must be at least 2")
    net = TransitionNetwork(_PROV)
    last = n_states - 1
    states = [
        _state(f"c{i}", **({"absorbing": True, "absorbing_kind": "sink"} if i == last else {}))
        for i in range(n_states)
    ]
    for s in states:
        net.add_state(s)
    for i in range(last):
        _add_edge(net, states[i], states[i + 1], rate_hz, i % 2)
    return net, states[0].state_id


def _positive(**values: float) -> None:
    for name, value in values.items():
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")


def check_first_passage(
    rate_hz: float,
    *,
    n_trajectories: int,
    seed: int,
    z: float = DEFAULT_Z,
    expected_override: float | None = None,
) -> BenchmarkResult:
    """Mean first-passage time against ``1/k`` within ``z`` standard errors.

    ``expected_override`` exists to build a negative control that must fail.
    """
    _positive(rate_hz=rate_hz, z=z)
    if n_trajectories < 30:
        raise ValueError("n_trajectories must be at least 30 for a standard error")
    net, start = decay_network(rate_hz)
    res = run_ensemble(
        net, start, TEMPERATURE_K, n_trajectories=n_trajectories, seed=seed, require="exploratory"
    )
    times = res.first_passage_times_s
    mean = math.fsum(times) / len(times)
    var = math.fsum((t - mean) ** 2 for t in times) / (len(times) - 1)
    half = z * math.sqrt(var / len(times))
    expected = expected_override if expected_override is not None else 1.0 / rate_hz
    ok = len(times) == n_trajectories and abs(mean - expected) <= half
    return BenchmarkResult(
        f"first-passage-mean(k={rate_hz:g})",
        expected,
        mean,
        mean - half,
        mean + half,
        n_trajectories,
        "pass" if ok else "fail",
    )


def check_branching(
    k1_hz: float,
    k2_hz: float,
    *,
    n_trajectories: int,
    seed: int,
    expected_override: float | None = None,
) -> BenchmarkResult:
    """Fraction absorbed in channel 1 against ``k1/(k1+k2)``; Wilson interval must contain it."""
    _positive(k1_hz=k1_hz, k2_hz=k2_hz)
    if n_trajectories < 30:
        raise ValueError("n_trajectories must be at least 30")
    net, start = branching_network(k1_hz, k2_hz)
    res = run_ensemble(
        net, start, TEMPERATURE_K, n_trajectories=n_trajectories, seed=seed, require="exploratory"
    )
    outcome = res.outcomes.get("recombined")
    expected = expected_override if expected_override is not None else k1_hz / (k1_hz + k2_hz)
    if outcome is None:
        return BenchmarkResult("branching-ratio", expected, 0.0, 0.0, 1.0, n_trajectories, "fail")
    ok = outcome.wilson_low <= expected <= outcome.wilson_high
    return BenchmarkResult(
        f"branching-ratio(k1={k1_hz:g},k2={k2_hz:g})",
        expected,
        outcome.probability,
        outcome.wilson_low,
        outcome.wilson_high,
        n_trajectories,
        "pass" if ok else "fail",
    )


def check_arrhenius(
    barrier_eV: float, temperature_K: float, attempt_frequency_hz: float
) -> BenchmarkResult:
    """``htst_rate`` against ``nu * exp(-Ea/kT)`` computed independently, to 1e-12 relative."""
    _positive(temperature_K=temperature_K, attempt_frequency_hz=attempt_frequency_hz)
    expected = attempt_frequency_hz * math.exp(-barrier_eV / (KB_EV_PER_K * temperature_K))
    observed = htst_rate(barrier_eV, temperature_K, attempt_frequency_hz)
    ok = math.isclose(observed, expected, rel_tol=1e-12)
    return BenchmarkResult(
        f"arrhenius(Ea={barrier_eV:g}eV,T={temperature_K:g}K)",
        expected,
        observed,
        expected,
        expected,
        0,
        "pass" if ok else "fail",
    )


def run_kinetic_benchmarks(*, seed: int, n_trajectories: int = 2000) -> KineticBenchmarkReport:
    """The standard set. Fails if any single benchmark fails."""
    results = (
        check_first_passage(2.0, n_trajectories=n_trajectories, seed=seed),
        check_first_passage(1.0e6, n_trajectories=n_trajectories, seed=seed),
        check_branching(1.0, 3.0, n_trajectories=n_trajectories, seed=seed),
        check_branching(5.0, 5.0, n_trajectories=n_trajectories, seed=seed),
        check_arrhenius(0.5, 300.0, 1.0e13),
        check_arrhenius(1.3, 900.0, 5.0e12),
    )
    return KineticBenchmarkReport(
        seed=seed,
        results=results,
        verdict="pass" if all(r.verdict == "pass" for r in results) else "fail",
        supports=("analytical-benchmarks",),
        caveats=(
            "synthetic networks only",
            "does not assess attempt frequencies or network completeness",
            "a pass shows the engine matches known maths, not any material",
        ),
    )


__all__ = [
    "BenchmarkResult",
    "KineticBenchmarkReport",
    "branching_network",
    "chain_network",
    "check_arrhenius",
    "check_branching",
    "check_first_passage",
    "decay_network",
    "run_kinetic_benchmarks",
]
