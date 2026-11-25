"""Gillespie continuous-time kinetic Monte Carlo on a transition network.

Direct method: waiting time ``dt = -ln(u1)/R``, event chosen by ``u2 * R`` over the cumulative
rates. A state with no outgoing events or zero total rate is *stuck*; reaching the horizon or
the event cap is censoring. Neither is reported as an absorbing outcome.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from matsim_agents.kinetics.graph import TransitionNetwork
from matsim_agents.kinetics.rates import Require, event_rate
from matsim_agents.seeding import numpy_rng

Termination = Literal["absorbed", "horizon", "max_events", "stuck"]


class KmcStep(BaseModel):
    model_config = ConfigDict(frozen=True)

    time_s: float
    transition_id: str
    source_id: str
    target_id: str


class KmcTrajectory(BaseModel):
    model_config = ConfigDict(frozen=True)

    seed: int
    stream: tuple[str | int, ...]
    temperature_K: float
    initial_state_id: str
    final_state_id: str
    final_time_s: float
    termination: Termination
    absorbing_kind: str | None = None
    n_events: int
    steps: tuple[KmcStep, ...] = Field(default=())
    assumptions: tuple[str, ...] = ()


def run_kmc(
    network: TransitionNetwork,
    initial_state_id: str,
    temperature_K: float,
    *,
    seed: int,
    stream: tuple[str | int, ...] = (),
    horizon_s: float = math.inf,
    max_events: int = 100_000,
    require: Require = "qualified",
    fallback_attempt_frequency_hz: float | None = None,
    record_steps: bool = True,
) -> KmcTrajectory:
    """One trajectory. Needs a finite ``horizon_s`` or a finite ``max_events``."""
    if horizon_s <= 0 or math.isnan(horizon_s):
        raise ValueError("horizon_s must be positive")
    if max_events < 1:
        raise ValueError("max_events must be at least 1")
    network.state(initial_state_id)
    rng = numpy_rng(seed, "kmc", *stream)

    rate_cache: dict[str, list[tuple[str, str, float]]] = {}
    assumptions: set[str] = set()

    def channels(sid: str) -> list[tuple[str, str, float]]:
        if sid not in rate_cache:
            rows = []
            for event in network.outgoing(sid):
                r = event_rate(
                    event,
                    temperature_K,
                    require=require,
                    fallback_attempt_frequency_hz=fallback_attempt_frequency_hz,
                )
                assumptions.update(r.assumptions)
                rows.append((event.transition_id, event.target_id, r.rate_per_s))
            rate_cache[sid] = rows
        return rate_cache[sid]

    state, now, steps = initial_state_id, 0.0, []
    termination: Termination = "max_events"
    n = 0
    while True:
        current = network.state(state)
        if current.absorbing:
            termination = "absorbed"
            break
        if n >= max_events:
            termination = "max_events"
            break
        rows = channels(state)
        total = math.fsum(r for _, _, r in rows)
        if total <= 0.0:
            termination = "stuck"
            break
        dt = -math.log1p(-rng.random()) / total
        if now + dt > horizon_s:
            now, termination = horizon_s, "horizon"
            break
        pick, acc = rng.random() * total, 0.0
        chosen = rows[-1]
        for row in rows:
            acc += row[2]
            if pick < acc and row[2] > 0:
                chosen = row
                break
        if chosen[2] <= 0:
            chosen = next(row for row in reversed(rows) if row[2] > 0)
        now += dt
        n += 1
        if record_steps:
            steps.append(
                KmcStep(time_s=now, transition_id=chosen[0], source_id=state, target_id=chosen[1])
            )
        state = chosen[1]

    return KmcTrajectory(
        seed=seed,
        stream=tuple(stream),
        temperature_K=temperature_K,
        initial_state_id=initial_state_id,
        final_state_id=state,
        final_time_s=now,
        termination=termination,
        absorbing_kind=network.state(state).absorbing_kind if termination == "absorbed" else None,
        n_events=n,
        steps=tuple(steps),
        assumptions=tuple(sorted(assumptions)),
    )


__all__ = ["KmcStep", "KmcTrajectory", "Termination", "run_kmc"]
