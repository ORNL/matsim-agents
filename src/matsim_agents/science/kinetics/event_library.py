"""Generalised kinetic event library for cluster dynamics.

Each event type declares how it may change the inventory ``{V, I, He, H, <solutes>}``:

* ``conserve``: closed; the inventory is unchanged (migration, trapping, association).
* ``annihilate``: ``k`` vacancies and ``k`` interstitials disappear together (recombination).
* ``create_pair``: ``k`` vacancies and ``k`` interstitials appear together (Frenkel generation).
* ``open``: the change must be explained by ``OpenSystemTerm`` records naming the reservoir or
  sink (gas production, implantation, release, transmutation, absorption at a sink), with a
  required sign.

The gas mechanisms of :mod:`gas_mechanisms` stay valid and are checked by their own rules. An
unknown mechanism is rejected. Host-atom depletion by transmutation is not part of the
inventory (the host is the lattice), so it is not tracked here. Precipitate formation has no
validated model and is listed as future work in :mod:`cluster_dynamics`.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from matsim_agents.science.kinetics.gas_mechanisms import (
    GAS,
    GAS_MECHANISM_IDS,
    GasMechanismError,
    OpenSystemTerm,
    check_event_inventory,
)
from matsim_agents.science.kinetics.transitions import list_mechanisms, register_mechanism

RESERVED_LABELS = ("V", "I", "He", "H")
Rule = Literal["conserve", "annihilate", "create_pair", "open"]
Sign = Literal["any", "gain", "loss"]


@dataclass(frozen=True)
class EventSpec:
    name: str
    rule: Rule
    reversible: bool
    description: str
    need: Mapping[str, int]
    sign: Sign = "any"
    species: tuple[str, ...] | None = None  # allowed species of open terms


def _e(name, rule, description, *, reversible=False, need=None, sign="any", species=None):
    return EventSpec(name, rule, reversible, description, dict(need or {}), sign, species)


EVENTS: Mapping[str, EventSpec] = {
    s.name: s
    for s in (
        _e("frenkel_generation", "create_pair", "thermal creation of a vacancy-interstitial pair"),
        _e("vacancy_migration", "conserve", "vacancy or vacancy-cluster move", reversible=True),
        _e(
            "interstitial_migration",
            "conserve",
            "interstitial or interstitial-cluster move",
            reversible=True,
            need={"I": 1},
        ),
        _e(
            "frenkel_recombination",
            "annihilate",
            "vacancy and interstitial annihilate",
            need={"V": 1, "I": 1},
        ),
        _e("cluster_association", "conserve", "defects or clusters combine"),
        _e("cluster_dissociation", "conserve", "a cluster releases a defect or cluster"),
        _e("defect_trapping", "conserve", "a mobile defect is captured by a trap"),
        _e("defect_detrapping", "conserve", "a trapped defect is released"),
        _e(
            "gas_implantation",
            "open",
            "gas atoms enter from an external reservoir",
            sign="gain",
            species=GAS,
        ),
        _e(
            "gas_production",
            "open",
            "gas atoms produced in place (for example by (n,p) or (n,alpha))",
            sign="gain",
            species=GAS,
        ),
        _e(
            "gas_release",
            "open",
            "gas atoms leave to a surface or boundary",
            sign="loss",
            species=GAS,
        ),
        _e("transmutation", "open", "nuclear transmutation changes the species inventory"),
        _e("sink_absorption", "open", "a defect is absorbed at a sink", sign="loss"),
    )
}
EVENT_IDS: tuple[str, ...] = tuple(EVENTS)
SUPPORTED_MECHANISMS: tuple[str, ...] = (*GAS_MECHANISM_IDS, *EVENT_IDS)

_registered = set(list_mechanisms())
for _spec in EVENTS.values():
    if _spec.name not in _registered:
        register_mechanism(_spec.name, reversible=_spec.reversible)


def _change(source: Mapping[str, int], target: Mapping[str, int]) -> dict[str, int]:
    keys = set(source) | set(target)
    return {
        k: target.get(k, 0) - source.get(k, 0)
        for k in sorted(keys)
        if target.get(k, 0) != source.get(k, 0)
    }


def _check_open(spec: EventSpec, change: dict[str, int], terms: Sequence[OpenSystemTerm]) -> None:
    if not terms:
        raise GasMechanismError(
            f"{spec.name} is an open event and needs explicit source/sink terms"
        )
    explained: Counter = Counter()
    for t in terms:
        if spec.species is not None and t.species not in spec.species:
            raise GasMechanismError(f"{spec.name} cannot exchange {t.species}")
        if (spec.sign == "gain" and t.delta < 0) or (spec.sign == "loss" and t.delta > 0):
            raise GasMechanismError(
                f"{spec.name} must be a {spec.sign}, got {t.delta:+d} {t.species}"
            )
        explained[t.species] += t.delta
    explained = {k: v for k, v in sorted(explained.items()) if v}
    if explained != change:
        raise GasMechanismError(
            f"{spec.name}: source/sink terms {explained} do not equal the inventory change {change}"
        )


def check_event(
    mechanism: str,
    source: Mapping[str, int],
    target: Mapping[str, int],
    open_terms: Sequence[OpenSystemTerm] = (),
) -> None:
    """Raise ``GasMechanismError`` unless the inventory change is justified by the event type."""
    if mechanism in GAS_MECHANISM_IDS:
        check_event_inventory(mechanism, source, target, open_terms)
        return
    spec = EVENTS.get(mechanism)
    if spec is None:
        raise GasMechanismError(
            f"unsupported mechanism {mechanism!r}; supported: {', '.join(SUPPORTED_MECHANISMS)}"
        )
    for key, minimum in spec.need.items():
        if source.get(key, 0) < minimum:
            raise GasMechanismError(
                f"{mechanism} needs at least {minimum} {key} in the source state, "
                f"found {source.get(key, 0)}"
            )
    change = _change(source, target)
    if spec.rule == "open":
        _check_open(spec, change, open_terms)
        return
    if open_terms:
        raise GasMechanismError(f"{mechanism} is a closed event and cannot carry open-system terms")
    if spec.rule == "conserve":
        if change:
            raise GasMechanismError(
                f"{mechanism} changes the inventory by {change} without explicit source/sink "
                "terms; closed-system events must conserve it"
            )
        return
    sign = -1 if spec.rule == "annihilate" else 1
    pairs = sign * change.get("V", 0)
    if pairs < 1 or change != {"I": sign * pairs, "V": sign * pairs}:
        raise GasMechanismError(
            f"{mechanism} must {'remove' if sign < 0 else 'add'} equal numbers of vacancies and "
            f"interstitials and change nothing else, got {change}"
        )


__all__ = [
    "EVENTS",
    "EVENT_IDS",
    "RESERVED_LABELS",
    "SUPPORTED_MECHANISMS",
    "EventSpec",
    "check_event",
]
