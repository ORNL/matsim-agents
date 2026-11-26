"""Inject WP3.9 defect sources into a cluster-dynamics network, guarding double counting.

A cascade deposits correlated defects, so a cluster source puts a whole cluster into one
network state; independent defects (for example dpa-based isolated Frenkel pairs) put a
single defect in per event. Both are legitimate, but one damage origin must enter through
one channel per defect species, otherwise the same displacements are counted twice.

Checks: ``independent`` channels need a single-defect state; a declared cluster size must
match the state; two channels with the same origin may not share a defect class or mix
``cascade`` and ``dpa`` bases on overlapping species. Not checked: that the yields and a
dpa value were derived from the same spectrum (``origin`` is a caller-supplied label),
interstitial bookkeeping (the network tracks V, He and H only), or any physics of the
rates. Nothing here is validated against data.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Literal

from matsim_agents.science.defects.source_terms import DefectSource, cluster_source_term
from matsim_agents.science.kinetics.cluster_dynamics import (
    ClusterNetwork,
    SourceTerm,
)

Kind = Literal["cascade", "dpa"]
Correlation = Literal["correlated", "independent"]


class InjectionError(ValueError):
    """The requested source channels would misrepresent or double count damage."""


@dataclass(frozen=True)
class SourceChannel:
    source: DefectSource
    state_key: str
    kind: Kind
    correlation: Correlation
    origin: str
    cluster_size: int | None = None


@dataclass(frozen=True)
class InjectionReport:
    terms: tuple[SourceTerm, ...]
    species_rate_per_s: dict[str, float]
    not_tracked: tuple[str, ...]


def _size(network: ClusterNetwork, key: str) -> int:
    return sum(network.state(key).inventory.values())


def check_channels(network: ClusterNetwork, channels: list[SourceChannel]) -> None:
    seen_class: set[tuple[str, str]] = set()
    species_kinds: dict[tuple[str, str], set[str]] = {}
    for c in channels:
        size = _size(network, c.state_key)
        if c.correlation == "independent" and size != 1:
            raise InjectionError(
                f"independent channel {c.source.defect_class!r} targets {c.state_key} "
                f"with {size} defects; independent defects enter one at a time"
            )
        if c.cluster_size is not None and c.cluster_size != size:
            raise InjectionError(
                f"channel {c.source.defect_class!r} declares cluster size {c.cluster_size} "
                f"but {c.state_key} holds {size}"
            )
        ident = (c.origin, c.source.defect_class)
        if ident in seen_class:
            raise InjectionError(
                f"defect class {c.source.defect_class!r} from origin {c.origin!r} is injected twice"
            )
        seen_class.add(ident)
        for species in network.state(c.state_key).inventory:
            species_kinds.setdefault((c.origin, species), set()).add(c.kind)
    for (origin, species), kinds in species_kinds.items():
        if len(kinds) > 1:
            raise InjectionError(
                f"origin {origin!r} feeds species {species} through both cascade and dpa "
                "channels; the same damage would be counted twice"
            )


def inject_sources(
    network: ClusterNetwork,
    channels: list[SourceChannel],
    *,
    n_target_atoms: float,
    reservoir: str,
) -> InjectionReport:
    """Validate all channels first, then add one source term per channel."""
    check_channels(network, channels)
    terms = tuple(
        cluster_source_term(
            c.source, state_key=c.state_key, n_target_atoms=n_target_atoms, reservoir=reservoir
        )
        for c in channels
    )
    for t in terms:
        network.add_source(t)
    rates: Counter = Counter()
    for t in terms:
        for species, count in network.state(t.state_key).inventory.items():
            rates[species] += t.rate_per_s * count
    return InjectionReport(
        terms=terms,
        species_rate_per_s=dict(sorted(rates.items())),
        not_tracked=("interstitial balance unless the network states carry interstitials",),
    )


__all__ = [
    "InjectionError",
    "InjectionReport",
    "SourceChannel",
    "check_channels",
    "inject_sources",
]
