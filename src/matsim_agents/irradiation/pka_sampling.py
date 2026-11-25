"""Reproducible PKA event sampling from a species-resolved spectrum.

The spectrum fixes the joint probability of (species, energy bin). Within a bin the energy is
drawn uniformly (or log-uniformly); that is a sampling convention, not data. Each event is
placed on a lattice site whose element matches the recoiling atom's parent. Times are only
produced when the spectrum states that its rates are per target atom per second.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from matsim_agents.core.seeding import numpy_rng
from matsim_agents.irradiation.contracts import (
    RATE_BASIS_PER_ATOM_PER_S,
    ContractError,
    PKAEvent,
    PKASpectrum,
)


@dataclass(frozen=True)
class SamplingResult:
    events: tuple[PKAEvent, ...]
    excluded_rate_fraction: float
    excluded_species: tuple[str, ...]
    time_basis: str


def _site_lookup(lattice_elements: Sequence[str]) -> dict[str, np.ndarray]:
    by_element: dict[str, list[int]] = {}
    for index, element in enumerate(lattice_elements):
        by_element.setdefault(element, []).append(index)
    return {el: np.asarray(ix, dtype=np.int64) for el, ix in by_element.items()}


def sample_pka_events(
    spectrum: PKASpectrum,
    n_events: int,
    *,
    seed: int,
    lattice_elements: Sequence[str],
    neighbors: Sequence[Sequence[int]] | None = None,
    within_bin: Literal["uniform", "log"] = "uniform",
    stream: str = "pka",
) -> SamplingResult:
    """Draw ``n_events`` PKAs on a lattice.

    ``lattice_elements[i]`` is the element at site ``i``. Spectrum entries whose parent element
    has no site in the lattice (for example emitted H or He) cannot be placed; they are excluded
    and the excluded share of the total rate is reported rather than hidden.
    """
    if n_events < 0:
        raise ValueError("n_events must be non-negative")
    if not lattice_elements:
        raise ContractError("the lattice has no sites")
    if neighbors is not None and len(neighbors) != len(lattice_elements):
        raise ValueError("neighbors must have one entry per site")
    sites = _site_lookup(lattice_elements)

    placeable = [s for s in spectrum.spectra if s.site_element in sites]
    excluded = [s for s in spectrum.spectra if s.site_element not in sites]
    if not placeable:
        raise ContractError("no spectrum species has a matching site in the lattice")
    total_all = spectrum.total_rate
    kept_total = math.fsum(s.total_rate for s in placeable)
    if kept_total <= 0:
        raise ContractError("the placeable species have zero PKA production")

    # Flatten (species entry, bin) into one categorical distribution.
    flat_rates: list[float] = []
    flat_index: list[tuple[int, int]] = []
    for si, entry in enumerate(placeable):
        for bi, rate in enumerate(entry.rates):
            if rate > 0:
                flat_rates.append(rate)
                flat_index.append((si, bi))
    weights = np.asarray(flat_rates, dtype=float)
    probabilities = weights / weights.sum()

    rng = numpy_rng(seed, stream)
    choices = rng.choice(len(probabilities), size=n_events, p=probabilities)

    per_atom = spectrum.rate_basis == RATE_BASIS_PER_ATOM_PER_S
    event_rate = kept_total * len(lattice_elements) if per_atom else None

    events: list[PKAEvent] = []
    clock = 0.0
    for i, choice in enumerate(choices):
        si, bi = flat_index[int(choice)]
        entry = placeable[si]
        lo, hi = entry.bin_low_mev[bi], entry.bin_high_mev[bi]
        if within_bin == "log" and lo > 0:
            energy = float(math.exp(rng.uniform(math.log(lo), math.log(hi))))
        else:
            energy = float(rng.uniform(lo, hi))
        energy = min(max(energy, lo), hi) or hi
        candidates = sites[entry.site_element]
        site = int(candidates[int(rng.integers(len(candidates)))])
        interval = float(rng.exponential(1.0 / event_rate)) if event_rate else None
        if interval is not None:
            clock += interval
        config = None
        if neighbors is not None:
            config = tuple(lattice_elements[j] for j in neighbors[site])
        events.append(
            PKAEvent(
                index=i,
                species=entry.species,
                energy_mev=energy,
                weight=1.0,
                time_s=clock if interval is not None else None,
                interval_s=interval,
                site_index=site,
                site_element=lattice_elements[site],
                chemical_configuration=config,
                probability=float(probabilities[int(choice)]),
            )
        )
    return SamplingResult(
        events=tuple(events),
        excluded_rate_fraction=1.0 - kept_total / total_all,
        excluded_species=tuple(sorted({s.species.key for s in excluded})),
        time_basis="per_target_atom_per_s" if per_atom else "unspecified",
    )


__all__ = ["SamplingResult", "sample_pka_events"]
