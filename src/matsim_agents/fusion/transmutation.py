"""Seeded transmutation configurations of the 1,024-site reference alloy (F9).

The parent is the BCC 8x8x8 reference supercell (W640 Ta128 Cr64 V96 Hf32 Mo64) with
stable site ids. Re and Os counts come from imported, isotope-resolved transmutation
evidence: summed isotope atom fractions times 1,024, rounded to the nearest integer,
and substituted on W sites chosen by a seeded generator. Hydrogen (H, D, T) and helium
(He-3, He-4) are counted per isotope and never merged; they are not placed, because
placing gas needs a site model that is not part of this module. Nothing is run here.
"""

from __future__ import annotations

import hashlib
import json
import math

from pydantic import BaseModel, ConfigDict, Field

from matsim_agents.campaign.radiation_properties import (
    REFERENCE_ALLOY_COUNTS,
    REFERENCE_REPEATS,
    REFERENCE_SITES,
)
from matsim_agents.fusion.nuclear_evidence import (
    ExternalNuclearEvidence,
    NuclearEvidenceError,
    NuclearFamily,
)
from matsim_agents.seeding import numpy_rng

TRANSMUTATION_VERSION = "fusion-pfc-transmutation-v1"
SITE_MAP_SEED = 0
HOST = "W"
PRODUCTS = ("Re", "Os")
# Quantities that are fractions per atom of material, convertible to site counts.
_PER_ATOM = {"atom_fraction": 1.0, "concentration_appm": 1e-6}


class Site(BaseModel):
    model_config = ConfigDict(frozen=True)

    site_id: str
    frac: tuple[float, float, float]
    element: str


class GasInventory(BaseModel):
    """Per-isotope gas counts in the supercell; each is kept separate and unplaced."""

    model_config = ConfigDict(frozen=True)

    hydrogen: dict[str, float] = Field(default_factory=dict, description="H, D, T atoms.")
    helium: dict[str, float] = Field(default_factory=dict, description="He-3, He-4 atoms.")
    placed: bool = False


class Substitution(BaseModel):
    model_config = ConfigDict(frozen=True)

    site_id: str
    from_element: str
    to_element: str


class TransmutedConfiguration(BaseModel):
    model_config = ConfigDict(frozen=True)

    version: str = TRANSMUTATION_VERSION
    evidence_id: str
    quantity: str
    seed: int
    expected_counts: dict[str, float]
    counts: dict[str, int]
    substitutions: tuple[Substitution, ...]
    composition: dict[str, int]
    gas: GasInventory
    synthetic: bool
    notes: tuple[str, ...]

    def digest(self) -> str:
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return "ftx-" + hashlib.sha256(payload.encode()).hexdigest()[:24]


def reference_site_map(seed: int = SITE_MAP_SEED) -> tuple[Site, ...]:
    """The 1,024 BCC sites with stable ids and a seeded, fixed element assignment."""
    nx, ny, nz = REFERENCE_REPEATS
    coords = [
        ((i + b) / nx, (j + b) / ny, (k + b) / nz)
        for i in range(nx)
        for j in range(ny)
        for k in range(nz)
        for b in (0.0, 0.5)
    ]
    elements = [el for el, n in sorted(REFERENCE_ALLOY_COUNTS.items()) for _ in range(n)]
    order = numpy_rng(seed, "fusion-transmutation", "site-map").permutation(REFERENCE_SITES)
    return tuple(
        Site(site_id=f"s{n:04d}", frac=coords[n], element=elements[int(order[n])])
        for n in range(REFERENCE_SITES)
    )


def _per_atom(quantity: str) -> float:
    if quantity not in _PER_ATOM:
        raise NuclearEvidenceError(
            f"{quantity} is not a per-atom amount; use atom_fraction or concentration_appm"
        )
    return _PER_ATOM[quantity]


def _element_fraction(evidence: ExternalNuclearEvidence, quantity: str, element: str) -> float:
    scale = _per_atom(quantity)
    seen: set[str] = set()
    total = 0.0
    for q in evidence.quantities:
        if q.name != quantity or q.isotope is None or q.isotope.element != element:
            continue
        if q.isotope.key in seen:
            raise NuclearEvidenceError(f"duplicate {quantity} for {q.isotope.key}")
        seen.add(q.isotope.key)
        total += q.value * scale
    return total


def transmuted_configuration(
    evidence: ExternalNuclearEvidence,
    *,
    quantity: str = "atom_fraction",
    seed: int = 0,
    sites: tuple[Site, ...] | None = None,
) -> TransmutedConfiguration:
    """Substitute Re/Os on seeded W sites from transmutation evidence; gas is counted only."""
    if evidence.family != NuclearFamily.TRANSMUTATION:
        raise NuclearEvidenceError("a transmutation configuration needs transmutation evidence")
    sites = sites or reference_site_map()
    if len(sites) != REFERENCE_SITES:
        raise NuclearEvidenceError(f"site map has {len(sites)} sites, not {REFERENCE_SITES}")
    scale = _per_atom(quantity)
    expected = {el: _element_fraction(evidence, quantity, el) * REFERENCE_SITES for el in PRODUCTS}
    counts = {el: int(math.floor(v + 0.5)) for el, v in expected.items()}
    hosts = [s for s in sites if s.element == HOST]
    if sum(counts.values()) > len(hosts):
        raise NuclearEvidenceError(
            f"{sum(counts.values())} substitutions exceed the {len(hosts)} {HOST} sites"
        )
    order = numpy_rng(seed, "fusion-transmutation", evidence.evidence_id()).permutation(len(hosts))
    chosen = iter(hosts[int(i)] for i in order)
    subs = sorted(
        (
            Substitution(site_id=next(chosen).site_id, from_element=HOST, to_element=el)
            for el in PRODUCTS
            for _ in range(counts[el])
        ),
        key=lambda s: s.site_id,
    )
    composition = dict(REFERENCE_ALLOY_COUNTS)
    composition[HOST] -= sum(counts.values())
    composition.update({el: n for el, n in counts.items() if n})
    gas = GasInventory(
        hydrogen={
            k: v * scale * REFERENCE_SITES for k, v in evidence.hydrogen_isotopes(quantity).items()
        },
        helium={
            k: v * scale * REFERENCE_SITES for k, v in evidence.helium_isotopes(quantity).items()
        },
    )
    notes = [
        "Counts round expected atoms in 1,024 sites to the nearest integer.",
        "Gas isotopes are counted, not placed; placement needs a separate site model.",
        "No electronic-structure calculation has been run on this configuration.",
    ]
    if evidence.synthetic:
        notes.append("Evidence is SYNTHETIC; the configuration is a demonstration only.")
    return TransmutedConfiguration(
        evidence_id=evidence.evidence_id(),
        quantity=quantity,
        seed=seed,
        expected_counts=expected,
        counts=counts,
        substitutions=tuple(subs),
        composition=composition,
        gas=gas,
        synthetic=evidence.synthetic,
        notes=tuple(notes),
    )


__all__ = [
    "GasInventory",
    "Site",
    "Substitution",
    "TRANSMUTATION_VERSION",
    "TransmutedConfiguration",
    "reference_site_map",
    "transmuted_configuration",
]
