"""Substitutional transmutation configurations (host -> products) on a fixed BCC parent.

A transmutation changes the chemical species of a lattice atom and keeps its site identity:
``atom_id`` and ``lattice_site`` are those of the parent (see :mod:`point_defects`). Only the
scenario's ``host`` species is replaced. This module places already-decided product counts (for
example W -> Re, Os); the counts come from an external nuclear inventory (never from DFT) and
enter only through the scenario.

The default cap ``sum(products) / N_sites <= 0.01`` is a campaign limit, not a physical
bound; a scenario may raise it only with a justification and an inventory source. Sampled
distributions are initial geometries, not claims about irradiation-induced spatial statistics.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from typing import Literal

import numpy as np
from ase import Atoms
from ase.geometry import get_distances
from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from matsim_agents.core.seeding import numpy_rng
from matsim_agents.science.defects.point_defects import DefectConfiguration, geometry_fingerprint
from matsim_agents.science.structures.alloy_ensemble import occupation_fingerprint
from matsim_agents.science.structures.alloys import (
    AlloySpec,
    first_neighbor_list,
    second_neighbor_list,
)

DEFAULT_MAX_SOLUTE_FRACTION = 0.01
# "Isolated" means no other solute within this many lattice parameters (beyond the 2NN shell).
ISOLATION_LATTICE_PARAMETERS = 1.2

Distribution = Literal["random", "isolated", "pair", "enriched", "near_defect"]
EvidenceLevel = Literal[
    "synthetic", "predicted", "numerically_validated", "scientifically_qualified"
]


class TransmutationError(ValueError):
    """A transmutation scenario cannot be realized on the given structure."""


class TransmutationScenario(BaseModel):
    """Counts and provenance of one transmutation scenario (not a predicted composition)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_id: str = Field(min_length=1)
    parent_alloy_id: str = Field(min_length=1)
    exposure_scenario: str = Field(min_length=1)
    host: str = Field(min_length=1, description="The species that transmutes.")
    products: dict[str, StrictInt] = Field(
        min_length=1, description="Product species -> atom count, in a fixed order."
    )
    distribution: Distribution = "random"
    seed: StrictInt = Field(ge=0)
    max_solute_fraction: float = Field(
        default=DEFAULT_MAX_SOLUTE_FRACTION, gt=0, le=1, allow_inf_nan=False
    )
    limit_justification: str | None = None
    inventory_source: str | None = None
    evidence_level: EvidenceLevel = "synthetic"

    @model_validator(mode="after")
    def _consistent(self) -> TransmutationScenario:
        if any(n < 0 for n in self.products.values()):
            raise ValueError("product counts must be non-negative")
        if self.host in self.products:
            raise ValueError("the host cannot also be a product")
        if self.n_solutes < 1:
            raise ValueError("a transmutation scenario needs at least one product atom")
        if self.max_solute_fraction > DEFAULT_MAX_SOLUTE_FRACTION and not (
            self.limit_justification and self.inventory_source
        ):
            raise ValueError(
                "raising max_solute_fraction above the 0.01 campaign default requires "
                "limit_justification and inventory_source"
            )
        if self.distribution == "pair":
            counts = [n for n in self.products.values() if n]
            if len(self.products) != 2 or len(counts) != 2 or counts[0] != counts[1]:
                raise ValueError("pair needs exactly two products with equal positive counts")
        return self

    @property
    def n_solutes(self) -> int:
        return sum(self.products.values())


class TransmutedSite(BaseModel):
    model_config = ConfigDict(frozen=True)

    atom_id: StrictInt = Field(ge=0)
    lattice_site: StrictInt = Field(ge=0)
    original_species: str
    final_species: str
    position: tuple[float, float, float]
    first_shell: dict[str, int]
    second_shell: dict[str, int]


class TransmutationRecord(BaseModel):
    """Provenance of one generated configuration; the geometry stays in the ``Atoms``."""

    model_config = ConfigDict(frozen=True)

    config_id: str = Field(min_length=1)
    parent_id: str = Field(min_length=1)
    parent_ordering_id: str
    scenario: TransmutationScenario
    n_lattice_sites: StrictInt = Field(gt=0)
    n_atoms: StrictInt = Field(gt=0)
    parent_composition: dict[str, int]
    final_composition: dict[str, int]
    host: str
    product_counts: dict[str, int]
    solute_fraction: float
    sites: tuple[TransmutedSite, ...]
    geometry_fingerprint: str
    evidence_level: EvidenceLevel


def _lattice_parameter(atoms: Atoms) -> float:
    raw = atoms.info.get("alloy_spec")
    if raw is None:
        raise TransmutationError("structure has no alloy_spec; pass lattice_parameter")
    return AlloySpec.model_validate_json(raw).lattice_parameter


def _atom_ids(atoms: Atoms) -> np.ndarray:
    if "atom_id" in atoms.arrays:
        return np.asarray(atoms.get_array("atom_id"), dtype=np.int64)
    return np.arange(len(atoms), dtype=np.int64)


def _min_distance_to(atoms: Atoms, index: int, others: list[int]) -> float:
    if not others:
        return math.inf
    _, d = get_distances(
        atoms.positions[index][None], atoms.positions[others], cell=atoms.cell, pbc=atoms.pbc
    )
    return float(d.min())


def _ranked_by_distance(
    atoms: Atoms, eligible: np.ndarray, centers: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Eligible indices ordered by minimum-image distance to the nearest center, ties random."""
    _, d = get_distances(atoms.positions[eligible], centers, cell=atoms.cell, pbc=atoms.pbc)
    nearest = np.round(d.min(axis=1), 6)
    return eligible[np.lexsort((rng.random(len(eligible)), nearest))]


def _select_isolated(
    atoms: Atoms, eligible: np.ndarray, n: int, a: float, rng: np.random.Generator
) -> list[int]:
    chosen: list[int] = []
    for idx in rng.permutation(eligible):
        if _min_distance_to(atoms, int(idx), chosen) > ISOLATION_LATTICE_PARAMETERS * a:
            chosen.append(int(idx))
            if len(chosen) == n:
                return chosen
    raise TransmutationError(f"cannot place {n} isolated solutes; placed {len(chosen)}")


def _select_pairs(
    atoms: Atoms, eligible: np.ndarray, n_pairs: int, a: float, rng: np.random.Generator
) -> list[tuple[int, int]]:
    first_i, first_j, _ = first_neighbor_list(atoms, a)
    allowed = set(int(i) for i in eligible)
    partners: dict[int, list[int]] = {}
    for i, j in zip(first_i, first_j, strict=True):
        if int(i) in allowed and int(j) in allowed:
            partners.setdefault(int(i), []).append(int(j))
    pairs: list[tuple[int, int]] = []
    used: list[int] = []
    for idx in rng.permutation(eligible):
        idx = int(idx)
        options = partners.get(idx, [])
        if not options:
            continue
        mate = int(options[int(rng.integers(len(options)))])
        if _min_distance_to(atoms, idx, used) <= ISOLATION_LATTICE_PARAMETERS * a:
            continue
        if _min_distance_to(atoms, mate, used) <= ISOLATION_LATTICE_PARAMETERS * a:
            continue
        pairs.append((idx, mate))
        used += [idx, mate]
        if len(pairs) == n_pairs:
            return pairs
    raise TransmutationError(f"cannot place {n_pairs} separated pairs; placed {len(pairs)}")


def _shell_counts(atoms: Atoms, a: float) -> tuple[list[dict[str, int]], list[dict[str, int]]]:
    symbols = atoms.get_chemical_symbols()
    out = []
    for lists in (first_neighbor_list(atoms, a), second_neighbor_list(atoms, a)):
        counts: list[Counter] = [Counter() for _ in symbols]
        for i, j in zip(lists[0], lists[1], strict=True):
            counts[int(i)][symbols[int(j)]] += 1
        out.append([dict(sorted(c.items())) for c in counts])
    return out[0], out[1]


def apply_transmutation(
    parent: Atoms,
    parent_id: str,
    scenario: TransmutationScenario,
    *,
    n_lattice_sites: int | None = None,
    lattice_parameter: float | None = None,
    defect: DefectConfiguration | None = None,
) -> tuple[Atoms, TransmutationRecord]:
    """Replace host atoms of ``parent`` by the products of ``scenario``; count and ids kept.

    ``parent`` may already hold a defect (then ``defect`` supplies its sites and
    ``n_lattice_sites`` defaults to ``defect.n_parent_atoms``); only host atoms with a parent
    ``atom_id`` are eligible. The concentration limit is measured against the lattice-site count.
    """
    if n_lattice_sites is None:
        n_lattice_sites = defect.n_parent_atoms if defect is not None else len(parent)
    a = lattice_parameter if lattice_parameter is not None else _lattice_parameter(parent)
    limit = math.floor(scenario.max_solute_fraction * n_lattice_sites + 1e-9)
    if scenario.n_solutes > limit:
        raise TransmutationError(
            f"{scenario.n_solutes} product atoms exceed the scenario limit {limit} "
            f"({scenario.max_solute_fraction} of {n_lattice_sites} sites)"
        )
    ids = _atom_ids(parent)
    symbols = parent.get_chemical_symbols()
    unknown = [s for s in scenario.products if s in symbols]
    if unknown:
        raise TransmutationError(f"parent already contains transmutation products: {unknown}")
    eligible = np.array(
        [i for i, s in enumerate(symbols) if s == scenario.host and ids[i] < n_lattice_sites],
        dtype=np.int64,
    )
    if scenario.n_solutes > len(eligible):
        raise TransmutationError(
            f"{scenario.n_solutes} solutes requested but only {len(eligible)} "
            f"{scenario.host} atoms available"
        )

    rng = numpy_rng(scenario.seed, "transmutation", scenario.distribution)
    final_species: dict[int, str] = {}
    dist = scenario.distribution
    if dist == "random":
        picked = [int(i) for i in rng.permutation(eligible)[: scenario.n_solutes]]
    elif dist == "isolated":
        picked = _select_isolated(parent, eligible, scenario.n_solutes, a, rng)
    elif dist == "pair":
        first, second = scenario.products
        pairs = _select_pairs(parent, eligible, scenario.products[first], a, rng)
        picked = []
        for i, j in pairs:
            final_species[i], final_species[j] = first, second
            picked += [i, j]
    elif dist == "enriched":
        center = parent.positions[[int(rng.choice(eligible))]]
        picked = [
            int(i) for i in _ranked_by_distance(parent, eligible, center, rng)[: scenario.n_solutes]
        ]
    else:  # near_defect
        if defect is None or not defect.sites:
            raise TransmutationError("near_defect needs a DefectConfiguration with sites")
        centers = np.array([s.position for s in defect.sites], dtype=float)
        picked = [
            int(i)
            for i in _ranked_by_distance(parent, eligible, centers, rng)[: scenario.n_solutes]
        ]
    if not final_species:
        labels = [el for el, n in scenario.products.items() for _ in range(n)]
        for idx, label in zip(picked, rng.permutation(labels), strict=True):
            final_species[idx] = str(label)

    result = parent.copy()
    result.set_array("atom_id", ids)
    new_symbols = list(symbols)
    for idx, label in final_species.items():
        new_symbols[idx] = label
    result.set_chemical_symbols(new_symbols)
    first, second = _shell_counts(result, a)
    sites = tuple(
        TransmutedSite(
            atom_id=int(ids[idx]),
            lattice_site=int(ids[idx]),
            original_species=symbols[idx],
            final_species=final_species[idx],
            position=tuple(float(x) for x in parent.positions[idx]),
            first_shell=first[idx],
            second_shell=second[idx],
        )
        for idx in sorted(final_species, key=lambda i: int(ids[i]))
    )
    placed = Counter(s.final_species for s in sites)
    product_counts = {el: placed.get(el, 0) for el in scenario.products}
    if product_counts != scenario.products or len(result) != len(parent):
        raise AssertionError("transmutation changed the requested product counts or atom count")
    fingerprint = geometry_fingerprint(result)
    digest = hashlib.sha256(
        json.dumps(
            {
                "parent": parent_id,
                "scenario": scenario.model_dump(mode="json"),
                "sites": [s.atom_id for s in sites],
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:12]
    record = TransmutationRecord(
        config_id=f"transmuted-{digest}",
        parent_id=parent_id,
        parent_ordering_id=occupation_fingerprint(parent),
        scenario=scenario,
        n_lattice_sites=n_lattice_sites,
        n_atoms=len(result),
        parent_composition=dict(sorted(Counter(symbols).items())),
        final_composition=dict(sorted(Counter(result.get_chemical_symbols()).items())),
        host=scenario.host,
        product_counts=product_counts,
        solute_fraction=scenario.n_solutes / n_lattice_sites,
        sites=sites,
        geometry_fingerprint=fingerprint,
        evidence_level=scenario.evidence_level,
    )
    return result, record


__all__ = [
    "DEFAULT_MAX_SOLUTE_FRACTION",
    "Distribution",
    "TransmutationError",
    "TransmutationRecord",
    "TransmutationScenario",
    "TransmutedSite",
    "apply_transmutation",
]
