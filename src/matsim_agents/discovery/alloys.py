"""Generic, typed BCC alloy supercell generation with short-range-order control.

The generator is element-agnostic: any set of elements and integer counts is accepted,
independently of the frozen E3 structure generator in ``benchmarks/e3``. A cell is a
``2 * nx * ny * nz`` conventional BCC supercell (1024 atoms for 8x8x8), occupied so that
the requested composition is met exactly.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.neighborlist import neighbor_list
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    field_serializer,
    field_validator,
    model_validator,
)
from pymatgen.core import Element

CrystalStructure = Literal["bcc"]
OrderingMode = Literal["random"]

DEFAULT_REPEATS = (8, 8, 8)
DEFAULT_LATTICE_PARAMETER = 3.16
BCC_FIRST_SHELL = 8
BCC_SECOND_SHELL = 6
BCC_FIRST_SHELL_RATIO = float(np.sqrt(3) / 2)
BCC_SECOND_SHELL_RATIO = 1.0
BCC_THIRD_SHELL_RATIO = float(np.sqrt(2))


class AlloySpec(BaseModel):
    """Validated description of one BCC alloy supercell.

    ``counts`` holds the exact nonnegative integer number of atoms per element. Their sum
    must equal the site count of the ``repeats`` supercell, so composition is never rounded.
    """

    model_config = ConfigDict(frozen=True)

    crystal_structure: CrystalStructure = "bcc"
    counts: Mapping[str, StrictInt]
    repeats: tuple[int, int, int] = DEFAULT_REPEATS
    lattice_parameter: float = Field(default=DEFAULT_LATTICE_PARAMETER, gt=0, allow_inf_nan=False)
    ordering: OrderingMode = "random"
    seed: int = Field(default=0, ge=0)

    @field_serializer("counts")
    def _serialize_counts(self, value: Mapping[str, int]) -> dict[str, int]:
        return dict(value)

    @field_validator("counts")
    @classmethod
    def _validate_counts(cls, value: Mapping[str, int]) -> Mapping[str, int]:
        for symbol, amount in value.items():
            try:
                Element(symbol)
            except ValueError as exc:
                raise ValueError(f"unknown element symbol: {symbol!r}") from exc
            if amount < 0:
                raise ValueError(f"count for {symbol} must be nonnegative, got {amount}")
        nonzero = {symbol: amount for symbol, amount in value.items() if amount > 0}
        if not nonzero:
            raise ValueError("composition must contain at least one atom")
        return MappingProxyType(dict(sorted(nonzero.items())))

    @field_validator("repeats")
    @classmethod
    def _validate_repeats(cls, value: tuple[int, int, int]) -> tuple[int, int, int]:
        if any(n <= 0 for n in value):
            raise ValueError(f"supercell repeats must be positive, got {value}")
        return value

    @model_validator(mode="after")
    def _validate_site_count(self) -> AlloySpec:
        sites = bcc_site_count(self.repeats)
        total = sum(self.counts.values())
        if total != sites:
            raise ValueError(
                f"counts sum to {total} but the {self.repeats} BCC supercell has {sites} sites"
            )
        return self

    @property
    def n_sites(self) -> int:
        return bcc_site_count(self.repeats)

    @property
    def composition(self) -> dict[str, float]:
        """Atomic fractions of each element."""
        total = sum(self.counts.values())
        return {symbol: amount / total for symbol, amount in self.counts.items()}


def bcc_site_count(repeats: tuple[int, int, int]) -> int:
    """Number of atoms in a conventional BCC supercell with the given repeats."""
    nx, ny, nz = repeats
    return 2 * nx * ny * nz


def build_bcc_supercell(spec: AlloySpec) -> Atoms:
    """Periodic conventional cubic BCC supercell with unoccupied (placeholder) species.

    The cell is built from the two-atom conventional BCC unit cell, replicated
    ``spec.repeats`` times along each axis, with ``cubic=True`` so the cell is orthorhombic.
    """
    unit = bulk("W", crystalstructure="bcc", a=spec.lattice_parameter, cubic=True)
    atoms = unit.repeat(spec.repeats)
    if len(atoms) != spec.n_sites:
        raise AssertionError(f"BCC supercell has {len(atoms)} sites, expected {spec.n_sites}")
    return atoms


def first_neighbor_list(
    atoms: Atoms, lattice_parameter: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Periodic directed (i, j, vector) pairs within the first BCC shell, via ASE cell lists."""
    # Midpoint of the gap between the first and second shells.
    cutoff = 0.5 * (BCC_FIRST_SHELL_RATIO + BCC_SECOND_SHELL_RATIO) * lattice_parameter
    i, j, vectors = neighbor_list("ijD", atoms, cutoff)
    return i, j, vectors


def second_neighbor_list(
    atoms: Atoms, lattice_parameter: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Periodic directed (i, j, vector) pairs within the second BCC shell (distance a)."""
    inner = 0.5 * (BCC_FIRST_SHELL_RATIO + BCC_SECOND_SHELL_RATIO) * lattice_parameter
    outer = 0.5 * (BCC_SECOND_SHELL_RATIO + BCC_THIRD_SHELL_RATIO) * lattice_parameter
    i, j, vectors = neighbor_list("ijD", atoms, outer)
    keep = np.linalg.norm(vectors, axis=1) >= inner
    return i[keep], j[keep], vectors[keep]


@dataclass(frozen=True, eq=False)
class ShellSRO:
    """Warren-Cowley data for one shell; matrices are indexed by ``SROResult.species``.

    ``pair_counts[i, j]`` counts directed pairs (atom of species i, neighbour of species j),
    so each bond is counted once from each end and the matrix is symmetric.
    """

    pair_counts: np.ndarray
    conditional_probability: np.ndarray
    alpha: np.ndarray


@dataclass(frozen=True, eq=False)
class SROResult:
    """Measured first- and second-shell Warren-Cowley parameters of one structure."""

    species: tuple[str, ...]
    composition: np.ndarray
    first_shell: ShellSRO
    second_shell: ShellSRO


def _shell_sro(
    index: np.ndarray, i: np.ndarray, j: np.ndarray, composition: np.ndarray
) -> ShellSRO:
    n = len(composition)
    counts = np.bincount(index[i] * n + index[j], minlength=n * n).reshape(n, n)
    if not np.array_equal(counts, counts.T):
        raise ValueError("directed pair counts are not reciprocal; neighbour list is inconsistent")
    neighbours = counts.sum(axis=1)
    if (neighbours == 0).any():
        raise ValueError("a species has no neighbours in this shell")
    conditional = counts / neighbours[:, None]
    alpha = 1.0 - conditional / composition[None, :]
    for array in (counts, conditional, alpha):
        array.setflags(write=False)
    return ShellSRO(counts, conditional, alpha)


def warren_cowley_sro(
    atoms: Atoms, lattice_parameter: float, species: tuple[str, ...] | None = None
) -> SROResult:
    """Measured Warren-Cowley parameters alpha_ij = 1 - P(j|i) / c_j for the first two BCC shells.

    c_j is the actual atomic fraction in ``atoms`` and P(j|i) the measured probability that a
    neighbour of an i atom is j, so a finite random alloy reports its nonzero sampled SRO.
    ``species`` fixes the matrix ordering (default: sorted symbols present).
    """
    if len(atoms) == 0:
        raise ValueError("structure has no atoms")
    if not atoms.pbc.all():
        raise ValueError("structure must be periodic in all directions")
    if not (math.isfinite(lattice_parameter) and lattice_parameter > 0):
        raise ValueError(f"lattice parameter must be positive and finite, got {lattice_parameter}")

    symbols = atoms.get_chemical_symbols()
    order = tuple(sorted(set(symbols)) if species is None else species)
    if len(set(order)) != len(order):
        raise ValueError(f"duplicate species in {order}")
    unlisted = sorted(set(symbols) - set(order))
    if unlisted:
        raise ValueError(f"structure contains species not listed: {unlisted}")
    position = {symbol: k for k, symbol in enumerate(order)}
    index = np.array([position[symbol] for symbol in symbols])
    counts = np.bincount(index, minlength=len(order))
    absent = [symbol for symbol, count in zip(order, counts, strict=True) if count == 0]
    if absent:
        raise ValueError(f"species absent from structure: {absent}")
    composition = counts / len(atoms)
    composition.setflags(write=False)

    first_i, first_j, _ = first_neighbor_list(atoms, lattice_parameter)
    second_i, second_j, _ = second_neighbor_list(atoms, lattice_parameter)
    return SROResult(
        species=order,
        composition=composition,
        first_shell=_shell_sro(index, first_i, first_j, composition),
        second_shell=_shell_sro(index, second_i, second_j, composition),
    )


def assign_species(spec: AlloySpec) -> list[str]:
    """Species per site: exact counts, shuffled by a generator seeded from ``spec.seed``."""
    symbols = [symbol for symbol, count in spec.counts.items() for _ in range(count)]
    rng = np.random.default_rng(spec.seed)
    return [symbols[i] for i in rng.permutation(len(symbols))]


def generate_alloy(spec: AlloySpec) -> Atoms:
    """Build a BCC alloy supercell whose species counts match ``spec`` exactly."""
    atoms = build_bcc_supercell(spec)
    atoms.set_chemical_symbols(assign_species(spec))
    atoms.info["alloy_spec"] = spec.model_dump_json()
    return atoms
