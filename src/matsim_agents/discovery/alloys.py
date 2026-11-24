"""Generic, typed BCC alloy supercell generation with short-range-order control.

The generator is element-agnostic: any set of elements and integer counts is accepted,
independently of the frozen E3 structure generator in ``benchmarks/e3``. A cell is a
``2 * nx * ny * nz`` conventional BCC supercell (1024 atoms for 8x8x8), occupied so that
the requested composition is met exactly.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from ase import Atoms
from ase.build import bulk
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pymatgen.core import Element

CrystalStructure = Literal["bcc"]
OrderingMode = Literal["random"]

DEFAULT_REPEATS = (8, 8, 8)
DEFAULT_LATTICE_PARAMETER = 3.16
BCC_FIRST_SHELL = 8
BCC_SECOND_SHELL = 6


class AlloySpec(BaseModel):
    """Validated description of one BCC alloy supercell.

    ``counts`` holds the exact nonnegative integer number of atoms per element. Their sum
    must equal the site count of the ``repeats`` supercell, so composition is never rounded.
    """

    model_config = ConfigDict(frozen=True)

    crystal_structure: CrystalStructure = "bcc"
    counts: dict[str, int] = Field(min_length=1)
    repeats: tuple[int, int, int] = DEFAULT_REPEATS
    lattice_parameter: float = Field(default=DEFAULT_LATTICE_PARAMETER, gt=0)
    ordering: OrderingMode = "random"
    seed: int = Field(default=0, ge=0)

    @field_validator("counts")
    @classmethod
    def _validate_counts(cls, value: dict[str, int]) -> dict[str, int]:
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
        return dict(sorted(nonzero.items()))

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
