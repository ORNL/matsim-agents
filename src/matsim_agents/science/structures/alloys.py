"""Generic, typed alloy supercell generation with short-range-order control.

The generator is element- and lattice-agnostic. A parent lattice is a named ASE structure
(``sc``, ``bcc``, ``fcc``, ``hcp``, ``diamond``; conventional or primitive cell) or a custom
unit cell given as a cell matrix and fractional site positions. The supercell is either a
diagonal ``repeats`` triple or any nonsingular 3x3 integer supercell matrix, so the site count
is ``len(unit cell) * |det(repeats)|`` (1024 atoms for the default 8x8x8 conventional bcc
cell). Every site is substitutional and occupied so that the requested composition is met
exactly. This module is independent of the frozen E3 structure generator in ``benchmarks/e3``.

Neighbour shells are geometric: shell radii are the distinct pair distances shared by most
sites of the structure, so the same SRO and coordination code serves every lattice.

Seeds are required, nonnegative integers. Each function draws from its own named stream of
``matsim_agents.core.seeding``: ``generate_alloy`` uses ``(seed, "alloy-occupation")`` and
``generate_targeted_sro`` uses ``(seed, "targeted-sro")``, so equal seeds passed to different
functions never share draws.
"""

from __future__ import annotations

import math
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

import numpy as np
from ase import Atoms
from ase.build import bulk, make_supercell
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

from matsim_agents.core.seeding import numpy_rng

CrystalStructure = Literal["sc", "bcc", "fcc", "hcp", "diamond", "custom"]
CellChoice = Literal["conventional", "primitive"]
OrderingMode = Literal["random"]
SROStatus = Literal["converged", "not_converged"]
Vector3 = tuple[float, float, float]
IntVector3 = tuple[int, int, int]
Repeats = IntVector3 | tuple[IntVector3, IntVector3, IntVector3]

DEFAULT_REPEATS = (8, 8, 8)
DEFAULT_LATTICE_PARAMETER = 3.16
IDEAL_HCP_C_OVER_A = math.sqrt(8 / 3)
NAMED_STRUCTURES: tuple[str, ...] = ("sc", "bcc", "fcc", "hcp", "diamond")
_SHELL_TOL = 0.02  # a gap wider than this fraction of the lattice parameter separates shells
_ALPHA_TOL = 1e-9
_ROW_TOL = 1e-6


class UnitCell(BaseModel):
    """A custom parent unit cell: rows of ``cell`` in units of the lattice parameter."""

    model_config = ConfigDict(frozen=True)

    cell: tuple[Vector3, Vector3, Vector3]
    scaled_positions: tuple[Vector3, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate(self) -> UnitCell:
        cell = np.array(self.cell, dtype=float)
        if not np.isfinite(cell).all() or abs(np.linalg.det(cell)) < 1e-8:
            raise ValueError("unit cell must be finite and nonsingular")
        frac = np.array(self.scaled_positions, dtype=float)
        if not np.isfinite(frac).all():
            raise ValueError("scaled positions must be finite")
        delta = frac[:, None, :] - frac[None, :, :]
        delta -= np.rint(delta)
        distance = np.linalg.norm(delta @ cell, axis=-1) + np.eye(len(frac))
        if (distance < 1e-6).any():
            raise ValueError("unit cell has coincident sites")
        return self

    @classmethod
    def from_atoms(cls, atoms: Atoms) -> tuple[UnitCell, float]:
        """Unit cell and length scale (the first cell-vector length, in angstrom) of ``atoms``.

        Species are ignored: the sites become the substitutional sites of the parent lattice.
        """
        if not atoms.pbc.all():
            raise ValueError("a reference structure must be periodic in all three directions")
        cell = np.array(atoms.cell.array, dtype=float)
        scale = float(np.linalg.norm(cell[0]))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("reference structure has no valid cell")
        frac = atoms.get_scaled_positions(wrap=True)
        unit = cls(
            cell=tuple(tuple(float(x) for x in row) for row in cell / scale),
            scaled_positions=tuple(tuple(float(x) for x in row) for row in frac),
        )
        return unit, scale


def reference_atoms(reference: object) -> Atoms:
    """Coerce an ASE ``Atoms``, a pymatgen ``Structure`` or a structure file to ``Atoms``."""
    if isinstance(reference, Atoms):
        return reference.copy()
    if isinstance(reference, (str, os.PathLike)):
        from ase.io import read

        return read(os.fspath(reference))
    try:
        from pymatgen.core import Structure
        from pymatgen.io.ase import AseAtomsAdaptor
    except ImportError as error:  # pragma: no cover - pymatgen is a hard dependency
        raise TypeError(f"unsupported reference structure {type(reference)!r}") from error
    if isinstance(reference, Structure):
        return AseAtomsAdaptor.get_atoms(reference)
    raise TypeError(
        f"reference must be ase.Atoms, pymatgen Structure or a file path, got {type(reference)!r}"
    )


def _repeats_matrix(repeats: Repeats) -> np.ndarray:
    matrix = np.array(repeats, dtype=np.int64)
    return np.diag(matrix) if matrix.ndim == 1 else matrix


class LatticeSpec(BaseModel):
    """Geometry of a periodic supercell on any parent lattice, without chemical occupation.

    ``repeats`` is a diagonal triple or a full integer supercell matrix (rows are supercell
    vectors in units of the unit-cell vectors). ``unit_cell`` is required for, and only allowed
    with, ``crystal_structure="custom"``; ``c_over_a`` only applies to ``hcp`` (default ideal).
    """

    model_config = ConfigDict(frozen=True)

    crystal_structure: CrystalStructure = "bcc"
    repeats: Repeats = DEFAULT_REPEATS
    lattice_parameter: float = Field(default=DEFAULT_LATTICE_PARAMETER, gt=0, allow_inf_nan=False)
    cell_choice: CellChoice = "conventional"
    c_over_a: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    unit_cell: UnitCell | None = None

    @field_validator("repeats")
    @classmethod
    def _validate_repeats(cls, value: Repeats) -> Repeats:
        matrix = np.array(value)
        if matrix.ndim == 1 and (matrix <= 0).any():
            raise ValueError(f"supercell repeats must be positive, got {value}")
        if matrix.ndim == 2 and round(np.linalg.det(matrix)) == 0:
            raise ValueError(f"supercell matrix must be nonsingular, got {value}")
        return value

    @model_validator(mode="after")
    def _validate_lattice(self) -> LatticeSpec:
        if (self.crystal_structure == "custom") != (self.unit_cell is not None):
            raise ValueError('unit_cell is required for, and only allowed with, "custom"')
        if self.c_over_a is not None and self.crystal_structure != "hcp":
            raise ValueError("c_over_a only applies to hcp")
        return self

    @property
    def sites_per_unit_cell(self) -> int:
        return len(build_unit_cell(self))

    @property
    def n_sites(self) -> int:
        return self.sites_per_unit_cell * abs(round(np.linalg.det(_repeats_matrix(self.repeats))))

    @property
    def lattice_key(self) -> str:
        """JSON of the undecorated supercell; hashable and identical for any occupation."""
        return self.model_dump_json(
            include={
                "crystal_structure",
                "repeats",
                "lattice_parameter",
                "cell_choice",
                "c_over_a",
                "unit_cell",
            }
        )


class AlloySpec(LatticeSpec):
    """Validated description of one substitutional alloy supercell on any parent lattice.

    ``counts`` holds the exact nonnegative integer number of atoms per element. Their sum must
    equal the site count of the supercell, so composition is never rounded.
    """

    counts: Mapping[str, StrictInt]
    ordering: OrderingMode = "random"
    seed: StrictInt = Field(ge=0)

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

    @model_validator(mode="after")
    def _validate_counts_fill_sites(self) -> AlloySpec:
        sites = self.n_sites
        total = sum(self.counts.values())
        if total != sites:
            raise ValueError(
                f"counts sum to {total} but the {self.repeats} {self.crystal_structure} "
                f"supercell has {sites} sites"
            )
        return self

    @classmethod
    def from_reference(
        cls,
        reference: object,
        *,
        counts: Mapping[str, int],
        repeats: Repeats,
        seed: int,
    ) -> AlloySpec:
        """Alloy on the lattice of any reference crystal (ASE, pymatgen, CIF or other file).

        The reference supplies geometry only (cell and site positions, any space group, any
        number of sites per cell). Its species are discarded; ``counts`` and ``seed`` set the
        occupation. ``repeats`` is a triple or an integer supercell matrix of the reference cell.
        """
        unit, scale = UnitCell.from_atoms(reference_atoms(reference))
        return cls(
            crystal_structure="custom",
            unit_cell=unit,
            lattice_parameter=scale,
            counts=counts,
            repeats=repeats,
            seed=seed,
        )

    @property
    def composition(self) -> dict[str, float]:
        """Atomic fractions of each element."""
        total = sum(self.counts.values())
        return {symbol: amount / total for symbol, amount in self.counts.items()}


def build_unit_cell(spec: LatticeSpec) -> Atoms:
    """Periodic parent unit cell of ``spec`` with placeholder species ``X``."""
    a = spec.lattice_parameter
    if spec.unit_cell is not None:
        return Atoms(
            ["X"] * len(spec.unit_cell.scaled_positions),
            cell=np.array(spec.unit_cell.cell) * a,
            scaled_positions=spec.unit_cell.scaled_positions,
            pbc=True,
        )
    if spec.crystal_structure == "hcp":
        return bulk("X", "hcp", a=a, covera=spec.c_over_a or IDEAL_HCP_C_OVER_A)
    cubic = spec.cell_choice == "conventional"
    return bulk("X", spec.crystal_structure, a=a, cubic=cubic)


def build_supercell(spec: LatticeSpec) -> Atoms:
    """Periodic supercell of ``spec`` with unoccupied (placeholder ``X``) species.

    Diagonal ``repeats`` replicate the unit cell along its axes (ASE ``repeat`` order); a full
    matrix uses ``ase.build.make_supercell``.
    """
    unit = build_unit_cell(spec)
    matrix = np.array(spec.repeats)
    atoms = unit.repeat(spec.repeats) if matrix.ndim == 1 else make_supercell(unit, matrix)
    if len(atoms) != spec.n_sites:
        raise AssertionError(f"supercell has {len(atoms)} sites, expected {spec.n_sites}")
    return atoms


def shell_radii(atoms: Atoms, lattice_parameter: float, n_shells: int = 3) -> tuple[float, ...]:
    """Radii of the first ``n_shells`` neighbour shells shared by most sites of ``atoms``.

    Sorted directed pair distances are split wherever consecutive values differ by more than
    ``0.02 * lattice_parameter`` (so thermally or chemically distorted shells stay whole), and a
    group counts as a lattice shell when it holds at least ``len(atoms) / 2`` pairs, so the few
    distances introduced by isolated point defects do not displace the lattice shells.
    ``lattice_parameter`` only sets the length scale of the search.
    """
    if not (math.isfinite(lattice_parameter) and lattice_parameter > 0):
        raise ValueError(f"lattice parameter must be positive and finite, got {lattice_parameter}")
    tol = _SHELL_TOL * lattice_parameter
    radius = 1.5 * lattice_parameter
    while radius <= 8 * lattice_parameter:
        distances = np.sort(neighbor_list("d", atoms, radius))
        groups = np.split(distances, np.flatnonzero(np.diff(distances) > tol) + 1)
        shells = [
            float(g.mean()) for g in groups if len(g) >= len(atoms) / 2 and g[-1] < radius - tol
        ]
        if len(shells) >= n_shells:
            return tuple(shells[:n_shells])
        radius *= 1.5
    raise ValueError(f"could not find {n_shells} neighbour shells within 8 lattice parameters")


def _shell_lists(
    atoms: Atoms, radii: tuple[float, ...]
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    inner = 0.5 * (radii[0] + radii[1])
    outer = 0.5 * (radii[1] + radii[2])
    i, j, vectors = neighbor_list("ijD", atoms, outer)
    near = np.linalg.norm(vectors, axis=1) < inner
    return (i[near], j[near], vectors[near]), (i[~near], j[~near], vectors[~near])


def first_neighbor_list(
    atoms: Atoms, lattice_parameter: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Periodic directed (i, j, vector) pairs within the first neighbour shell."""
    return _shell_lists(atoms, shell_radii(atoms, lattice_parameter))[0]


def second_neighbor_list(
    atoms: Atoms, lattice_parameter: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Periodic directed (i, j, vector) pairs within the second neighbour shell."""
    return _shell_lists(atoms, shell_radii(atoms, lattice_parameter))[1]


@dataclass(frozen=True, eq=False)
class ShellTopology:
    """Directed first- and second-shell (i, j) neighbour pairs of one fixed periodic cell.

    Depends only on site positions and cell, so one instance can serve every occupation of the
    same lattice. It is not re-validated against the structure it is used with.
    """

    n_atoms: int
    first: tuple[np.ndarray, np.ndarray]
    second: tuple[np.ndarray, np.ndarray]


def shell_topology(atoms: Atoms, lattice_parameter: float) -> ShellTopology:
    """Neighbour pairs of the first two shells, computed once with ASE cell lists."""
    first, second = _shell_lists(atoms, shell_radii(atoms, lattice_parameter))
    return ShellTopology(len(atoms), first[:2], second[:2])


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


def _pair_counts(index: np.ndarray, i: np.ndarray, j: np.ndarray, n: int) -> np.ndarray:
    return np.bincount(index[i] * n + index[j], minlength=n * n).reshape(n, n)


def _conditional_and_alpha(
    counts: np.ndarray, composition: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    conditional = counts / counts.sum(axis=1)[:, None]
    return conditional, 1.0 - conditional / composition[None, :]


def _shell_sro(
    index: np.ndarray, i: np.ndarray, j: np.ndarray, composition: np.ndarray
) -> ShellSRO:
    counts = _pair_counts(index, i, j, len(composition))
    if not np.array_equal(counts, counts.T):
        raise ValueError("directed pair counts are not reciprocal; neighbour list is inconsistent")
    if (counts.sum(axis=1) == 0).any():
        raise ValueError("a species has no neighbours in this shell")
    conditional, alpha = _conditional_and_alpha(counts, composition)
    for array in (counts, conditional, alpha):
        array.setflags(write=False)
    return ShellSRO(counts, conditional, alpha)


def warren_cowley_sro(
    atoms: Atoms,
    lattice_parameter: float,
    species: tuple[str, ...] | None = None,
    *,
    topology: ShellTopology | None = None,
) -> SROResult:
    """Measured Warren-Cowley parameters alpha_ij = 1 - P(j|i) / c_j for the first two shells.

    c_j is the actual atomic fraction in ``atoms`` and P(j|i) the measured probability that a
    neighbour of an i atom is j, so a finite random alloy reports its nonzero sampled SRO.
    ``species`` fixes the matrix ordering (default: sorted symbols present). ``topology`` reuses
    precomputed neighbour pairs of the same lattice instead of searching again.
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

    if topology is None:
        topology = shell_topology(atoms, lattice_parameter)
    elif topology.n_atoms != len(atoms):
        raise ValueError(f"topology has {topology.n_atoms} atoms, structure has {len(atoms)}")
    return SROResult(
        species=order,
        composition=composition,
        first_shell=_shell_sro(index, *topology.first, composition),
        second_shell=_shell_sro(index, *topology.second, composition),
    )


class SROTarget(BaseModel):
    """Target Warren-Cowley alpha for one species pair in shell 1 or 2.

    alpha_ij equals alpha_ji for a periodic lattice (pair counts are symmetric), so ``pair``
    is unordered.
    """

    model_config = ConfigDict(frozen=True)

    shell: Literal[1, 2]
    pair: tuple[str, str]
    alpha: float = Field(allow_inf_nan=False)
    weight: float = Field(default=1.0, gt=0, allow_inf_nan=False)


class InvalidSROTargetError(ValueError):
    """Targets violate a necessary physical or mathematical constraint (no search is run)."""


@dataclass(frozen=True, eq=False)
class TargetedSROResult:
    """Outcome of ``generate_targeted_sro``: a bounded-search configuration, not an equilibrium.

    ``objective`` is the weighted squared deviation of the achieved targeted alphas, recomputed
    from ``atoms``, and ``sro`` the full measured descriptor of ``atoms``. ``converged`` only
    says that ``objective <= tolerance``; a valid target that is not reached leaves it False
    (``status == "not_converged"``) with the best SRO reported. Invalid targets never produce a
    result: they raise ``InvalidSROTargetError``.
    """

    atoms: Atoms
    sro: SROResult
    objective: float
    initial_objective: float
    attempts: int
    accepted_swaps: int
    seed: int
    tolerance: float
    converged: bool

    @property
    def status(self) -> SROStatus:
        return "converged" if self.converged else "not_converged"


@dataclass(frozen=True, eq=False)
class TargetComparison:
    """One targeted (shell, pair) entry: requested alpha against the measured alpha."""

    shell: int
    pair: tuple[str, str]
    target: float
    achieved: float
    deviation: float
    weight: float


@dataclass(frozen=True, eq=False)
class TargetedSROValidation:
    """Independent re-measurement of a ``TargetedSROResult`` from its returned structure."""

    comparisons: tuple[TargetComparison, ...]
    objective: float
    max_abs_deviation: float
    status: SROStatus
    reproducibility_checked: bool


def _alpha_lower_bound(composition: np.ndarray, a: int, b: int) -> float:
    """P(b|a) <= 1 and P(a|b) <= 1 give alpha_ab >= 1 - 1 / max(c_a, c_b)."""
    return 1.0 - 1.0 / max(composition[a], composition[b])


def _alpha_upper_bound(composition: np.ndarray, a: int) -> float:
    """Diagonal bound: the z N_a bonds of a-atoms hold at most z min(N_a, N_k) non-a ends per k."""
    c_a = composition[a]
    others = sum(min(c_a, c) for k, c in enumerate(composition) if k != a)
    p_min = max(0.0, 1.0 - others / c_a)
    return 1.0 - p_min / c_a


def _canonical_targets(
    targets: Sequence[SROTarget], species: tuple[str, ...], composition: np.ndarray
) -> dict[tuple[int, int, int], SROTarget]:
    """Necessary-condition checks on targets; merge symmetric duplicates by (shell, i, j), i <= j.

    These are cheap bounds from probability normalization and finite counts, not a proof that
    a target is achievable. Violations raise ``InvalidSROTargetError``.
    """
    position = {symbol: k for k, symbol in enumerate(species)}
    merged: dict[tuple[int, int, int], SROTarget] = {}
    for target in targets:
        if target.shell not in (1, 2):
            raise InvalidSROTargetError(f"unsupported coordination shell: {target.shell!r}")
        if not math.isfinite(target.alpha):
            raise InvalidSROTargetError(f"target alpha must be finite, got {target.alpha}")
        missing = [symbol for symbol in target.pair if symbol not in position]
        if missing:
            raise InvalidSROTargetError(f"target species not in structure: {missing}")
        a, b = sorted(position[symbol] for symbol in target.pair)
        lower = _alpha_lower_bound(composition, a, b)
        upper = 1.0 if a != b else _alpha_upper_bound(composition, a)
        if not lower - _ALPHA_TOL <= target.alpha <= upper + _ALPHA_TOL:
            raise InvalidSROTargetError(
                f"target alpha {target.alpha} for {target.pair} is outside the attainable "
                f"range [{lower:.6g}, {upper:.6g}] for this composition"
            )
        key = (target.shell, a, b)
        if key in merged and abs(merged[key].alpha - target.alpha) > _ALPHA_TOL:
            raise InvalidSROTargetError(
                f"contradictory targets for shell {target.shell} pair {target.pair}: "
                f"alpha_ij = alpha_ji but got {merged[key].alpha} and {target.alpha}"
            )
        merged.setdefault(key, target)
    n = len(species)
    for shell in (1, 2):
        for a in range(n):
            row = [merged.get((shell, min(a, b), max(a, b))) for b in range(n)]
            given = [(c, t) for c, t in zip(composition, row, strict=True) if t is not None]
            if not given:
                continue
            if len(given) == n:
                total = sum(c * t.alpha for c, t in given)
                if abs(total) > _ROW_TOL:
                    raise InvalidSROTargetError(
                        f"targets for shell {shell} species {species[a]} violate "
                        f"sum_j c_j alpha_ij = 0 (got {total:.3g})"
                    )
            else:
                # Specified P(j|i) = c_j (1 - alpha_ij) cannot already exceed total probability 1.
                probability = sum(c * (1.0 - t.alpha) for c, t in given)
                if probability > 1.0 + _ROW_TOL:
                    raise InvalidSROTargetError(
                        f"targets for shell {shell} species {species[a]} imply specified "
                        f"conditional probabilities summing to {probability:.6g} > 1"
                    )
    return merged


def _target_objective(
    alpha: Mapping[int, np.ndarray], merged: Mapping[tuple[int, int, int], SROTarget]
) -> float:
    """Single objective arithmetic for the optimizer and the validator."""
    return sum(t.weight * float(alpha[s][a, b] - t.alpha) ** 2 for (s, a, b), t in merged.items())


def generate_targeted_sro(
    atoms: Atoms,
    lattice_parameter: float,
    targets: Sequence[SROTarget],
    *,
    seed: int,
    tolerance: float = 1e-4,
    max_attempts: int = 20_000,
    topology: ShellTopology | None = None,
) -> TargetedSROResult:
    """Swap species between lattice sites to approach target Warren-Cowley parameters.

    Greedy, bounded search: each attempt swaps two sites of different species and is kept only
    if it strictly lowers L = sum w * (alpha - alpha_target)^2 over the listed targets. Sites,
    atom count and exact species counts never change. The search stops when L <= ``tolerance``
    or after ``max_attempts`` swaps. A finite periodic cell only admits discrete alpha values,
    so a target inside the attainable range may still be missed; the achieved SRO is reported
    either way. Targets listed for (i, j) and (j, i) are the same quantity and are merged
    (first weight kept); conflicting values, unknown species, values outside the attainable
    range, or full rows breaking sum_j c_j alpha_ij = 0 raise ``ValueError``. ``topology``
    reuses neighbour pairs computed for the same lattice by ``shell_topology``.
    """
    if not targets:
        raise ValueError("at least one target is required")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a nonnegative integer, got {seed!r}")
    if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts < 0:
        raise ValueError(f"max_attempts must be a nonnegative integer, got {max_attempts!r}")
    if not (math.isfinite(tolerance) and tolerance >= 0):
        raise ValueError(f"tolerance must be finite and nonnegative, got {tolerance}")

    initial = warren_cowley_sro(atoms, lattice_parameter, topology=topology)
    species = initial.species
    if len(species) < 2:
        raise ValueError("at least two species are required to swap")
    n_atoms = len(atoms)
    if topology is None:
        topology = shell_topology(atoms, lattice_parameter)
    pairs = {1: topology.first, 2: topology.second}
    for shell, (i, _) in pairs.items():
        coordination = np.bincount(i, minlength=n_atoms)
        if coordination.min() == 0 or coordination.min() != coordination.max():
            raise ValueError(f"shell {shell} coordination is not uniform: not a perfect lattice")

    merged = _canonical_targets(targets, species, initial.composition)
    n = len(species)
    composition = initial.composition

    def objective(index: np.ndarray) -> float:
        alpha = {
            shell: _conditional_and_alpha(_pair_counts(index, i, j, n), composition)[1]
            for shell, (i, j) in pairs.items()
        }
        return _target_objective(alpha, merged)

    position = {symbol: k for k, symbol in enumerate(species)}
    index = np.array([position[symbol] for symbol in atoms.get_chemical_symbols()])
    current = initial_objective = objective(index)
    rng = numpy_rng(seed, "targeted-sro")
    attempts = accepted = 0
    # Only strict improvements are kept, so the current state is always the best one found.
    while current > tolerance and attempts < max_attempts:
        attempts += 1
        a = int(rng.integers(n_atoms))
        b = int(rng.choice(np.flatnonzero(index != index[a])))
        index[a], index[b] = index[b], index[a]
        trial = objective(index)
        if trial < current:
            current = trial
            accepted += 1
        else:
            index[a], index[b] = index[b], index[a]

    result = atoms.copy()
    result.set_chemical_symbols([species[k] for k in index])
    return TargetedSROResult(
        atoms=result,
        sro=warren_cowley_sro(result, lattice_parameter, species, topology=topology),
        objective=current,
        initial_objective=initial_objective,
        attempts=attempts,
        accepted_swaps=accepted,
        seed=seed,
        tolerance=tolerance,
        converged=current <= tolerance,
    )


def validate_targeted_sro(
    start: Atoms,
    result: TargetedSROResult,
    lattice_parameter: float,
    targets: Sequence[SROTarget],
    *,
    check_reproducibility: bool = True,
    atol: float = 1e-9,
) -> TargetedSROValidation:
    """Re-measure ``result.atoms`` and check every reported quantity against it.

    Nothing cached by the optimizer is trusted: alpha, the objective and the convergence flag
    are recomputed from the returned structure; counts, sites and cell are compared with
    ``start``; and (by default) the search is replayed from ``result.seed`` with the same
    tolerance and ``result.attempts`` budget, which must reproduce the structure exactly.
    Any inconsistency raises ``ValueError``; otherwise the achieved-versus-target comparison is
    returned, with ``status`` taken from the recomputed objective.
    """
    final, initial = result.atoms, start
    if len(final) != len(initial):
        raise ValueError("returned structure has a different number of atoms")
    if Counter(final.get_chemical_symbols()) != Counter(initial.get_chemical_symbols()):
        raise ValueError("elemental counts changed")
    if not np.array_equal(final.positions, initial.positions):
        raise ValueError("lattice-site positions changed")
    if not (np.array_equal(final.cell, initial.cell) and np.array_equal(final.pbc, initial.pbc)):
        raise ValueError("periodic cell changed")

    measured = warren_cowley_sro(final, lattice_parameter, result.sro.species)
    if not np.allclose(result.sro.composition, measured.composition, rtol=0.0, atol=atol):
        raise ValueError("reported composition does not match the returned structure")
    for name in ("first_shell", "second_shell"):
        reported, fresh = getattr(result.sro, name), getattr(measured, name)
        if not (
            np.array_equal(reported.pair_counts, fresh.pair_counts)
            and np.allclose(
                reported.conditional_probability,
                fresh.conditional_probability,
                rtol=0.0,
                atol=atol,
            )
            and np.allclose(reported.alpha, fresh.alpha, rtol=0.0, atol=atol)
        ):
            raise ValueError(f"reported {name} SRO does not match the returned structure")

    merged = _canonical_targets(targets, measured.species, measured.composition)
    comparisons = []
    for (shell, a, b), target in merged.items():
        alpha = (measured.first_shell if shell == 1 else measured.second_shell).alpha[a, b]
        comparisons.append(
            TargetComparison(
                shell=shell,
                pair=(measured.species[a], measured.species[b]),
                target=target.alpha,
                achieved=float(alpha),
                deviation=float(alpha - target.alpha),
                weight=target.weight,
            )
        )
    objective = _target_objective(
        {1: measured.first_shell.alpha, 2: measured.second_shell.alpha}, merged
    )
    if not math.isclose(objective, result.objective, rel_tol=1e-9, abs_tol=atol):
        raise ValueError(f"reported objective {result.objective} != recomputed {objective}")
    if (objective <= result.tolerance) != result.converged:
        raise ValueError("convergence flag disagrees with the recomputed objective")

    initial_measured = warren_cowley_sro(initial, lattice_parameter, measured.species)
    initial_objective = _target_objective(
        {1: initial_measured.first_shell.alpha, 2: initial_measured.second_shell.alpha}, merged
    )
    if not math.isclose(initial_objective, result.initial_objective, rel_tol=1e-9, abs_tol=atol):
        raise ValueError("reported initial objective does not match the start structure")

    if check_reproducibility:
        replay = generate_targeted_sro(
            initial,
            lattice_parameter,
            targets,
            seed=result.seed,
            tolerance=result.tolerance,
            max_attempts=result.attempts,
        )
        if replay.atoms.get_chemical_symbols() != final.get_chemical_symbols():
            raise ValueError("result is not reproducible from its seed")

    return TargetedSROValidation(
        comparisons=tuple(comparisons),
        objective=float(objective),
        max_abs_deviation=max(abs(c.deviation) for c in comparisons),
        status="converged" if objective <= result.tolerance else "not_converged",
        reproducibility_checked=check_reproducibility,
    )


def _shuffled_symbols(counts: Mapping[str, int], seed: int) -> list[str]:
    symbols = [symbol for symbol, count in counts.items() for _ in range(count)]
    rng = numpy_rng(seed, "alloy-occupation")
    return [symbols[i] for i in rng.permutation(len(symbols))]


def assign_species(spec: AlloySpec) -> list[str]:
    """Species per site: exact counts, shuffled by a generator seeded from ``spec.seed``."""
    return _shuffled_symbols(spec.counts, spec.seed)


def generate_alloy(spec: AlloySpec) -> Atoms:
    """Build an alloy supercell whose species counts match ``spec`` exactly."""
    atoms = build_supercell(spec)
    atoms.set_chemical_symbols(assign_species(spec))
    atoms.info["alloy_spec"] = spec.model_dump_json()
    return atoms
