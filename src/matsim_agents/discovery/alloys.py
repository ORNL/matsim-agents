"""Generic, typed BCC alloy supercell generation with short-range-order control.

The generator is element-agnostic: any set of elements and integer counts is accepted,
independently of the frozen E3 structure generator in ``benchmarks/e3``. A cell is a
``2 * nx * ny * nz`` conventional BCC supercell (1024 atoms for 8x8x8), occupied so that
the requested composition is met exactly.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
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
SROStatus = Literal["converged", "not_converged"]

DEFAULT_REPEATS = (8, 8, 8)
DEFAULT_LATTICE_PARAMETER = 3.16
BCC_FIRST_SHELL = 8
BCC_SECOND_SHELL = 6
BCC_FIRST_SHELL_RATIO = float(np.sqrt(3) / 2)
BCC_SECOND_SHELL_RATIO = 1.0
BCC_THIRD_SHELL_RATIO = float(np.sqrt(2))
_ALPHA_TOL = 1e-9
_ROW_TOL = 1e-6


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


def generate_targeted_sro(
    atoms: Atoms,
    lattice_parameter: float,
    targets: Sequence[SROTarget],
    *,
    seed: int,
    tolerance: float = 1e-4,
    max_attempts: int = 20_000,
) -> TargetedSROResult:
    """Swap species between sites of a BCC alloy to approach target Warren-Cowley parameters.

    Greedy, bounded search: each attempt swaps two sites of different species and is kept only
    if it strictly lowers L = sum w * (alpha - alpha_target)^2 over the listed targets. Sites,
    atom count and exact species counts never change. The search stops when L <= ``tolerance``
    or after ``max_attempts`` swaps. A finite periodic cell only admits discrete alpha values,
    so a target inside the attainable range may still be missed; the achieved SRO is reported
    either way. Targets listed for (i, j) and (j, i) are the same quantity and are merged
    (first weight kept); conflicting values, unknown species, values outside the attainable
    range, or full rows breaking sum_j c_j alpha_ij = 0 raise ``ValueError``.
    """
    if not targets:
        raise ValueError("at least one target is required")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a nonnegative integer, got {seed!r}")
    if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts < 0:
        raise ValueError(f"max_attempts must be a nonnegative integer, got {max_attempts!r}")
    if not (math.isfinite(tolerance) and tolerance >= 0):
        raise ValueError(f"tolerance must be finite and nonnegative, got {tolerance}")

    initial = warren_cowley_sro(atoms, lattice_parameter)
    species = initial.species
    if len(species) < 2:
        raise ValueError("at least two species are required to swap")
    n_atoms = len(atoms)
    pairs = {
        1: first_neighbor_list(atoms, lattice_parameter)[:2],
        2: second_neighbor_list(atoms, lattice_parameter)[:2],
    }
    for shell, expected in ((1, BCC_FIRST_SHELL), (2, BCC_SECOND_SHELL)):
        if not (np.bincount(pairs[shell][0], minlength=n_atoms) == expected).all():
            raise ValueError(f"shell {shell} is not {expected}-fold coordinated: not a BCC lattice")

    merged = _canonical_targets(targets, species, initial.composition)
    n = len(species)
    shells = np.array([key[0] for key in merged])
    rows = np.array([key[1] for key in merged])
    cols = np.array([key[2] for key in merged])
    wanted = np.array([t.alpha for t in merged.values()])
    weights = np.array([t.weight for t in merged.values()])
    species_counts = np.rint(initial.composition * n_atoms)
    coordination = {1: BCC_FIRST_SHELL, 2: BCC_SECOND_SHELL}
    # alpha_ij = 1 - counts_ij * N / (z * N_i * N_j)
    scale = {
        s: n_atoms / (coordination[s] * np.outer(species_counts, species_counts)) for s in (1, 2)
    }

    def objective(index: np.ndarray) -> float:
        alpha = {}
        for shell, (i, j) in pairs.items():
            counts = np.bincount(index[i] * n + index[j], minlength=n * n).reshape(n, n)
            alpha[shell] = 1.0 - counts * scale[shell]
        achieved = np.array([alpha[s][a, b] for s, a, b in zip(shells, rows, cols, strict=True)])
        return float(np.sum(weights * (achieved - wanted) ** 2))

    position = {symbol: k for k, symbol in enumerate(species)}
    index = np.array([position[symbol] for symbol in atoms.get_chemical_symbols()])
    current = initial_objective = objective(index)
    rng = np.random.default_rng(seed)
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
        sro=warren_cowley_sro(result, lattice_parameter, species),
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
    for name in ("first_shell", "second_shell"):
        reported, fresh = getattr(result.sro, name), getattr(measured, name)
        if not (
            np.array_equal(reported.pair_counts, fresh.pair_counts)
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
    objective = sum(c.weight * c.deviation**2 for c in comparisons)
    if not math.isclose(objective, result.objective, rel_tol=1e-9, abs_tol=atol):
        raise ValueError(f"reported objective {result.objective} != recomputed {objective}")
    if (objective <= result.tolerance) != result.converged:
        raise ValueError("convergence flag disagrees with the recomputed objective")

    initial_measured = warren_cowley_sro(initial, lattice_parameter, measured.species)
    initial_objective = sum(
        t.weight
        * (
            (initial_measured.first_shell if s == 1 else initial_measured.second_shell).alpha[a, b]
            - t.alpha
        )
        ** 2
        for (s, a, b), t in merged.items()
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
