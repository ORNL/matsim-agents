"""Reproducible ensembles of independent alloy occupations at fixed composition.

Each configuration gets its own random stream derived from one ensemble seed with
``matsim_agents.core.seeding.stream_seed``. Independent seeds do not guarantee distinct or
statistically uncorrelated occupations: exact duplicates are detected and reported (never
retried), and symmetry-equivalent configurations are not detected here. Targeted-SRO members
are bounded-search configurations, not thermodynamic equilibrium structures.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import warnings
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

import numpy as np
from ase import Atoms
from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from matsim_agents.core.seeding import stream_seed
from matsim_agents.science.structures.alloys import (
    AlloySpec,
    ShellSRO,
    SROResult,
    SROTarget,
    UnitCell,
    _shuffled_symbols,
    build_supercell,
    generate_targeted_sro,
    shell_topology,
    warren_cowley_sro,
)

EnsembleMode = Literal["random", "targeted_sro"]


class AlloyEnsembleSpec(BaseModel):
    """Validated description of an ensemble: one parent alloy, a mode, a size and a seed."""

    model_config = ConfigDict(frozen=True)

    alloy: AlloySpec
    n_configurations: StrictInt = Field(gt=0)
    mode: EnsembleMode
    seed: StrictInt = Field(ge=0)
    sro_targets: tuple[SROTarget, ...] = ()
    max_swap_attempts: StrictInt = Field(default=1000, ge=0)
    tolerance: float = Field(default=1e-4, ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def _validate_targets(self) -> AlloyEnsembleSpec:
        if self.mode == "targeted_sro" and not self.sro_targets:
            raise ValueError("targeted_sro mode requires at least one SRO target")
        if self.mode == "random" and self.sro_targets:
            raise ValueError("random mode does not accept SRO targets")
        return self


@dataclass(frozen=True, eq=False)
class ConfigurationRecord:
    """Provenance and ordering descriptors of one configuration (no coordinates).

    ``seed`` is the ``AlloySpec.seed`` that ``generate_alloy`` would use for the starting
    occupation; ``optimizer_seed`` is the ``seed`` passed to ``generate_targeted_sro``. In random
    mode ``optimizer_seed``, ``converged``, ``objective`` and ``attempts`` are None.
    A non-converged targeted configuration is a valid outcome, not an error.
    """

    config_id: str
    parent: AlloySpec
    mode: EnsembleMode
    seed: int
    optimizer_seed: int | None
    counts: dict[str, int]
    fingerprint: str
    sro: SROResult
    converged: bool | None
    objective: float | None
    attempts: int | None

    def metadata(self) -> dict[str, Any]:
        """JSON-serializable record; SRO matrices are indexed by ``species``."""

        def shell(data: ShellSRO) -> dict[str, Any]:
            return {
                "pair_counts": data.pair_counts.tolist(),
                "alpha": data.alpha.tolist(),
            }

        return {
            "config_id": self.config_id,
            "parent": self.parent.model_dump(mode="json"),
            "mode": self.mode,
            "seed": self.seed,
            "optimizer_seed": self.optimizer_seed,
            "counts": dict(self.counts),
            "fingerprint": self.fingerprint,
            "species": list(self.sro.species),
            "first_shell": shell(self.sro.first_shell),
            "second_shell": shell(self.sro.second_shell),
            "converged": self.converged,
            "objective": self.objective,
            "attempts": self.attempts,
        }


@dataclass(frozen=True, eq=False)
class AlloyEnsemble:
    """All configurations of one ensemble, with the structures kept in memory.

    ``duplicate_groups`` lists the IDs of configurations sharing an identical occupation (same
    fingerprint); it says nothing about symmetry-equivalent occupations.
    """

    spec: AlloyEnsembleSpec
    records: tuple[ConfigurationRecord, ...]
    structures: tuple[Atoms, ...]
    duplicate_groups: tuple[tuple[str, ...], ...]

    @property
    def n_unique(self) -> int:
        """Number of distinct exact occupations."""
        return len({record.fingerprint for record in self.records})

    @property
    def expected_random_overlap(self) -> float:
        """E[site overlap] of two independent uniform shuffles of this composition: sum_i c_i^2."""
        return float(sum(c * c for c in self.spec.alloy.composition.values()))

    def overlap_matrix(self) -> np.ndarray:
        """Fraction of sites with the same species for every pair of configurations.

        A descriptive diagnostic only: values near ``expected_random_overlap`` are consistent
        with, but do not prove, statistical independence. Targeted-SRO members are not expected
        to match the random value.
        """
        numbers = np.stack([atoms.numbers for atoms in self.structures])
        return np.array([(numbers == row).mean(axis=1) for row in numbers])

    def overlap_summary(self) -> dict[str, float | None]:
        """Min, mean and max overlap over distinct pairs (None for a single configuration)."""
        matrix = self.overlap_matrix()
        pairs = matrix[np.triu_indices(len(matrix), k=1)]
        stats = (pairs.min(), pairs.mean(), pairs.max()) if len(pairs) else (None, None, None)
        keys = ("min", "mean", "max")
        summary = {k: None if v is None else float(v) for k, v in zip(keys, stats, strict=True)}
        return summary | {"expected_random": self.expected_random_overlap}

    def metadata(self) -> dict[str, Any]:
        return {
            "spec": self.spec.model_dump(mode="json"),
            "n_unique": self.n_unique,
            "duplicate_groups": [list(group) for group in self.duplicate_groups],
            "overlap": self.overlap_summary(),
            "configurations": [record.metadata() for record in self.records],
        }


def occupation_fingerprint(atoms: Atoms) -> str:
    """SHA-256 of the per-site species sequence; identical occupations of one lattice match."""
    symbols = atoms.get_chemical_symbols()
    species = sorted(set(symbols))
    position = {symbol: k for k, symbol in enumerate(species)}
    index = np.fromiter((position[s] for s in symbols), dtype="<u2", count=len(symbols))
    digest = hashlib.sha256()
    digest.update("|".join(species).encode())
    digest.update(b"\n")
    digest.update(index.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True, eq=False)
class ExactDeduplication:
    """Result of exact-occupation deduplication; the input ensemble is not modified.

    ``unique`` holds the first record of each distinct occupation in input order.
    ``duplicate_of`` maps each later duplicate's ``config_id`` to its representative's.
    """

    unique: tuple[ConfigurationRecord, ...]
    duplicate_of: dict[str, str]


def find_exact_duplicates(ensemble: AlloyEnsemble) -> ExactDeduplication:
    """Group configurations whose species sit on identical lattice sites.

    Fingerprints filter candidates and each match is confirmed by comparing the occupation
    arrays, so a hash collision cannot merge distinct occupations. Detection only: nothing is
    removed from ``ensemble``, and symmetry-equivalent occupations are not merged.
    """
    representatives: dict[str, list[int]] = defaultdict(list)
    unique: list[ConfigurationRecord] = []
    duplicate_of: dict[str, str] = {}
    for k, record in enumerate(ensemble.records):
        numbers = ensemble.structures[k].numbers
        match = next(
            (m for m in representatives[record.fingerprint] if _same(ensemble, m, numbers)),
            None,
        )
        if match is None:
            representatives[record.fingerprint].append(k)
            unique.append(record)
        else:
            duplicate_of[record.config_id] = ensemble.records[match].config_id
    return ExactDeduplication(tuple(unique), duplicate_of)


def _same(ensemble: AlloyEnsemble, index: int, numbers: np.ndarray) -> bool:
    return bool(np.array_equal(ensemble.structures[index].numbers, numbers))


@dataclass(frozen=True, eq=False)
class TranslationDeduplication:
    """Deduplication under exact identity and periodic translations of the parent lattice.

    ``unique`` keeps the earliest member of every class. ``exact_duplicate_of`` and
    ``translation_equivalent_to`` map later configuration IDs to their representative's ID:
    the first when the occupation is identical to the representative's, the second when it is
    the representative's occupation shifted by a lattice translation. Point-group operations are
    not considered, so rotated or reflected copies remain unique here.
    """

    unique: tuple[ConfigurationRecord, ...]
    exact_duplicate_of: dict[str, str]
    translation_equivalent_to: dict[str, str]


_KEY_GRID = 1 << 20


def _site_keys(frac: np.ndarray) -> np.ndarray:
    """One int64 per fractional position on a 2**-20 grid, periodic in each axis."""
    grid = np.rint(np.mod(frac, 1.0) * _KEY_GRID).astype(np.int64) % _KEY_GRID
    return (grid[..., 0] * _KEY_GRID + grid[..., 1]) * _KEY_GRID + grid[..., 2]


@lru_cache(maxsize=8)
def _supercell_operations(lattice_key: str) -> tuple[np.ndarray, np.ndarray]:
    """Site permutations of the undecorated periodic supercell described by ``lattice_key``.

    Returns ``(translations, points)``. ``translations`` holds one row per pure lattice
    translation that maps the site set onto itself (shape (n_translations, n_sites)), and
    ``points`` one row per distinct rotation part of the space group (shape (n_operations,
    n_sites)), each taken with one of its fractional translations. Row 0 of each is the identity.
    Operations come from spglib on the supercell itself, so axes are exchanged only when the
    supercell allows it and nonsymmorphic lattices (hcp, diamond) are handled. ``numbers[row]``
    is the occupation moved by that operation. Combining every point row with every translation
    row gives the full symmetry group of the periodic cell.
    """
    import spglib

    fields = json.loads(lattice_key)
    unit_cell = fields.pop("unit_cell")
    probe = AlloySpec.model_construct(
        **fields, unit_cell=unit_cell and UnitCell.model_validate(unit_cell)
    )
    spec = AlloySpec(**fields, unit_cell=unit_cell, counts={"H": probe.n_sites}, seed=0)
    lattice = build_supercell(spec)
    frac = lattice.get_scaled_positions(wrap=True)
    keys = _site_keys(frac)
    order = np.argsort(keys)
    sorted_keys = keys[order]
    cell = (lattice.cell.array, frac, np.ones(len(lattice), dtype=int))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        symmetry = spglib.get_symmetry(cell, symprec=1e-4 * spec.lattice_parameter)
    if symmetry is None:
        raise AssertionError("spglib found no symmetry for the parent supercell")

    def permutation(rotation: np.ndarray, shift: np.ndarray) -> np.ndarray:
        moved = _site_keys(frac @ rotation.T + shift)
        where = np.searchsorted(sorted_keys, moved).clip(max=len(keys) - 1)
        if not np.array_equal(sorted_keys[where], moved):
            raise AssertionError("a symmetry operation left the site lattice")
        return order[where].astype(np.int32)

    identity = np.eye(3, dtype=int)
    translations: dict[bytes, np.ndarray] = {}
    points: dict[bytes, np.ndarray] = {}
    for rotation, shift in zip(symmetry["rotations"], symmetry["translations"], strict=True):
        if np.array_equal(rotation, identity):
            row = permutation(rotation, shift)
            translations.setdefault(row.tobytes(), row)
        elif rotation.tobytes() not in points:
            points[rotation.tobytes()] = permutation(rotation, shift)
    base = np.arange(len(lattice), dtype=np.int32)
    translations.setdefault(base.tobytes(), base)
    t_rows = [base] + [row for row in translations.values() if not np.array_equal(row, base)]
    p_rows = [base, *points.values()]
    result = np.stack(t_rows), np.stack(p_rows)
    for array in result:
        array.setflags(write=False)
    return result


def find_translation_equivalents(ensemble: AlloyEnsemble) -> TranslationDeduplication:
    """Group configurations that are identical or related by a periodic lattice translation.

    Every structure must sit on the ideal supercell of ``ensemble.spec.alloy`` (same cell and
    site positions), otherwise ``ValueError`` is raised. An occupation's canonical form is the
    smallest of its translated copies; translations form a group, so equal canonical forms mean
    equivalence exactly. SRO matrices are never used. Nothing in ``ensemble`` is modified.
    """
    alloy = ensemble.spec.alloy
    _require_parent_lattice(ensemble)
    exact = find_exact_duplicates(ensemble)
    index = {record.config_id: k for k, record in enumerate(ensemble.records)}
    perms = _supercell_operations(alloy.lattice_key)[0]
    representatives: dict[bytes, ConfigurationRecord] = {}
    final: dict[str, str] = {}
    for record in exact.unique:
        numbers = ensemble.structures[index[record.config_id]].numbers.astype(np.uint8)
        key = min(row.tobytes() for row in numbers[perms])
        final[record.config_id] = representatives.setdefault(key, record).config_id
    kept = {record.config_id for record in representatives.values()}
    # A duplicate of a configuration that was itself folded into another class is not identical
    # to the final representative, so it is reported as translation-equivalent.
    exact_of = {d: r for d, r in exact.duplicate_of.items() if r in kept}
    translated = {
        record.config_id: final[exact.duplicate_of.get(record.config_id, record.config_id)]
        for record in ensemble.records
        if record.config_id not in kept and record.config_id not in exact_of
    }
    return TranslationDeduplication(tuple(representatives.values()), exact_of, translated)


def _require_parent_lattice(ensemble: AlloyEnsemble) -> None:
    lattice = build_supercell(ensemble.spec.alloy)
    for record, atoms in zip(ensemble.records, ensemble.structures, strict=True):
        if (
            len(atoms) != len(lattice)
            or not atoms.pbc.all()
            or not np.allclose(atoms.cell.array, lattice.cell.array)
            or not np.allclose(atoms.positions, lattice.positions)
        ):
            raise ValueError(f"{record.config_id} is not on the parent supercell")


@dataclass(frozen=True, eq=False)
class SymmetryDeduplication:
    """Deduplication under the space-group operations of the undecorated periodic supercell.

    Operations are the periodic lattice translations combined with the point operations that
    preserve the supercell, so for a cubic lattice a noncubic cell has fewer than 48.
    ``unique`` keeps the earliest member of every class. The three maps send later configuration
    IDs to their representative's ID and are mutually exclusive: ``exact_duplicate_of`` for an
    identical occupation, ``translation_equivalent_to`` for a pure translation of it, and
    ``point_group_equivalent_to`` when a rotation or reflection (with or without translation) is
    needed. Chemical species are never permuted.
    """

    unique: tuple[ConfigurationRecord, ...]
    exact_duplicate_of: dict[str, str]
    translation_equivalent_to: dict[str, str]
    point_group_equivalent_to: dict[str, str]
    n_point_operations: int
    status: Literal["complete", "incomplete"]
    unchecked: tuple[str, ...]
    n_canonicalized: int


def _lexmin(rows: np.ndarray) -> np.ndarray:
    """Lexicographically smallest row, narrowing the candidates one column at a time."""
    for column in range(rows.shape[1]):
        values = rows[:, column]
        rows = rows[values == values.min()]
        if len(rows) == 1:
            break
    return rows[0]


def _pair_signature(record: ConfigurationRecord) -> tuple[Any, ...]:
    """Integer shell pair counts: invariant under every lattice symmetry, so a filter only."""
    sro = record.sro
    return (
        sro.species,
        sro.first_shell.pair_counts.tobytes(),
        sro.second_shell.pair_counts.tobytes(),
    )


def find_symmetry_equivalents(
    ensemble: AlloyEnsemble, *, max_canonicalizations: int = 256
) -> SymmetryDeduplication:
    """Group configurations related by translations, rotations or reflections of the parent.

    Structures must sit on the ideal supercell of ``ensemble.spec.alloy`` or ``ValueError`` is
    raised. The group is complete for that periodic cell: all lattice translations modulo the
    supercell times every point operation that preserves it.

    Exact duplicates are removed first. Configurations whose integer first- and second-shell pair
    counts differ cannot be equivalent and are never compared; equal counts only nominate
    candidates. Each candidate is reduced to the smallest of its images under the whole group
    (about ``n_point_operations * n_sites**2`` byte lookups), and equal forms mean equivalence
    exactly. At most ``max_canonicalizations`` candidates are reduced; any others are kept as
    unique, listed in ``unchecked`` and the status is ``"incomplete"``. SRO values are never used
    to establish equivalence and nothing in ``ensemble`` is modified.
    """
    if max_canonicalizations < 0:
        raise ValueError("max_canonicalizations must be nonnegative")
    alloy = ensemble.spec.alloy
    _require_parent_lattice(ensemble)
    exact = find_exact_duplicates(ensemble)
    index = {record.config_id: k for k, record in enumerate(ensemble.records)}
    shifts, points = _supercell_operations(alloy.lattice_key)

    candidates: dict[tuple[Any, ...], list[ConfigurationRecord]] = defaultdict(list)
    for record in exact.unique:
        candidates[_pair_signature(record)].append(record)

    representatives: list[ConfigurationRecord] = []
    final: dict[str, str] = {}
    related: dict[str, str] = {}
    unchecked: list[str] = []
    n_canonicalized = 0
    for group in candidates.values():
        classes: dict[bytes, ConfigurationRecord] = {}
        translation_keys: dict[str, bytes] = {}
        for record in group:
            cid = record.config_id
            if len(group) == 1 or n_canonicalized >= max_canonicalizations:
                if len(group) > 1:
                    unchecked.append(cid)
                representatives.append(record)
                final[cid] = cid
                continue
            n_canonicalized += 1
            numbers = ensemble.structures[index[cid]].numbers.astype(np.uint8)
            # Row 0 of ``points`` is the identity, so forms[0] is the translation-only form.
            forms = [_lexmin(numbers[p][shifts]).tobytes() for p in points]
            translation_keys[cid] = forms[0]
            representative = classes.setdefault(min(forms), record)
            final[cid] = representative.config_id
            if representative is record:
                representatives.append(record)
            else:
                same = forms[0] == translation_keys[representative.config_id]
                related[cid] = "translation" if same else "point_group"
    representatives.sort(key=lambda record: index[record.config_id])

    kept = {record.config_id for record in representatives}
    exact_of: dict[str, str] = {}
    translation: dict[str, str] = {}
    point_group: dict[str, str] = {}
    for record in ensemble.records:
        cid = record.config_id
        if cid in kept:
            continue
        source = exact.duplicate_of.get(cid, cid)
        if source == cid:
            kind = related[cid]
        elif source in kept:
            kind = "exact"
        else:
            kind = related[source]
        target = {"exact": exact_of, "translation": translation, "point_group": point_group}[kind]
        target[cid] = final[source]
    return SymmetryDeduplication(
        unique=tuple(representatives),
        exact_duplicate_of=exact_of,
        translation_equivalent_to=translation,
        point_group_equivalent_to=point_group,
        n_point_operations=len(points),
        status="incomplete" if unchecked else "complete",
        unchecked=tuple(unchecked),
        n_canonicalized=n_canonicalized,
    )


@dataclass(frozen=True, eq=False)
class CrossCellDeduplication:
    """Equivalence of decorated BCC crystals that may be given in different supercells.

    ``unique`` lists representative IDs in input order and ``equivalent_to`` maps every other ID
    to its representative. IDs in ``unchecked`` ran out of comparison budget and are kept as
    unique, in which case ``status`` is ``"incomplete"``.
    """

    unique: tuple[str, ...]
    equivalent_to: dict[str, str]
    status: Literal["complete", "incomplete"]
    unchecked: tuple[str, ...]
    n_comparisons: int


@dataclass(frozen=True, eq=False)
class _PeriodicSites:
    cell: np.ndarray
    grid: np.ndarray
    adjugate: np.ndarray
    volume: int
    codes: np.ndarray
    order: np.ndarray
    numbers: np.ndarray


def _encode(keys: np.ndarray, volume: int) -> np.ndarray:
    return (keys[..., 0] * volume + keys[..., 1]) * volume + keys[..., 2]


def _on_bcc_lattice(points: np.ndarray) -> bool:
    parity = points % 2
    return bool((parity == parity[:, :1]).all())


def _bcc_sites(config_id: str, atoms: Atoms, lattice_parameter: float) -> _PeriodicSites:
    """Integer description of a periodic BCC structure in units of a/2, origin on site 0.

    Sites are reduced modulo the cell exactly via the integer adjugate of the cell matrix.
    """
    half = lattice_parameter / 2
    scaled_cell = atoms.cell.array / half
    scaled = (atoms.positions - atoms.positions[0]) / half
    cell = np.rint(scaled_cell).astype(np.int64)
    grid = np.rint(scaled).astype(np.int64)
    det = round(float(np.linalg.det(cell)))
    if not (
        atoms.pbc.all()
        and np.allclose(scaled_cell, cell, atol=1e-4)
        and np.allclose(scaled, grid, atol=1e-4)
        and _on_bcc_lattice(cell)
        and _on_bcc_lattice(grid)
        and abs(det) == 4 * len(atoms)
    ):
        raise ValueError(f"{config_id} is not a periodic BCC supercell with a={lattice_parameter}")
    adjugate = np.rint(np.linalg.inv(cell) * det).astype(np.int64) * np.sign(det)
    volume = abs(det)
    codes = _encode((grid @ adjugate) % volume, volume)
    order = np.argsort(codes)
    if np.unique(codes).size != len(codes):
        raise ValueError(f"{config_id} has two atoms on the same periodic site")
    return _PeriodicSites(cell, grid, adjugate, volume, codes[order], order, atoms.numbers.copy())


@lru_cache(maxsize=1)
def _cubic_operations() -> np.ndarray:
    """The 48 signed permutation matrices of m-3m; index 0 is the identity."""
    eye = np.eye(3, dtype=np.int64)
    ops = [
        eye[list(axes)] * np.array(signs)[:, None]
        for axes in itertools.permutations(range(3))
        for signs in itertools.product((1, -1), repeat=3)
    ]
    return np.stack(ops)


def _align_cubic_axes(config_id: str, atoms: Atoms, lattice_parameter: float) -> Atoms:
    """Copy of ``atoms`` rigidly rotated so its cubic axes lie along x, y and z.

    The axes are read from the second-shell vectors (length ``a``, along the cube edges) of atom 0.
    """
    from ase.neighborlist import neighbor_list

    first, vectors = neighbor_list("iD", atoms, 1.05 * lattice_parameter)
    vectors = vectors[first == 0]
    edges = vectors[np.isclose(np.linalg.norm(vectors, axis=1), lattice_parameter, rtol=1e-3)]
    if len(edges) != 6:
        raise ValueError(f"{config_id} is not a periodic BCC supercell with a={lattice_parameter}")
    u = edges[0] / lattice_parameter
    v = next((e / lattice_parameter for e in edges if abs(e @ u) < 1e-3 * lattice_parameter), None)
    if v is None:
        raise ValueError(f"{config_id} is not a periodic BCC supercell with a={lattice_parameter}")
    rotation = np.array([u, v, np.cross(u, v)])
    aligned = atoms.copy()
    aligned.set_cell(atoms.cell.array @ rotation.T, scale_atoms=False)
    aligned.positions = atoms.positions @ rotation.T
    return aligned


_SHELL_OFFSETS = (
    np.array(list(itertools.product((1, -1), repeat=3)), dtype=np.int64),
    np.array([s * row for row in np.eye(3, dtype=np.int64) for s in (2, -2)]),
)


def _environment_signature(site: _PeriodicSites) -> bytes:
    """Multiset of (species, first-shell and second-shell neighbour counts) over all sites.

    Invariant under every translation and cubic point operation, so unequal signatures rule out
    equivalence; equal signatures only nominate a pair for the exact comparison.
    """
    species, kind = np.unique(site.numbers, return_inverse=True)
    columns = [kind[:, None]]
    for offsets in _SHELL_OFFSETS:
        keys = ((site.grid[:, None] + offsets[None]) @ site.adjugate) % site.volume
        neighbours = kind[site.order[np.searchsorted(site.codes, _encode(keys, site.volume))]]
        columns.append((neighbours[..., None] == np.arange(len(species))).sum(axis=1))
    rows, counts = np.unique(np.hstack(columns), axis=0, return_counts=True)
    return species.tobytes() + rows.astype(np.int32).tobytes() + counts.astype(np.int32).tobytes()


def _maps_onto(a: _PeriodicSites, b: _PeriodicSites) -> bool:
    """True if some cubic operation plus lattice translation carries crystal ``a`` onto ``b``."""
    shifts = b.grid @ b.adjugate
    for rotation in _cubic_operations():
        # Equal volumes, so the rotated lattice of ``a`` lying in that of ``b`` means equality.
        if ((a.cell @ rotation.T @ b.adjugate) % b.volume).any():
            continue
        base = a.grid @ rotation.T @ b.adjugate
        codes = _encode((base[None] + shifts[:, None]) % b.volume, b.volume)
        images = b.numbers[b.order[np.searchsorted(b.codes, codes)]]
        if (images == a.numbers).all(axis=1).any():
            return True
    return False


def find_cross_cell_equivalents(
    structures: Mapping[str, Atoms], lattice_parameter: float, *, max_comparisons: int = 1024
) -> CrossCellDeduplication:
    """Group decorated BCC crystals that are the same infinite crystal, whatever cell describes it.

    Each structure may use any integer BCC supercell (non-orthogonal included), any site order,
    any origin and any rigid orientation. Two crystals are equivalent if a cubic point operation
    and a translation map one lattice and occupation onto the other.

    Every structure gets a symmetry-invariant local-environment signature in O(n_sites) and is
    compared only with representatives sharing it, so distinct configurations cost no pairwise
    work. Each remaining comparison is exact and costs at most ``48 * n_sites**2`` lookups. When
    ``max_comparisons`` is reached the remaining structures are kept as unique and listed in
    ``unchecked``.
    """
    if max_comparisons < 0:
        raise ValueError("max_comparisons must be nonnegative")
    sites = {
        cid: _bcc_sites(cid, _align_cubic_axes(cid, atoms, lattice_parameter), lattice_parameter)
        for cid, atoms in structures.items()
    }
    buckets: dict[bytes, list[str]] = defaultdict(list)
    representatives: list[str] = []
    equivalent_to: dict[str, str] = {}
    unchecked: list[str] = []
    n_comparisons = 0
    for cid, site in sites.items():
        bucket = buckets[_environment_signature(site)]
        match, skipped = None, False
        for rep in bucket:
            other = sites[rep]
            if n_comparisons >= max_comparisons:
                skipped = True
                break
            n_comparisons += 1
            if _maps_onto(site, other):
                match = rep
                break
        if match is not None:
            equivalent_to[cid] = match
        else:
            representatives.append(cid)
            bucket.append(cid)
            if skipped:
                unchecked.append(cid)
    return CrossCellDeduplication(
        unique=tuple(representatives),
        equivalent_to=equivalent_to,
        status="incomplete" if unchecked else "complete",
        unchecked=tuple(unchecked),
        n_comparisons=n_comparisons,
    )


@dataclass(frozen=True, eq=False)
class EnsembleSROStatistics:
    """Mean and sample standard deviation of alpha over the unique configurations.

    Matrices are indexed by ``species`` (rows and columns). ``*_std`` is None when fewer than two
    configurations are included, because the sample deviation is undefined there.
    """

    species: tuple[str, ...]
    n_unique: int
    first_shell_mean: np.ndarray
    first_shell_std: np.ndarray | None
    second_shell_mean: np.ndarray
    second_shell_std: np.ndarray | None


def ensemble_sro_statistics(ensemble: AlloyEnsemble) -> EnsembleSROStatistics:
    """Per-pair mean and sample (M - 1) standard deviation of the stored SRO matrices.

    Only the first occurrence of each exact occupation is included; the stored
    ``ConfigurationRecord.sro`` is used, so no neighbour search is repeated. Variance across
    configurations does not establish convergence or statistical independence.
    """
    unique = find_exact_duplicates(ensemble).unique
    species = unique[0].sro.species
    if any(record.sro.species != species for record in unique):
        raise ValueError("configurations use different species orderings")

    def summarize(shell: str) -> tuple[np.ndarray, np.ndarray | None]:
        alpha = np.stack([getattr(record.sro, shell).alpha for record in unique])
        std = alpha.std(axis=0, ddof=1) if len(unique) > 1 else None
        return alpha.mean(axis=0), std

    first_mean, first_std = summarize("first_shell")
    second_mean, second_std = summarize("second_shell")
    return EnsembleSROStatistics(
        species, len(unique), first_mean, first_std, second_mean, second_std
    )


def generate_alloy_ensemble(
    spec: AlloySpec,
    *,
    n_configurations: int,
    mode: str,
    seed: int,
    sro_targets: Sequence[SROTarget] | None = None,
    max_swap_attempts: int = 1000,
    tolerance: float = 1e-4,
) -> AlloyEnsemble:
    """Generate ``n_configurations`` independently seeded occupations of one parent lattice.

    Per-configuration seeds come from ``matsim_agents.core.seeding.stream_seed``; the global
    NumPy state is untouched. The lattice and its neighbour shells are built once and shared.
    ``random`` mode shuffles the exact species counts. ``targeted_sro`` mode starts from such an
    occupation and runs ``generate_targeted_sro`` for at most ``max_swap_attempts`` swaps, then
    records the achieved SRO and convergence status whether or not the tolerance was met.
    Invalid sizes, modes or targets raise ``ValueError``.
    """
    ensemble = AlloyEnsembleSpec(
        alloy=spec,
        n_configurations=n_configurations,
        mode=mode,
        seed=seed,
        sro_targets=tuple(sro_targets or ()),
        max_swap_attempts=max_swap_attempts,
        tolerance=tolerance,
    )
    lattice = build_supercell(spec)
    topology = shell_topology(lattice, spec.lattice_parameter)
    species = tuple(spec.counts)

    records: list[ConfigurationRecord] = []
    structures: list[Atoms] = []
    for k in range(n_configurations):
        occupation_seed = stream_seed(seed, "alloy-ensemble", "occupation", k)
        optimizer_seed = stream_seed(seed, "alloy-ensemble", "optimizer", k)
        atoms = lattice.copy()
        atoms.set_chemical_symbols(_shuffled_symbols(spec.counts, occupation_seed))
        converged = objective = attempts = None
        used_optimizer_seed = None
        if ensemble.mode == "targeted_sro":
            out = generate_targeted_sro(
                atoms,
                spec.lattice_parameter,
                ensemble.sro_targets,
                seed=optimizer_seed,
                tolerance=ensemble.tolerance,
                max_attempts=ensemble.max_swap_attempts,
                topology=topology,
            )
            atoms = out.atoms
            converged, objective, attempts = out.converged, out.objective, out.attempts
            used_optimizer_seed = optimizer_seed
        sro = warren_cowley_sro(atoms, spec.lattice_parameter, species, topology=topology)
        records.append(
            ConfigurationRecord(
                config_id=f"{ensemble.mode}-{k:04d}",
                parent=spec,
                mode=ensemble.mode,
                seed=occupation_seed,
                optimizer_seed=used_optimizer_seed,
                counts=dict(spec.counts),
                fingerprint=occupation_fingerprint(atoms),
                sro=sro,
                converged=converged,
                objective=objective,
                attempts=attempts,
            )
        )
        structures.append(atoms)

    by_fingerprint: dict[str, list[str]] = defaultdict(list)
    for record in records:
        by_fingerprint[record.fingerprint].append(record.config_id)
    duplicates = tuple(tuple(ids) for ids in by_fingerprint.values() if len(ids) > 1)
    return AlloyEnsemble(ensemble, tuple(records), tuple(structures), duplicates)
