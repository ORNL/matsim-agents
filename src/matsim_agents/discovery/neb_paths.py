"""NEB path geometry: identity-based correspondence and periodic displacements.

Atoms are matched between structures by the integer ``atom_id`` array, never by ASE index, so
reordered arrays are handled. Displacements use the true minimum image: the fractional vector is
rounded in the Minkowski-reduced cell and the 27 neighbouring images are searched. This is exact
for skewed cells, where plain rounding is not. ASE 3.28 ``find_mic`` was observed to return
non-shortest vectors for single-vector calls on a skewed cell while being exact for a large batch
(``tests/test_neb_paths.py``), so it is not relied on here. A displacement whose shortest image is
not unique is ambiguous (for example exactly half a cell vector) and is reported or rejected; it
is never resolved silently.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
from ase import Atoms
from ase.geometry import minkowski_reduce

from matsim_agents.discovery.neb_contracts import AtomCorrespondence

CELL_TOLERANCE = 1e-6
AMBIGUITY_TOLERANCE = 1e-6  # angstrom; shortest images closer than this to each other tie


@dataclass(frozen=True)
class PeriodicDisplacements:
    """Per-atom minimum-image displacement from ``initial`` to ``final``, in initial order."""

    atom_ids: tuple[int, ...]
    species: tuple[str, ...]
    vectors: np.ndarray  # (N, 3), angstrom
    lengths: np.ndarray  # (N,)
    unwrapped_final: np.ndarray  # initial.positions + vectors; may lie outside the cell
    ambiguous_atom_ids: tuple[int, ...]

    @property
    def max_length(self) -> float:
        return float(self.lengths.max()) if len(self.lengths) else 0.0


def _identities(atoms: Atoms, name: str) -> np.ndarray:
    if "atom_id" not in atoms.arrays:
        raise ValueError(f"{name} structure has no atom_id array; identities are required")
    ids = np.asarray(atoms.get_array("atom_id"))
    if not np.issubdtype(ids.dtype, np.integer):
        raise ValueError(f"{name} atom_id array must be integer")
    if len(np.unique(ids)) != len(ids):
        raise ValueError(f"{name} structure has duplicate atom_id values")
    if (ids < 0).any():
        raise ValueError(f"{name} atom_id values must be nonnegative")
    if not np.isfinite(atoms.positions).all():
        raise ValueError(f"{name} structure has non-finite coordinates")
    return ids


def _check_cell(initial: Atoms, final: Atoms) -> None:
    if not (initial.pbc.all() and final.pbc.all()):
        raise ValueError("structures must be fully periodic")
    cell = initial.cell.array
    if not np.isfinite(cell).all() or abs(np.linalg.det(cell)) < 1e-12:
        raise ValueError("cell must be finite and nonsingular")
    if not np.allclose(cell, final.cell.array, rtol=0.0, atol=CELL_TOLERANCE):
        raise ValueError("cells differ; only fixed-cell paths are supported")


def correspond_atoms(initial: Atoms, final: Atoms) -> AtomCorrespondence:
    """One-to-one correspondence by ``atom_id`` in initial order (validates species and cell)."""
    initial_ids = _identities(initial, "initial")
    final_ids = _identities(final, "final")
    _check_cell(initial, final)
    if len(initial_ids) != len(final_ids):
        raise ValueError(f"atom counts differ: {len(initial_ids)} initial, {len(final_ids)} final")
    if set(initial_ids.tolist()) != set(final_ids.tolist()):
        raise ValueError("structures do not contain the same atomic identities")
    final_index = {int(atom_id): k for k, atom_id in enumerate(final_ids)}
    final_indices = tuple(final_index[int(atom_id)] for atom_id in initial_ids)
    for k, atom_id in enumerate(initial_ids):
        a, b = initial[k].symbol, final[final_indices[k]].symbol
        if a != b:
            raise ValueError(f"species of atom {int(atom_id)} differs: {a} initial, {b} final")
    return AtomCorrespondence(
        atom_ids=tuple(int(i) for i in initial_ids),
        species=tuple(initial.get_chemical_symbols()),
        initial_indices=tuple(range(len(initial_ids))),
        final_indices=final_indices,
    )


def minimum_image(
    vectors: np.ndarray, cell: np.ndarray, tolerance: float = AMBIGUITY_TOLERANCE
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shortest periodic image of each vector: ``(vectors, lengths, tied)``.

    ``tied`` marks vectors with a second image within ``tolerance`` of the shortest length.
    """
    reduced = np.asarray(minkowski_reduce(cell, pbc=True)[0], dtype=float)
    fractional = np.asarray(vectors, dtype=float) @ np.linalg.inv(reduced)
    shifts = np.array(list(product((-1, 0, 1), repeat=3)), dtype=float)
    candidates = (fractional - np.round(fractional))[:, None, :] + shifts[None, :, :]
    images = candidates @ reduced
    lengths = np.linalg.norm(images, axis=2)
    best = np.argmin(lengths, axis=1)
    rows = np.arange(len(images))
    shortest = lengths[rows, best]
    tied = (lengths <= shortest[:, None] + tolerance).sum(axis=1) > 1
    return images[rows, best], shortest, tied


def periodic_displacements(
    initial: Atoms,
    final: Atoms,
    correspondence: AtomCorrespondence | None = None,
    *,
    allow_ambiguous: bool = False,
    ambiguity_tolerance: float = AMBIGUITY_TOLERANCE,
) -> PeriodicDisplacements:
    """Minimum-image displacement of every atom, matched by identity.

    ``correspondence`` defaults to :func:`correspond_atoms`. If given, it must agree with the
    structures' identities and species. Raises ``ValueError`` on an ambiguous displacement unless
    ``allow_ambiguous`` (then the ids are listed in the result; the vector is one shortest image).
    """
    if not (np.isfinite(ambiguity_tolerance) and ambiguity_tolerance >= 0):
        raise ValueError("ambiguity_tolerance must be nonnegative and finite")
    expected = correspond_atoms(initial, final)
    if correspondence is not None and correspondence != expected:
        raise ValueError("correspondence does not match the structures")
    correspondence = expected
    order = np.array(correspondence.final_indices, dtype=int)
    raw = final.positions[order] - initial.positions
    vectors, lengths, tie = minimum_image(raw, initial.cell.array, ambiguity_tolerance)
    ambiguous = tuple(int(i) for i in np.array(correspondence.atom_ids)[tie])
    if ambiguous and not allow_ambiguous:
        raise ValueError(f"ambiguous periodic displacement for atom_id {list(ambiguous)}")
    return PeriodicDisplacements(
        atom_ids=correspondence.atom_ids,
        species=correspondence.species,
        vectors=vectors,
        lengths=np.asarray(lengths, dtype=float),
        unwrapped_final=initial.positions + vectors,
        ambiguous_atom_ids=ambiguous,
    )


__all__ = [
    "PeriodicDisplacements",
    "correspond_atoms",
    "minimum_image",
    "periodic_displacements",
]
