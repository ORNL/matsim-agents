"""Wigner-Seitz occupancy analysis of a structure against a reference lattice.

Each atom is assigned to its nearest reference site under periodic boundary conditions, so a site
holds zero atoms (vacant), one (singly occupied) or several (multiply occupied). Occupancy is
geometric only: a singly occupied site holding another species is an antisite (a chemical
exchange) and is never counted as a vacancy or interstitial.

Alignment: ``cell_mode="affine"`` maps a homogeneously deformed cell onto the reference cell, and
``align="median"`` removes a rigid translation estimated from the median atom-to-site offset.
A translation must stay below half the nearest site spacing, otherwise the assignment itself is
wrong. Thermal noise is not filtered; ``n_ambiguous`` counts atoms nearly equidistant from two
sites, which is a measure of how sensitive the counts are to it. Nothing here is a physical
validation of any defect count.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from ase import Atoms
from ase.geometry import find_mic
from scipy.optimize import linear_sum_assignment

Align = Literal["none", "median"]
CellMode = Literal["fixed", "affine"]

_CHUNK = 128
_MAX_ALIGN_ITERATIONS = 25


@dataclass(frozen=True)
class WignerSeitzFrenkelPair:
    """A vacant site paired with an interstitial atom by minimum total periodic separation."""

    vacancy_site: int
    interstitial_atom: int
    separation: float  # angstrom


@dataclass(frozen=True)
class WignerSeitzAntisite:
    site: int
    atom: int
    reference_species: str
    occupant_species: str


@dataclass(frozen=True)
class WignerSeitzResult:
    """Occupancy of ``n_sites`` reference sites by ``n_atoms`` atoms.

    ``atom_site[i]`` is the site atom ``i`` belongs to and ``occupancy[s]`` the atom count of site
    ``s``. ``interstitial_atoms`` are the surplus atoms of multiply occupied sites (one atom per
    site is kept as the site's own: the nearest one of the reference species if any, else the
    nearest). ``translation`` is the rigid shift removed from the atoms, in angstrom.
    """

    n_atoms: int
    n_sites: int
    atom_site: tuple[int, ...]
    occupancy: tuple[int, ...]
    vacant_sites: tuple[int, ...]
    singly_occupied_sites: tuple[int, ...]
    multiply_occupied_sites: tuple[int, ...]
    interstitial_atoms: tuple[int, ...]
    interstitial_host_sites: tuple[int, ...]
    antisites: tuple[WignerSeitzAntisite, ...]
    frenkel_pairs: tuple[WignerSeitzFrenkelPair, ...]
    translation: tuple[float, float, float]
    cell_mode: str
    align: str
    ambiguity_margin: float
    n_ambiguous: int
    max_nearest_distance: float

    @property
    def n_vacancies(self) -> int:
        return len(self.vacant_sites)

    @property
    def n_interstitials(self) -> int:
        return len(self.interstitial_atoms)

    @property
    def n_frenkel_pairs(self) -> int:
        return len(self.frenkel_pairs)


def _nearest_sites(positions: np.ndarray, reference: Atoms):
    """Nearest and second-nearest reference site of every position, by minimum image."""
    n_sites = len(reference)
    nearest = np.empty(len(positions), dtype=np.int64)
    d1 = np.empty(len(positions))
    d2 = np.empty(len(positions))
    offsets = np.empty((len(positions), 3))
    for start in range(0, len(positions), _CHUNK):
        block = positions[start : start + _CHUNK]
        vectors = reference.positions[None, :, :] - block[:, None, :]
        flat, lengths = find_mic(vectors.reshape(-1, 3), reference.cell, pbc=True)
        flat = flat.reshape(len(block), n_sites, 3)
        lengths = lengths.reshape(len(block), n_sites)
        order = np.argsort(lengths, axis=1, kind="stable")[:, :2]
        rows = np.arange(len(block))
        stop = start + len(block)
        nearest[start:stop] = order[:, 0]
        d1[start:stop] = lengths[rows, order[:, 0]]
        d2[start:stop] = lengths[rows, order[:, 1]]
        offsets[start:stop] = -flat[rows, order[:, 0]]  # atom minus its nearest site
    return nearest, d1, d2, offsets


def _validate(reference: Atoms, current: Atoms, cell_mode: str, align: str) -> None:
    if cell_mode not in ("fixed", "affine"):
        raise ValueError(f"cell_mode must be 'fixed' or 'affine', got {cell_mode!r}")
    if align not in ("none", "median"):
        raise ValueError(f"align must be 'none' or 'median', got {align!r}")
    if len(reference) < 2:
        raise ValueError("at least two reference sites are required")
    if len(current) == 0:
        raise ValueError("the analysed structure has no atoms")
    if not reference.pbc.all() or not current.pbc.all():
        raise ValueError("Wigner-Seitz analysis requires fully periodic structures")
    if not np.isfinite(current.positions).all() or not np.isfinite(current.cell.array).all():
        raise ValueError("the analysed structure has non-finite coordinates or cell")
    if (
        abs(np.linalg.det(reference.cell.array)) < 1e-12
        or abs(np.linalg.det(current.cell.array)) < 1e-12
    ):
        raise ValueError("a cell is singular")
    if cell_mode == "fixed" and not np.allclose(
        reference.cell.array, current.cell.array, rtol=0.0, atol=1e-6
    ):
        raise ValueError("cells differ; use cell_mode='affine' for a deformed or rescaled cell")


def wigner_seitz_analysis(
    reference: Atoms,
    current: Atoms,
    *,
    cell_mode: CellMode = "fixed",
    align: Align = "median",
    ambiguity_margin: float = 0.1,
) -> WignerSeitzResult:
    """Occupancy of the sites of ``reference`` by the atoms of ``current``.

    ``ambiguity_margin`` (angstrom) only feeds the ``n_ambiguous`` diagnostic.
    """
    _validate(reference, current, cell_mode, align)
    if not np.isfinite(ambiguity_margin) or ambiguity_margin < 0:
        raise ValueError(f"ambiguity_margin must be nonnegative and finite, got {ambiguity_margin}")

    if cell_mode == "affine":
        positions = current.get_scaled_positions(wrap=False) @ reference.cell.array
    else:
        positions = current.positions.copy()

    translation = np.zeros(3)
    if align == "median":
        for _ in range(_MAX_ALIGN_ITERATIONS):
            _, _, _, offsets = _nearest_sites(positions - translation, reference)
            step = np.median(offsets, axis=0)
            translation = translation + step
            if np.linalg.norm(step) < 1e-8:
                break
    shifted = positions - translation
    nearest, d1, d2, _ = _nearest_sites(shifted, reference)

    n_sites = len(reference)
    occupancy = np.bincount(nearest, minlength=n_sites)
    ref_symbols = reference.get_chemical_symbols()
    symbols = current.get_chemical_symbols()
    members: dict[int, list[int]] = {}
    for atom, site in enumerate(nearest):
        members.setdefault(int(site), []).append(atom)

    interstitials: list[int] = []
    hosts: list[int] = []
    antisites: list[WignerSeitzAntisite] = []
    for site, atoms in sorted(members.items()):
        atoms = sorted(atoms, key=lambda a: (d1[a], a))
        if len(atoms) == 1:
            keep = atoms[0]
        else:
            same = [a for a in atoms if symbols[a] == ref_symbols[site]]
            keep = same[0] if same else atoms[0]
            for atom in atoms:
                if atom != keep:
                    interstitials.append(atom)
                    hosts.append(site)
        if symbols[keep] != ref_symbols[site]:
            antisites.append(WignerSeitzAntisite(site, keep, ref_symbols[site], symbols[keep]))

    vacant = tuple(int(s) for s in np.flatnonzero(occupancy == 0))
    pairs = _pair_frenkel(reference, shifted, vacant, interstitials)
    return WignerSeitzResult(
        n_atoms=len(current),
        n_sites=n_sites,
        atom_site=tuple(int(s) for s in nearest),
        occupancy=tuple(int(o) for o in occupancy),
        vacant_sites=vacant,
        singly_occupied_sites=tuple(int(s) for s in np.flatnonzero(occupancy == 1)),
        multiply_occupied_sites=tuple(int(s) for s in np.flatnonzero(occupancy > 1)),
        interstitial_atoms=tuple(interstitials),
        interstitial_host_sites=tuple(hosts),
        antisites=tuple(antisites),
        frenkel_pairs=pairs,
        translation=(float(translation[0]), float(translation[1]), float(translation[2])),
        cell_mode=cell_mode,
        align=align,
        ambiguity_margin=float(ambiguity_margin),
        n_ambiguous=int(np.count_nonzero(d2 - d1 < ambiguity_margin)),
        max_nearest_distance=float(d1.max()),
    )


def _pair_frenkel(
    reference: Atoms,
    positions: np.ndarray,
    vacant: tuple[int, ...],
    interstitials: list[int],
) -> tuple[WignerSeitzFrenkelPair, ...]:
    """Pair vacancies with interstitials at minimum total periodic distance."""
    if not vacant or not interstitials:
        return ()
    vac_positions = reference.positions[list(vacant)]
    int_positions = positions[interstitials]
    vectors = int_positions[None, :, :] - vac_positions[:, None, :]
    _, lengths = find_mic(vectors.reshape(-1, 3), reference.cell, pbc=True)
    lengths = lengths.reshape(len(vacant), len(interstitials))
    rows, cols = linear_sum_assignment(lengths)
    return tuple(
        WignerSeitzFrenkelPair(vacant[r], interstitials[c], float(lengths[r, c]))
        for r, c in sorted(zip(rows, cols, strict=True))
    )


__all__ = [
    "WignerSeitzAntisite",
    "WignerSeitzFrenkelPair",
    "WignerSeitzResult",
    "wigner_seitz_analysis",
]
