"""Geometric mapping of relaxed atoms onto parent lattice sites (A3.7a).

Each atom of a (relaxed) structure is assigned to its nearest parent lattice site by
minimum-image distance. The mapping never forces an assignment: an atom far from every site is
``off_lattice``, an atom nearly equidistant from its two closest sites is ``ambiguous``, and
atoms that claim the same site are all marked ``conflict``. Sites near ambiguous atoms are
``uncertain`` rather than vacant. The fixed cell is required to match; rigid drift is not
corrected. Purely geometric: species and defect classification are handled elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from ase import Atoms
from ase.geometry import get_distances

from matsim_agents.discovery.radiation_defects import DefectSite

AtomStatus = Literal["on_site", "off_lattice", "ambiguous", "conflict"]

DEFAULT_SITE_TOLERANCE = 0.25
DEFAULT_AMBIGUITY_MARGIN = 0.05


@dataclass(frozen=True)
class SiteAssignment:
    """Result of mapping atoms to parent sites; distances are in angstrom.

    ``atom_site[i]`` is the site index for ``on_site`` and ``conflict`` atoms and ``None``
    otherwise. ``nearest_site``/``second_site`` are the two closest sites of every atom whatever
    its status. ``site_occupants[s]`` lists the ``on_site`` atoms of site ``s``.
    """

    n_atoms: int
    n_sites: int
    site_tolerance: float
    ambiguity_margin: float
    status: tuple[AtomStatus, ...]
    atom_site: tuple[int | None, ...]
    nearest_site: tuple[int, ...]
    second_site: tuple[int, ...]
    nearest_distance: tuple[float, ...]
    second_distance: tuple[float, ...]
    site_occupants: tuple[tuple[int, ...], ...]
    vacant_sites: tuple[int, ...]
    uncertain_sites: tuple[int, ...]
    conflict_sites: tuple[int, ...]

    def atoms_with_status(self, status: AtomStatus) -> tuple[int, ...]:
        return tuple(i for i, s in enumerate(self.status) if s == status)

    @property
    def off_lattice(self) -> tuple[int, ...]:
        return self.atoms_with_status("off_lattice")

    @property
    def ambiguous(self) -> tuple[int, ...]:
        return self.atoms_with_status("ambiguous")

    @property
    def conflicts(self) -> tuple[int, ...]:
        return self.atoms_with_status("conflict")

    @property
    def resolved(self) -> bool:
        """True when no atom is ambiguous or in conflict (off-lattice atoms are resolved)."""
        return not self.ambiguous and not self.conflicts


def assign_lattice_sites(
    parent: Atoms,
    relaxed: Atoms,
    lattice_parameter: float,
    *,
    site_tolerance: float = DEFAULT_SITE_TOLERANCE,
    ambiguity_margin: float = DEFAULT_AMBIGUITY_MARGIN,
    cell_tolerance: float = 1e-6,
) -> SiteAssignment:
    """Assign each ``relaxed`` atom to a ``parent`` site.

    ``site_tolerance`` and ``ambiguity_margin`` are fractions of ``lattice_parameter``. An atom
    is ``off_lattice`` when its nearest site is farther than ``site_tolerance``; otherwise it is
    ``ambiguous`` when the second-nearest site is within ``ambiguity_margin`` of the nearest.
    Remaining atoms claim their nearest site, and atoms sharing a site are all ``conflict``.
    """
    for name, value in (
        ("lattice_parameter", lattice_parameter),
        ("site_tolerance", site_tolerance),
        ("ambiguity_margin", ambiguity_margin),
        ("cell_tolerance", cell_tolerance),
    ):
        if isinstance(value, bool) or not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be nonnegative and finite, got {value!r}")
    if lattice_parameter <= 0 or site_tolerance <= 0:
        raise ValueError("lattice_parameter and site_tolerance must be positive")
    if len(parent) < 2:
        raise ValueError("at least two parent sites are required")
    if len(relaxed) == 0:
        raise ValueError("relaxed structure has no atoms")
    if not parent.pbc.all() or not relaxed.pbc.all():
        raise ValueError("site mapping requires fully periodic structures")
    if not np.allclose(parent.cell.array, relaxed.cell.array, rtol=0.0, atol=cell_tolerance):
        raise ValueError("parent and relaxed cells differ; only fixed-cell mapping is supported")
    if not np.isfinite(relaxed.positions).all():
        raise ValueError("relaxed positions contain non-finite values")

    _, distances = get_distances(relaxed.positions, parent.positions, cell=parent.cell, pbc=True)
    order = np.argsort(distances, axis=1, kind="stable")[:, :2]
    rows = np.arange(len(relaxed))
    nearest = order[:, 0]
    second = order[:, 1]
    d1 = distances[rows, nearest]
    d2 = distances[rows, second]

    tolerance = site_tolerance * lattice_parameter
    margin = ambiguity_margin * lattice_parameter
    status: list[AtomStatus] = []
    for a in range(len(relaxed)):
        if d1[a] > tolerance:
            status.append("off_lattice")
        elif d2[a] - d1[a] < margin:
            status.append("ambiguous")
        else:
            status.append("on_site")

    claims: dict[int, list[int]] = {}
    for a, state in enumerate(status):
        if state == "on_site":
            claims.setdefault(int(nearest[a]), []).append(a)
    conflict_sites = tuple(sorted(s for s, atoms in claims.items() if len(atoms) > 1))
    for site in conflict_sites:
        for a in claims[site]:
            status[a] = "conflict"

    uncertain: set[int] = set()
    for a, state in enumerate(status):
        if state == "ambiguous":
            uncertain.update((int(nearest[a]), int(second[a])))
    occupants = tuple(tuple(claims.get(s, ())) for s in range(len(parent)))
    vacant = tuple(s for s in range(len(parent)) if not occupants[s] and s not in uncertain)
    atom_site = tuple(
        int(nearest[a]) if status[a] in ("on_site", "conflict") else None
        for a in range(len(relaxed))
    )
    return SiteAssignment(
        n_atoms=len(relaxed),
        n_sites=len(parent),
        site_tolerance=float(tolerance),
        ambiguity_margin=float(margin),
        status=tuple(status),
        atom_site=atom_site,
        nearest_site=tuple(int(x) for x in nearest),
        second_site=tuple(int(x) for x in second),
        nearest_distance=tuple(float(x) for x in d1),
        second_distance=tuple(float(x) for x in d2),
        site_occupants=occupants,
        vacant_sites=vacant,
        uncertain_sites=tuple(sorted(uncertain - {s for s in range(len(parent)) if occupants[s]})),
        conflict_sites=conflict_sites,
    )


@dataclass(frozen=True)
class Antisite:
    """A relaxed atom of a different species than the parent atom at the site it occupies."""

    lattice_site: int
    atom_id: int
    parent_species: str
    occupant_species: str


@dataclass(frozen=True)
class SiteIdentification:
    """Defect sites read from a :class:`SiteAssignment`; nothing here is forced.

    ``vacancies`` and ``interstitials`` are :class:`DefectSite` records (vacancies at the parent
    position with the parent species, interstitials at the relaxed position). ``unresolved_atoms``
    holds ambiguous and conflicting atoms and ``uncertain_sites`` the sites they touch; neither
    is reported as a vacancy, interstitial or antisite.
    """

    vacancies: tuple[DefectSite, ...]
    interstitials: tuple[DefectSite, ...]
    antisites: tuple[Antisite, ...]
    unresolved_atoms: tuple[int, ...]
    uncertain_sites: tuple[int, ...]
    resolved: bool

    @property
    def final_sites(self) -> tuple[DefectSite, ...]:
        return (*self.vacancies, *self.interstitials)


def identify_defect_sites(
    parent: Atoms, relaxed: Atoms, assignment: SiteAssignment
) -> SiteIdentification:
    """Vacancies, interstitials and antisites implied by ``assignment``.

    Atom ids come from the ``atom_id`` array of ``relaxed`` when present, else the atom index.
    A vacancy's ``atom_id`` is its site index (parent numbering). ``resolved`` is False whenever
    any atom is ambiguous or in conflict.
    """
    if assignment.n_atoms != len(relaxed) or assignment.n_sites != len(parent):
        raise ValueError("assignment does not match the given parent and relaxed structures")
    ids = (
        relaxed.get_array("atom_id").tolist()
        if "atom_id" in relaxed.arrays
        else list(range(len(relaxed)))
    )
    parent_symbols = parent.get_chemical_symbols()
    symbols = relaxed.get_chemical_symbols()
    vacancies = tuple(
        DefectSite(
            role="vacancy",
            species=parent_symbols[site],
            atom_id=site,
            lattice_site=site,
            position=tuple(float(x) for x in parent.positions[site]),
        )
        for site in assignment.vacant_sites
    )
    interstitials = tuple(
        DefectSite(
            role="interstitial",
            species=symbols[a],
            atom_id=int(ids[a]),
            position=tuple(float(x) for x in relaxed.positions[a]),
        )
        for a in assignment.off_lattice
    )
    antisites = tuple(
        Antisite(site, int(ids[a]), parent_symbols[site], symbols[a])
        for a, site in enumerate(assignment.atom_site)
        if site is not None
        and assignment.status[a] == "on_site"
        and symbols[a] != parent_symbols[site]
    )
    unresolved = tuple(sorted((*assignment.ambiguous, *assignment.conflicts)))
    return SiteIdentification(
        vacancies=vacancies,
        interstitials=interstitials,
        antisites=antisites,
        unresolved_atoms=unresolved,
        uncertain_sites=assignment.uncertain_sites,
        resolved=assignment.resolved,
    )


__all__ = [
    "DEFAULT_AMBIGUITY_MARGIN",
    "DEFAULT_SITE_TOLERANCE",
    "Antisite",
    "AtomStatus",
    "SiteAssignment",
    "SiteIdentification",
    "assign_lattice_sites",
    "identify_defect_sites",
]
