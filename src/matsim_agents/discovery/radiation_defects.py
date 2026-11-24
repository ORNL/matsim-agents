"""Typed contracts for radiation defects and their relation to a pristine parent structure.

Records hold identities, site references and provenance only; coordinates of the full structure
stay in the ASE ``Atoms`` object. Stable atom identities are integers independent of ASE
indices: parent atoms keep ``atom_id`` equal to their parent index, and each inserted atom gets
the next unused id (``>= n_parent_atoms``). ``lattice_site`` is always a parent site index.
``parent_ordering_id`` is the parent's chemical-ordering reference, e.g. the
``occupation_fingerprint`` of ``alloy_ensemble``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from types import MappingProxyType
from typing import Annotated, Literal

import numpy as np
from ase import Atoms
from ase.geometry import get_distances, wrap_positions
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    field_serializer,
    field_validator,
    model_validator,
)
from pymatgen.core import Element

from matsim_agents.discovery.alloy_ensemble import occupation_fingerprint

DefectType = Literal["vacancy", "interstitial", "frenkel_pair", "divacancy"]
SiteRole = Literal["vacancy", "interstitial", "displaced"]
OutcomeClass = Literal[
    "persisted", "recombined_restored", "recombined_antisite", "transformed", "unresolved"
]
ValidationStatus = Literal["unvalidated", "valid", "invalid"]

# Allowed number of parent lattice sites named by a spec; a new defect type adds one entry.
SITE_COUNTS: Mapping[str, tuple[int, int]] = MappingProxyType(
    {"vacancy": (1, 1), "divacancy": (2, 2), "frenkel_pair": (1, 1), "interstitial": (0, 1)}
)

Finite = Annotated[float, Field(allow_inf_nan=False)]
Vec3 = tuple[Finite, Finite, Finite]
ParameterValue = StrictBool | StrictInt | Finite | str


def _check_species(symbol: str) -> str:
    try:
        Element(symbol)
    except ValueError as exc:
        raise ValueError(f"unknown element symbol: {symbol!r}") from exc
    return symbol


class DefectSpec(BaseModel):
    """Requested defect before creation.

    ``site_indices`` are parent lattice sites (vacancy sites, the displaced atom's site, or an
    interstitial's centring site). ``species`` are the incoming species: exactly one for an
    interstitial, none otherwise. ``parameters`` carries generator settings such as orientation
    or separation.
    """

    model_config = ConfigDict(frozen=True)

    defect_type: DefectType
    site_indices: tuple[StrictInt, ...] = ()
    species: tuple[str, ...] = ()
    parameters: Mapping[str, ParameterValue] = Field(default_factory=dict)
    seed: StrictInt | None = Field(default=None, ge=0)

    @field_serializer("parameters")
    def _serialize_parameters(self, value: Mapping[str, ParameterValue]) -> dict:
        return dict(value)

    @field_validator("parameters")
    @classmethod
    def _freeze_parameters(
        cls, value: Mapping[str, ParameterValue]
    ) -> Mapping[str, ParameterValue]:
        return MappingProxyType(dict(sorted(value.items())))

    @field_validator("species")
    @classmethod
    def _validate_species(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(_check_species(symbol) for symbol in value)

    @model_validator(mode="after")
    def _validate_sites(self) -> DefectSpec:
        low, high = SITE_COUNTS[self.defect_type]
        n = len(self.site_indices)
        if not low <= n <= high:
            raise ValueError(f"{self.defect_type} needs {low}..{high} site indices, got {n}")
        if any(site < 0 for site in self.site_indices):
            raise ValueError("site indices must be nonnegative")
        if len(set(self.site_indices)) != n:
            raise ValueError("site indices must be distinct")
        expected = 1 if self.defect_type == "interstitial" else 0
        if len(self.species) != expected:
            raise ValueError(
                f"{self.defect_type} needs {expected} species, got {len(self.species)}"
            )
        return self


class DefectSite(BaseModel):
    """One defect location.

    ``vacancy``: ``lattice_site`` is the vacated parent site, ``atom_id`` the removed atom and
    ``position`` the original position. ``displaced``: a lattice atom moved but still bound to
    its site. ``interstitial``: an off-lattice atom (``lattice_site`` is None); ``atom_id`` is
    new for an inserted atom or a parent id for a relocated one (Frenkel pair).
    """

    model_config = ConfigDict(frozen=True)

    role: SiteRole
    species: str
    atom_id: StrictInt = Field(ge=0)
    lattice_site: StrictInt | None = Field(default=None, ge=0)
    position: Vec3

    @field_validator("species")
    @classmethod
    def _validate_species(cls, value: str) -> str:
        return _check_species(value)

    @model_validator(mode="after")
    def _validate_site_reference(self) -> DefectSite:
        if (self.lattice_site is None) != (self.role == "interstitial"):
            raise ValueError(
                f"{self.role} site must{'' if self.role == 'interstitial' else ' not'} "
                "be off-lattice (lattice_site None)"
            )
        return self


class DefectConfiguration(BaseModel):
    """Generated defect structure's metadata and its link to the parent (no full geometry)."""

    model_config = ConfigDict(frozen=True)

    config_id: str = Field(min_length=1)
    parent_id: str = Field(min_length=1)
    parent_ordering_id: str | None = None
    spec: DefectSpec
    n_parent_atoms: StrictInt = Field(gt=0)
    n_atoms: StrictInt = Field(gt=0)
    cell: tuple[Vec3, Vec3, Vec3]
    sites: tuple[DefectSite, ...]
    geometry_fingerprint: str | None = None
    validation_status: ValidationStatus = "unvalidated"
    validation_notes: tuple[str, ...] = ()

    @property
    def defect_type(self) -> DefectType:
        return self.spec.defect_type

    @model_validator(mode="after")
    def _validate_consistency(self) -> DefectConfiguration:
        vacancies = [s for s in self.sites if s.role == "vacancy"]
        interstitials = [s for s in self.sites if s.role == "interstitial"]
        for site in self.sites:
            if site.role != "interstitial" and site.atom_id >= self.n_parent_atoms:
                raise ValueError(f"{site.role} atom_id {site.atom_id} is not a parent atom")
            if site.lattice_site is not None and site.lattice_site >= self.n_parent_atoms:
                raise ValueError(f"lattice_site {site.lattice_site} is outside the parent")
        if len({s.lattice_site for s in vacancies}) != len(vacancies):
            raise ValueError("vacancy sites must be distinct")
        if len({s.atom_id for s in interstitials}) != len(interstitials):
            raise ValueError("interstitial atom ids must be distinct")
        removed = {s.atom_id for s in vacancies}
        for site in interstitials:
            if site.atom_id < self.n_parent_atoms and site.atom_id not in removed:
                raise ValueError(f"relocated atom {site.atom_id} has no matching vacancy")
        expected = self.n_parent_atoms - len(vacancies) + len(interstitials)
        if self.n_atoms != expected:
            raise ValueError(f"n_atoms is {self.n_atoms} but the sites imply {expected}")
        return self


class DefectOutcome(BaseModel):
    """State of a defect after relaxation or later characterization.

    An outcome whose relaxation did not converge (or is unknown) can only be ``unresolved``.
    """

    model_config = ConfigDict(frozen=True)

    config_id: str = Field(min_length=1)
    classification: OutcomeClass
    relaxation_converged: bool | None = None
    final_sites: tuple[DefectSite, ...] = ()
    geometry_fingerprint: str | None = None
    notes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _require_convergence(self) -> DefectOutcome:
        if self.classification != "unresolved" and self.relaxation_converged is not True:
            raise ValueError(
                "only a converged relaxation can have a classification other than unresolved"
            )
        return self


def make_defect_config_id(parent_id: str, spec: DefectSpec) -> str:
    """Deterministic ID from the parent ID and the full spec (including parameters and seed)."""
    payload = json.dumps(
        {"parent": parent_id, "spec": spec.model_dump(mode="json")}, sort_keys=True, allow_nan=False
    )
    digest = hashlib.sha256(payload.encode()).hexdigest()[:12]
    return f"{spec.defect_type}-{digest}"


def geometry_fingerprint(atoms: Atoms) -> str:
    """SHA-256 of the cell, species and exact coordinates, in atom order."""
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(atoms.cell.array, dtype="<f8").tobytes())
    digest.update(np.ascontiguousarray(atoms.numbers, dtype="<i8").tobytes())
    digest.update(np.ascontiguousarray(atoms.positions, dtype="<f8").tobytes())
    return digest.hexdigest()


def generate_vacancy(
    parent: Atoms,
    parent_id: str,
    site: int,
    *,
    parent_ordering_id: str | None = None,
) -> tuple[Atoms, DefectConfiguration]:
    """Remove the atom at parent index ``site``; no relaxation, no placeholder atom.

    The parent is not modified. The returned structure carries an integer ``atom_id`` array;
    parent atoms are numbered by their parent index (see module docstring), so the surviving
    atoms keep their identities although ASE indices shift. ``parent_ordering_id`` defaults to
    ``occupation_fingerprint(parent)``.
    """
    if not parent_id:
        raise ValueError("parent_id must be a nonempty string")
    if isinstance(site, bool) or not isinstance(site, int | np.integer):
        raise ValueError(f"site must be an integer index, got {site!r}")
    site = int(site)
    if not 0 <= site < len(parent):
        raise ValueError(f"site {site} is outside the {len(parent)}-atom parent")

    spec = DefectSpec(defect_type="vacancy", site_indices=(site,))
    removed = DefectSite(
        role="vacancy",
        species=parent[site].symbol,
        atom_id=site,
        lattice_site=site,
        position=tuple(float(x) for x in parent.positions[site]),
    )
    defective = parent.copy()
    defective.set_array("atom_id", np.arange(len(parent), dtype=np.int64))
    del defective[site]
    config = DefectConfiguration(
        config_id=make_defect_config_id(parent_id, spec),
        parent_id=parent_id,
        parent_ordering_id=parent_ordering_id or occupation_fingerprint(parent),
        spec=spec,
        n_parent_atoms=len(parent),
        n_atoms=len(defective),
        cell=tuple(tuple(float(x) for x in row) for row in parent.cell.array),
        sites=(removed,),
        geometry_fingerprint=geometry_fingerprint(defective),
        validation_status="valid",
    )
    return defective, config


# Dumbbell axis in units of the unit vectors along the three cell vectors.
DUMBBELL_ORIENTATIONS: Mapping[str, tuple[int, int, int]] = MappingProxyType(
    {"111": (1, 1, 1), "110": (1, 1, 0), "100": (1, 0, 0)}
)
_AXIS_TOL = 1e-6


def _cubic_axes(parent: Atoms, lattice_parameter: float) -> np.ndarray:
    """Unit vectors of the conventional cubic axes: the cell vectors of a BCC supercell."""
    cell = parent.cell.array
    lengths = np.linalg.norm(cell, axis=1)
    axes = cell / lengths[:, None]
    multiples = lengths / lattice_parameter
    if (
        not np.allclose(axes @ axes.T, np.eye(3), atol=_AXIS_TOL)
        or not np.allclose(multiples, np.rint(multiples), atol=_AXIS_TOL)
        or (np.rint(multiples) < 1).any()
    ):
        raise ValueError(
            f"cell is not an orthogonal supercell of a cubic lattice with a={lattice_parameter}"
        )
    return axes


def generate_dumbbell_interstitial(
    parent: Atoms,
    parent_id: str,
    site: int,
    species: str,
    orientation: str,
    lattice_parameter: float,
    *,
    separation: float | None = None,
    min_distance: float | None = None,
    parent_ordering_id: str | None = None,
) -> tuple[Atoms, DefectConfiguration]:
    """Split the lattice atom at ``site`` into a dumbbell with an incoming ``species`` atom.

    The two atoms are centred on the original site and separated by ``separation`` (angstrom,
    default ``0.5 * lattice_parameter``) along ``orientation`` (``"111"``, ``"110"`` or
    ``"100"``), taken relative to the cell vectors so a rotated cell rotates the dumbbell. The
    original atom moves to ``-separation / 2`` along the axis and keeps its parent index and
    ``atom_id``; the incoming atom is appended with ``atom_id = len(parent)`` at
    ``+separation / 2``. Nothing is relaxed, and the result is only an initial candidate: its
    stability is not implied. Any pair closer than ``min_distance`` (minimum image, default
    ``0.35 * lattice_parameter``) is rejected as a severe overlap.
    """
    if not parent_id:
        raise ValueError("parent_id must be a nonempty string")
    if isinstance(site, bool) or not isinstance(site, int | np.integer):
        raise ValueError(f"site must be an integer index, got {site!r}")
    site = int(site)
    if not 0 <= site < len(parent):
        raise ValueError(f"site {site} is outside the {len(parent)}-atom parent")
    if orientation not in DUMBBELL_ORIENTATIONS:
        raise ValueError(
            f"unsupported orientation {orientation!r}; use one of {sorted(DUMBBELL_ORIENTATIONS)}"
        )
    if not (np.isfinite(lattice_parameter) and lattice_parameter > 0):
        raise ValueError(f"lattice_parameter must be positive and finite, got {lattice_parameter}")
    separation = 0.5 * lattice_parameter if separation is None else float(separation)
    min_distance = 0.35 * lattice_parameter if min_distance is None else float(min_distance)
    if not (np.isfinite(separation) and separation > 0):
        raise ValueError(f"separation must be positive and finite, got {separation}")
    if not (np.isfinite(min_distance) and min_distance > 0):
        raise ValueError(f"min_distance must be positive and finite, got {min_distance}")
    if not parent.pbc.all():
        raise ValueError("parent must be periodic in all directions")

    spec = DefectSpec(
        defect_type="interstitial",
        site_indices=(site,),
        species=(species,),
        parameters={
            "orientation": orientation,
            "separation": separation,
            "min_distance": min_distance,
            "lattice_parameter": float(lattice_parameter),
        },
    )
    axes = _cubic_axes(parent, lattice_parameter)
    direction = np.array(DUMBBELL_ORIENTATIONS[orientation]) @ axes
    direction /= np.linalg.norm(direction)
    centre = parent.positions[site]
    moved, added = wrap_positions(
        np.array([centre - 0.5 * separation * direction, centre + 0.5 * separation * direction]),
        parent.cell,
        pbc=True,
    )

    defective = parent.copy()
    defective.set_array("atom_id", np.arange(len(parent), dtype=np.int64))
    defective.positions[site] = moved
    defective.extend(Atoms(species, positions=[added]))
    defective.arrays["atom_id"][-1] = len(parent)

    for k in (site, len(parent)):
        _, dist = get_distances(
            defective.positions[k], defective.positions, cell=defective.cell, pbc=True
        )
        dist[0, k] = np.inf
        if dist.min() < min_distance:
            raise ValueError(
                f"severe overlap: atom {k} is {dist.min():.3f} A from another atom "
                f"(minimum {min_distance:.3f} A)"
            )

    config = DefectConfiguration(
        config_id=make_defect_config_id(parent_id, spec),
        parent_id=parent_id,
        parent_ordering_id=parent_ordering_id or occupation_fingerprint(parent),
        spec=spec,
        n_parent_atoms=len(parent),
        n_atoms=len(defective),
        cell=tuple(tuple(float(x) for x in row) for row in parent.cell.array),
        sites=(
            DefectSite(
                role="displaced",
                species=parent[site].symbol,
                atom_id=site,
                lattice_site=site,
                position=tuple(float(x) for x in moved),
            ),
            DefectSite(
                role="interstitial",
                species=species,
                atom_id=len(parent),
                position=tuple(float(x) for x in added),
            ),
        ),
        geometry_fingerprint=geometry_fingerprint(defective),
        validation_status="valid",
    )
    return defective, config


__all__ = [
    "DUMBBELL_ORIENTATIONS",
    "SITE_COUNTS",
    "DefectConfiguration",
    "DefectOutcome",
    "DefectSite",
    "DefectSpec",
    "generate_dumbbell_interstitial",
    "generate_vacancy",
    "geometry_fingerprint",
    "make_defect_config_id",
]
