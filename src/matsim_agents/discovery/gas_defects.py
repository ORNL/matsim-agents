"""Hydrogen/helium vacancy complexes ``V_n He_m H_p`` on a fixed BCC parent.

``N_atoms = N_parent - n + m + p`` is validated for every configuration. Surviving parent atoms
keep ``atom_id`` equal to their parent index; each inserted H/He gets the next unused id
(``>= N_parent``), so an id of a removed lattice atom is never reused. The vacancy -> original
site -> inserted-gas mapping is recorded.

Geometries are unrelaxed initial placements. Nothing here asserts a metastable state; use
:mod:`point_defects` outcomes after relaxation for that. Pairs closer than the minimum distances
are rejected (``0.35 a`` to lattice atoms, matching the interstitial generator, and
``GAS_GAS_MIN_DISTANCE`` between inserted atoms).
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Literal

import numpy as np
from ase import Atoms
from ase.geometry import find_mic, get_distances, wrap_positions
from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator

from matsim_agents.discovery.alloy_ensemble import occupation_fingerprint
from matsim_agents.discovery.bcc_defects import generate_bcc_interstitial
from matsim_agents.discovery.point_defects import DefectSite, geometry_fingerprint

GAS_SPECIES = ("H", "He")
GAS_GAS_MIN_DISTANCE = 0.6  # Angstrom; He-He in dense clusters is ~1.2 A
CAVITY_RADIUS_FRACTION = 0.2  # of the lattice parameter, for several gas atoms in one cavity
Placement = Literal["tetrahedral", "octahedral", "vacancy", "cluster_centroid"]
NOT_ASSERTED = "initial unrelaxed geometry; metastability is not asserted"

_CAVITY_DIRECTIONS = tuple(
    [tuple(s * (k == axis) for k in range(3)) for axis in range(3) for s in (1, -1)]
    + [
        tuple(s / 3**0.5 for s in (sx, sy, sz))
        for sx in (1, -1)
        for sy in (1, -1)
        for sz in (1, -1)
    ]
)


class GasDefectError(ValueError):
    """A gas-complex request is invalid or its geometry is rejected."""


class GasPlacement(BaseModel):
    """One inserted gas atom.

    ``tetrahedral``/``octahedral``: site next to parent atom ``anchor_site`` (which must not be
    vacated), ``variant`` picks among the equivalent offsets. ``vacancy``: in the cavity of the
    vacated parent site ``anchor_site``. ``cluster_centroid``: at the minimum-image centroid of
    all vacancies.
    """

    model_config = ConfigDict(frozen=True)

    species: str
    placement: Placement
    anchor_site: StrictInt | None = Field(default=None, ge=0)
    variant: StrictInt = Field(default=0, ge=0)

    @field_validator("species")
    @classmethod
    def _gas(cls, value: str) -> str:
        if value not in GAS_SPECIES:
            raise ValueError(f"gas species must be one of {GAS_SPECIES}, got {value!r}")
        return value

    @model_validator(mode="after")
    def _anchor(self) -> GasPlacement:
        if self.placement != "cluster_centroid" and self.anchor_site is None:
            raise ValueError(f"{self.placement} placement needs anchor_site")
        if self.placement == "cluster_centroid" and self.anchor_site is not None:
            raise ValueError("cluster_centroid placement takes no anchor_site")
        return self


class GasComplexSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    vacancy_sites: tuple[StrictInt, ...] = ()
    gas: tuple[GasPlacement, ...] = ()
    lattice_parameter: float = Field(gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def _nonempty(self) -> GasComplexSpec:
        if not self.vacancy_sites and not self.gas:
            raise ValueError("a complex needs at least one vacancy or one gas atom")
        if any(s < 0 for s in self.vacancy_sites) or len(set(self.vacancy_sites)) != len(
            self.vacancy_sites
        ):
            raise ValueError("vacancy sites must be distinct nonnegative indices")
        return self


class GasAtom(BaseModel):
    model_config = ConfigDict(frozen=True)

    atom_id: StrictInt = Field(ge=0)
    species: str
    placement: Placement
    anchor_site: StrictInt | None
    trapped_vacancy_site: StrictInt | None  # parent site of the cavity it was placed in
    position: tuple[float, float, float]


class GasComplexConfiguration(BaseModel):
    """Identities and provenance; geometry stays in the ``Atoms``."""

    model_config = ConfigDict(frozen=True)

    config_id: str = Field(min_length=1)
    parent_id: str = Field(min_length=1)
    parent_ordering_id: str
    notation: str
    n_vacancies: StrictInt = Field(ge=0)
    n_he: StrictInt = Field(ge=0)
    n_h: StrictInt = Field(ge=0)
    n_parent_atoms: StrictInt = Field(gt=0)
    n_atoms: StrictInt = Field(gt=0)
    vacancies: tuple[DefectSite, ...]
    gas_atoms: tuple[GasAtom, ...]
    spec: GasComplexSpec
    geometry_fingerprint: str
    validation_notes: tuple[str, ...] = (NOT_ASSERTED,)

    @model_validator(mode="after")
    def _invariant(self) -> GasComplexConfiguration:
        he = sum(g.species == "He" for g in self.gas_atoms)
        h = sum(g.species == "H" for g in self.gas_atoms)
        if (self.n_vacancies, self.n_he, self.n_h) != (len(self.vacancies), he, h):
            raise ValueError("V/He/H counts do not match the recorded sites")
        if self.n_atoms != self.n_parent_atoms - self.n_vacancies + self.n_he + self.n_h:
            raise ValueError("n_atoms violates N = N_parent - n + m + p")
        gas_ids = [g.atom_id for g in self.gas_atoms]
        removed = {v.atom_id for v in self.vacancies}
        if len(set(gas_ids)) != len(gas_ids) or any(i < self.n_parent_atoms for i in gas_ids):
            raise ValueError("gas atom ids must be unique and not parent ids")
        if removed & set(gas_ids):
            raise ValueError("a gas atom reuses a removed atom id")
        return self


def complex_notation(n_vacancies: int, n_he: int, n_h: int) -> str:
    """``V_n He_m H_p`` compact form, e.g. ``V1He2H1``; zero terms omitted."""
    parts = [(f"V{n_vacancies}" if n_vacancies else ""), (f"He{n_he}" if n_he else "")]
    return "".join(parts) + (f"H{n_h}" if n_h else "")


def _atom_ids(parent: Atoms) -> np.ndarray:
    if "atom_id" in parent.arrays:
        return np.asarray(parent.get_array("atom_id"), dtype=np.int64)
    return np.arange(len(parent), dtype=np.int64)


def _cavity_position(parent: Atoms, centre: np.ndarray, rank: int, count: int, a: float):
    if count == 1:
        return centre
    if rank >= len(_CAVITY_DIRECTIONS):
        raise GasDefectError(f"at most {len(_CAVITY_DIRECTIONS)} gas atoms fit one cavity layout")
    offset = np.array(_CAVITY_DIRECTIONS[rank]) * CAVITY_RADIUS_FRACTION * a
    return centre + offset


def generate_gas_complex(
    parent: Atoms,
    parent_id: str,
    spec: GasComplexSpec,
    *,
    parent_ordering_id: str | None = None,
    min_distance: float | None = None,
) -> tuple[Atoms, GasComplexConfiguration]:
    """Create ``V_n He_m H_p`` from ``parent``; the parent is not modified, nothing is relaxed."""
    if not parent_id:
        raise GasDefectError("parent_id must be a nonempty string")
    n_parent = len(parent)
    a = spec.lattice_parameter
    min_distance = 0.35 * a if min_distance is None else float(min_distance)
    if any(s >= n_parent for s in spec.vacancy_sites):
        raise GasDefectError(f"a vacancy site is outside the {n_parent}-atom parent")
    ids = _atom_ids(parent)
    vacated = set(spec.vacancy_sites)
    for g in spec.gas:
        if g.placement in ("tetrahedral", "octahedral"):
            if g.anchor_site >= n_parent:
                raise GasDefectError(f"anchor site {g.anchor_site} is outside the parent")
            if g.anchor_site in vacated:
                raise GasDefectError(f"anchor site {g.anchor_site} is vacated")
        elif g.placement == "vacancy" and g.anchor_site not in vacated:
            raise GasDefectError(f"vacancy placement at {g.anchor_site}, which is not a vacancy")
        elif g.placement == "cluster_centroid" and not vacated:
            raise GasDefectError("cluster_centroid placement needs vacancies")

    cavity_total = Counter(g.anchor_site for g in spec.gas if g.placement == "vacancy")
    cavity_total.update(
        {None: sum(g.placement == "cluster_centroid" for g in spec.gas)}
        if any(g.placement == "cluster_centroid" for g in spec.gas)
        else {}
    )
    cavity_rank: Counter = Counter()
    positions: list[np.ndarray] = []
    gas_atoms: list[GasAtom] = []
    for k, g in enumerate(spec.gas):
        trapped = None
        if g.placement in ("tetrahedral", "octahedral"):
            try:
                probe, _ = generate_bcc_interstitial(
                    parent,
                    parent_id,
                    g.anchor_site,
                    g.species,
                    g.placement,
                    a,
                    variant=g.variant,
                    min_distance=1e-3,
                )
            except ValueError as exc:
                raise GasDefectError(str(exc)) from exc
            position = probe.positions[-1]
        else:
            if g.placement == "vacancy":
                trapped = g.anchor_site
                centre = parent.positions[g.anchor_site]
            else:
                origin = parent.positions[spec.vacancy_sites[0]]
                rel, _ = find_mic(
                    parent.positions[list(spec.vacancy_sites)] - origin, parent.cell, pbc=True
                )
                centre = origin + rel.mean(axis=0)
            key = g.anchor_site if g.placement == "vacancy" else None
            rank = cavity_rank[key]
            cavity_rank[key] += 1
            position = _cavity_position(parent, centre, rank, cavity_total[key], a)
        (position,) = wrap_positions(np.asarray(position)[None], parent.cell, pbc=True)
        positions.append(position)
        gas_atoms.append(
            GasAtom(
                atom_id=n_parent + k,
                species=g.species,
                placement=g.placement,
                anchor_site=g.anchor_site,
                trapped_vacancy_site=trapped,
                position=tuple(float(x) for x in position),
            )
        )

    keep = np.array([i for i in range(n_parent) if i not in vacated], dtype=np.int64)
    if positions:
        new = np.array(positions)
        _, d = get_distances(new, parent.positions[keep], cell=parent.cell, pbc=True)
        if d.size and d.min() < min_distance:
            raise GasDefectError(
                f"severe overlap: inserted atom {d.min():.3f} A from a lattice atom "
                f"(minimum {min_distance:.3f} A)"
            )
        if len(new) > 1:
            _, dgg = get_distances(new, cell=parent.cell, pbc=True)
            dgg = dgg[np.triu_indices(len(new), 1)]
            if dgg.min() < GAS_GAS_MIN_DISTANCE:
                raise GasDefectError(
                    f"gas atoms {dgg.min():.3f} A apart (minimum {GAS_GAS_MIN_DISTANCE} A)"
                )

    result = parent[keep.tolist()]
    result.set_array("atom_id", ids[keep])
    if gas_atoms:
        result.extend(
            Atoms([g.species for g in spec.gas], positions=np.array(positions), cell=parent.cell)
        )
        result.arrays["atom_id"][len(keep) :] = [g.atom_id for g in gas_atoms]
    result.set_cell(parent.cell)
    result.set_pbc(parent.pbc)

    vacancies = tuple(
        DefectSite(
            role="vacancy",
            species=parent[s].symbol,
            atom_id=int(ids[s]),
            lattice_site=int(s),
            position=tuple(float(x) for x in parent.positions[s]),
        )
        for s in spec.vacancy_sites
    )
    n_he = sum(g.species == "He" for g in gas_atoms)
    n_h = len(gas_atoms) - n_he
    digest = hashlib.sha256(
        json.dumps(
            {"parent": parent_id, "spec": spec.model_dump(mode="json")}, sort_keys=True
        ).encode()
    ).hexdigest()[:12]
    config = GasComplexConfiguration(
        config_id=f"gascomplex-{digest}",
        parent_id=parent_id,
        parent_ordering_id=parent_ordering_id or occupation_fingerprint(parent),
        notation=complex_notation(len(vacancies), n_he, n_h),
        n_vacancies=len(vacancies),
        n_he=n_he,
        n_h=n_h,
        n_parent_atoms=n_parent,
        n_atoms=len(result),
        vacancies=vacancies,
        gas_atoms=tuple(gas_atoms),
        spec=spec,
        geometry_fingerprint=geometry_fingerprint(result),
    )
    return result, config


__all__ = [
    "GAS_GAS_MIN_DISTANCE",
    "GAS_SPECIES",
    "GasAtom",
    "GasComplexConfiguration",
    "GasComplexSpec",
    "GasDefectError",
    "GasPlacement",
    "complex_notation",
    "generate_gas_complex",
]
