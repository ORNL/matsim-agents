"""Candidate endpoint pairs for migration events (generation only; nothing is relaxed).

Each generator returns the unmodified initial structure, a new final structure, and an
``NebTransition`` whose endpoints are ``generated`` candidates and whose mechanism is a
hypothesis that still needs geometric and energetic validation. Atom identities (the integer
``atom_id`` array) are preserved, so a hop moves one atom between sites without renumbering.

Conventions: parent atom ``k`` belongs to parent site ``k``; a vacancy ``DefectSite`` names the
parent site and its parent atom, even after earlier hops. Lattice sites of moving atoms are found
from positions against the pristine ``parent`` (minimum image).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from math import sqrt

import numpy as np
from ase import Atoms
from ase.geometry import get_distances

from matsim_agents.discovery.neb_contracts import (
    MigrationMechanism,
    NebTransition,
    build_correspondence,
    endpoint_from_atoms,
)
from matsim_agents.discovery.neb_paths import minimum_image
from matsim_agents.discovery.radiation_defects import (
    DefectConfiguration,
    DefectSite,
    DefectSpec,
    ParameterValue,
    geometry_fingerprint,
    make_defect_config_id,
)

NEIGHBOR_TOLERANCE = 0.1  # relative tolerance on the BCC nearest-neighbour distance


@dataclass(frozen=True)
class MigrationPair:
    """Initial and final candidate structures with their transition record."""

    initial: Atoms
    final: Atoms
    transition: NebTransition
    final_config: DefectConfiguration | None


def _index_of(atoms: Atoms, atom_id: int) -> int:
    ids = atoms.get_array("atom_id")
    hits = np.flatnonzero(ids == atom_id)
    if len(hits) != 1:
        raise ValueError(f"atom_id {atom_id} is not present exactly once in the structure")
    return int(hits[0])


def _require_identities(atoms: Atoms, config: DefectConfiguration) -> None:
    if "atom_id" not in atoms.arrays:
        raise ValueError("structure has no atom_id array; stable identities are required")
    if len(atoms) != config.n_atoms:
        raise ValueError(f"structure has {len(atoms)} atoms but the defect has {config.n_atoms}")
    if not np.isfinite(atoms.positions).all():
        raise ValueError("structure has non-finite coordinates")


def _check_parent(parent: Atoms, config: DefectConfiguration) -> None:
    if len(parent) != config.n_parent_atoms:
        raise ValueError(
            f"parent has {len(parent)} atoms but the defect names {config.n_parent_atoms}"
        )
    if not np.allclose(parent.cell.array, np.array(config.cell), atol=1e-6):
        raise ValueError("parent cell differs from the defect configuration cell")


def _nn_distance(lattice_parameter: float) -> float:
    if not (np.isfinite(lattice_parameter) and lattice_parameter > 0):
        raise ValueError(f"lattice_parameter must be positive and finite, got {lattice_parameter}")
    return sqrt(3.0) / 2.0 * float(lattice_parameter)


def _mic_vector(atoms_cell: Atoms, source: np.ndarray, target: np.ndarray) -> np.ndarray:
    vector, _, _ = minimum_image(
        np.asarray(target - source, dtype=float)[None, :], atoms_cell.cell.array
    )
    return vector[0]


def _distance(atoms_cell: Atoms, a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(_mic_vector(atoms_cell, a, b)))


def _parent_site_of(parent: Atoms, position: np.ndarray, tolerance: float) -> int:
    _, distances = get_distances(position, parent.positions, cell=parent.cell, pbc=parent.pbc)
    site = int(np.argmin(distances[0]))
    if distances[0, site] > tolerance:
        raise ValueError(
            f"atom is {distances[0, site]:.3f} A from the nearest parent site "
            f"(limit {tolerance:.3f})"
        )
    return site


def _min_pair_distance(atoms: Atoms, index: int) -> float:
    others = np.delete(np.arange(len(atoms)), index)
    if len(others) == 0:
        return float("inf")
    _, distances = get_distances(
        atoms.positions[index], atoms.positions[others], cell=atoms.cell, pbc=atoms.pbc
    )
    return float(distances.min())


def _check_no_overlap(atoms: Atoms, moved: list[int], min_distance: float) -> None:
    if not (np.isfinite(min_distance) and min_distance > 0):
        raise ValueError(f"min_distance must be positive and finite, got {min_distance}")
    for index in moved:
        closest = _min_pair_distance(atoms, index)
        if closest < min_distance:
            raise ValueError(
                f"final structure puts atom_id {int(atoms.get_array('atom_id')[index])} "
                f"{closest:.3f} A from another atom (minimum {min_distance:.3f} A)"
            )


def _transition_id(initial_id: str, final_id: str, mechanism: MigrationMechanism) -> str:
    payload = json.dumps(
        {"i": initial_id, "f": final_id, "m": mechanism.model_dump(mode="json")},
        sort_keys=True,
        allow_nan=False,
    )
    return f"{mechanism.kind}-{hashlib.sha256(payload.encode()).hexdigest()[:12]}"


def _final_id(config_id: str, mechanism: MigrationMechanism, final: Atoms) -> str:
    payload = json.dumps(
        {"m": mechanism.model_dump(mode="json"), "g": geometry_fingerprint(final)},
        sort_keys=True,
        allow_nan=False,
    )
    return f"{config_id}->{mechanism.kind}-{hashlib.sha256(payload.encode()).hexdigest()[:12]}"


def _assemble(
    initial: Atoms,
    final: Atoms,
    config: DefectConfiguration,
    final_config: DefectConfiguration | None,
    mechanism: MigrationMechanism,
) -> MigrationPair:
    common = {
        "parent_id": config.parent_id,
        "parent_ordering_id": config.parent_ordering_id,
        "defect_type": config.defect_type,
    }
    final_id = _final_id(config.config_id, mechanism, final)
    initial_endpoint = endpoint_from_atoms(
        initial, config.config_id, "initial", defect_config=config, **common
    )
    final_endpoint = endpoint_from_atoms(
        final, final_id, "final", defect_config=final_config, **common
    )
    transition = NebTransition(
        transition_id=_transition_id(config.config_id, final_id, mechanism),
        initial=initial_endpoint,
        final=final_endpoint,
        correspondence=build_correspondence(initial_endpoint, final_endpoint),
        mechanism=mechanism,
    )
    return MigrationPair(initial, final, transition, final_config)


def _single_vacancy(config: DefectConfiguration) -> DefectSite:
    if config.defect_type != "vacancy":
        raise ValueError(f"vacancy hop needs a vacancy configuration, got {config.defect_type}")
    vacancies = [s for s in config.sites if s.role == "vacancy"]
    if len(vacancies) != 1 or len(config.sites) != 1:
        raise ValueError("vacancy hop needs exactly one vacancy site")
    return vacancies[0]


def generate_vacancy_hop(
    initial: Atoms,
    config: DefectConfiguration,
    parent: Atoms,
    mover_atom_id: int,
    lattice_parameter: float,
) -> MigrationPair:
    """Move the nearest-neighbour atom ``mover_atom_id`` into the vacancy.

    The mover must sit one BCC nearest-neighbour distance from the vacancy. The final vacancy is
    at the mover's former parent site. In an alloy the hop changes the occupation of both sites;
    this is recorded as ``occupant_exchange`` (the site the mover enters held a different
    species in the parent). The displacement is the minimum-image vector. Nothing is relaxed.
    """
    _require_identities(initial, config)
    _check_parent(parent, config)
    vacancy = _single_vacancy(config)
    nn = _nn_distance(lattice_parameter)
    tolerance = NEIGHBOR_TOLERANCE * nn
    index = _index_of(initial, int(mover_atom_id))
    vacancy_position = np.array(vacancy.position)
    mover_position = initial.positions[index].copy()
    distance = _distance(initial, mover_position, vacancy_position)
    if abs(distance - nn) > tolerance:
        raise ValueError(
            f"atom_id {mover_atom_id} is {distance:.3f} A from the vacancy, "
            f"not a nearest neighbour ({nn:.3f} A)"
        )
    source_site = _parent_site_of(parent, mover_position, tolerance)
    if source_site == vacancy.lattice_site:
        raise ValueError("mover already sits on the vacancy site")
    vector = _mic_vector(initial, mover_position, vacancy_position)

    final = initial.copy()
    final.positions[index] = mover_position + vector  # the short periodic image
    mover_species = initial[index].symbol
    spec = DefectSpec(
        defect_type="vacancy",
        site_indices=(source_site,),
        parameters={
            "kind": "vacancy_hop",
            "derived_from": config.config_id,
            "mover_atom_id": int(mover_atom_id),
        },
    )
    final_config = DefectConfiguration(
        config_id=make_defect_config_id(config.parent_id, spec),
        parent_id=config.parent_id,
        parent_ordering_id=config.parent_ordering_id,
        spec=spec,
        n_parent_atoms=config.n_parent_atoms,
        n_atoms=config.n_atoms,
        cell=config.cell,
        sites=(
            DefectSite(
                role="vacancy",
                species=parent[source_site].symbol,
                atom_id=source_site,
                lattice_site=source_site,
                position=tuple(float(x) for x in parent.positions[source_site]),
            ),
        ),
        geometry_fingerprint=geometry_fingerprint(final),
        validation_status="unvalidated",
        validation_notes=("generated hop endpoint; not relaxed",),
    )
    parameters: dict[str, ParameterValue] = {
        "from_site": int(source_site),
        "to_site": int(vacancy.lattice_site),
        "mover_species": mover_species,
        "displacement_length": float(np.linalg.norm(vector)),
        "displacement_x": float(vector[0]),
        "displacement_y": float(vector[1]),
        "displacement_z": float(vector[2]),
        "occupant_exchange": bool(parent[int(vacancy.lattice_site)].symbol != mover_species),
        "lattice_parameter": float(lattice_parameter),
    }
    mechanism = MigrationMechanism(
        kind="vacancy_hop",
        moving_atom_ids=(int(mover_atom_id),),
        parameters=parameters,
        notes=("nearest-neighbour hop hypothesis; the saddle path is not known",),
    )
    return _assemble(initial, final, config, final_config, mechanism)


def vacancy_neighbors(
    initial: Atoms, config: DefectConfiguration, lattice_parameter: float
) -> tuple[int, ...]:
    """``atom_id`` of every atom one nearest-neighbour distance from the vacancy, sorted."""
    _require_identities(initial, config)
    vacancy = _single_vacancy(config)
    nn = _nn_distance(lattice_parameter)
    _, distances = get_distances(
        np.array(vacancy.position), initial.positions, cell=initial.cell, pbc=initial.pbc
    )
    close = np.flatnonzero(np.abs(distances[0] - nn) <= NEIGHBOR_TOLERANCE * nn)
    return tuple(sorted(int(initial.get_array("atom_id")[i]) for i in close))


def generate_interstitial_migration(
    initial: Atoms,
    config: DefectConfiguration,
    new_positions: dict[int, tuple[float, float, float]],
    *,
    min_distance: float = 1.5,
) -> MigrationPair:
    """Explicit endpoint interface: place interstitial or displaced atoms at given positions.

    ``new_positions`` maps the ``atom_id`` of an interstitial or displaced defect site to its
    final Cartesian position (positions are never inferred; the mechanism, for example a hop or
    a dumbbell rotation, is the caller's hypothesis). Overlaps below ``min_distance`` are
    rejected. The pair needs later validation and relaxation.
    """
    _require_identities(initial, config)
    if config.defect_type not in ("interstitial", "frenkel_pair"):
        raise ValueError(f"interstitial migration needs an interstitial, got {config.defect_type}")
    if not new_positions:
        raise ValueError("new_positions must name at least one atom")
    mobile = {s.atom_id: s for s in config.sites if s.role in ("interstitial", "displaced")}
    final = initial.copy()
    moved: list[int] = []
    for atom_id, target in new_positions.items():
        if atom_id not in mobile:
            raise ValueError(f"atom_id {atom_id} is not an interstitial or displaced atom")
        target_array = np.array(target, dtype=float)
        if target_array.shape != (3,) or not np.isfinite(target_array).all():
            raise ValueError(f"new position of atom_id {atom_id} must be 3 finite numbers")
        index = _index_of(initial, atom_id)
        final.positions[index] = target_array
        moved.append(index)
    _check_no_overlap(final, moved, min_distance)
    if np.allclose(final.positions, initial.positions):
        raise ValueError("final structure equals the initial structure")

    sites = []
    for site in config.sites:
        if site.atom_id in new_positions and site.role in ("interstitial", "displaced"):
            target = tuple(float(x) for x in new_positions[site.atom_id])
            sites.append(
                site.model_copy(
                    update={
                        "position": target,
                        "initial_position": site.initial_position or site.position,
                    }
                )
            )
        else:
            sites.append(site)
    parameters: dict[str, ParameterValue] = {
        **config.spec.parameters,
        "kind": "interstitial_migration",
        "derived_from": config.config_id,
        "moved_atom_ids": ",".join(str(i) for i in sorted(new_positions)),
    }
    for atom_id in sorted(new_positions):
        for axis, value in zip("xyz", new_positions[atom_id], strict=True):
            parameters[f"final_{atom_id}_{axis}"] = float(value)
    spec = config.spec.model_copy(update={"parameters": parameters})
    final_config = config.model_copy(
        update={
            "config_id": make_defect_config_id(config.parent_id, spec),
            "spec": spec,
            "sites": tuple(sites),
            "geometry_fingerprint": geometry_fingerprint(final),
            "validation_status": "unvalidated",
            "validation_notes": ("generated migration endpoint; not relaxed",),
        }
    )
    displacements = {
        atom_id: float(
            np.linalg.norm(
                _mic_vector(initial, initial.positions[_index_of(initial, atom_id)], np.array(t))
            )
        )
        for atom_id, t in new_positions.items()
    }
    mechanism = MigrationMechanism(
        kind="interstitial_migration",
        moving_atom_ids=tuple(sorted(new_positions)),
        parameters={
            "min_distance": float(min_distance),
            "max_displacement": max(displacements.values()),
        },
        notes=("caller-specified final positions; the migration mechanism is unvalidated",),
    )
    return _assemble(initial, final, config, final_config, mechanism)


def _ordering_restored(final: Atoms, parent: Atoms, tolerance: float) -> bool:
    """True when every atom sits on a distinct parent site holding the parent's species."""
    _, distances = get_distances(
        final.positions, parent.positions, cell=parent.cell, pbc=parent.pbc
    )
    sites = np.argmin(distances, axis=1)
    if len(set(sites.tolist())) != len(final) or len(final) != len(parent):
        raise ValueError("final structure is not a complete lattice; cannot compare orderings")
    if distances[np.arange(len(final)), sites].max() > tolerance:
        raise ValueError("final structure has atoms away from lattice sites")
    return bool(
        np.array_equal(
            np.array(final.get_chemical_symbols())[np.argsort(sites)],
            np.array(parent.get_chemical_symbols()),
        )
    )


def generate_frenkel_recombination(
    initial: Atoms,
    config: DefectConfiguration,
    parent: Atoms,
    lattice_parameter: float,
    *,
    exchange_with: int | None = None,
) -> MigrationPair:
    """Return the interstitial atom to the lattice, recombining the Frenkel pair.

    Default: the atom goes back to its own vacancy site (a candidate for the restored
    ordering). With ``exchange_with`` (``atom_id`` of a nearest neighbour of the vacancy) that
    lattice atom moves into the vacancy and the interstitial atom takes its place (a
    replacement chain). Either way the final structure must be a complete lattice; whether the
    chemical ordering is ``restored`` or ``exchanged`` is computed from the species on the
    parent sites and recorded, never assumed. The final state has no defect, so
    ``final_config`` is None. The vacancy-interstitial correspondence is recorded.
    """
    _require_identities(initial, config)
    _check_parent(parent, config)
    if config.defect_type != "frenkel_pair":
        raise ValueError(f"recombination needs a Frenkel pair, got {config.defect_type}")
    vacancy = next(s for s in config.sites if s.role == "vacancy")
    interstitial = next(s for s in config.sites if s.role == "interstitial")
    nn = _nn_distance(lattice_parameter)
    tolerance = NEIGHBOR_TOLERANCE * nn
    mover_id = interstitial.atom_id
    mover_index = _index_of(initial, mover_id)
    target = np.array(vacancy.position)
    final = initial.copy()
    moving = [mover_id]
    if exchange_with is None:
        final.positions[mover_index] = target
        recombination = "direct"
    else:
        partner_index = _index_of(initial, int(exchange_with))
        if int(exchange_with) == mover_id:
            raise ValueError("exchange_with must be a lattice atom, not the interstitial")
        partner_position = initial.positions[partner_index].copy()
        distance = _distance(initial, partner_position, target)
        if abs(distance - nn) > tolerance:
            raise ValueError(
                f"atom_id {exchange_with} is {distance:.3f} A from the vacancy, "
                f"not a nearest neighbour ({nn:.3f} A)"
            )
        final.positions[partner_index] = partner_position + _mic_vector(
            initial, partner_position, target
        )
        final.positions[mover_index] = partner_position
        moving.append(int(exchange_with))
        recombination = "exchange"
    restored = _ordering_restored(final, parent, tolerance)
    _check_no_overlap(final, [_index_of(final, i) for i in moving], 0.5 * nn)

    parameters: dict[str, ParameterValue] = {
        "recombination": recombination,
        "ordering": "restored" if restored else "exchanged",
        "vacancy_site": int(vacancy.lattice_site),
        "vacancy_atom_id": int(vacancy.atom_id),
        "interstitial_atom_id": int(mover_id),
        "separation": float(
            _distance(initial, initial.positions[mover_index], target),
        ),
        "lattice_parameter": float(lattice_parameter),
    }
    mechanism = MigrationMechanism(
        kind="frenkel_recombination",
        moving_atom_ids=tuple(moving),
        parameters=parameters,
        notes=("recombination hypothesis; barrier and spontaneity are unknown",),
    )
    return _assemble(initial, final, config, None, mechanism)


__all__ = [
    "MigrationPair",
    "generate_frenkel_recombination",
    "generate_interstitial_migration",
    "generate_vacancy_hop",
    "vacancy_neighbors",
]
