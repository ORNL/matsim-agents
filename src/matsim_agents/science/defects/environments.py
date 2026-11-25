"""Local chemical environments of generated point defects.

Pure geometry and bookkeeping: neighbour species and distances around vacancies (taken from the
unrelaxed parent lattice) and around interstitial atoms (taken from the defective structure),
plus the periodic vacancy-interstitial separation and a parent-versus-defective comparison. No
energies are computed and no stability or qualification is implied.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
from ase import Atoms
from ase.geometry import get_distances

from matsim_agents.science.defects.point_defects import DefectConfiguration
from matsim_agents.science.structures.alloy_ensemble import occupation_fingerprint
from matsim_agents.science.structures.alloys import first_neighbor_list, second_neighbor_list

Composition = tuple[tuple[str, int], ...]


def _composition(symbols: list[str]) -> Composition:
    return tuple(sorted(Counter(symbols).items()))


@dataclass(frozen=True)
class VacancyEnvironment:
    """Shell composition around one vacated lattice site, read from the parent lattice.

    Neighbours that are themselves vacated (divacancy) are listed in ``vacant_first`` /
    ``vacant_second`` and excluded from the compositions.
    """

    lattice_site: int
    species: str
    first_shell: tuple[int, ...]
    second_shell: tuple[int, ...]
    first_composition: Composition
    second_composition: Composition
    vacant_first: tuple[int, ...]
    vacant_second: tuple[int, ...]


@dataclass(frozen=True)
class InterstitialEnvironment:
    """Nearest atoms around one interstitial (or displaced) atom of the defective structure.

    ``nearest_*`` hold the ``n_nearest`` closest other atoms (minimum image) in distance order,
    ties broken by index. ``coordination`` counts atoms within ``(1 + shell_tolerance)`` times
    the nearest distance.
    """

    atom_id: int
    species: str
    nearest_indices: tuple[int, ...]
    nearest_species: tuple[str, ...]
    nearest_distances: tuple[float, ...]
    nearest_distance: float
    coordination: int
    coordination_composition: Composition


@dataclass(frozen=True)
class DefectEnvironment:
    """Environment metadata of one defect configuration relative to its parent."""

    config_id: str
    parent_composition: Composition
    defective_composition: Composition
    composition_change: tuple[tuple[str, int], ...]
    vacancies: tuple[VacancyEnvironment, ...]
    interstitials: tuple[InterstitialEnvironment, ...]
    vacancy_interstitial_separation: float | None


def _check_pair(parent: Atoms, defective: Atoms, config: DefectConfiguration) -> None:
    if len(parent) != config.n_parent_atoms:
        raise ValueError(f"parent has {len(parent)} atoms, config expects {config.n_parent_atoms}")
    if len(defective) != config.n_atoms:
        raise ValueError(f"defective has {len(defective)} atoms, config expects {config.n_atoms}")
    if occupation_fingerprint(parent) != config.parent_ordering_id:
        raise ValueError("parent occupation does not match the configuration's parent ordering")
    if not parent.pbc.all() or not defective.pbc.all():
        raise ValueError("environments require fully periodic structures")


def vacancy_environments(
    parent: Atoms, config: DefectConfiguration, lattice_parameter: float
) -> tuple[VacancyEnvironment, ...]:
    """First- and second-shell composition around each vacancy site, from the parent lattice."""
    if not np.isfinite(lattice_parameter) or lattice_parameter <= 0:
        raise ValueError(f"lattice_parameter must be positive and finite, got {lattice_parameter}")
    vacant = {s.lattice_site for s in config.sites if s.role == "vacancy"}
    symbols = parent.get_chemical_symbols()
    first_i, first_j, _ = first_neighbor_list(parent, lattice_parameter)
    second_i, second_j, _ = second_neighbor_list(parent, lattice_parameter)
    result = []
    for site in sorted(s for s in vacant if s is not None):
        first = tuple(sorted(int(j) for j in first_j[first_i == site]))
        second = tuple(sorted(int(j) for j in second_j[second_i == site]))
        result.append(
            VacancyEnvironment(
                lattice_site=site,
                species=symbols[site],
                first_shell=first,
                second_shell=second,
                first_composition=_composition([symbols[j] for j in first if j not in vacant]),
                second_composition=_composition([symbols[j] for j in second if j not in vacant]),
                vacant_first=tuple(j for j in first if j in vacant),
                vacant_second=tuple(j for j in second if j in vacant),
            )
        )
    return tuple(result)


def interstitial_environments(
    defective: Atoms,
    config: DefectConfiguration,
    *,
    n_nearest: int = 8,
    shell_tolerance: float = 0.1,
) -> tuple[InterstitialEnvironment, ...]:
    """Nearest-atom environment of each interstitial site, from the defective structure."""
    if isinstance(n_nearest, bool) or not isinstance(n_nearest, int) or n_nearest < 1:
        raise ValueError(f"n_nearest must be a positive integer, got {n_nearest!r}")
    if not np.isfinite(shell_tolerance) or shell_tolerance < 0:
        raise ValueError(f"shell_tolerance must be nonnegative and finite, got {shell_tolerance}")
    if "atom_id" not in defective.arrays:
        raise ValueError("defective structure has no 'atom_id' array")
    ids = defective.get_array("atom_id").tolist()
    symbols = defective.get_chemical_symbols()
    result = []
    for site in config.sites:
        if site.role != "interstitial":
            continue
        if ids.count(site.atom_id) != 1:
            raise ValueError(f"atom_id {site.atom_id} is not unique in the defective structure")
        index = ids.index(site.atom_id)
        _, distances = get_distances(
            defective.positions[index], defective.positions, cell=defective.cell, pbc=True
        )
        distances = distances[0].copy()
        distances[index] = np.inf
        order = np.lexsort((np.arange(len(distances)), distances))[: min(n_nearest, len(ids) - 1)]
        nearest = float(distances[order[0]])
        shell = [int(k) for k in np.flatnonzero(distances <= nearest * (1.0 + shell_tolerance))]
        result.append(
            InterstitialEnvironment(
                atom_id=site.atom_id,
                species=symbols[index],
                nearest_indices=tuple(int(k) for k in order),
                nearest_species=tuple(symbols[k] for k in order),
                nearest_distances=tuple(float(distances[k]) for k in order),
                nearest_distance=nearest,
                coordination=len(shell),
                coordination_composition=_composition([symbols[k] for k in shell]),
            )
        )
    return tuple(result)


def vacancy_interstitial_separation(config: DefectConfiguration) -> float | None:
    """Smallest minimum-image distance between any vacancy and any interstitial site.

    ``None`` when the configuration has no vacancy or no interstitial.
    """
    vacancies = [s.position for s in config.sites if s.role == "vacancy"]
    interstitials = [s.position for s in config.sites if s.role == "interstitial"]
    if not vacancies or not interstitials:
        return None
    _, distances = get_distances(
        np.array(vacancies), np.array(interstitials), cell=np.array(config.cell), pbc=True
    )
    return float(distances.min())


def defect_environment(
    parent: Atoms,
    defective: Atoms,
    config: DefectConfiguration,
    lattice_parameter: float,
    *,
    n_nearest: int = 8,
    shell_tolerance: float = 0.1,
) -> DefectEnvironment:
    """Parent-versus-defective environment metadata for one generated configuration."""
    _check_pair(parent, defective, config)
    before = Counter(parent.get_chemical_symbols())
    after = Counter(defective.get_chemical_symbols())
    change = tuple(
        (symbol, after[symbol] - before[symbol])
        for symbol in sorted(set(before) | set(after))
        if after[symbol] != before[symbol]
    )
    return DefectEnvironment(
        config_id=config.config_id,
        parent_composition=tuple(sorted(before.items())),
        defective_composition=tuple(sorted(after.items())),
        composition_change=change,
        vacancies=vacancy_environments(parent, config, lattice_parameter),
        interstitials=interstitial_environments(
            defective, config, n_nearest=n_nearest, shell_tolerance=shell_tolerance
        ),
        vacancy_interstitial_separation=vacancy_interstitial_separation(config),
    )


__all__ = [
    "DefectEnvironment",
    "InterstitialEnvironment",
    "VacancyEnvironment",
    "defect_environment",
    "interstitial_environments",
    "vacancy_environments",
    "vacancy_interstitial_separation",
]
