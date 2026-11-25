"""Formation energies of classified point defects.

A thin, strict wrapper around
:func:`matsim_agents.science.defects.defects.defect_formation_energies`. The caller
supplies the host and defect total energies, the method each came from, and explicit
chemical-potential references (one mapping per vertex). Nothing is computed here: the module
checks that the evidence belongs together and then applies

    E_f = E_defect - E_host - sum_i dn_i * mu_i

with ``dn`` the species-count change between the parent and the relaxed structure. A
configuration is rejected, not guessed at, when the relaxation did not converge, the outcome is
unresolved or belongs to another configuration or structure, the energy methods differ, or the
stoichiometry or a chemical potential is inconsistent. Neutral defects only; magnetism is not
handled (see ``defects.MAGNETISM_LIMITATION`` and ``defects.CHARGE_LIMITATION``). A number from
this module is only as good as the supplied energies and says nothing about DFT/MLIP
qualification of the method.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from ase import Atoms

from matsim_agents.science.defects.defects import (
    CHARGE_LIMITATION,
    MAGNETISM_LIMITATION,
    defect_formation_energies,
    stoichiometry_change,
)
from matsim_agents.science.defects.point_defects import (
    DefectConfiguration,
    DefectOutcome,
    geometry_fingerprint,
)
from matsim_agents.science.structures.alloy_ensemble import occupation_fingerprint


@dataclass(frozen=True)
class ClassifiedDefectEnergy:
    """Formation energy of one defect at every supplied chemical-potential vertex (eV)."""

    config_id: str
    classification: str
    method: str
    stoichiometry_change: tuple[tuple[str, int], ...]
    formation_energy_eV_by_vertex: tuple[tuple[str, float], ...]
    formation_energy_min_eV: float
    formation_energy_max_eV: float
    stoichiometry_conserving: bool
    limitations: tuple[str, ...]


def _finite(name: str, value: float) -> float:
    if isinstance(value, bool) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return float(value)


def classified_defect_formation_energy(
    parent: Atoms,
    relaxed: Atoms,
    config: DefectConfiguration,
    outcome: DefectOutcome,
    *,
    host_energy_eV: float,
    defect_energy_eV: float,
    host_method: str,
    defect_method: str,
    chemical_potentials: Mapping[str, Mapping[str, float]],
) -> ClassifiedDefectEnergy:
    """Formation energy of a converged, classified defect against explicit references.

    ``parent`` is the host supercell the energy ``host_energy_eV`` belongs to, ``relaxed`` the
    structure of ``defect_energy_eV``. ``chemical_potentials[vertex][element]`` are absolute
    chemical potentials in eV/atom; they must cover every element whose count changes.
    """
    if not host_method or not defect_method:
        raise ValueError("host_method and defect_method must be nonempty strings")
    if host_method != defect_method:
        raise ValueError(
            f"host and defect energies come from different methods "
            f"({host_method!r} vs {defect_method!r})"
        )
    host_energy = _finite("host_energy_eV", host_energy_eV)
    defect_energy = _finite("defect_energy_eV", defect_energy_eV)
    if outcome.config_id != config.config_id:
        raise ValueError("outcome belongs to a different defect configuration")
    if outcome.relaxation_converged is not True:
        raise ValueError("relaxation did not converge; no formation energy is reported")
    if outcome.classification == "unresolved":
        raise ValueError("defect outcome is unresolved; no formation energy is reported")
    if outcome.classification == "recombined_restored":
        raise ValueError("defect recombined to the perfect lattice; there is no defect to score")
    if outcome.geometry_fingerprint != geometry_fingerprint(relaxed):
        raise ValueError("relaxed structure does not match the classified outcome")
    if len(parent) != config.n_parent_atoms:
        raise ValueError(f"parent has {len(parent)} atoms, config expects {config.n_parent_atoms}")
    if occupation_fingerprint(parent) != config.parent_ordering_id:
        raise ValueError("parent is not the host ordering this defect was generated from")
    if not chemical_potentials:
        raise ValueError("at least one chemical-potential vertex is required")
    for vertex, mu in chemical_potentials.items():
        for element, value in mu.items():
            _finite(f"chemical potential {element} at {vertex}", value)

    change = stoichiometry_change(parent.get_chemical_symbols(), relaxed.get_chemical_symbols())
    expected = Counter(parent.get_chemical_symbols())
    for site in config.sites:
        if site.role == "vacancy":
            expected[site.species] -= 1
        elif site.role == "interstitial":
            expected[site.species] += 1
    if Counter(relaxed.get_chemical_symbols()) != +expected:
        raise ValueError("relaxed composition differs from the configuration's expected one")

    energies = defect_formation_energies(
        parent.get_chemical_symbols(),
        host_energy,
        relaxed.get_chemical_symbols(),
        defect_energy,
        {vertex: dict(mu) for vertex, mu in chemical_potentials.items()},
    )
    return ClassifiedDefectEnergy(
        config_id=config.config_id,
        classification=outcome.classification,
        method=host_method,
        stoichiometry_change=tuple(change.items()),
        formation_energy_eV_by_vertex=tuple(energies.items()),
        formation_energy_min_eV=min(energies.values()),
        formation_energy_max_eV=max(energies.values()),
        stoichiometry_conserving=not change,
        limitations=(MAGNETISM_LIMITATION, CHARGE_LIMITATION),
    )


__all__ = ["ClassifiedDefectEnergy", "classified_defect_formation_energy"]
