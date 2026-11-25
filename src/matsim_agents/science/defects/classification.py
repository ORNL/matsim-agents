"""Outcome classification of a relaxed point-defect structure.

Rule-based and purely geometric. The classification compares the defect sites found by
:func:`~matsim_agents.science.defects.mapping.identify_defect_sites` with the sites
the configuration started with. Anything that cannot be decided is ``unresolved`` with the
reasons in ``notes``: an unconverged or unknown relaxation, a changed cell, non-finite
coordinates, a changed atom count or composition, or ambiguous or conflicting atoms. Split
and collective defects (dumbbells, crowdions; configurations with ``displaced`` sites) cannot
be represented by one-atom-per-site mapping, so a change that would otherwise be
``transformed`` is ``unresolved`` for them. A classification is a geometric label for one
structure and is not evidence of energetic stability or of DFT/MLIP qualification.

Rules for a resolved structure, with V0 and I0 the initial vacancy sites and interstitial count:

- ``recombined_restored``: no vacancy, interstitial or antisite remains.
- ``recombined_antisite``: no vacancy or interstitial remains, but species antisites do.
- ``persisted``: vacancies are exactly V0, the interstitial count equals I0 and no antisite
  appears.
- ``transformed``: anything else, for example a migrated vacancy, a changed defect count or
  antisites next to remaining vacancies or interstitials.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from ase import Atoms

from matsim_agents.science.defects.mapping import (
    DEFAULT_AMBIGUITY_MARGIN,
    DEFAULT_SITE_TOLERANCE,
    assign_lattice_sites,
    identify_defect_sites,
)
from matsim_agents.science.defects.point_defects import (
    DefectConfiguration,
    DefectOutcome,
    geometry_fingerprint,
)
from matsim_agents.science.structures.alloy_ensemble import occupation_fingerprint


def _expected_composition(parent: Atoms, config: DefectConfiguration) -> Counter[str]:
    expected = Counter(parent.get_chemical_symbols())
    for site in config.sites:
        if site.role == "vacancy":
            expected[site.species] -= 1
        elif site.role == "interstitial":
            expected[site.species] += 1
    return +expected


def classify_defect_outcome(
    parent: Atoms,
    relaxed: Atoms,
    config: DefectConfiguration,
    lattice_parameter: float,
    *,
    relaxation_converged: bool | None,
    site_tolerance: float = DEFAULT_SITE_TOLERANCE,
    ambiguity_margin: float = DEFAULT_AMBIGUITY_MARGIN,
    cell_tolerance: float = 1e-6,
) -> DefectOutcome:
    """Classify ``relaxed`` against ``parent`` and the initial ``config``.

    Raises ``ValueError`` when the inputs do not belong together (wrong parent ordering, or a
    non-periodic structure). Evidence problems give an ``unresolved`` outcome instead.
    """
    if occupation_fingerprint(parent) != config.parent_ordering_id:
        raise ValueError("parent occupation does not match the configuration's parent ordering")
    if len(parent) != config.n_parent_atoms:
        raise ValueError(f"parent has {len(parent)} atoms, config expects {config.n_parent_atoms}")
    if not parent.pbc.all() or not relaxed.pbc.all():
        raise ValueError("classification requires fully periodic structures")

    fingerprint = geometry_fingerprint(relaxed) if len(relaxed) else None
    notes: list[str] = []

    def unresolved(*reasons: str, sites=()) -> DefectOutcome:
        return DefectOutcome(
            config_id=config.config_id,
            classification="unresolved",
            relaxation_converged=relaxation_converged,
            final_sites=tuple(sites),
            geometry_fingerprint=fingerprint,
            notes=(*notes, *reasons),
        )

    if relaxation_converged is not True:
        label = "unknown" if relaxation_converged is None else "did not converge"
        return unresolved(f"relaxation {label}; no classification is made")
    if len(relaxed) != config.n_atoms:
        return unresolved(f"relaxed structure has {len(relaxed)} atoms, expected {config.n_atoms}")
    if Counter(relaxed.get_chemical_symbols()) != _expected_composition(parent, config):
        return unresolved("relaxed composition differs from the configuration's expected one")
    if not np.isfinite(relaxed.positions).all():
        return unresolved("relaxed positions contain non-finite values")
    if not np.allclose(parent.cell.array, relaxed.cell.array, rtol=0.0, atol=cell_tolerance):
        return unresolved("cell changed during relaxation; only fixed-cell mapping is supported")

    assignment = assign_lattice_sites(
        parent,
        relaxed,
        lattice_parameter,
        site_tolerance=site_tolerance,
        ambiguity_margin=ambiguity_margin,
        cell_tolerance=cell_tolerance,
    )
    found = identify_defect_sites(parent, relaxed, assignment)
    if not found.resolved:
        reasons = []
        if assignment.ambiguous:
            reasons.append(f"ambiguous atoms {list(assignment.ambiguous)}")
        if assignment.conflicts:
            reasons.append(f"atoms {list(assignment.conflicts)} share lattice sites")
        return unresolved("site mapping is not resolved: " + "; ".join(reasons))

    initial_vacancies = {s.lattice_site for s in config.sites if s.role == "vacancy"}
    initial_interstitials = sum(s.role == "interstitial" for s in config.sites)
    final_vacancies = {s.lattice_site for s in found.vacancies}
    if not final_vacancies and not found.interstitials:
        label = "recombined_antisite" if found.antisites else "recombined_restored"
    elif (
        final_vacancies == initial_vacancies
        and len(found.interstitials) == initial_interstitials
        and not found.antisites
    ):
        label = "persisted"
    elif any(s.role == "displaced" for s in config.sites):
        return unresolved(
            "split or collective defect (displaced atoms) changed under relaxation; site mapping "
            "cannot represent it, so no classification is made"
        )
    else:
        label = "transformed"
    if found.antisites:
        notes.append(
            "antisites at lattice sites " + str(sorted(a.lattice_site for a in found.antisites))
        )
    return DefectOutcome(
        config_id=config.config_id,
        classification=label,
        relaxation_converged=True,
        final_sites=found.final_sites,
        geometry_fingerprint=fingerprint,
        notes=tuple(notes),
    )


__all__ = ["classify_defect_outcome"]
