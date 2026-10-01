"""Shared coverage checks for MLIP surrogate hull references."""

from __future__ import annotations

from collections.abc import Iterable
from itertools import combinations
from pathlib import Path
from typing import Any


def reference_coverage(formulas: Iterable[str]) -> tuple[set[str], set[frozenset[str]]]:
    """Return pure-element endpoints and multielement subsystems in a reference set."""
    from pymatgen.core import Composition

    elemental_endpoints: set[str] = set()
    covered_subsystems: set[frozenset[str]] = set()
    for formula in formulas:
        elements = frozenset(str(element) for element in Composition(formula).elements)
        if len(elements) == 1:
            elemental_endpoints.update(elements)
        else:
            covered_subsystems.add(elements)
    return elemental_endpoints, covered_subsystems


def surrogate_hull_coverage(
    formulas: Iterable[str],
    target_formula: str,
    reference_manifest: str,
    *,
    elemental_endpoints: set[str] | None = None,
) -> dict[str, object]:
    """Describe whether references can support an MLIP proxy hull for a target."""
    from pymatgen.core import Composition

    manifest_endpoints, covered_subsystems = reference_coverage(formulas)
    available_endpoints = (
        manifest_endpoints if elemental_endpoints is None else elemental_endpoints
    )
    target_elements = {str(element) for element in Composition(target_formula).elements}
    required_subsystems = {frozenset(pair) for pair in combinations(target_elements, 2)}
    missing_elements = target_elements - available_endpoints
    missing_subsystems = required_subsystems - covered_subsystems
    return {
        "evidence_level": "mlip_proxy",
        "reference_manifest": reference_manifest,
        "provisional": bool(missing_elements or missing_subsystems),
        "missing_elemental_references": sorted(missing_elements),
        "missing_binary_subsystems": sorted(
            "-".join(sorted(pair)) for pair in missing_subsystems
        ),
    }


def evaluate_surrogate_hull(
    reference_manifest: Path,
    target_formula: str,
    labels: list[dict[str, Any]],
    calculator: Any,
    *,
    model_identifier: str,
    unary_max_steps: int = 200,
    unary_fmax_eV_per_A: float = 0.02,
    unary_maxstep_A: float = 0.01,
    minimum_unique_unary: int = 2,
) -> dict[str, object]:
    """Evaluate a model-specific proxy hull with relaxed unary polymorphs."""
    from ase.io import read as ase_read
    from pymatgen.analysis.phase_diagram import PhaseDiagram
    from pymatgen.core import Composition
    from pymatgen.entries.computed_entries import ComputedEntry

    from matsim_agents.campaign.unary_references import (
        load_reference_phase_specs,
        relax_unary_references,
        unary_cache_directory,
    )

    specs = load_reference_phase_specs(reference_manifest)
    target_composition = Composition(target_formula)
    target_elements = {str(element) for element in target_composition.elements}
    search_elements = {
        str(composition.elements[0])
        for spec in specs
        if len((composition := Composition(str(spec["formula"]))).elements) == 1
    }
    settings = {
        "schema_version": 1,
        "max_steps": unary_max_steps,
        "fmax_eV_per_A": unary_fmax_eV_per_A,
        "maxstep_A": unary_maxstep_A,
        "relax_cell": True,
        "elements": sorted(search_elements),
    }
    unary_search = relax_unary_references(
        specs,
        search_elements,
        calculator,
        model_identifier=model_identifier,
        output_dir=unary_cache_directory(reference_manifest, model_identifier, settings),
        max_steps=unary_max_steps,
        fmax_eV_per_A=unary_fmax_eV_per_A,
        maxstep_A=unary_maxstep_A,
        relax_cell=True,
    )
    coverage = surrogate_hull_coverage(
        (str(spec["formula"]) for spec in specs),
        target_formula,
        str(reference_manifest),
        elemental_endpoints=set(unary_search.selected_endpoints),
    )
    coverage["unary_reference_search"] = unary_search.model_dump()
    undercovered = sorted(
        element
        for element in target_elements
        if unary_search.unique_converged_counts.get(element, 0) < minimum_unique_unary
    )
    coverage["minimum_unique_unary_polymorphs"] = minimum_unique_unary
    coverage["undercovered_unary_elements"] = undercovered
    coverage["provisional"] = bool(coverage["provisional"] or undercovered)
    if unary_search.missing_elements:
        return coverage

    entries = []
    for phase in unary_search.phases:
        if (
            not phase.converged
            or phase.duplicate_of is not None
            or phase.total_energy_eV is None
        ):
            continue
        composition = Composition(phase.formula)
        corrected_total = (
            phase.total_energy_eV
            + phase.energy_correction_eV_per_atom * composition.num_atoms
        )
        entries.append(
            ComputedEntry(composition, corrected_total, entry_id=phase.phase_id)
        )

    for spec in specs:
        composition = Composition(str(spec["formula"]))
        if len(composition.elements) == 1:
            continue
        atoms = ase_read(str(spec["path"]))
        atoms.calc = calculator
        correction = float(spec.get("energy_correction_eV_per_atom", 0.0))
        total_energy = float(atoms.get_potential_energy()) + correction * len(atoms)
        entries.append(
            ComputedEntry(
                composition,
                total_energy,
                entry_id=str(spec["phase_id"]),
            )
        )

    targets = []
    for index, label in enumerate(labels):
        entry = ComputedEntry(
            target_composition,
            float(label["energy_eV"]),
            entry_id=f"target-{index}",
        )
        entries.append(entry)
        targets.append((label, entry))

    diagram = PhaseDiagram(entries)
    for label, entry in targets:
        decomposition, hull_energy = diagram.get_decomp_and_e_above_hull(entry)
        label["surrogate_formation_energy_eV_per_atom"] = float(
            diagram.get_form_energy_per_atom(entry)
        )
        label["surrogate_energy_above_hull_eV_per_atom"] = float(hull_energy)
        label["surrogate_decomposition"] = {
            str(product.entry_id or product.composition.reduced_formula): float(fraction)
            for product, fraction in decomposition.items()
        }
    return coverage
