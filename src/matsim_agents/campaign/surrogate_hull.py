"""Shared coverage checks for MLIP surrogate hull references."""

from __future__ import annotations

import hashlib
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
    available_endpoints = manifest_endpoints if elemental_endpoints is None else elemental_endpoints
    target_elements = {str(element) for element in Composition(target_formula).elements}
    required_subsystems = {frozenset(pair) for pair in combinations(target_elements, 2)}
    missing_elements = target_elements - available_endpoints
    missing_subsystems = required_subsystems - covered_subsystems
    return {
        "evidence_level": "mlip_proxy",
        "reference_manifest": reference_manifest,
        "provisional": bool(missing_elements or missing_subsystems),
        "missing_elemental_references": sorted(missing_elements),
        "missing_binary_subsystems": sorted("-".join(sorted(pair)) for pair in missing_subsystems),
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
    relax_targets: bool = True,
) -> dict[str, object]:
    """Evaluate a model-specific proxy hull with every phase relaxed by that model.

    Unary polymorphs, multielement references and (unless ``relax_targets`` is false)
    the targets are relaxed with the same controls, so all hull entries are minima of
    one potential energy surface. The ``unary_*`` controls apply to every phase.
    """
    from ase.io import read as ase_read
    from pymatgen.analysis.phase_diagram import PhaseDiagram
    from pymatgen.core import Composition
    from pymatgen.entries.computed_entries import ComputedEntry

    from matsim_agents.campaign.unary_references import _safe_name, relax_structure

    target_composition = Composition(target_formula)
    model_hull = _model_hull_entries(
        reference_manifest,
        {str(element) for element in target_composition.elements},
        calculator,
        target_formula=target_formula,
        model_identifier=model_identifier,
        max_steps=unary_max_steps,
        fmax_eV_per_A=unary_fmax_eV_per_A,
        maxstep_A=unary_maxstep_A,
        minimum_unique_unary=minimum_unique_unary,
    )
    coverage = model_hull["coverage"]
    if coverage["missing_elemental_references"]:
        return coverage
    entries = model_hull["entries"]
    coverage["target_geometry"] = "model_relaxed" if relax_targets else "fixed_geometry"

    target_dir = Path(model_hull["cache_dir"]) / "targets"
    target_dir.mkdir(parents=True, exist_ok=True)
    targets = []
    unconverged_targets = []
    for index, label in enumerate(labels):
        structure_path = str(label["structure_path"])
        atoms = ase_read(structure_path)
        composition = Composition(atoms.get_chemical_formula())
        if composition.reduced_composition != target_composition.reduced_composition:
            raise ValueError(
                f"Target structure composition {composition.reduced_formula} does not match "
                f"target formula {target_composition.reduced_formula}"
            )
        energy = float(label["energy_eV"])
        if relax_targets:
            digest = hashlib.sha256(Path(structure_path).read_bytes()).hexdigest()[:16]
            name = f"target-{index:03d}-{_safe_name(Path(structure_path).stem)}-{digest}"
            try:
                converged, steps, max_force, energy = relax_structure(
                    structure_path,
                    calculator,
                    output_path=target_dir / f"{name}.extxyz",
                    log_path=target_dir / f"{name}.log",
                    max_steps=unary_max_steps,
                    fmax_eV_per_A=unary_fmax_eV_per_A,
                    maxstep_A=unary_maxstep_A,
                    relax_cell=True,
                )
            except Exception as exc:  # noqa: BLE001
                label["surrogate_hull_status"] = f"relaxation_failed: {exc!r}"
                unconverged_targets.append(structure_path)
                continue
            label["surrogate_relaxed_structure_path"] = str(
                (target_dir / f"{name}.extxyz").resolve()
            )
            label["surrogate_relaxed_energy_eV"] = energy
            label["surrogate_relaxation_steps"] = steps
            label["surrogate_residual_max_force_eV_per_A"] = max_force
            if not converged:
                label["surrogate_hull_status"] = "relaxation_unconverged"
                unconverged_targets.append(structure_path)
                continue
        entry = ComputedEntry(composition, energy, entry_id=f"target-{index}")
        entries.append(entry)
        targets.append((label, entry))
    coverage["unconverged_targets"] = unconverged_targets
    coverage["provisional"] = bool(
        coverage["provisional"] or unconverged_targets or not relax_targets
    )
    if not targets:
        return coverage

    diagram = PhaseDiagram(entries)
    for label, entry in targets:
        decomposition, hull_energy = diagram.get_decomp_and_e_above_hull(entry)
        label["surrogate_hull_status"] = "evaluated"
        label["surrogate_formation_energy_eV_per_atom"] = float(
            diagram.get_form_energy_per_atom(entry)
        )
        label["surrogate_energy_above_hull_eV_per_atom"] = float(hull_energy)
        label["surrogate_decomposition"] = {
            str(product.entry_id or product.composition.reduced_formula): float(fraction)
            for product, fraction in decomposition.items()
        }
    return coverage


def _model_hull_entries(
    reference_manifest: Path,
    target_elements: set[str],
    calculator: Any,
    *,
    target_formula: str,
    model_identifier: str,
    max_steps: int,
    fmax_eV_per_A: float,
    maxstep_A: float,
    minimum_unique_unary: int,
) -> dict[str, Any]:
    """Model-relaxed unary and multielement hull entries with their coverage audit."""
    from ase.io import read as ase_read
    from pymatgen.core import Composition
    from pymatgen.entries.computed_entries import ComputedEntry

    from matsim_agents.campaign.unary_references import (
        load_reference_phase_specs,
        relax_compound_references,
        relax_unary_references,
        unary_cache_directory,
    )
    from matsim_agents.discovery.defects import MAGNETISM_LIMITATION, magnetic_elements

    specs = load_reference_phase_specs(reference_manifest)
    search_elements = {
        str(composition.elements[0])
        for spec in specs
        if len((composition := Composition(str(spec["formula"]))).elements) == 1
    }
    settings = {
        "schema_version": 1,
        "max_steps": max_steps,
        "fmax_eV_per_A": fmax_eV_per_A,
        "maxstep_A": maxstep_A,
        "relax_cell": True,
        "elements": sorted(search_elements),
    }
    cache_dir = unary_cache_directory(reference_manifest, model_identifier, settings)
    unary_search = relax_unary_references(
        specs,
        search_elements,
        calculator,
        model_identifier=model_identifier,
        output_dir=cache_dir,
        max_steps=max_steps,
        fmax_eV_per_A=fmax_eV_per_A,
        maxstep_A=maxstep_A,
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
    present_magnetic = magnetic_elements(
        {str(element) for spec in specs for element in Composition(str(spec["formula"])).elements}
        | target_elements
    )
    coverage["magnetic_elements_present"] = present_magnetic
    coverage["limitations"] = [MAGNETISM_LIMITATION]
    if present_magnetic:
        coverage["provisional"] = True
    if coverage["missing_elemental_references"]:
        return {"coverage": coverage, "entries": [], "cache_dir": str(cache_dir)}

    entries = []
    for phase in unary_search.phases:
        if (
            not phase.converged
            or phase.duplicate_of is not None
            or phase.corrected_energy_per_atom_eV is None
        ):
            continue
        composition = Composition(phase.element)
        entries.append(
            ComputedEntry(
                composition,
                phase.corrected_energy_per_atom_eV * composition.num_atoms,
                entry_id=phase.phase_id,
            )
        )

    compound_search = relax_compound_references(
        specs,
        calculator,
        model_identifier=model_identifier,
        output_dir=cache_dir / "compounds",
        max_steps=max_steps,
        fmax_eV_per_A=fmax_eV_per_A,
        maxstep_A=maxstep_A,
        relax_cell=True,
    )
    coverage["compound_reference_search"] = compound_search.model_dump()
    excluded = []
    for phase in compound_search.phases:
        if not phase.converged or phase.total_energy_eV is None:
            excluded.append(phase.phase_id)
            continue
        atoms = ase_read(str(phase.optimized_structure_path))
        composition = Composition(atoms.get_chemical_formula())
        if composition.reduced_composition != Composition(phase.formula).reduced_composition:
            raise ValueError(
                f"Reference phase {phase.phase_id} structure composition "
                f"{composition.reduced_formula} does not match declared formula "
                f"{Composition(phase.formula).reduced_formula}"
            )
        entries.append(
            ComputedEntry(
                composition,
                phase.total_energy_eV + phase.energy_correction_eV_per_atom * len(atoms),
                entry_id=phase.phase_id,
            )
        )
    coverage["excluded_unconverged_compound_references"] = excluded
    coverage["provisional"] = bool(coverage["provisional"] or excluded)
    return {"coverage": coverage, "entries": entries, "cache_dir": str(cache_dir)}


def evaluate_surrogate_defects(
    reference_manifest: Path,
    host_path: str | Path,
    defect_paths: dict[str, str | Path],
    calculator: Any,
    *,
    model_identifier: str,
    max_steps: int = 200,
    fmax_eV_per_A: float = 0.02,
    maxstep_A: float = 0.01,
    minimum_unique_unary: int = 2,
    host_hull_tolerance_eV_per_atom: float = 0.0,
) -> dict[str, Any]:
    """MLIP neutral defect energies against the same model-relaxed hull.

    The host supercell is cell-relaxed; each defect keeps the host cell and relaxes
    its positions (dilute limit). All energies come from ``calculator``.
    """
    from ase.io import read as ase_read
    from ase.io import write as ase_write

    from matsim_agents.campaign.unary_references import _safe_name, relax_structure
    from matsim_agents.discovery.defects import defect_report

    host_atoms = ase_read(str(host_path))
    host_formula = host_atoms.get_chemical_formula(mode="hill", empirical=True)
    elements = set(host_atoms.get_chemical_symbols())
    for path in defect_paths.values():
        elements.update(ase_read(str(path)).get_chemical_symbols())
    model_hull = _model_hull_entries(
        reference_manifest,
        elements,
        calculator,
        target_formula=host_formula,
        model_identifier=model_identifier,
        max_steps=max_steps,
        fmax_eV_per_A=fmax_eV_per_A,
        maxstep_A=maxstep_A,
        minimum_unique_unary=minimum_unique_unary,
    )
    coverage = model_hull["coverage"]
    if coverage["missing_elemental_references"]:
        return {"coverage": coverage, "status": "missing_elemental_references"}

    defect_dir = Path(model_hull["cache_dir"]) / "defects"
    defect_dir.mkdir(parents=True, exist_ok=True)
    relaxations: dict[str, dict[str, Any]] = {}
    host_relaxed = defect_dir / "host.extxyz"
    converged, steps, max_force, host_energy = relax_structure(
        host_path,
        calculator,
        output_path=host_relaxed,
        log_path=defect_dir / "host.log",
        max_steps=max_steps,
        fmax_eV_per_A=fmax_eV_per_A,
        maxstep_A=maxstep_A,
        relax_cell=True,
    )
    relaxations["host"] = {
        "converged": converged,
        "steps": steps,
        "residual_max_force_eV_per_A": max_force,
        "energy_eV": host_energy,
        "relaxed_structure_path": str(host_relaxed.resolve()),
    }
    relaxed_host = ase_read(host_relaxed)
    defects = []
    for defect_id, path in defect_paths.items():
        atoms = ase_read(str(path))
        # Dilute limit: the defect keeps the relaxed host lattice.
        atoms.set_cell(relaxed_host.cell, scale_atoms=True)
        start = defect_dir / f"{_safe_name(defect_id)}-start.extxyz"
        relaxed = defect_dir / f"{_safe_name(defect_id)}.extxyz"
        ase_write(start, atoms)
        converged, steps, max_force, energy = relax_structure(
            start,
            calculator,
            output_path=relaxed,
            log_path=defect_dir / f"{_safe_name(defect_id)}.log",
            max_steps=max_steps,
            fmax_eV_per_A=fmax_eV_per_A,
            maxstep_A=maxstep_A,
            relax_cell=False,
        )
        relaxations[defect_id] = {
            "converged": converged,
            "steps": steps,
            "residual_max_force_eV_per_A": max_force,
            "energy_eV": energy,
            "relaxed_structure_path": str(relaxed.resolve()),
        }
        if converged:
            defects.append(
                {
                    "defect_id": defect_id,
                    "symbols": atoms.get_chemical_symbols(),
                    "energy_eV": energy,
                }
            )
    unconverged = sorted(name for name, item in relaxations.items() if not item["converged"])
    if not relaxations["host"]["converged"]:
        return {
            "coverage": coverage,
            "relaxations": relaxations,
            "status": "host_relaxation_unconverged",
        }
    report = defect_report(
        model_hull["entries"],
        host_formula,
        relaxed_host.get_chemical_symbols(),
        host_energy,
        defects,
        host_hull_tolerance_eV_per_atom=host_hull_tolerance_eV_per_atom,
    )
    report["evidence_level"] = "mlip_proxy"
    report["model_identifier"] = model_identifier
    report["coverage"] = coverage
    report["relaxations"] = relaxations
    report["unconverged_relaxations"] = unconverged
    report["provisional"] = bool(coverage["provisional"] or unconverged)
    return report
