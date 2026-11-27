"""DFT relaxation settings and the reference energies that hull ranking needs."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from matsim_agents.core.composition import parse_composition
from matsim_agents.core.relaxation import RelaxationResult
from matsim_agents.science.thermodynamics.stability import (
    ElementalReferenceEntry,
    ReferenceEnergySet,
    ReferencePhaseEntry,
)
from matsim_agents.workflows.active_learning.config import ALConfig
from matsim_agents.workflows.candidate_registry import (
    structure_content_hash,
)
from matsim_agents.workflows.multi_formula.formula_config import (
    CampaignDFTRefinementConfig,
    ReferenceStructureSpec,
)
from matsim_agents.workflows.relaxation import (
    DFTBackendConfig,
    GeometryControls,
    RelaxationMode,
    ScientificRelaxationConfig,
    ScientificRelaxationResult,
)


def _vasp_template_settings(al_cfg: ALConfig) -> dict[str, Any]:
    vasp = al_cfg.dft.vasp
    if vasp is None:
        raise ValueError("VASP refinement requires a dft.vasp configuration")
    try:
        from pymatgen.io.vasp.inputs import Incar
    except ImportError as exc:  # pragma: no cover - hull ranking already requires pymatgen
        raise RuntimeError("VASP refinement requires pymatgen to parse INCAR settings") from exc

    text = vasp.incar_template.read_text(encoding="utf-8").format(natoms=1)
    incar = dict(Incar.from_str(text))
    incar.update(vasp.extra_incar)
    field_map = {
        "ENCUT": "encut_ev",
        "PREC": "prec",
        "EDIFF": "ediff",
        "ISYM": "isym",
        "ISMEAR": "ismear",
        "SIGMA": "sigma",
        "KSPACING": "kspacing",
        "KGAMMA": "kgamma",
        "ISPIN": "ispin",
        "LREAL": "lreal",
        "LWAVE": "lwave",
        "LCHARG": "lcharg",
        "ALGO": "algo",
        "NELM": "nelm",
        "NELMIN": "nelmin",
    }
    settings = {field_map[key]: value for key, value in incar.items() if key in field_map}
    relaxation_keys = {"IBRION", "ISIF", "NSW", "EDIFFG"}
    settings["extra_incar"] = {
        key: value
        for key, value in incar.items()
        if key not in field_map and key not in relaxation_keys
    }
    if vasp.kpoints_template is not None:
        settings["kpoints_template"] = str(vasp.kpoints_template)
        settings.pop("kspacing", None)
    return settings


def _merge_dft_settings(
    settings: dict[str, Any], overrides: dict[str, Any] | None
) -> dict[str, Any]:
    merged = dict(settings)
    for key, value in (overrides or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged


def _dft_relaxation_config(
    structure_path: str,
    output_root: Path,
    al_cfg: ALConfig,
    refinement: CampaignDFTRefinementConfig,
    settings_overrides: dict[str, Any] | None = None,
) -> ScientificRelaxationConfig:
    launch = al_cfg.resources.place().pool("dft").launches[0]
    if al_cfg.dft.backend == "vasp":
        vasp = al_cfg.dft.vasp
        if vasp is None:
            raise ValueError("campaign VASP refinement requires a dft.vasp block")
        settings = _merge_dft_settings(_vasp_template_settings(al_cfg), settings_overrides)
        return ScientificRelaxationConfig(
            mode=RelaxationMode.DFT,
            structure_path=structure_path,
            output_root=str(output_root),
            geometry=GeometryControls(relax_cell=refinement.relax_cell),
            dft=DFTBackendConfig(
                backend="vasp",
                launcher=[
                    "bash",
                    str(vasp.vasp_wrapper),
                    ".",
                    str(vasp.vasp_bin),
                    "{nodes}",
                    "{ranks_per_node}",
                    "{threads_per_rank}",
                ],
                potcar_dir=str(vasp.potcar_dir),
                settings=settings,
                timeout_sec=vasp.timeout_sec,
                environment=vasp.runtime_environment(),
                launch=launch,
            ),
            dft_approved=True,
            max_steps=refinement.max_steps,
            force_tolerance_eV_per_A=refinement.force_tolerance_eV_per_A,
        )

    qe = al_cfg.dft.qe
    if al_cfg.dft.backend != "qe" or qe is None:
        raise ValueError(f"unsupported campaign DFT refinement backend: {al_cfg.dft.backend}")
    if qe.pw_template is not None:
        raise ValueError("campaign DFT refinement does not support qe.pw_template")
    settings = {
        "ecutwfc_ry": qe.ecutwfc_ry,
        "ecutrho_ry": qe.ecutrho_ry,
        "kpts": qe.kpts,
        "koffset": qe.koffset,
        "occupations": qe.occupations,
        "smearing": qe.smearing,
        "degauss_ry": qe.degauss_ry,
        "pseudopotentials": qe.pseudopotentials,
        "extra_control": qe.extra_control,
        "extra_system": qe.extra_system,
        "extra_electrons": qe.extra_electrons,
    }
    settings = _merge_dft_settings(settings, settings_overrides)
    return ScientificRelaxationConfig(
        mode=RelaxationMode.DFT,
        structure_path=structure_path,
        output_root=str(output_root),
        geometry=GeometryControls(relax_cell=refinement.relax_cell),
        dft=DFTBackendConfig(
            backend="qe",
            launcher=[
                "bash",
                str(qe.pw_wrapper),
                "{work_dir}",
                str(qe.pw_bin),
                "{input}",
                "{nodes}",
                "{ranks_per_node}",
                "{threads_per_rank}",
            ],
            pseudo_dir=str(qe.pseudo_dir),
            settings={key: value for key, value in settings.items() if value is not None},
            timeout_sec=qe.timeout_sec,
            launch=launch,
        ),
        dft_approved=True,
        max_steps=refinement.max_steps,
        force_tolerance_eV_per_A=refinement.force_tolerance_eV_per_A,
    )


def _converged_dft_relaxation(
    source: RelaxationResult,
    result: ScientificRelaxationResult,
) -> RelaxationResult:
    if not result.stages:
        raise RuntimeError("DFT relaxation produced no stage result")
    stage = result.stages[-1]
    if not stage.converged or stage.energy_eV is None or stage.optimized_structure_path is None:
        raise RuntimeError(stage.failure_reason or "DFT relaxation did not converge")
    if stage.max_force_eV_per_A is None:
        raise RuntimeError("DFT relaxation did not report a residual force")
    return RelaxationResult(
        structure_path=source.structure_path,
        optimized_structure_path=stage.optimized_structure_path,
        trajectory_path=stage.artifacts.get("stdout", ""),
        log_csv_path=stage.artifacts.get("stdout", ""),
        final_energy_eV=stage.energy_eV,
        final_max_force_eV_per_A=stage.max_force_eV_per_A,
        num_steps=stage.steps,
        converged=True,
        notes=f"converged DFT relaxation; run_id={result.run_id}",
    )


def _rebase_compound_references(
    references: ReferenceEnergySet, element: str, new_energy: float
) -> None:
    """Recompute formation energies from raw DFT energies after a unary update."""
    updates: list[tuple[ReferencePhaseEntry, float]] = []
    for entry in references.phase_entries:
        composition = parse_composition(entry.formula)
        if composition is None:
            raise ValueError(f"Could not parse persisted reference formula {entry.formula!r}")
        if element not in composition.elements:
            continue
        if entry.energy_per_atom_eV is None:
            raise ValueError(
                f"cannot rebase persisted reference {entry.phase_id!r} without raw energy"
            )
        baseline = (
            sum(
                amount
                * (
                    new_energy
                    if symbol == element
                    else references.elemental_energies_eV_per_atom[symbol]
                )
                for symbol, amount in composition.elements.items()
            )
            / composition.total_atoms
        )
        updates.append(
            (
                entry,
                entry.energy_per_atom_eV
                - baseline
                + entry.corrections.get("energy_correction_eV_per_atom", 0.0),
            )
        )
    for entry, energy in updates:
        entry.formation_energy_eV_per_atom = energy


def _generate_reference_energies(
    al_cfg: ALConfig,
    refinement: CampaignDFTRefinementConfig,
    output_root: Path,
    relaxation_runner: Callable[[ScientificRelaxationConfig], ScientificRelaxationResult],
) -> tuple[ReferenceEnergySet, int, float]:
    references = refinement.reference_energies or ReferenceEnergySet(
        identifier=f"campaign-{refinement.method_signature}",
        method_signature=refinement.method_signature,
        backend=al_cfg.dft.backend,
        elemental_energies_eV_per_atom={},
        completeness_policy=refinement.reference_completeness_policy,
    )
    if references.backend is not None and references.backend != al_cfg.dft.backend:
        raise ValueError(
            f"reference energies use {references.backend}, but refinement uses {al_cfg.dft.backend}"
        )
    if references.backend is None and (
        references.elemental_energies_eV_per_atom or references.phase_entries
    ):
        raise ValueError(
            "populated reference energies lack backend provenance; regenerate or migrate them"
        )
    references.backend = al_cfg.dft.backend
    specs = refinement.reference_phases
    phase_ids = [spec.phase_id for spec in specs]
    if len(phase_ids) != len(set(phase_ids)):
        raise ValueError("reference structure phase IDs must be unique")
    existing_phase_ids = {entry.phase_id for entry in references.phase_entries}
    known_elemental_reference_ids = {
        entry.phase_id for entry in references.elemental_reference_candidates
    }
    known_elemental_reference_ids.update(
        entry.phase_id for entry in references.elemental_entries.values()
    )
    pending: list[tuple[ReferenceStructureSpec, Any]] = []
    for spec in specs:
        composition = parse_composition(spec.formula)
        if composition is None:
            raise ValueError(f"Could not parse reference formula {spec.formula!r}")
        if len(composition.elements) == 1:
            if spec.phase_id in known_elemental_reference_ids:
                continue
        elif spec.phase_id in existing_phase_ids:
            continue
        pending.append((spec, composition))
    pending.sort(key=lambda item: len(item[1].elements))

    calculations = 0
    elapsed_seconds = 0.0
    for spec, composition in pending:
        started = time.monotonic()
        result = relaxation_runner(
            _dft_relaxation_config(
                str(spec.structure_path),
                output_root / "references" / spec.phase_id,
                al_cfg,
                refinement.model_copy(
                    update={
                        "relax_cell": (
                            spec.relax_cell
                            if spec.relax_cell is not None
                            else refinement.relax_cell
                        )
                    }
                ),
                spec.settings,
            )
        )
        elapsed_seconds += time.monotonic() - started
        calculations += 1
        source = RelaxationResult(
            structure_path=str(spec.structure_path),
            optimized_structure_path=str(spec.structure_path),
            trajectory_path="",
            log_csv_path="",
            final_energy_eV=0.0,
            final_max_force_eV_per_A=0.0,
            num_steps=0,
            converged=True,
        )
        relaxed = _converged_dft_relaxation(source, result)
        from ase.io import read
        from pymatgen.core import Composition as PymatgenComposition

        atoms = read(relaxed.optimized_structure_path)
        cell_composition = PymatgenComposition(atoms.get_chemical_formula())
        declared_composition = PymatgenComposition(spec.formula)
        if cell_composition.reduced_composition != declared_composition.reduced_composition:
            raise ValueError(
                f"relaxed reference composition {cell_composition.formula} does not match "
                f"declared formula {spec.formula}"
            )
        energy_per_atom = relaxed.final_energy_eV / len(atoms)
        if len(composition.elements) == 1:
            element = next(iter(composition.elements))
            corrected_energy = energy_per_atom + spec.energy_correction_eV_per_atom
            entry = ElementalReferenceEntry(
                element=element,
                phase_id=spec.phase_id,
                reference_formula=spec.formula,
                energy_eV_per_atom=corrected_energy,
                method_signature=refinement.method_signature,
                backend=al_cfg.dft.backend,
                structure_path=relaxed.optimized_structure_path,
                structure_hash=structure_content_hash(relaxed.optimized_structure_path),
                total_energy_eV=relaxed.final_energy_eV,
                source=spec.source,
                provenance=spec.provenance,
                corrections={"energy_correction_eV_per_atom": spec.energy_correction_eV_per_atom},
            )
            current_energy = references.elemental_energies_eV_per_atom.get(element)
            if current_energy is None or corrected_energy < current_energy:
                _rebase_compound_references(references, element, corrected_energy)
                references.elemental_energies_eV_per_atom[element] = corrected_energy
                references.elemental_entries[element] = entry
            references.elemental_reference_candidates.append(entry)
            continue
        cell_amounts = cell_composition.get_el_amt_dict()
        missing = set(cell_amounts) - set(references.elemental_energies_eV_per_atom)
        if missing:
            raise ValueError(
                f"reference structures lack elemental references for {sorted(missing)}"
            )
        elemental_total = sum(
            amount * references.elemental_energies_eV_per_atom[element]
            for element, amount in cell_amounts.items()
        )
        formation_energy = (relaxed.final_energy_eV - elemental_total) / len(
            atoms
        ) + spec.energy_correction_eV_per_atom
        references.phase_entries.append(
            ReferencePhaseEntry(
                phase_id=spec.phase_id,
                formula=spec.formula,
                formation_energy_eV_per_atom=formation_energy,
                method_signature=refinement.method_signature,
                backend=al_cfg.dft.backend,
                structure_path=relaxed.optimized_structure_path,
                structure_hash=structure_content_hash(relaxed.optimized_structure_path),
                total_energy_eV=relaxed.final_energy_eV,
                energy_per_atom_eV=energy_per_atom,
                source=spec.source,
                provenance=spec.provenance,
                corrections={"energy_correction_eV_per_atom": spec.energy_correction_eV_per_atom},
            )
        )
    references = ReferenceEnergySet.model_validate(references.model_dump())
    refinement.reference_energies = references
    return references, calculations, elapsed_seconds


def _pending_reference_calculation_count(refinement: CampaignDFTRefinementConfig) -> int:
    references = refinement.reference_energies
    phase_ids = (
        {entry.phase_id for entry in references.phase_entries} if references is not None else set()
    )
    known_elemental_reference_ids = (
        {entry.phase_id for entry in references.elemental_reference_candidates}
        if references is not None
        else set()
    )
    if references is not None:
        known_elemental_reference_ids.update(
            entry.phase_id for entry in references.elemental_entries.values()
        )
    pending = 0
    for spec in refinement.reference_phases:
        composition = parse_composition(spec.formula)
        if composition is None:
            raise ValueError(f"Could not parse reference formula {spec.formula!r}")
        if len(composition.elements) == 1:
            if spec.phase_id not in known_elemental_reference_ids:
                pending += 1
        elif spec.phase_id not in phase_ids:
            pending += 1
    return pending
