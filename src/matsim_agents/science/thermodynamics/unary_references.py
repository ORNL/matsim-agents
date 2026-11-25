"""Model-specific relaxation and selection of unary hull references."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field


class UnaryReferencePhase(BaseModel):
    """Outcome for one unary polymorph evaluated by one MLIP."""

    element: str
    phase_id: str
    formula: str
    source: str
    provenance: dict[str, str] = Field(default_factory=dict)
    initial_structure_path: str
    optimized_structure_path: str | None = None
    converged: bool = False
    num_steps: int = 0
    max_force_eV_per_A: float | None = None
    total_energy_eV: float | None = None
    energy_per_atom_eV: float | None = None
    corrected_energy_per_atom_eV: float | None = None
    energy_above_endpoint_eV_per_atom: float | None = None
    energy_correction_eV_per_atom: float = 0.0
    pressure_GPa: float = 0.0
    magnetic_state: str = "model_default_or_implicit"
    failure: str | None = None
    duplicate_of: str | None = None
    selected_endpoint: bool = False


class UnaryReferenceSearchResult(BaseModel):
    """Auditable unary search for one MLIP and reference manifest."""

    model_identifier: str
    phases: list[UnaryReferencePhase] = Field(default_factory=list)
    selected_endpoints: dict[str, str] = Field(default_factory=dict)
    generated_counts: dict[str, int] = Field(default_factory=dict)
    converged_counts: dict[str, int] = Field(default_factory=dict)
    unique_converged_counts: dict[str, int] = Field(default_factory=dict)
    missing_elements: list[str] = Field(default_factory=list)
    provisional: bool = True


class CompoundReferencePhase(BaseModel):
    """One multielement hull reference relaxed by one MLIP."""

    phase_id: str
    formula: str
    source: str
    provenance: dict[str, str] = Field(default_factory=dict)
    initial_structure_path: str
    optimized_structure_path: str | None = None
    converged: bool = False
    num_steps: int = 0
    max_force_eV_per_A: float | None = None
    total_energy_eV: float | None = None
    energy_correction_eV_per_atom: float = 0.0
    magnetic_state: str = "model_default_or_implicit"
    failure: str | None = None


class CompoundReferenceSearchResult(BaseModel):
    """Auditable relaxation of the multielement hull references for one MLIP."""

    model_identifier: str
    phases: list[CompoundReferencePhase] = Field(default_factory=list)


def relax_structure(
    structure_path: str | Path,
    calculator: Any,
    *,
    output_path: Path,
    log_path: Path,
    max_steps: int,
    fmax_eV_per_A: float,
    maxstep_A: float,
    relax_cell: bool,
) -> tuple[bool, int, float, float]:
    """Relax one structure with FIRE; return (converged, steps, max force, energy)."""
    from ase.filters import ExpCellFilter
    from ase.io import read, write
    from ase.optimize import FIRE

    from matsim_agents.backends.mlip.relaxation import _NumericalStressCalculator

    atoms = read(str(structure_path))
    atoms.calc = calculator
    optimizable: Any = atoms
    if relax_cell and bool(np.all(atoms.pbc)):
        atoms.calc = _NumericalStressCalculator(calculator)
        optimizable = ExpCellFilter(atoms)
    optimizer = FIRE(optimizable, logfile=str(log_path), maxstep=maxstep_A)
    converged = bool(optimizer.run(fmax=fmax_eV_per_A, steps=max_steps))
    forces = np.asarray(optimizable.get_forces(), dtype=float)
    max_force = float(np.linalg.norm(forces, axis=1).max())
    energy = float(atoms.get_potential_energy())
    write(output_path, atoms)
    return converged, int(optimizer.nsteps), max_force, energy


def relax_compound_references(
    phases: list[dict[str, Any]],
    calculator: Any,
    *,
    model_identifier: str,
    output_dir: Path,
    max_steps: int = 200,
    fmax_eV_per_A: float = 0.02,
    maxstep_A: float = 0.01,
    relax_cell: bool = True,
) -> CompoundReferenceSearchResult:
    """Relax every multielement reference with the same controls as the unary search."""
    from pymatgen.core import Composition

    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "compound_reference_search.json"
    if result_path.is_file():
        return CompoundReferenceSearchResult.model_validate_json(
            result_path.read_text(encoding="utf-8")
        )
    records: list[CompoundReferencePhase] = []
    for spec in phases:
        if len(Composition(str(spec["formula"])).elements) < 2:
            continue
        phase_id = str(spec.get("phase_id", spec["formula"]))
        artifact_name = (
            f"{_safe_name(phase_id)}-{hashlib.sha256(phase_id.encode('utf-8')).hexdigest()}"
        )
        record = CompoundReferencePhase(
            phase_id=phase_id,
            formula=str(spec["formula"]),
            source=str(spec.get("source", "manifest")),
            provenance={
                str(key): str(value) for key, value in dict(spec.get("provenance", {})).items()
            },
            initial_structure_path=str(spec["path"]),
            energy_correction_eV_per_atom=float(spec.get("energy_correction_eV_per_atom", 0.0)),
            magnetic_state=str(spec.get("magnetic_state", "model_default_or_implicit")),
        )
        optimized_path = output_dir / f"{artifact_name}.extxyz"
        try:
            converged, steps, max_force, energy = relax_structure(
                spec["path"],
                calculator,
                output_path=optimized_path,
                log_path=output_dir / f"{artifact_name}.log",
                max_steps=max_steps,
                fmax_eV_per_A=fmax_eV_per_A,
                maxstep_A=maxstep_A,
                relax_cell=relax_cell and bool(spec.get("relax_cell", True)),
            )
            record.optimized_structure_path = str(optimized_path.resolve())
            record.converged = converged
            record.num_steps = steps
            record.max_force_eV_per_A = max_force
            record.total_energy_eV = energy
        except Exception as exc:  # noqa: BLE001
            record.failure = repr(exc)
        records.append(record)
    result = CompoundReferenceSearchResult(model_identifier=model_identifier, phases=records)
    result_path.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return result


def load_reference_phase_specs(reference_manifest: Path) -> list[dict[str, Any]]:
    """Load normalized phase specifications and resolve paths against the manifest."""
    raw = json.loads(reference_manifest.read_text(encoding="utf-8"))
    raw_phases = raw.get("phases", raw)
    specs: list[dict[str, Any]] = []
    for phase_id, raw_spec in raw_phases.items():
        spec = {"path": raw_spec} if isinstance(raw_spec, str) else dict(raw_spec)
        path = Path(str(spec["path"])).expanduser()
        if not path.is_absolute():
            path = (reference_manifest.parent / path).resolve()
        spec["path"] = str(path)
        spec.setdefault("phase_id", str(phase_id))
        spec.setdefault("formula", str(phase_id))
        specs.append(spec)
    return specs


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")
    return cleaned or "phase"


def unary_cache_directory(
    reference_manifest: Path, model_identifier: str, settings: dict[str, Any]
) -> Path:
    """Return a stable cache directory keyed by manifest, model, and controls."""
    structure_digests = {
        str(spec["phase_id"]): hashlib.sha256(Path(spec["path"]).read_bytes()).hexdigest()
        for spec in load_reference_phase_specs(reference_manifest)
    }
    payload = {
        "artifact_schema_version": 2,
        "manifest_sha256": hashlib.sha256(reference_manifest.read_bytes()).hexdigest(),
        "structure_sha256": structure_digests,
        "model_identifier": model_identifier,
        "settings": settings,
    }
    key = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:20]
    return reference_manifest.parent / "mlip-unary-relaxations" / key


def relax_unary_references(
    phases: list[dict[str, Any]],
    target_elements: set[str],
    calculator: Any,
    *,
    model_identifier: str,
    output_dir: Path,
    max_steps: int = 200,
    fmax_eV_per_A: float = 0.02,
    maxstep_A: float = 0.01,
    relax_cell: bool = True,
) -> UnaryReferenceSearchResult:
    """Relax all unary candidates and select the minimum endpoint per element."""
    from ase.io import read
    from pymatgen.core import Composition

    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "unary_reference_search.json"
    if result_path.is_file():
        return UnaryReferenceSearchResult.model_validate_json(
            result_path.read_text(encoding="utf-8")
        )
    records: list[UnaryReferencePhase] = []
    generated_counts = {element: 0 for element in sorted(target_elements)}

    for spec in phases:
        composition = Composition(str(spec["formula"]))
        elements = [str(element) for element in composition.elements]
        if len(elements) != 1 or elements[0] not in target_elements:
            continue
        element = elements[0]
        generated_counts[element] += 1
        phase_id = str(spec.get("phase_id", spec["formula"]))
        artifact_name = (
            f"{_safe_name(phase_id)}-{hashlib.sha256(phase_id.encode('utf-8')).hexdigest()}"
        )
        source = str(spec.get("source", "manifest"))
        provenance = {
            str(key): str(value) for key, value in dict(spec.get("provenance", {})).items()
        }
        correction = float(spec.get("energy_correction_eV_per_atom", 0.0))
        record = UnaryReferencePhase(
            element=element,
            phase_id=phase_id,
            formula=str(spec["formula"]),
            source=source,
            provenance=provenance,
            initial_structure_path=str(spec["path"]),
            energy_correction_eV_per_atom=correction,
            pressure_GPa=float(spec.get("pressure_GPa", 0.0)),
            magnetic_state=str(spec.get("magnetic_state", "model_default_or_implicit")),
        )
        try:
            optimized_path = output_dir / f"{artifact_name}.extxyz"
            converged, steps, max_force, energy = relax_structure(
                spec["path"],
                calculator,
                output_path=optimized_path,
                log_path=output_dir / f"{artifact_name}.log",
                max_steps=max_steps,
                fmax_eV_per_A=fmax_eV_per_A,
                maxstep_A=maxstep_A,
                relax_cell=relax_cell and bool(spec.get("relax_cell", True)),
            )
            n_atoms = len(read(optimized_path))
            record.optimized_structure_path = str(optimized_path.resolve())
            record.converged = converged
            record.num_steps = steps
            record.max_force_eV_per_A = max_force
            record.total_energy_eV = energy
            record.energy_per_atom_eV = energy / n_atoms
            record.corrected_energy_per_atom_eV = energy / n_atoms + correction
        except Exception as exc:  # noqa: BLE001
            record.failure = repr(exc)
        records.append(record)

    from ase.io import read
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.io.ase import AseAtomsAdaptor

    matcher = StructureMatcher(primitive_cell=True, attempt_supercell=True)
    for element in sorted(target_elements):
        representatives: list[tuple[str, Any]] = []
        for record in sorted(
            records,
            key=lambda item: (
                float(item.corrected_energy_per_atom_eV)
                if item.corrected_energy_per_atom_eV is not None
                else float("inf"),
                item.phase_id,
            ),
        ):
            if (
                record.element != element
                or not record.converged
                or record.optimized_structure_path is None
            ):
                continue
            structure = AseAtomsAdaptor.get_structure(read(record.optimized_structure_path))
            duplicate = next(
                (
                    phase_id
                    for phase_id, representative in representatives
                    if matcher.fit(representative, structure)
                ),
                None,
            )
            if duplicate is None:
                representatives.append((record.phase_id, structure))
            else:
                record.duplicate_of = duplicate

    selected_endpoints: dict[str, str] = {}
    converged_counts: dict[str, int] = {}
    unique_converged_counts: dict[str, int] = {}
    for element in sorted(target_elements):
        all_converged = [
            record
            for record in records
            if record.element == element
            and record.converged
            and record.corrected_energy_per_atom_eV is not None
        ]
        converged = [record for record in all_converged if record.duplicate_of is None]
        converged_counts[element] = len(all_converged)
        unique_converged_counts[element] = len(converged)
        if not converged:
            continue
        selected = min(
            converged,
            key=lambda record: (
                float(record.corrected_energy_per_atom_eV),
                record.phase_id,
            ),
        )
        selected.selected_endpoint = True
        selected_endpoints[element] = selected.phase_id
        endpoint_energy = float(selected.corrected_energy_per_atom_eV)
        for record in all_converged:
            record.energy_above_endpoint_eV_per_atom = max(
                0.0,
                float(record.corrected_energy_per_atom_eV) - endpoint_energy,
            )

    missing = sorted(target_elements - set(selected_endpoints))
    result = UnaryReferenceSearchResult(
        model_identifier=model_identifier,
        phases=records,
        selected_endpoints=selected_endpoints,
        generated_counts=generated_counts,
        converged_counts=converged_counts,
        unique_converged_counts=unique_converged_counts,
        missing_elements=missing,
        provisional=bool(missing),
    )
    result_path.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return result
