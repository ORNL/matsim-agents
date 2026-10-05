"""Fixed-geometry, DFT-labelled elemental references for energy comparisons."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.io import read

log = logging.getLogger(__name__)


class ElementalReferenceSpec(TypedDict):
    structure_path: str
    structure_sha256: str
    energy_eV: float


class ElementalReferenceManifest(TypedDict):
    backend: str
    method_signature: str
    references: dict[str, ElementalReferenceSpec]


def validate_dataset_reference_method(
    dataset_path: str | Path,
    *,
    reference_backend: str,
    reference_method_signature: str,
) -> None:
    """Check the recorded compound DFT protocol when a dataset sidecar exists."""
    path = Path(dataset_path)
    sidecar = path.with_suffix(path.suffix + ".manifest.json")
    if not sidecar.is_file():
        log.warning(
            "No DFT method sidecar for %s; the caller must verify elemental/compound "
            "DFT method compatibility",
            path,
        )
        return
    metadata = json.loads(sidecar.read_text())
    if not isinstance(metadata, dict):
        raise ValueError("DFT dataset manifest must be a JSON object")
    if metadata.get("sha256") != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError("DFT dataset manifest hash does not match the compound data")
    if (
        metadata.get("dft_backend") != reference_backend
        or metadata.get("method_signature") != reference_method_signature
    ):
        raise ValueError("elemental references and compound labels use different DFT methods")


@dataclass
class ElementalEnergyReferences:
    dft_eV_per_atom: dict[str, float]
    model_eV_per_atom: dict[str, float]
    provenance: dict[str, object]

    def formation_energy(self, atoms: Atoms, energy_eV: float, *, model: bool) -> float:
        if not len(atoms) or not np.isfinite(energy_eV):
            raise ValueError("formation energy requires a nonempty structure and finite energy")
        references = self.model_eV_per_atom if model else self.dft_eV_per_atom
        missing = set(atoms.get_chemical_symbols()) - references.keys()
        if missing:
            raise ValueError(f"elemental references lack coverage for {sorted(missing)}")
        baseline = sum(references[symbol] for symbol in atoms.get_chemical_symbols())
        formation = (energy_eV - baseline) / len(atoms)
        if not np.isfinite(formation):
            raise ValueError("formation energy is not finite")
        return float(formation)


def load_elemental_reference_manifest(
    manifest_path: str | Path,
    *,
    required_elements: set[str],
) -> tuple[ElementalReferenceManifest, bytes, list[tuple[str, Atoms, float, str]]]:
    """Validate DFT labels and fixed pure-element geometries without inference."""
    path = Path(manifest_path).resolve()
    payload = path.read_bytes()
    manifest = json.loads(payload)
    if not isinstance(manifest, dict):
        raise ValueError("elemental reference manifest must be a JSON object")
    signature = manifest.get("method_signature")
    backend = manifest.get("backend")
    if (
        not isinstance(backend, str)
        or backend not in {"qe", "vasp"}
        or not isinstance(signature, str)
        or not signature.strip()
    ):
        raise ValueError("elemental reference manifest requires a DFT backend and method_signature")
    specs = manifest.get("references")
    if not isinstance(specs, dict) or not specs:
        raise ValueError("elemental reference manifest requires a nonempty references object")
    missing = required_elements - specs.keys()
    if missing:
        raise ValueError(f"elemental reference manifest lacks coverage for {sorted(missing)}")

    prepared = []
    validated_specs: dict[str, ElementalReferenceSpec] = {}
    for element, spec in specs.items():
        if not isinstance(spec, dict) or not isinstance(spec.get("structure_path"), str):
            raise ValueError(f"reference {element!r} requires a structure_path string")
        if not spec["structure_path"].strip():
            raise ValueError(f"reference {element!r} requires a nonempty structure_path")
        energy = spec.get("energy_eV")
        if isinstance(energy, bool) or not isinstance(energy, (int, float)):
            raise ValueError(f"DFT reference energy for {element} must be numeric")
        structure_path = Path(spec["structure_path"])
        if not structure_path.is_absolute():
            structure_path = path.parent / structure_path
        frames = read(structure_path, index=":")
        if len(frames) != 1:
            raise ValueError(f"reference {element!r} must declare exactly one geometry")
        atoms = frames[0]
        if not len(atoms) or set(atoms.get_chemical_symbols()) != {element}:
            raise ValueError(f"reference {element!r} must contain only that pure element")
        dft_energy = float(energy)
        if not np.isfinite(dft_energy):
            raise ValueError(f"DFT reference energy for {element} must be finite")
        digest = hashlib.sha256(structure_path.read_bytes()).hexdigest()
        if spec.get("structure_sha256") != digest:
            raise ValueError(f"DFT reference structure hash mismatch for {element}")
        validated_specs[element] = {
            "structure_path": spec["structure_path"],
            "energy_eV": dft_energy,
            "structure_sha256": digest,
        }
        prepared.append((element, atoms, dft_energy, digest))
    return (
        {
            "backend": backend,
            "method_signature": signature,
            "references": validated_specs,
        },
        payload,
        prepared,
    )


def predict_elemental_references(
    manifest_path: str | Path,
    calculator: Calculator,
    *,
    required_elements: set[str],
) -> ElementalEnergyReferences:
    """Evaluate each declared pure reference before any compound predictions."""
    manifest, payload, prepared = load_elemental_reference_manifest(
        manifest_path, required_elements=required_elements
    )

    dft_references = {}
    model_references = {}
    records = {}
    for element, atoms, dft_energy, digest in prepared:
        probe = atoms.copy()
        probe.calc = calculator
        model_energy = float(probe.get_potential_energy())
        if not np.isfinite(model_energy):
            raise ValueError(f"model reference energy for {element} must be finite")
        dft_references[element] = dft_energy / len(atoms)
        model_references[element] = model_energy / len(atoms)
        records[element] = {
            "structure_sha256": digest,
            "n_atoms": len(atoms),
            "dft_energy_eV": dft_energy,
            "model_energy_eV": model_energy,
            "baseline_error_eV_per_atom": (model_energy - dft_energy) / len(atoms),
        }
    return ElementalEnergyReferences(
        dft_references,
        model_references,
        {
            "manifest_sha256": hashlib.sha256(payload).hexdigest(),
            "backend": manifest["backend"],
            "method_signature": manifest["method_signature"],
            "references": records,
        },
    )
