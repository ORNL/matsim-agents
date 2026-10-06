"""Prepare physical formation-energy labels without modifying raw DFT datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write

from matsim_agents.active_learning.dataset_governance import (
    DatasetManifest,
    DatasetValidationSummary,
    sha256_file,
    write_dataset_manifest,
)
from matsim_agents.discovery.energy_references import (
    load_elemental_reference_manifest,
    validate_dataset_reference_method,
)

FORMATION_ENERGY_REFERENCE = "dft:declared_elemental_formation_energy"


def prepare_formation_training_dataset(
    dataset_path: str | Path,
    elemental_reference_manifest: str | Path,
    output_dir: str | Path,
) -> Path:
    """Create a new immutable dataset snapshot with total-cell formation labels.

    Requires verified native-total-energy labels and a compatible elemental
    manifest. This function never launches DFT or fits elemental offsets.
    """
    source = Path(dataset_path).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise ValueError(f"Formation training snapshot already exists: {output}")
    if source == output or output in source.parents:
        raise ValueError("Formation training output must not contain the source dataset")
    metadata_path = source.with_suffix(source.suffix + ".manifest.json")
    metadata = DatasetManifest.model_validate_json(metadata_path.read_text())
    if metadata.energy_reference != f"{metadata.dft_backend}:native_total_energy":
        raise ValueError("Formation conversion requires native DFT total-energy labels")
    source_bytes = source.read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != metadata.sha256:
        raise ValueError("DFT dataset manifest hash does not match the training data")
    frames = list(read(source, index=":"))
    if not frames:
        raise ValueError("Formation training requires a nonempty dataset")
    required_elements = {symbol for frame in frames for symbol in frame.get_chemical_symbols()}
    manifest, reference_bytes, references = load_elemental_reference_manifest(
        elemental_reference_manifest, required_elements=required_elements
    )
    validate_dataset_reference_method(
        source,
        reference_backend=manifest["backend"],
        reference_method_signature=manifest["method_signature"],
        require_sidecar=True,
    )
    energies = {element: energy / len(atoms) for element, atoms, energy, _ in references}
    reference_id = hashlib.sha256(reference_bytes).hexdigest()
    converted = []
    for index, atoms in enumerate(frames):
        if not len(atoms):
            raise ValueError(f"Training frame {index} is empty")
        if atoms.info.get("dft_backend", metadata.dft_backend) != metadata.dft_backend:
            raise ValueError(f"Training frame {index} has incompatible DFT backend")
        total_energy = (
            float(atoms.info["energy"])
            if "energy" in atoms.info
            else float(atoms.get_potential_energy())
        )
        forces = (
            np.asarray(atoms.arrays["forces"], dtype=float)
            if "forces" in atoms.arrays
            else np.asarray(atoms.get_forces(), dtype=float)
        )
        if not np.isfinite(total_energy):
            raise ValueError(f"Training frame {index} has nonfinite DFT energy")
        if forces.shape != (len(atoms), 3) or not np.isfinite(forces).all():
            raise ValueError(f"Training frame {index} has invalid DFT forces")
        baseline = float(sum(energies[symbol] for symbol in atoms.get_chemical_symbols()))
        formation = total_energy - baseline
        if not np.isfinite(formation):
            raise ValueError(f"Training frame {index} has nonfinite formation energy")
        frame = atoms.copy()
        # Remove alternative total-energy labels so consumers cannot select them.
        frame.info.pop("energy", None)
        frame.info.pop("REF_energy", None)
        frame.arrays.pop("forces", None)
        frame.arrays.pop("REF_forces", None)
        frame.info.update(
            dft_total_energy_eV=total_energy,
            elemental_baseline_energy_eV=baseline,
            energy_convention=FORMATION_ENERGY_REFERENCE,
            elemental_reference_id=reference_id,
        )
        results = {"energy": formation, "forces": forces.copy()}
        if atoms.calc is not None and "stress" in atoms.calc.results:
            stress = np.asarray(atoms.calc.results["stress"], dtype=float)
            if not np.isfinite(stress).all():
                raise ValueError(f"Training frame {index} has nonfinite DFT stress")
            results["stress"] = stress.copy()
        frame.calc = SinglePointCalculator(frame, **results)
        converted.append(frame)

    if sha256_file(source) != metadata.sha256:
        raise ValueError("Training data changed during formation-energy preparation")
    output.mkdir(parents=True)
    target = output / "formation.extxyz"
    write(target, converted, format="extxyz")
    write_dataset_manifest(
        target,
        dft_backend=metadata.dft_backend,
        method_signature=metadata.method_signature,
        energy_reference=FORMATION_ENERGY_REFERENCE,
        validation=DatasetValidationSummary(accepted=len(converted)),
        parent_dataset_id=metadata.dataset_id,
        split_role=metadata.split_role,
    )
    provenance = {
        "schema_version": 1,
        "energy_convention": FORMATION_ENERGY_REFERENCE,
        "energy_units": "eV",
        "energy_normalization": "total_cell",
        "forces_units": "eV/angstrom",
        "dft_backend": manifest["backend"],
        "method_signature": manifest["method_signature"],
        "source_dataset_sha256": metadata.sha256,
        "formation_dataset_sha256": sha256_file(target),
        "elemental_reference_id": reference_id,
        "elemental_energies_eV_per_atom": energies,
        "source_elemental_manifest_sha256": reference_id,
    }
    (output / "energy-convention.json").write_text(
        json.dumps(provenance, indent=2, allow_nan=False) + "\n"
    )
    reference_dir = output / "references"
    reference_dir.mkdir()
    snapshot_references = {}
    for element, atoms, energy, _ in references:
        geometry = reference_dir / f"{element}.extxyz"
        write(geometry, atoms, format="extxyz")
        snapshot_references[element] = {
            "structure_path": str(geometry.relative_to(output)),
            "structure_sha256": sha256_file(geometry),
            "energy_eV": energy,
        }
    (output / "elemental-references.json").write_text(
        json.dumps(
            {**json.loads(reference_bytes), **manifest, "references": snapshot_references},
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--elemental-reference-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    target = prepare_formation_training_dataset(
        args.dataset, args.elemental_reference_manifest, args.output_dir
    )
    print(target)


if __name__ == "__main__":
    main()
