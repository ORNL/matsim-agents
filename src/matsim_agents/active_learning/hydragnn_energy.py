"""Restore checkpoint-declared elemental baselines at the ASE calculator boundary."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from pydantic import BaseModel, Field

from matsim_agents.active_learning.dataset_governance import sha256_file
from matsim_agents.active_learning.formation_training import FORMATION_ENERGY_REFERENCE
from matsim_agents.discovery.energy_references import load_elemental_reference_manifest


class HydraGNNEnergyConvention(BaseModel):
    schema_version: int
    energy_convention: str
    energy_units: str
    energy_normalization: str
    checkpoint: str
    checkpoint_sha256: str
    elemental_energies_eV_per_atom: dict[str, float] = Field(min_length=1)


def restore_hydragnn_total_energy(calculator, logdir: Path, checkpoint: str | None = None):
    """Leave unmarked foundation models unchanged; verify and restore marked ones."""
    path = logdir / "energy-convention.json"
    if not path.exists():
        for metadata_name in ("routing.json", "newhead.json"):
            metadata_path = logdir / metadata_name
            if metadata_path.exists():
                metadata = json.loads(metadata_path.read_text())
                if "reference_energies" in metadata or "energy_convention" in metadata:
                    raise ValueError(
                        "HydraGNN fine-tuned checkpoint lacks verified energy convention; "
                        "retrain with declared elemental references"
                    )
        return calculator
    convention = HydraGNNEnergyConvention.model_validate_json(path.read_text())
    if (
        convention.schema_version != 1
        or convention.energy_convention != FORMATION_ENERGY_REFERENCE
        or convention.energy_units != "eV"
        or convention.energy_normalization != "total_cell"
    ):
        raise ValueError("Unsupported HydraGNN checkpoint energy convention")
    checkpoint_path = Path(checkpoint) if checkpoint is not None else Path(convention.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = logdir / checkpoint_path
    if (
        checkpoint_path.name != convention.checkpoint
        or sha256_file(checkpoint_path) != convention.checkpoint_sha256
    ):
        raise ValueError("HydraGNN energy metadata does not match selected checkpoint")
    energies = convention.elemental_energies_eV_per_atom
    if not all(np.isfinite(value) for value in energies.values()):
        raise ValueError("HydraGNN elemental baseline is not finite")
    snapshot_dir = logdir / "training-reference"
    snapshot_metadata = json.loads((snapshot_dir / "energy-convention.json").read_text())
    metadata = json.loads(path.read_text())
    for name, digest in metadata["inference_artifact_sha256"].items():
        if Path(name).name != name or sha256_file(logdir / name) != digest:
            raise ValueError(f"HydraGNN inference artifact hash mismatch: {name}")
    if any(metadata.get(key) != value for key, value in snapshot_metadata.items()):
        raise ValueError("HydraGNN checkpoint convention differs from training provenance")
    _, _, references = load_elemental_reference_manifest(
        snapshot_dir / "elemental-references.json", required_elements=set(energies)
    )
    if energies != {element: energy / len(atoms) for element, atoms, energy, _ in references}:
        raise ValueError("HydraGNN checkpoint baseline differs from training references")
    if sha256_file(snapshot_dir / "formation.extxyz") != metadata["formation_dataset_sha256"]:
        raise ValueError("HydraGNN formation training snapshot hash mismatch")
    if getattr(calculator, "energy_convention", None) == "dft:native_total_energy":
        raise ValueError("HydraGNN elemental baseline has already been restored")

    class TotalEnergyHydraGNNCalculator(Calculator):
        implemented_properties = calculator.implemented_properties
        energy_convention = "dft:native_total_energy"

        def __getattr__(self, name):
            return getattr(calculator, name)

        def reset(self):
            super().reset()
            calculator.reset()

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            symbols = atoms.get_chemical_symbols()
            missing = set(symbols) - energies.keys()
            if missing:
                raise ValueError(
                    f"HydraGNN training references lack coverage for {sorted(missing)}"
                )
            baseline = sum(energies[symbol] for symbol in symbols)
            calculator.calculate(atoms, properties, system_changes)
            self.results = deepcopy(calculator.results)
            for prop in ("energy", "free_energy"):
                if prop not in self.results:
                    continue
                value = self.results[prop]
                value = float(value) + baseline
                if not np.isfinite(value):
                    raise ValueError("HydraGNN reconstructed total energy is not finite")
                self.results[prop] = value

    return TotalEnergyHydraGNNCalculator()
