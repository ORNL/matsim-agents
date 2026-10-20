"""Put DFT energy labels on a foundation MLIP's energy scale before fine-tuning.

A foundation model's energies follow its pretraining DFT setup (UMA OMat and
MACE-MP: VASP PBE). Quantum ESPRESSO, or VASP with other settings, gives total
energies that differ from it by large per-element constants. Fine-tuning on raw
totals would force the model to absorb those constants. Instead each label is
shifted by the elemental reference offsets:

    E_label = E_DFT - sum over atoms i of (e_DFT[el_i] - e_model[el_i])

where ``e_DFT`` is the DFT energy per atom of the declared elemental reference
and ``e_model`` the model's energy per atom on the same geometry. The DFT
formation energy relative to the references is unchanged, and the fine-tuned
model stays on its foundation's scale. A model fine-tuned here records its
alignment in ``energy-alignment.json``; fine-tuning it again reuses the recorded
model references, so successive fine-tunes keep one scale.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Literal

from ase import Atoms
from pydantic import BaseModel, Field

from matsim_agents.datasets.energy_references import load_elemental_reference_manifest

ENERGY_ALIGNMENT_FILE = "energy-alignment.json"
ALIGNMENT_CONVENTION = "dft_formation_on_model_elemental_scale"
# Directories searched, from a model file's own upwards, for the alignment of the
# fine-tune that wrote it; fairchem nests checkpoints as
# <output>/<run>/checkpoints/final/inference_ckpt.pt.
_SEARCH_DEPTH = 4


class EnergyAlignment(BaseModel):
    """Per-element offsets from DFT labels to a model's energy scale."""

    schema_version: Literal[1] = 1
    convention: Literal["dft_formation_on_model_elemental_scale"] = ALIGNMENT_CONVENTION
    manifest_sha256: str
    backend: str
    method_signature: str
    dft_eV_per_atom: dict[str, float] = Field(min_length=1)
    model_eV_per_atom: dict[str, float] = Field(min_length=1)
    model_references_from: str = Field(
        description="The model whose elemental energies define the scale, or the alignment "
        "file they were inherited from."
    )
    source_dataset_sha256: str | None = None

    def offset_eV(self, symbols: Iterable[str]) -> float:
        total = 0.0
        for symbol in symbols:
            if symbol not in self.dft_eV_per_atom or symbol not in self.model_eV_per_atom:
                raise ValueError(f"energy alignment has no reference for {symbol}")
            total += self.dft_eV_per_atom[symbol] - self.model_eV_per_atom[symbol]
        return total

    def aligned_energy(self, atoms: Atoms, energy_eV: float) -> float:
        """DFT total energy of ``atoms`` on the model's energy scale."""
        return float(energy_eV) - self.offset_eV(atoms.get_chemical_symbols())

    def write(self, directory: str | Path) -> Path:
        path = Path(directory) / ENERGY_ALIGNMENT_FILE
        path.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return path


def recorded_alignment(model: str | Path) -> tuple[EnergyAlignment, Path] | None:
    """The alignment written with a model fine-tuned here, if ``model`` is one."""

    path = Path(str(model))
    if not path.exists():
        return None
    directory = path if path.is_dir() else path.parent
    for candidate in [directory, *directory.parents][:_SEARCH_DEPTH]:
        record = candidate / ENERGY_ALIGNMENT_FILE
        if record.is_file():
            return EnergyAlignment.model_validate_json(record.read_text()), record
    return None


def energy_alignment(
    manifest_path: str | Path,
    elements: set[str],
    model_energy: Callable[[Atoms], float],
    *,
    base_model: str | Path,
) -> EnergyAlignment:
    """Offsets for ``elements`` from the DFT references to ``base_model``'s scale.

    ``model_energy`` returns the base model's total energy of a structure; it is
    called only for elements whose model reference the base model does not
    already record (see the module docstring).
    """

    manifest, payload, prepared = load_elemental_reference_manifest(
        manifest_path, required_elements=elements
    )
    inherited = recorded_alignment(base_model)
    dft: dict[str, float] = {}
    model: dict[str, float] = {}
    for element, atoms, dft_energy, _digest in prepared:
        if element not in elements:
            continue
        dft[element] = dft_energy / len(atoms)
        if inherited is not None and element in inherited[0].model_eV_per_atom:
            model[element] = inherited[0].model_eV_per_atom[element]
        else:
            probe = atoms.copy()
            model[element] = float(model_energy(probe)) / len(atoms)
    computed = sorted(set(model) - set(inherited[0].model_eV_per_atom if inherited else ()))
    if inherited is None:
        source = str(base_model)
    elif computed:
        source = f"{inherited[1]} (computed with {base_model} for {', '.join(computed)})"
    else:
        source = str(inherited[1])
    return EnergyAlignment(
        manifest_sha256=hashlib.sha256(payload).hexdigest(),
        backend=manifest["backend"],
        method_signature=manifest["method_signature"],
        dft_eV_per_atom=dft,
        model_eV_per_atom=model,
        model_references_from=source,
    )


__all__ = [
    "ALIGNMENT_CONVENTION",
    "ENERGY_ALIGNMENT_FILE",
    "EnergyAlignment",
    "energy_alignment",
    "recorded_alignment",
]
