"""Physical reference conversion contracts; synthetic labels, no DFT execution."""

import json

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write

from matsim_agents.active_learning.formation_training import (
    FORMATION_ENERGY_REFERENCE,
    prepare_formation_training_dataset,
)
from matsim_agents.discovery.energy_references import load_elemental_reference_manifest


def _dataset(tmp_path, elemental_manifest, dataset_method_sidecar, *, role="training_pool"):
    reference = elemental_manifest({"Nb": -10.0, "O": -5.0})
    frames = []
    for formula, energy in (("NbO2", -23.0), ("NbO2", -22.0), ("Nb2O4", -46.0)):
        atoms = Atoms(formula, positions=np.arange(len(Atoms(formula)) * 3).reshape(-1, 3))
        atoms.calc = SinglePointCalculator(
            atoms, energy=energy, forces=np.full((len(atoms), 3), 0.2), stress=np.arange(6) * 0.01
        )
        frames.append(atoms)
    dataset = tmp_path / "raw.extxyz"
    write(dataset, frames)
    sidecar = dataset_method_sidecar(dataset, reference)
    metadata = json.loads(sidecar.read_text())
    metadata["split_role"] = role
    sidecar.write_text(json.dumps(metadata))
    return dataset, reference


@pytest.mark.parametrize("role", ["training_pool", "validation"])
def test_physical_formation_targets_preserve_polymorph_differences_and_multiplicity(
    tmp_path, elemental_manifest, dataset_method_sidecar, role
):
    dataset, reference = _dataset(tmp_path, elemental_manifest, dataset_method_sidecar, role=role)
    original = dataset.read_bytes()
    output = tmp_path / "snapshot"
    target = prepare_formation_training_dataset(dataset, reference, output)
    raw = read(dataset, index=":")
    formed = read(target, index=":")
    assert [frame.get_potential_energy() for frame in formed] == pytest.approx([-3, -2, -6])
    assert formed[1].get_potential_energy() - formed[0].get_potential_energy() == pytest.approx(1)
    assert formed[2].get_potential_energy() / len(formed[2]) == pytest.approx(
        formed[0].get_potential_energy() / len(formed[0])
    )
    for source, frame in zip(raw, formed, strict=True):
        np.testing.assert_array_equal(source.get_forces(), frame.get_forces())
        np.testing.assert_array_equal(source.get_stress(), frame.get_stress())
        np.testing.assert_array_equal(source.positions, frame.positions)
        assert frame.info["dft_total_energy_eV"] == source.get_potential_energy()
        assert (
            frame.get_potential_energy() + frame.info["elemental_baseline_energy_eV"]
        ) == pytest.approx(source.get_potential_energy())
        assert frame.info["energy_convention"] == FORMATION_ENERGY_REFERENCE
    assert dataset.read_bytes() == original
    metadata = json.loads(target.with_suffix(".extxyz.manifest.json").read_text())
    assert metadata["energy_reference"] == FORMATION_ENERGY_REFERENCE
    assert metadata["split_role"] == role
    assert (
        metadata["parent_dataset_id"]
        == json.loads(dataset.with_suffix(".extxyz.manifest.json").read_text())["dataset_id"]
    )
    convention = json.loads((output / "energy-convention.json").read_text())
    assert convention["energy_normalization"] == "total_cell"
    assert convention["elemental_energies_eV_per_atom"] == {"Nb": -10, "O": -5}
    reference.unlink()
    for file in tmp_path.glob("elemental-*.extxyz"):
        file.unlink()
    _, _, snapshot = load_elemental_reference_manifest(
        output / "elemental-references.json", required_elements={"Nb", "O"}
    )
    assert len(snapshot) == 2


@pytest.mark.parametrize("defect", ["missing", "signature", "backend", "hash", "convention"])
def test_rejects_unverified_or_incompatible_raw_data(
    tmp_path, elemental_manifest, dataset_method_sidecar, defect
):
    dataset, reference = _dataset(tmp_path, elemental_manifest, dataset_method_sidecar)
    sidecar = dataset.with_suffix(".extxyz.manifest.json")
    if defect == "missing":
        sidecar.unlink()
    else:
        metadata = json.loads(sidecar.read_text())
        key, value = {
            "signature": ("method_signature", "other"),
            "backend": ("dft_backend", "vasp"),
            "hash": ("sha256", "other"),
            "convention": ("energy_reference", FORMATION_ENERGY_REFERENCE),
        }[defect]
        metadata[key] = value
        if defect == "backend":
            metadata["energy_reference"] = "vasp:native_total_energy"
        sidecar.write_text(json.dumps(metadata))
    with pytest.raises((ValueError, FileNotFoundError)):
        prepare_formation_training_dataset(dataset, reference, tmp_path / "snapshot")
    assert not (tmp_path / "snapshot").exists()


def test_rejects_double_conversion(tmp_path, elemental_manifest, dataset_method_sidecar):
    dataset, reference = _dataset(tmp_path, elemental_manifest, dataset_method_sidecar)
    formed = prepare_formation_training_dataset(dataset, reference, tmp_path / "first")
    with pytest.raises(ValueError, match="native DFT total"):
        prepare_formation_training_dataset(formed, reference, tmp_path / "second")


@pytest.mark.parametrize("defect", ["missing_element", "geometry_hash", "nonfinite_reference"])
def test_rejects_invalid_elemental_references(
    tmp_path, elemental_manifest, dataset_method_sidecar, defect
):
    dataset, reference = _dataset(tmp_path, elemental_manifest, dataset_method_sidecar)
    manifest = json.loads(reference.read_text())
    if defect == "missing_element":
        del manifest["references"]["O"]
    elif defect == "geometry_hash":
        manifest["references"]["Nb"]["structure_sha256"] = "invalid"
    else:
        manifest["references"]["Nb"]["energy_eV"] = float("nan")
    reference.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        prepare_formation_training_dataset(dataset, reference, tmp_path / "snapshot")
    assert not (tmp_path / "snapshot").exists()


@pytest.mark.parametrize("defect", ["energy", "forces", "missing_forces"])
def test_rejects_invalid_training_labels(
    tmp_path, elemental_manifest, dataset_method_sidecar, defect
):
    dataset, reference = _dataset(tmp_path, elemental_manifest, dataset_method_sidecar)
    frame = read(dataset)
    results = {"energy": -23.0, "forces": np.zeros((len(frame), 3))}
    if defect == "missing_forces":
        del results["forces"]
    elif defect == "energy":
        results["energy"] = float("inf")
    else:
        results["forces"][0, 0] = float("nan")
    frame.calc = SinglePointCalculator(frame, **results)
    write(dataset, frame)
    dataset_method_sidecar(dataset, reference)
    with pytest.raises((ValueError, RuntimeError)):
        prepare_formation_training_dataset(dataset, reference, tmp_path / "snapshot")
    assert not (tmp_path / "snapshot").exists()


def test_refuses_overwriting_snapshot(tmp_path, elemental_manifest, dataset_method_sidecar):
    dataset, reference = _dataset(tmp_path, elemental_manifest, dataset_method_sidecar)
    output = tmp_path / "snapshot"
    prepare_formation_training_dataset(dataset, reference, output)
    with pytest.raises(ValueError, match="already exists"):
        prepare_formation_training_dataset(dataset, reference, output)


def test_molecular_reference_normalizes_by_atom_count(
    tmp_path, elemental_manifest, dataset_method_sidecar
):
    reference = elemental_manifest({"O": -5.0})
    atoms = Atoms("O3", positions=[[0, 0, 0], [0, 0, 1], [0, 1, 0]])
    atoms.calc = SinglePointCalculator(atoms, energy=-13.0, forces=np.zeros((3, 3)))
    dataset = tmp_path / "raw.extxyz"
    write(dataset, atoms)
    dataset_method_sidecar(dataset, reference)
    formed = read(prepare_formation_training_dataset(dataset, reference, tmp_path / "snapshot"))
    assert formed.get_potential_energy() == pytest.approx(2.0)
    assert formed.info["elemental_baseline_energy_eV"] == pytest.approx(-15.0)
