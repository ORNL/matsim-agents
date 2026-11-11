import json

import numpy as np
import pytest
from ase.build import molecule
from ase.io import read

from benchmarks.e4.aurora_dft import six_class_inputs as inputs


def test_rebuilt_phosphorene_is_threefold_puckered_with_preserved_vacuum():
    unit = inputs.phosphorene_unit_cell()
    assert np.allclose(unit.cell.lengths(), [4.3763, 3.3136, 20.0])
    geometry = inputs.validate_phosphorene(unit, 0)
    assert geometry["coordination_histogram"] == [0, 0, 0, 4]
    assert 2.20 < geometry["bond_length_range_A"][0] < geometry["bond_length_range_A"][1] < 2.26
    for vacancies, undercoordinated in ((1, 3), (2, 4)):
        atoms, variation = inputs.phosphorene_variant(3, 4, vacancies)
        assert len(atoms) == 48 - vacancies
        assert inputs.validate_phosphorene(atoms, vacancies)["coordination_histogram"][2] == (
            undercoordinated
        )
        assert variation["vacancies"] == vacancies


def test_zn_formate_validator_rejects_incomplete_chemistry():
    with pytest.raises(ValueError, match="unexpected species|not Zn\\(HCOO\\)2"):
        inputs.validate_zn_formate(molecule("HCOOH"))


def test_start_preflight_mirrors_wrapper_requirements():
    good = "PREC = Accurate\nENCUT = 520\nKPAR = 12\n"
    inputs.check_start(good, 12, {"ENCUT": "520"})
    with pytest.raises(ValueError, match="KPAR"):
        inputs.check_start("PREC = Accurate\nENCUT = 520\n", 12, {})
    with pytest.raises(ValueError, match="KPAR"):
        inputs.check_start("PREC = Accurate\nKPAR = 8\n", 12, {})
    with pytest.raises(ValueError, match="PREC"):
        inputs.check_start("PREC = Normal\nKPAR = 12\n", 12, {})
    with pytest.raises(ValueError, match="ENCUT"):
        inputs.check_start(good, 12, {"ENCUT": "600"})


def test_committed_six_class_inputs_are_unique_and_provenanced():
    manifest = json.loads(inputs.MANIFEST.read_text())
    assert manifest["execution_authorized"] is False
    assert manifest["scheduler_submissions"] == 0
    counts = {name: len(records) for name, records in manifest["classes"].items()}
    assert counts == {
        "nb_ta_o": 14,
        "zn_formate_mof": 16,
        "phosphorene": 16,
        "fcc_hea": 12,
        "lifepo4": 8,
    }
    for name, records in manifest["classes"].items():
        assert {record["protocol_id"] for record in records} == {inputs.PROTOCOLS[name]}
        hashes = [record["source_poscar_sha256"] for record in records]
        assert len(hashes) == len(set(hashes))
        for record in records:
            source = inputs.REPO / record["source_poscar"]
            assert inputs.sha256(source) == record["source_poscar_sha256"]
            species = inputs.poscar_species(source.read_text())
            assert dict(species) == record["composition"]
            assert [c["element"] for c in record["paw_components"]] == [s for s, _ in species]
            assert not record["case_dir"].startswith("research/")
    mof = manifest["classes"]["zn_formate_mof"]
    assert {r["potcar_sha256"] for r in mof} == {inputs.MOF_E2_POTCAR_SHA256}
    assert all(r["chemistry_validation"]["periodic_bond_graph_components"] == 1 for r in mof)
    nbtao = manifest["classes"]["nb_ta_o"]
    assert all(r["e4_adapter"]["added_tags"] == {"KPAR": "12"} for r in nbtao)


def test_production_inputs_are_unique_valid_and_frozen():
    manifest = json.loads(inputs.FULL_MANIFEST.read_text())
    assert manifest["execution_authorized"] is False
    assert manifest["scheduler_submissions"] == 0
    assert set(manifest["classes"]) == set(inputs.FULL_CLASSES)
    for name, records in manifest["classes"].items():
        assert len(records) == inputs.UNIQUE_INPUTS
        assert len({r["calculation_id"] for r in records}) == inputs.UNIQUE_INPUTS
        assert len({r["source_poscar_sha256"] for r in records}) == inputs.UNIQUE_INPUTS
        assert {r["protocol_id"] for r in records} == {inputs.FULL_CLASSES[name]["protocol_id"]}
        assert {r["walltime_seconds"] for r in records} == {
            inputs.FULL_CLASSES[name]["task_cap_hours"] * 3600
        }
        for record in records:
            source = inputs.REPO / record["source_poscar"]
            assert inputs.sha256(source) == record["source_poscar_sha256"]
            assert record["input_hashes"]["POSCAR"] == record["source_poscar_sha256"]
    mof = manifest["classes"]["zn_formate_mof"]
    assert {r["potcar_sha256"] for r in mof} == {inputs.MOF_E2_POTCAR_SHA256}
    for record in mof[:4]:
        atoms = read(inputs.REPO / record["source_poscar"], format="vasp")
        assert inputs.validate_zn_formate(atoms)["periodic_bond_graph_components"] == 1
    phos = manifest["classes"]["phosphorene"]
    vacancies = {"pristine": 0, "monovacancy": 1}
    for record in phos[:8]:
        atoms = read(inputs.REPO / record["source_poscar"], format="vasp")
        atoms.pbc = [True, True, False]
        defect = record["variation"]["defect"]
        inputs.validate_strained_phosphorene(atoms, vacancies.get(defect, 2))
    nbtao = manifest["classes"]["nb_ta_o"]
    assert all(r["composition"] == {"Nb": 8, "Ta": 8, "O": 40} for r in nbtao)
    assert len({r["variation"]["ordering"] for r in nbtao}) == inputs.UNIQUE_INPUTS
    bcc = manifest["classes"]["bcc_refractory"]
    assert [r["source_calculation_id"] for r in bcc[:8]] == list(inputs.BCC_AUTHORIZED_STRONG)
