"""Synthetic elemental DFT orchestration contracts; no external calculations."""

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write
from pydantic import ValidationError

from matsim_agents.active_learning import elemental_dft
from matsim_agents.active_learning.config import DFTConfig, TrainerConfig
from matsim_agents.active_learning.dft_backend import DFTResult
from matsim_agents.active_learning.dft_protocol import dft_method_signature
from matsim_agents.active_learning.elemental_dft import (
    ElementalPhaseSpec,
    ElementalReferencePlan,
    prepare_elemental_references,
)
from matsim_agents.active_learning.formation_training import prepare_formation_training_dataset
from matsim_agents.active_learning.hydragnn_references import resolve_training_references


@pytest.fixture
def setup_references(tmp_path, monkeypatch):
    pseudo = tmp_path / "pseudo"
    pseudo.mkdir()
    for element in ("Si", "O"):
        (pseudo / f"{element}.upf").write_text(f"synthetic-{element}")
    binary = tmp_path / "pw.x"
    binary.write_text("synthetic executable")
    dft = DFTConfig.model_validate(
        {
            "backend": "qe",
            "qe": {
                "pw_bin": binary,
                "pw_wrapper": tmp_path / "wrapper",
                "pseudo_dir": pseudo,
                "pseudopotentials": {"Si": "Si.upf", "O": "O.upf"},
                "ecutwfc_ry": 50,
                "ecutrho_ry": 400,
                "kpts": [2, 2, 2],
                "occupations": "fixed",
            },
        }
    )
    phases = []
    for name, formula, kind in (
        ("si-a", "Si2", "bulk"),
        ("si-b", "Si4", "bulk"),
        ("o2-triplet", "O2", "molecule"),
    ):
        path = tmp_path / f"{name}.extxyz"
        write(
            path,
            Atoms(
                formula, positions=np.zeros((len(Atoms(formula)), 3)), cell=[10, 10, 10], pbc=True
            ),
        )
        phases.append(
            ElementalPhaseSpec(
                phase_id=name,
                element="Si" if name.startswith("si") else "O",
                structure_path=path,
                kind=kind,
                magnetic_state="nonmagnetic" if kind == "bulk" else "triplet",
                qe_magnetic_settings={} if kind == "bulk" else {"nspin": 2, "tot_magnetization": 2},
                kpts=None if kind == "bulk" else (1, 1, 1),
            )
        )
    calls = []
    results = []

    def factory(cfg):
        def run(spec):
            calls.append((cfg, spec))
            energy = {"si-a": -8.0, "si-b": -20.0, "o2-triplet": -12.0}[spec.job_id]
            result = DFTResult(
                backend=cfg.backend,
                work_dir=spec.work_dir,
                return_code=0,
                converged=True,
                energy_eV=energy,
                forces_eV_per_A=np.zeros((len(spec.atoms), 3)),
                stress_eV_per_A3=None,
                n_atoms=len(spec.atoms),
                wall_time_sec=1,
                final_atoms=spec.atoms.copy(),
            )
            return results[0](result) if results else result

        return SimpleNamespace(run_one=run)

    monkeypatch.setattr(elemental_dft, "make_backend", factory)
    return SimpleNamespace(
        plan=ElementalReferencePlan(phases=phases),
        dft=dft,
        calls=calls,
        results=results,
        cache=tmp_path / "cache",
        tmp=tmp_path,
    )


def _prepare(setup, name="snapshot", **kwargs):
    return prepare_elemental_references(
        setup.plan,
        setup.dft,
        required_elements={"Si", "O"},
        cache_dir=setup.cache,
        output_dir=setup.tmp / name,
        phases_approved=True,
        **kwargs,
    )


@pytest.mark.parametrize("changed", [None, "plan", "geometry", "method"])
def test_al_automatic_training_references_are_approved_bounded_and_frozen(
    setup_references, dataset_method_sidecar, changed
):
    setup = setup_references
    plan_path = setup.tmp / "phases.yaml"
    plan_path.write_text(setup.plan.model_dump_json())
    atoms = Atoms("SiO2", positions=np.zeros((3, 3)))
    atoms.calc = SinglePointCalculator(atoms, energy=-20, forces=np.zeros((3, 3)))
    dataset = setup.tmp / "train.extxyz"
    write(dataset, [atoms, atoms])
    expected = _prepare(setup, dft_approved=True)
    dataset_method_sidecar(dataset, expected)
    cfg = TrainerConfig(
        enabled=True,
        train_script=setup.tmp / "train.py",
        compare_after_training=False,
        promote_model=False,
        hydragnn_training_references={
            "phase_plan": plan_path,
            "cache_dir": setup.cache,
            "phases_approved": True,
            "dft_approved": False,
            "max_dft_calculations": 0,
        },
    )
    root = setup.tmp / "al-references"
    selected = resolve_training_references(cfg, dataset, dft_config=setup.dft, reference_root=root)
    assert selected == root / "elemental-references.json"
    assert len(setup.calls) == 3
    if changed == "plan":
        plan_path.write_text(plan_path.read_text() + "\n")
    elif changed == "geometry":
        geometry = setup.plan.phases[0].structure_path
        updated = read(geometry)
        updated.positions[0, 0] += 0.1
        write(geometry, updated)
    elif changed == "method":
        setup.dft.qe.ecutwfc_ry += 10
    if changed:
        with pytest.raises(ValueError, match="inputs changed"):
            resolve_training_references(cfg, dataset, dft_config=setup.dft, reference_root=root)
    else:
        assert (
            resolve_training_references(cfg, dataset, dft_config=setup.dft, reference_root=root)
            == selected
        )
    assert len(setup.calls) == 3


def test_al_reference_method_mismatch_fails_before_any_dft(
    setup_references, elemental_manifest, dataset_method_sidecar
):
    setup = setup_references
    phase_plan = setup.tmp / "phases.yaml"
    phase_plan.write_text(setup.plan.model_dump_json())
    dataset = setup.tmp / "train.extxyz"
    atoms = Atoms("SiO2", positions=np.zeros((3, 3)))
    atoms.calc = SinglePointCalculator(atoms, energy=-20, forces=np.zeros((3, 3)))
    write(dataset, [atoms, atoms])
    dataset_method_sidecar(dataset, elemental_manifest({"Si": -5, "O": -6}))
    cfg = TrainerConfig(
        hydragnn_training_references={
            "phase_plan": phase_plan,
            "cache_dir": setup.cache,
            "phases_approved": True,
            "dft_approved": True,
            "max_dft_calculations": 3,
        }
    )
    with pytest.raises(ValueError, match="different DFT methods"):
        resolve_training_references(
            cfg, dataset, dft_config=setup.dft, reference_root=setup.tmp / "al-references"
        )
    assert setup.calls == []


def test_selects_energy_per_atom_and_reuses_without_dft_approval(setup_references):
    setup = setup_references
    manifest = _prepare(setup, dft_approved=True)
    assert len(setup.calls) == 3
    payload = json.loads(manifest.read_text())
    assert payload["references"]["Si"]["energy_eV"] == -20
    assert payload["references"]["O"]["energy_eV"] == -12
    assert payload["method_signature"] == dft_method_signature(setup.dft, {"Si", "O"})
    assert payload["selection"]["dft_attempts"] == 3
    assert {
        item["phase"]["phase_id"] for item in payload["selection"]["phases"] if item["selected"]
    } == {"si-b", "o2-triplet"}
    oxygen_cfg = next(cfg for cfg, spec in setup.calls if spec.job_id == "o2-triplet")
    assert oxygen_cfg.qe.extra_system == {"nspin": 2, "tot_magnetization": 2}
    assert oxygen_cfg.qe.kpts == (1, 1, 1)
    assert setup.dft.qe.extra_system == {}
    assert setup.dft.qe.kpts == (2, 2, 2)
    reused = _prepare(setup, "second", dft_approved=False)
    assert len(setup.calls) == 3
    assert json.loads(reused.read_text())["references"] == payload["references"]
    assert json.loads(reused.read_text())["selection"]["dft_attempts"] == 0


def test_reference_manifest_feeds_formation_preparation(setup_references):
    from ase.calculators.singlepoint import SinglePointCalculator

    from matsim_agents.active_learning.dataset_governance import (
        DatasetValidationSummary,
        write_dataset_manifest,
    )

    setup = setup_references
    manifest = _prepare(setup, dft_approved=True)
    atoms = Atoms("SiO2")
    atoms.calc = SinglePointCalculator(atoms, energy=-19, forces=np.zeros((3, 3)))
    raw = setup.tmp / "mixture.extxyz"
    write(raw, atoms)
    write_dataset_manifest(
        raw,
        dft_backend="qe",
        method_signature=dft_method_signature(setup.dft, {"Si", "O"}),
        energy_reference="qe:native_total_energy",
        validation=DatasetValidationSummary(accepted=1),
    )
    formed = read(prepare_formation_training_dataset(raw, manifest, setup.tmp / "formed"))
    assert formed.get_potential_energy() == pytest.approx(-2)
    assert formed.info["elemental_baseline_energy_eV"] == -17
    saved_manifest = json.loads((setup.tmp / "formed/elemental-references.json").read_text())
    assert saved_manifest["selection"] == json.loads(manifest.read_text())["selection"]


def test_reference_calculation_cap_is_preflighted(setup_references):
    setup = setup_references
    with pytest.raises(ValueError, match="cap is 2"):
        _prepare(setup, dft_approved=True, max_dft_calculations=2)
    assert not setup.calls
    _prepare(setup, dft_approved=True, max_dft_calculations=3)
    _prepare(setup, "reuse", max_dft_calculations=0)
    assert len(setup.calls) == 3


def test_duplicate_geometry_aliases_share_one_calculation(setup_references):
    setup = setup_references
    duplicate = setup.plan.phases[0].model_copy(deep=True)
    duplicate.phase_id = "si-a-alias"
    setup.plan.phases.append(duplicate)
    _prepare(setup, dft_approved=True, max_dft_calculations=3)
    assert len(setup.calls) == 3


def test_qe_molecular_settings_reach_generated_input(setup_references):
    from matsim_agents.backends.dft.qe import QEBackend
    from matsim_agents.backends.dft.qe_relax import write_pw_input

    setup = setup_references
    phase = setup.plan.phases[-1]
    cfg = elemental_dft._phase_config(setup.dft, phase)
    atoms = read(phase.structure_path)
    settings = QEBackend(cfg.qe)._settings_for(atoms)
    target = setup.tmp / "O2.in"
    write_pw_input(atoms, settings, str(target))
    text = target.read_text().lower()
    assert "nspin = 2" in " ".join(text.split())
    assert "tot_magnetization = 2" in " ".join(text.split())
    assert "k_points automatic" in text
    assert "1 1 1" in text


def test_fixed_reference_snapshot_never_overwrites_selection(setup_references):
    setup = setup_references
    manifest = _prepare(setup, dft_approved=True)
    original = manifest.read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        _prepare(setup, dft_approved=True)
    assert manifest.read_bytes() == original
    assert len(setup.calls) == 3


def test_no_calculations_without_approval(setup_references):
    with pytest.raises(ValueError, match="require approval"):
        _prepare(setup_references)
    assert setup_references.calls == []
    assert not setup_references.cache.exists()


@pytest.mark.parametrize("approved,required", [(False, {"Si", "O"}), (True, {"Nb"}), (True, set())])
def test_phase_approval_and_coverage_before_launch(setup_references, approved, required):
    setup = setup_references
    with pytest.raises(ValueError):
        prepare_elemental_references(
            setup.plan,
            setup.dft,
            required_elements=required,
            cache_dir=setup.cache,
            output_dir=setup.tmp / "snapshot",
            phases_approved=approved,
            dft_approved=True,
        )
    assert not setup.calls


@pytest.mark.parametrize("defect", ["impure", "multiple", "nonperiodic", "bad_species"])
def test_invalid_phase_preflight_launches_nothing(setup_references, defect):
    setup = setup_references
    path = setup.plan.phases[-1].structure_path
    if defect == "impure":
        write(path, Atoms("SiO"))
    elif defect == "multiple":
        write(path, [Atoms("O2"), Atoms("O2")])
    elif defect == "nonperiodic":
        write(setup.plan.phases[0].structure_path, Atoms("Si2"))
    else:
        with pytest.raises(ValidationError):
            ElementalPhaseSpec(
                phase_id="bad",
                element="Zz",
                structure_path=path,
                kind="bulk",
                magnetic_state="nonmagnetic",
            )
        return
    with pytest.raises(ValueError):
        _prepare(setup, dft_approved=True)
    assert not setup.calls


@pytest.mark.parametrize("change", ["geometry", "pseudo", "spin", "cutoff"])
def test_cache_invalidates_scientific_changes(setup_references, change):
    setup = setup_references
    _prepare(setup, dft_approved=True)
    if change == "geometry":
        path = setup.plan.phases[0].structure_path
        atoms = read(path)
        atoms.positions[0, 0] += 0.01
        write(path, atoms)
    elif change == "pseudo":
        (setup.dft.qe.pseudo_dir / "Si.upf").write_text("changed pseudo")
    elif change == "spin":
        setup.plan.phases[0].qe_magnetic_settings = {"nspin": 2}
    else:
        setup.dft.qe.ecutwfc_ry = 60
    with pytest.raises(ValueError, match="require approval"):
        _prepare(setup, "second", dft_approved=False)
    assert len(setup.calls) == 3


@pytest.mark.parametrize("defect", ["failed", "nonfinite", "changed_geometry", "wrong_backend"])
def test_failed_dft_never_becomes_a_cached_reference(setup_references, defect):
    setup = setup_references

    def mutate(result):
        if defect == "failed":
            return replace(result, converged=False, return_code=1, notes="test failure")
        if defect == "nonfinite":
            return replace(result, energy_eV=float("nan"))
        if defect == "wrong_backend":
            return replace(result, backend="vasp")
        result.final_atoms.positions[0, 0] += 0.1
        return result

    setup.results.append(mutate)
    with pytest.raises((RuntimeError, ValueError)):
        _prepare(setup, dft_approved=True)
    assert not list(setup.cache.glob("*/result.json"))
    assert not (setup.tmp / "snapshot").exists()


@pytest.mark.parametrize("corruption", ["energy", "geometry", "key"])
def test_corrupt_cache_fails_explicitly(setup_references, corruption):
    setup = setup_references
    _prepare(setup, dft_approved=True)
    record = next(setup.cache.glob("*/result.json"))
    if corruption == "geometry":
        (record.parent / "geometry.extxyz").write_text("corrupt")
    else:
        payload = json.loads(record.read_text())
        payload["energy_eV" if corruption == "energy" else "cache_key"] = (
            float("nan") if corruption == "energy" else "invalid"
        )
        record.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        _prepare(setup, "second", dft_approved=True)
    assert len(setup.calls) == 3


def test_plan_resolves_relative_paths_and_rejects_duplicate_ids(setup_references):
    setup = setup_references
    payload = setup.plan.model_dump(mode="json")
    for phase in payload["phases"]:
        phase["structure_path"] = phase["phase_id"] + ".extxyz"
    path = setup.tmp / "plan.yaml"
    path.write_text(json.dumps(payload))
    restored = ElementalReferencePlan.from_yaml(path)
    assert restored == setup.plan
    payload["phases"].append(payload["phases"][0])
    with pytest.raises(ValidationError, match="unique"):
        ElementalReferencePlan.model_validate(payload)


def test_vasp_overrides_are_explicit(setup_references):
    setup = setup_references
    dft = DFTConfig.model_validate(
        {
            "backend": "vasp",
            "vasp": {
                "vasp_bin": setup.tmp / "vasp",
                "vasp_wrapper": setup.tmp / "wrapper",
                "incar_template": setup.tmp / "INCAR",
                "potcar_dir": setup.tmp / "potcar",
            },
        }
    )
    for element in ("O", "Si"):
        directory = dft.vasp.potcar_dir / element
        directory.mkdir(parents=True)
        (directory / "POTCAR").write_text("synthetic potential")
    for phase in setup.plan.phases:
        phase.kpts = None
        phase.qe_magnetic_settings = {}
    setup.plan.phases[-1].vasp_magnetic_settings = {"ISPIN": "2", "NUPDOWN": "2"}
    manifest = prepare_elemental_references(
        setup.plan,
        dft,
        required_elements={"Si", "O"},
        cache_dir=setup.cache,
        output_dir=setup.tmp / "snapshot",
        phases_approved=True,
        dft_approved=True,
    )
    assert json.loads(manifest.read_text())["backend"] == "vasp"
    assert setup.calls[0][0].vasp.extra_incar == {"ISPIN": "2", "NUPDOWN": "2"}
    assert dft.vasp.extra_incar == {}


def test_rejects_ignored_or_unpinned_qe_overrides(setup_references):
    setup = setup_references
    setup.dft.qe.pw_template = setup.tmp / "pw.template"
    with pytest.raises(ValueError, match="ignores magnetic"):
        _prepare(setup, dft_approved=True)
    setup.dft.qe.pw_template = None
    setup.dft.qe.kpts = None
    with pytest.raises(ValueError, match="must pin"):
        _prepare(setup, dft_approved=True)
    assert not setup.calls
