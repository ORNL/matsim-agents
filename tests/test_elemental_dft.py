"""Synthetic elemental DFT orchestration contracts; no external calculations."""

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write
from pydantic import ValidationError

from matsim_agents.active_learning import elemental_dft
from matsim_agents.active_learning.config import TrainerConfig
from matsim_agents.active_learning.dft_protocol import dft_method_signature
from matsim_agents.active_learning.elemental_dft import (
    ElementalPhaseSpec,
    ElementalReferencePlan,
    prepare_elemental_references,
)
from matsim_agents.active_learning.hydragnn_references import resolve_training_references
from matsim_agents.backends.dft.base import DFTResult
from matsim_agents.backends.dft.config import DFTConfig
from matsim_agents.datasets.formation_training import prepare_formation_training_dataset


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
                geometry_status="verified_converged",
                magnetic_state="nonmagnetic" if kind == "bulk" else "triplet",
                relaxation_stress_tolerance_eV_per_A3=0.01,
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
    required = kwargs.pop("required_elements", {"Si", "O"})
    return prepare_elemental_references(
        setup.plan,
        setup.dft,
        required_elements=required,
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
    dataset_method_sidecar(dataset, expected, method_signature_elements={"Si", "O"})
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


def test_phase_plan_references_use_recorded_scope_beyond_training_elements(
    setup_references, dataset_method_sidecar
):
    setup = setup_references
    plan_path = setup.tmp / "phases.yaml"
    plan_path.write_text(setup.plan.model_dump_json())

    si_atoms = Atoms("Si2", positions=np.zeros((2, 3)))
    si_atoms.calc = SinglePointCalculator(si_atoms, energy=-10, forces=np.zeros((2, 3)))
    dataset = setup.tmp / "train-si-only.extxyz"
    write(dataset, [si_atoms])
    expected = _prepare(setup, dft_approved=True, required_elements={"Si", "O"})
    dataset_method_sidecar(dataset, expected, method_signature_elements={"Si", "O"})

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

    manifest = json.loads(selected.read_text())
    assert set(manifest["references"]) == {"Si", "O"}
    assert manifest["method_signature"] == dft_method_signature(setup.dft, {"Si", "O"})
    assert len(setup.calls) == 3


@pytest.mark.parametrize(
    "method_signature_elements,error",
    [
        (None, "require the dataset manifest to record"),
        ({"O"}, "does not cover training elements"),
    ],
)
def test_phase_plan_references_reject_missing_or_incomplete_scope(
    setup_references, dataset_method_sidecar, method_signature_elements, error
):
    setup = setup_references
    plan_path = setup.tmp / "phases.yaml"
    plan_path.write_text(setup.plan.model_dump_json())
    si_atoms = Atoms("Si2", positions=np.zeros((2, 3)))
    si_atoms.calc = SinglePointCalculator(si_atoms, energy=-10, forces=np.zeros((2, 3)))
    dataset = setup.tmp / "train-si-only.extxyz"
    write(dataset, [si_atoms])
    expected = _prepare(setup, dft_approved=True, required_elements={"Si"})
    dataset_method_sidecar(
        dataset,
        expected,
        method_signature_elements=method_signature_elements,
    )
    cfg = TrainerConfig(
        enabled=True,
        train_script=setup.tmp / "train.py",
        hydragnn_training_references={
            "phase_plan": plan_path,
            "cache_dir": setup.cache,
            "phases_approved": True,
        },
    )

    with pytest.raises(ValueError, match=error):
        resolve_training_references(
            cfg, dataset, dft_config=setup.dft, reference_root=setup.tmp / "al-references"
        )


def test_unverified_phase_relaxes_then_runs_static_and_caches_geometry(
    setup_references, monkeypatch
):
    setup = setup_references
    phase = setup.plan.phases[0]
    phase.geometry_status = "requires_relaxation"
    relax_calls = []

    def fake_relax(phase, cfg, atoms, work_dir):
        relaxed = atoms.copy()
        relaxed.positions[0, 0] += 0.05
        relax_calls.append((phase, cfg, atoms.copy(), work_dir))
        return relaxed, 0.01, 4, np.zeros(6)

    monkeypatch.setattr(elemental_dft, "_run_reference_relaxation", fake_relax)
    with pytest.raises(ValueError, match="needs 4 DFT calculations"):
        _prepare(setup, dft_approved=True, max_dft_calculations=3)
    assert not setup.calls
    assert not relax_calls

    manifest = _prepare(setup, dft_approved=True, max_dft_calculations=4)
    assert len(relax_calls) == 1
    assert len(setup.calls) == 3
    relaxed_atoms = next(spec.atoms for _, spec in setup.calls if spec.job_id == phase.phase_id)
    assert relaxed_atoms.positions[0, 0] == pytest.approx(0.05)
    payload = json.loads(manifest.read_text())
    entry = next(
        item
        for item in payload["selection"]["phases"]
        if item["phase"]["phase_id"] == phase.phase_id
    )
    assert entry["geometry_status"] == "requires_relaxation"
    assert entry["relaxation"]["converged"] is True
    assert entry["relaxation"]["ionic_steps"] == 4
    assert entry["relaxation"]["final_stress_eV_per_A3"] == [0.0] * 6
    assert entry["calculation"]["geometry_sha256"] != entry["input_geometry_sha256"]
    assert payload["selection"]["dft_attempts"] == 4

    reused = _prepare(setup, "reuse", dft_approved=False, max_dft_calculations=0)
    assert len(relax_calls) == 1
    assert len(setup.calls) == 3
    assert json.loads(reused.read_text())["selection"]["dft_attempts"] == 0


@pytest.mark.parametrize("requires_relaxation", [False, True])
def test_relative_cache_uses_absolute_static_and_relaxation_work_dirs(
    setup_references, monkeypatch, requires_relaxation
):
    setup = setup_references
    monkeypatch.chdir(setup.tmp)
    setup.cache = Path("relative-cache")
    phase = setup.plan.phases[0]
    if requires_relaxation:
        phase.geometry_status = "requires_relaxation"

        def fake_relax(_phase, _cfg, atoms, work_dir):
            path = Path(work_dir)
            assert path.is_absolute()
            assert path.is_relative_to((setup.tmp / "relative-cache").resolve())
            return atoms.copy(), 0.01, 4, np.zeros(6)

        monkeypatch.setattr(elemental_dft, "_run_reference_relaxation", fake_relax)

    _prepare(
        setup,
        dft_approved=True,
        max_dft_calculations=4 if requires_relaxation else 3,
    )

    static_work_dirs = [Path(spec.work_dir) for _, spec in setup.calls]
    assert len(static_work_dirs) == 3
    assert all(path.is_absolute() for path in static_work_dirs)
    assert all(
        path.is_relative_to((setup.tmp / "relative-cache").resolve()) for path in static_work_dirs
    )


def test_unconverged_reference_relaxation_never_creates_reference_result(
    setup_references, monkeypatch
):
    setup = setup_references
    setup.plan.phases[0].geometry_status = "requires_relaxation"

    def failed_relax(*args):
        raise RuntimeError("test relaxation did not converge")

    monkeypatch.setattr(elemental_dft, "_run_reference_relaxation", failed_relax)
    with pytest.raises(RuntimeError, match="did not converge"):
        _prepare(setup, dft_approved=True, max_dft_calculations=4)
    assert not (setup.tmp / "snapshot").exists()
    assert not list((setup.cache / "relaxations").rglob("result.json"))


def test_bulk_relaxation_rejects_nonperiodic_final_geometry(setup_references, monkeypatch):
    setup = setup_references
    phase = setup.plan.phases[0]
    phase.geometry_status = "requires_relaxation"

    def fake_relax(_phase, _cfg, atoms, _work_dir):
        relaxed = atoms.copy()
        relaxed.pbc = False
        return relaxed, 0.01, 4, np.zeros(6)

    monkeypatch.setattr(elemental_dft, "_run_reference_relaxation", fake_relax)
    with pytest.raises(RuntimeError, match="invalid final geometry"):
        _prepare(setup, dft_approved=True, max_dft_calculations=4)
    assert all(spec.job_id != phase.phase_id for _, spec in setup.calls)
    assert not list((setup.cache / "relaxations").rglob("result.json"))


def test_bulk_cell_relaxation_requires_explicit_stress_tolerance(setup_references):
    phase = setup_references.plan.phases[0].model_dump()
    phase["geometry_status"] = "requires_relaxation"
    phase.pop("relaxation_stress_tolerance_eV_per_A3")

    with pytest.raises(ValidationError, match="requires an explicit"):
        ElementalPhaseSpec.model_validate(phase)


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
    dataset_method_sidecar(
        dataset,
        elemental_manifest({"Si": -5, "O": -6}),
        method_signature_elements={"Si", "O"},
    )
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


def test_qe_phase_kpoints_complete_shared_protocol(setup_references):
    setup = setup_references
    setup.dft.qe.kpts = None
    phase = setup.plan.phases[-1]
    assert phase.kpts == (1, 1, 1)

    resolved = elemental_dft._phase_config(setup.dft, phase)

    assert resolved.qe.kpts == (1, 1, 1)
    assert setup.dft.qe.kpts is None


@pytest.mark.parametrize("format_name,filename", [("cif", "Si.cif"), ("vasp", "POSCAR")])
def test_trusted_reference_formats_are_serialized_as_extxyz(
    setup_references, format_name, filename
):
    setup = setup_references
    phase = setup.plan.phases[1]
    original = read(phase.structure_path)
    original.positions[:, 0] = np.arange(len(original), dtype=float) * 0.17 + 0.123456789
    phase.structure_path = setup.tmp / filename
    write(phase.structure_path, original, format=format_name)
    source_digest = elemental_dft.sha256_file(phase.structure_path)

    manifest = _prepare(setup, dft_approved=True)
    selected = json.loads(manifest.read_text())["references"]["Si"]
    geometry = manifest.parent / selected["structure_path"]
    snapshot = read(geometry, format="extxyz")
    calculated_atoms = next(spec.atoms for _, spec in setup.calls if spec.job_id == phase.phase_id)
    assert snapshot.get_chemical_symbols() == original.get_chemical_symbols()
    np.testing.assert_allclose(snapshot.positions, original.positions)
    np.testing.assert_allclose(snapshot.cell.array, original.cell.array)
    np.testing.assert_array_equal(calculated_atoms.positions, snapshot.positions)
    np.testing.assert_array_equal(calculated_atoms.cell.array, snapshot.cell.array)
    np.testing.assert_array_equal(calculated_atoms.pbc, snapshot.pbc)
    assert selected["input_geometry_sha256"] == source_digest
    assert selected["structure_sha256"] == elemental_dft.sha256_file(geometry)
    assert selected["structure_sha256"] != source_digest
    reused = _prepare(setup, "reuse", dft_approved=False, max_dft_calculations=0)
    assert (
        json.loads(reused.read_text())["references"]
        == json.loads(manifest.read_text())["references"]
    )
    assert len(setup.calls) == 3


def test_reference_manifest_feeds_formation_preparation(setup_references):
    from ase.calculators.singlepoint import SinglePointCalculator

    from matsim_agents.datasets.governance import (
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


@pytest.mark.parametrize("change", ["wrapper", "environment"])
def test_reference_cache_includes_execution_identity(setup_references, monkeypatch, change):
    setup = setup_references
    wrapper = setup.dft.qe.pw_wrapper
    wrapper.write_text('ROCM_MODULE="${ROCM_MODULE:-rocm/6.2.4}"\n')
    monkeypatch.delenv("ROCM_MODULE", raising=False)
    method_signature = dft_method_signature(setup.dft, {"Si", "O"})
    _prepare(setup, "first", dft_approved=True)
    assert len(setup.calls) == 3

    if change == "wrapper":
        wrapper.write_text('ROCM_MODULE="${ROCM_MODULE:-rocm/6.2.5}"\n')
    else:
        monkeypatch.setenv("ROCM_MODULE", "rocm/6.3.0")

    assert dft_method_signature(setup.dft, {"Si", "O"}) == method_signature
    _prepare(setup, "second", dft_approved=True)
    assert len(setup.calls) == 6


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


def test_qe_relaxation_keeps_reference_sampling_and_spin(setup_references, monkeypatch):
    from matsim_agents.backends.dft import qe_relax

    setup = setup_references
    phase = setup.plan.phases[-1]
    phase.geometry_status = "requires_relaxation"
    cfg = elemental_dft._phase_config(setup.dft, phase)
    atoms = read(phase.structure_path)
    final_atoms = atoms.copy()
    final_atoms.calc = SinglePointCalculator(
        final_atoms,
        energy=-12.0,
        forces=np.tile([0.015, 0.0, 0.0], (len(final_atoms), 1)),
    )
    observed = {}
    original_recommend_settings = qe_relax.recommend_settings

    def capture_settings(*args, **kwargs):
        observed.update(kwargs)
        return original_recommend_settings(*args, **kwargs)

    def fake_run(argv, **kwargs):
        input_text = " ".join(Path(argv[4]).read_text().lower().split())
        assert "calculation = 'relax'" in input_text
        assert "ecutwfc = 5.00000000d+01" in input_text
        assert "nspin = 2" in input_text
        assert "tot_magnetization = 2" in input_text
        assert "1 1 1 0 0 0" in input_text
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(elemental_dft.subprocess, "run", fake_run)
    monkeypatch.setattr(qe_relax, "recommend_settings", capture_settings)
    monkeypatch.setattr(
        qe_relax,
        "parse_pw_stdout",
        lambda *args, **kwargs: SimpleNamespace(
            converged=True,
            return_code=0,
            final_max_force_ry_au=0.0212 / 25.711043,
            final_energy_ev=-12.0,
            bfgs_steps=5,
        ),
    )
    monkeypatch.setattr(elemental_dft, "read", lambda *args, **kwargs: final_atoms)
    relaxed, force, steps, stress = elemental_dft._run_reference_relaxation(
        phase, cfg, atoms, setup.tmp / "qe-relax"
    )
    assert relaxed.get_chemical_symbols() == atoms.get_chemical_symbols()
    assert force == pytest.approx(0.015)
    assert steps == 5
    assert stress is None
    assert observed["forc_conv_thr_ry_au"] == pytest.approx(
        phase.relaxation_force_tolerance_eV_per_A / (25.711043 * np.sqrt(3))
    )


@pytest.mark.parametrize(("stress_component", "accepted"), [(0.005, True), (0.02, False)])
def test_qe_cell_relaxation_checks_bulk_stress_tolerance(
    setup_references, monkeypatch, stress_component, accepted
):
    from matsim_agents.backends.dft import qe_relax

    setup = setup_references
    phase = setup.plan.phases[0].model_copy(
        update={"geometry_status": "requires_relaxation"}, deep=True
    )
    atoms = read(phase.structure_path)
    final_atoms = atoms.copy()
    final_atoms.calc = SinglePointCalculator(
        final_atoms,
        energy=-10.0,
        forces=np.zeros((len(final_atoms), 3)),
        stress=np.full(6, stress_component),
    )
    monkeypatch.setattr(
        qe_relax,
        "parse_pw_stdout",
        lambda *args, **kwargs: SimpleNamespace(
            converged=True,
            return_code=0,
            final_max_force_ry_au=0.0002,
            final_energy_ev=-10.0,
            bfgs_steps=5,
        ),
    )
    monkeypatch.setattr(
        elemental_dft.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0),
    )
    monkeypatch.setattr(elemental_dft, "read", lambda *args, **kwargs: final_atoms)

    if accepted:
        relaxed, _, _, stress = elemental_dft._run_reference_relaxation(
            phase, setup.dft, atoms, setup.tmp / "qe-cell-relax"
        )
        assert relaxed.get_chemical_symbols() == atoms.get_chemical_symbols()
        assert stress.tolist() == pytest.approx([stress_component] * 6)
    else:
        with pytest.raises(RuntimeError, match="stress threshold"):
            elemental_dft._run_reference_relaxation(
                phase, setup.dft, atoms, setup.tmp / "qe-cell-relax"
            )


@pytest.mark.parametrize(("stress_component", "accepted"), [(0.005, True), (0.02, False)])
def test_vasp_relaxation_checks_bulk_stress_tolerance(
    setup_references, monkeypatch, stress_component, accepted
):
    setup = setup_references
    template = setup.tmp / "INCAR"
    template.write_text("ENCUT = 520\nKSPACING = 0.2\n")
    potcar_dir = setup.tmp / "potcars"
    (potcar_dir / "O").mkdir(parents=True)
    (potcar_dir / "O" / "POTCAR").write_text("synthetic potential")
    dft = DFTConfig.model_validate(
        {
            "backend": "vasp",
            "vasp": {
                "vasp_bin": setup.tmp / "vasp",
                "vasp_wrapper": setup.tmp / "wrapper",
                "incar_template": template,
                "potcar_dir": potcar_dir,
            },
        }
    )
    phase = setup.plan.phases[0].model_copy(
        update={
            "geometry_status": "requires_relaxation",
            "qe_magnetic_settings": {},
            "kpts": None,
            "vasp_magnetic_settings": {"ISPIN": "2", "NUPDOWN": "2"},
            "relaxation_stress_tolerance_eV_per_A3": 0.01,
        },
        deep=True,
    )
    (potcar_dir / "Si").mkdir(parents=True)
    (potcar_dir / "Si" / "POTCAR").write_text("synthetic potential")
    cfg = elemental_dft._phase_config(dft, phase)
    atoms = read(phase.structure_path)
    from matsim_agents.backends.dft import vasp_relax

    def fake_run(argv, **kwargs):
        incar = (Path(kwargs["cwd"]) / "INCAR").read_text()
        assert "ENCUT = 520" in incar
        assert "ISPIN = 2" in incar
        assert "NUPDOWN = 2" in incar
        assert "IBRION = 2" in incar
        assert "ISIF = 3" in incar
        assert "NSW = 200" in incar
        assert "EDIFFG = -0.02" in incar
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(elemental_dft.subprocess, "run", fake_run)
    monkeypatch.setattr(
        vasp_relax,
        "parse_relax_outputs",
        lambda *args, **kwargs: SimpleNamespace(
            converged=True,
            return_code=0,
            final_atoms=atoms.copy(),
            final_energy_eV=-12.0,
            final_max_force_eV_per_A=0.01,
            final_stress_eV_per_A3=np.full(6, stress_component),
            n_ionic_steps=6,
        ),
    )
    if accepted:
        relaxed, force, steps, stress = elemental_dft._run_reference_relaxation(
            phase, cfg, atoms, setup.tmp / "vasp-relax"
        )
    else:
        with pytest.raises(RuntimeError, match="stress threshold"):
            elemental_dft._run_reference_relaxation(phase, cfg, atoms, setup.tmp / "vasp-relax")
        return
    assert relaxed.get_chemical_symbols() == atoms.get_chemical_symbols()
    assert force == pytest.approx(0.01)
    assert steps == 6
    assert stress.tolist() == pytest.approx([stress_component] * 6)


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
                geometry_status="verified_converged",
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
    assert not list(setup.cache.rglob("calculations/*/result.json"))
    assert not (setup.tmp / "snapshot").exists()


@pytest.mark.parametrize("corruption", ["energy", "geometry", "key"])
def test_corrupt_cache_fails_explicitly(setup_references, corruption):
    setup = setup_references
    _prepare(setup, dft_approved=True)
    record = next((setup.cache / "calculations").glob("*/result.json"))
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


@pytest.mark.parametrize(
    "defect,error",
    [
        ("convergence", "invalid cached elemental relaxation"),
        ("force", "invalid cached elemental relaxation"),
        ("missing_stress", "invalid cached elemental relaxation"),
        ("stress", "invalid cached elemental relaxation"),
        ("atom_count", "invalid cached elemental relaxation"),
        ("geometry", "invalid cached relaxed geometry"),
    ],
)
def test_post_lock_relaxation_cache_is_fully_validated(
    setup_references, monkeypatch, defect, error
):
    setup = setup_references
    phase = setup.plan.phases[0]
    phase.geometry_status = "requires_relaxation"
    phase_atoms = read(phase.structure_path)
    original_lock = elemental_dft.exclusive_lock
    inserted = False

    def insert_racing_cache(lock_path, **kwargs):
        nonlocal inserted
        lock_path = Path(lock_path)
        if not inserted and lock_path.parent.parent.name == "relaxations":
            inserted = True
            relaxed_atoms = phase_atoms.copy()
            if defect == "geometry":
                relaxed_atoms.set_chemical_symbols(["O"] * len(relaxed_atoms))
            if defect == "force":
                max_force = phase.relaxation_force_tolerance_eV_per_A + 0.1
            else:
                max_force = phase.relaxation_force_tolerance_eV_per_A
            max_stress = 0.02 if defect == "stress" else 0.005
            geometry_path = lock_path.parent / "geometry.extxyz"
            write(geometry_path, relaxed_atoms, format="extxyz")
            payload = {
                "cache_key": lock_path.parent.name,
                "input_geometry_sha256": elemental_dft.sha256_file(phase.structure_path),
                "relaxed_geometry_sha256": elemental_dft.sha256_file(geometry_path),
                "method_signature": elemental_dft.dft_method_signature(
                    elemental_dft._phase_config(setup.dft, phase), {phase.element}
                ),
                "execution_signature": elemental_dft.dft_execution_signature(
                    elemental_dft._phase_config(setup.dft, phase)
                ),
                "n_atoms": len(relaxed_atoms) + (1 if defect == "atom_count" else 0),
                "converged": defect != "convergence",
                "final_max_force_eV_per_A": max_force,
                "final_stress_eV_per_A3": (
                    None if defect == "missing_stress" else [max_stress] * 6
                ),
                "ionic_steps": 1,
                "work_dir": str(lock_path.parent),
            }
            (lock_path.parent / "result.json").write_text(json.dumps(payload))
        return original_lock(lock_path, **kwargs)

    monkeypatch.setattr(elemental_dft, "exclusive_lock", insert_racing_cache)
    with pytest.raises(ValueError, match=error):
        _prepare(setup, dft_approved=True, max_dft_calculations=4)
    assert inserted
    assert all(spec.job_id != phase.phase_id for _, spec in setup.calls)


def test_concurrent_reference_preflight_waits_for_locked_relaxation_cache(
    setup_references, monkeypatch
):
    setup = setup_references
    phase = setup.plan.phases[0]
    phase.geometry_status = "requires_relaxation"
    relaxation_started = threading.Event()
    allow_relaxation_to_finish = threading.Event()
    second_preflight_waiting = threading.Event()
    original_lock = elemental_dft.exclusive_lock
    thread_label = threading.local()

    def fake_relax(_phase, _cfg, atoms, _work_dir):
        relaxation_started.set()
        assert allow_relaxation_to_finish.wait(timeout=10)
        return atoms.copy(), 0.01, 4, np.zeros(6)

    def observe_lock(lock_path, **kwargs):
        if (
            getattr(thread_label, "value", None) == "second"
            and Path(lock_path).parent.parent.name == "relaxations"
        ):
            second_preflight_waiting.set()
        return original_lock(lock_path, **kwargs)

    monkeypatch.setattr(elemental_dft, "_run_reference_relaxation", fake_relax)
    monkeypatch.setattr(elemental_dft, "exclusive_lock", observe_lock)

    def prepare(label, name):
        thread_label.value = label
        return _prepare(setup, name, dft_approved=True, max_dft_calculations=4)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(prepare, "first", "snapshot-first")
        assert relaxation_started.wait(timeout=10)
        second = pool.submit(prepare, "second", "snapshot-second")
        assert second_preflight_waiting.wait(timeout=10)
        assert not second.done()
        allow_relaxation_to_finish.set()
        assert first.result(timeout=10).is_file()
        assert second.result(timeout=10).is_file()


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


@pytest.mark.parametrize(
    ("template_text", "rejects"),
    [
        ("ENCUT = 520; PSTRESS = 10\n", True),
        ("PSTRESS = 0; ENCUT = 520\n", False),
    ],
)
def test_vasp_template_parses_semicolon_separated_pressure(
    setup_references, template_text, rejects
):
    setup = setup_references
    template = setup.tmp / "INCAR"
    template.write_text(template_text)
    dft = DFTConfig.model_validate(
        {
            "backend": "vasp",
            "vasp": {
                "vasp_bin": setup.tmp / "vasp",
                "vasp_wrapper": setup.tmp / "wrapper",
                "incar_template": template,
                "potcar_dir": setup.tmp / "potcar",
            },
        }
    )
    phase = setup.plan.phases[0].model_copy(
        update={
            "geometry_status": "requires_relaxation",
            "qe_magnetic_settings": {},
            "kpts": None,
        },
        deep=True,
    )
    if rejects:
        with pytest.raises(ValueError, match="nonzero PSTRESS"):
            elemental_dft._phase_config(dft, phase)
    else:
        elemental_dft._phase_config(dft, phase)


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


@pytest.mark.parametrize("status", ["verified_converged", "requires_relaxation"])
@pytest.mark.parametrize("forbidden_key", ["calculation", "prefix", "outdir", "pseudo_dir"])
def test_qe_unconditionally_rejects_forbidden_control_overrides(
    setup_references, status, forbidden_key
):
    setup = setup_references
    phase = setup.plan.phases[0].model_copy(deep=True)
    phase.geometry_status = status
    setup.dft.qe.extra_control = {forbidden_key: "forbidden_value"}
    with pytest.raises(ValueError, match="control overrides conflict with"):
        elemental_dft._phase_config(setup.dft, phase)


def test_al_additive_reference_extension_across_iterations(
    setup_references, dataset_method_sidecar, monkeypatch
):
    setup = setup_references
    plan_path = setup.tmp / "phases.yaml"
    plan_path.write_text(setup.plan.model_dump_json())

    # Iteration 1: AL dataset only has Si
    si_atoms = Atoms("Si2", positions=np.zeros((2, 3)))
    si_atoms.calc = SinglePointCalculator(si_atoms, energy=-10, forces=np.zeros((2, 3)))
    iter1_dataset = setup.tmp / "train_iter1.extxyz"
    write(iter1_dataset, [si_atoms])
    # Iteration 1: AL dataset only has Si. Pre-cache Si calculations and run with dft_approved=False
    expected_si = _prepare(setup, "expected_si", dft_approved=True, required_elements={"Si"})
    dataset_method_sidecar(iter1_dataset, expected_si, method_signature_elements={"Si"})

    cfg1 = TrainerConfig(
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
    res1 = resolve_training_references(
        cfg1, iter1_dataset, dft_config=setup.dft, reference_root=root
    )
    assert res1 == root / "elemental-references.json"
    manifest1 = json.loads(res1.read_text())
    assert set(manifest1["references"].keys()) == {"Si"}
    assert manifest1["method_signature"] == dft_method_signature(setup.dft, {"Si"})
    assert manifest1["selection"]["dft_approved"] is False
    assert manifest1["selection"]["dft_attempts"] == 0
    assert (root / "Si.extxyz").exists()
    assert not (root / "O.extxyz").exists()
    calls_after_iter1 = len(setup.calls)

    # Iteration 2: AL dataset now has Si and O (newly labelled species O)
    # Run with dft_approved=True to calculate newly introduced species O
    from matsim_agents.datasets.governance import (
        DatasetValidationSummary,
        write_dataset_manifest,
    )

    sio2_atoms = Atoms("SiO2", positions=np.zeros((3, 3)))
    sio2_atoms.calc = SinglePointCalculator(sio2_atoms, energy=-20, forces=np.zeros((3, 3)))
    iter2_dataset = setup.tmp / "train_iter2.extxyz"
    write(iter2_dataset, [si_atoms, sio2_atoms])
    write_dataset_manifest(
        iter2_dataset,
        dft_backend=setup.dft.backend,
        method_signature=dft_method_signature(setup.dft, {"Si", "O"}),
        method_signature_elements={"Si", "O"},
        energy_reference=f"{setup.dft.backend}:native_total_energy",
        validation=DatasetValidationSummary(),
    )

    cfg2 = TrainerConfig(
        enabled=True,
        train_script=setup.tmp / "train.py",
        compare_after_training=False,
        promote_model=False,
        hydragnn_training_references={
            "phase_plan": plan_path,
            "cache_dir": setup.cache,
            "phases_approved": True,
            "dft_approved": True,
            "max_dft_calculations": 5,
        },
    )

    original_replace = Path.replace
    interrupted = False

    def interrupt_activation(path, target):
        nonlocal interrupted
        if not interrupted and path.name.startswith(f".{root.name}.next-"):
            interrupted = True
            raise RuntimeError("simulated publication interruption")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", interrupt_activation)
    with pytest.raises(RuntimeError, match="simulated publication interruption"):
        resolve_training_references(cfg2, iter2_dataset, dft_config=setup.dft, reference_root=root)
    monkeypatch.setattr(Path, "replace", original_replace)
    res2 = resolve_training_references(
        cfg2, iter2_dataset, dft_config=setup.dft, reference_root=root
    )
    assert interrupted
    assert res2 == root / "elemental-references.json"
    manifest2 = json.loads(res2.read_text())
    assert set(manifest2["references"].keys()) == {"Si", "O"}
    assert manifest2["method_signature"] == dft_method_signature(setup.dft, {"Si", "O"})
    assert manifest2["references"]["Si"] == manifest1["references"]["Si"]
    assert (root / "Si.extxyz").exists()
    assert (root / "O.extxyz").exists()
    assert len(setup.calls) > calls_after_iter1
    calls_after_iter2 = len(setup.calls)

    # Verify extension approval provenance and updated aggregate audit
    selection2 = manifest2["selection"]
    assert selection2["initial_approval"]["dft_approved"] is False
    assert selection2["initial_approval"]["dft_attempts"] == 0
    assert len(selection2["extensions"]) == 1
    assert selection2["extensions"][0]["elements"] == ["O"]
    assert selection2["extensions"][0]["dft_approved"] is True
    assert selection2["extensions"][0]["dft_attempts"] > 0
    assert selection2["dft_approved"] is True
    assert selection2["dft_attempts"] == selection2["extensions"][0]["dft_attempts"]

    # Iteration 3: same elements, no new DFT calls needed
    res3 = resolve_training_references(
        cfg2, iter2_dataset, dft_config=setup.dft, reference_root=root
    )
    assert res3 == root / "elemental-references.json"
    assert len(setup.calls) == calls_after_iter2

    # Verification: changing DFT method for existing element raises error
    setup.dft.qe.ecutwfc_ry += 10
    with pytest.raises(ValueError, match="inputs changed"):
        resolve_training_references(cfg2, iter2_dataset, dft_config=setup.dft, reference_root=root)


def test_manifest_reference_extension_accepts_expanded_real_method_scope(setup_references):
    from matsim_agents.datasets.governance import (
        DatasetValidationSummary,
        write_dataset_manifest,
    )

    setup = setup_references
    source = _prepare(setup, "source", dft_approved=True)
    config = TrainerConfig(
        enabled=True,
        train_script=setup.tmp / "train.py",
        hydragnn_training_references={"manifest": source},
    )
    root = setup.tmp / "manifest-references"
    si_atoms = Atoms("Si2", positions=np.zeros((2, 3)))
    si_atoms.calc = SinglePointCalculator(si_atoms, energy=-10, forces=np.zeros((2, 3)))
    iter1 = setup.tmp / "manifest-iter1.extxyz"
    write(iter1, [si_atoms])
    write_dataset_manifest(
        iter1,
        dft_backend="qe",
        method_signature=dft_method_signature(setup.dft, {"Si"}),
        method_signature_elements={"Si"},
        base_method_signature=dft_method_signature(setup.dft, set()),
        element_method_signatures={"Si": dft_method_signature(setup.dft, {"Si"})},
        energy_reference="qe:native_total_energy",
        validation=DatasetValidationSummary(),
    )
    first = resolve_training_references(config, iter1, dft_config=None, reference_root=root)
    first_manifest = json.loads(first.read_text())
    assert set(first_manifest["references"]) == {"Si"}

    sio_atoms = Atoms("SiO2", positions=np.zeros((3, 3)))
    sio_atoms.calc = SinglePointCalculator(sio_atoms, energy=-20, forces=np.zeros((3, 3)))
    iter2 = setup.tmp / "manifest-iter2.extxyz"
    write(iter2, [si_atoms, sio_atoms])
    write_dataset_manifest(
        iter2,
        dft_backend="qe",
        method_signature=dft_method_signature(setup.dft, {"Si", "O"}),
        method_signature_elements={"Si", "O"},
        base_method_signature=dft_method_signature(setup.dft, set()),
        element_method_signatures={
            element: dft_method_signature(setup.dft, {element}) for element in ("Si", "O")
        },
        energy_reference="qe:native_total_energy",
        validation=DatasetValidationSummary(),
    )
    second = resolve_training_references(config, iter2, dft_config=None, reference_root=root)
    second_manifest = json.loads(second.read_text())
    assert set(second_manifest["references"]) == {"Si", "O"}
    assert second_manifest["references"]["Si"] == first_manifest["references"]["Si"]
