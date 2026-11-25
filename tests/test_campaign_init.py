"""A campaign's own AL config and elemental references; repository examples are templates."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from ase.build import bulk
from ase.io import write
from typer.testing import CliRunner

from matsim_agents.active_learning import ALConfig
from matsim_agents.active_learning.campaign_init import (
    check_reference_coverage,
    elements_of,
    propose_reference_phases,
    write_campaign,
)
from matsim_agents.active_learning.config import TrainerConfig
from matsim_agents.active_learning.elemental_dft import ElementalReferencePlan
from matsim_agents.cli import app

ROOT = Path(__file__).resolve().parents[1]
UMA = ROOT / "examples/mlip/uma-s-1p1-omat.yaml"
QE_ENV = {"PW_BIN": "/bin/true", "PW_WRAPPER": "/bin/true", "PSEUDO_DIR": "/tmp"}


def _w_campaign(tmp_path: Path) -> Path:
    write_campaign(
        tmp_path / "w",
        mlip_config=UMA,
        dft_backend="qe",
        compositions=["W15Ta", "W14CrV", "W15Hf", "W15Mo"],
        ecutwfc_ry=51,
        ecutrho_ry=650,
        kpts=(2, 2, 2),
        reference_kpts=(16, 16, 16),
        preset="frontier",
    )
    return tmp_path / "w"


def test_campaign_covers_every_element_with_a_condensed_phase(tmp_path):
    campaign = _w_campaign(tmp_path)
    plan = ElementalReferencePlan.from_yaml(campaign / "elemental-phases.yaml")
    phases = {phase.element: phase for phase in plan.phases}
    assert set(phases) == {"W", "Ta", "Cr", "V", "Hf", "Mo"}
    assert {phase.kind for phase in plan.phases} == {"bulk"}
    assert phases["Hf"].phase_id == "hf-hcp"
    assert phases["W"].phase_id == "w-bcc"
    for phase in plan.phases:
        assert phase.geometry_status == "requires_relaxation"
        assert phase.structure_path.is_file()
        assert phase.kpts == (16, 16, 16)


def test_generated_config_approves_nothing(tmp_path):
    references = yaml.safe_load((_w_campaign(tmp_path) / "al.yaml").read_text())["trainer"][
        "training_references"
    ]
    assert references["phases_approved"] is False
    assert references["dft_approved"] is False


def test_generated_config_validates_after_user_approval(tmp_path, monkeypatch):
    for name, value in QE_ENV.items():
        monkeypatch.setenv(name, value)
    config = _w_campaign(tmp_path) / "al.yaml"
    config.write_text(config.read_text().replace("phases_approved: false", "phases_approved: true"))
    cfg = ALConfig.from_yaml(config)
    assert cfg.dft.qe.ecutwfc_ry == 51
    assert cfg.loop.max_dft_calculations == 0
    seeds = cfg.md.seed_source
    check_reference_coverage(cfg.trainer, elements_of(seeds.paths, seeds.compositions))


def test_generated_config_requires_phase_approval(tmp_path, monkeypatch):
    for name, value in QE_ENV.items():
        monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match="phases_approved"):
        ALConfig.from_yaml(_w_campaign(tmp_path) / "al.yaml")


def test_init_campaign_never_overwrites(tmp_path):
    campaign = _w_campaign(tmp_path)
    before = (campaign / "al.yaml").read_text()
    with pytest.raises(FileExistsError):
        _w_campaign(tmp_path)
    assert (campaign / "al.yaml").read_text() == before


def test_seed_structures_define_the_elements(tmp_path):
    seed = tmp_path / "NbO.extxyz"
    atoms = bulk("NaCl", "rocksalt", a=4.2)
    atoms.set_chemical_symbols(["Nb", "O"])
    write(seed, atoms)
    write_campaign(tmp_path / "c", mlip_config=UMA, dft_backend="qe", seed_paths=[seed])
    plan = ElementalReferencePlan.from_yaml(tmp_path / "c/elemental-phases.yaml")
    oxygen = next(phase for phase in plan.phases if phase.element == "O")
    assert {phase.element for phase in plan.phases} == {"Nb", "O"}
    assert oxygen.kind == "molecule"
    assert oxygen.magnetic_state == "triplet"
    assert oxygen.qe_magnetic_settings == {"nspin": 2, "tot_magnetization": 2}


def test_unsupported_and_magnetic_elements_are_flagged_not_guessed():
    proposal = propose_reference_phases(
        {"Cr", "Fe", "Gd", "S"}, backend="vasp", stress_tolerance_eV_per_A3=0.01
    )
    assert set(proposal.unresolved) == {"Gd", "S"}
    fe = next(phase for phase in proposal.phases if phase["element"] == "Fe")
    assert fe["magnetic_state"] == "ferromagnetic"
    assert fe["vasp_magnetic_settings"] == {"ISPIN": "2"}
    cr = next(phase for phase in proposal.phases if phase["element"] == "Cr")
    assert cr["magnetic_state"] == "antiferromagnetic"
    assert cr["vasp_magnetic_settings"] == {"ISPIN": "2", "MAGMOM": "1 -1"}
    assert any("spin-density wave" in note for note in proposal.review)


def test_cr_reference_is_a_declared_two_site_antiferromagnet(tmp_path):
    from matsim_agents.active_learning.elemental_dft import _site_magnetism
    from matsim_agents.backends.dft.qe_relax import QESettings, write_pw_input

    campaign = _w_campaign(tmp_path)
    plan = ElementalReferencePlan.from_yaml(campaign / "elemental-phases.yaml")
    cr = next(phase for phase in plan.phases if phase.element == "Cr")
    assert cr.phase_id == "cr-bcc-afm"
    assert cr.qe_site_starting_magnetization == (0.5, -0.5)
    from ase.io import read

    atoms, config = _site_magnetism(cr, read(cr.structure_path))
    assert len(atoms) == 2 and config.mode == "antiferromagnetic"
    (tmp_path / "Cr.UPF").write_text("")
    settings = QESettings(pseudo_dir=str(tmp_path), pseudopotentials={"Cr": "Cr.UPF"})
    text = Path(
        write_pw_input(atoms, settings, str(tmp_path / "pw.in"), magnetic_config=config)
    ).read_text()
    entries = dict(
        line.replace(" ", "").replace("d", "e").split("=", 1)
        for line in text.splitlines()
        if "=" in line
    )
    assert entries["nspin"] == "2" and entries["ntyp"] == "2"
    assert float(entries["starting_magnetization(1)"]) == 0.5
    assert float(entries["starting_magnetization(2)"]) == -0.5


def test_site_magnetization_must_match_atoms_and_excludes_species_settings():
    from ase.build import bulk as build

    from matsim_agents.active_learning.elemental_dft import ElementalPhaseSpec, _site_magnetism

    base = {
        "phase_id": "cr",
        "element": "Cr",
        "structure_path": "x.extxyz",
        "kind": "bulk",
        "geometry_status": "verified_converged",
        "magnetic_state": "antiferromagnetic",
    }
    with pytest.raises(ValueError, match="replaces"):
        ElementalPhaseSpec(
            **base,
            qe_site_starting_magnetization=(0.5, -0.5),
            qe_magnetic_settings={"nspin": 2},
        )
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        ElementalPhaseSpec(**base, qe_site_starting_magnetization=(1.5, -0.5))
    phase = ElementalPhaseSpec(**base, qe_site_starting_magnetization=(0.5, -0.5))
    with pytest.raises(ValueError, match="2 site magnetizations for 1 atoms"):
        _site_magnetism(phase, build("Cr"))


def test_reference_kpts_match_a_supercell_mesh_spacing():
    from matsim_agents.active_learning.campaign_init import kpts_for_spacing, kspacing_of_mesh

    spacing = kspacing_of_mesh([3.16 * 8] * 3, [3, 3, 3])
    assert kpts_for_spacing(bulk("W", "bcc", a=3.16 * 8, cubic=True), spacing) == [3, 3, 3]
    assert kpts_for_spacing(bulk("W", "bcc", a=3.16, cubic=True), spacing) == [24, 24, 24]
    proposal = propose_reference_phases(
        {"W", "H"}, backend="qe", stress_tolerance_eV_per_A3=0.01, reference_kspacing_inv_A=spacing
    )
    by_element = {phase["element"]: phase for phase in proposal.phases}
    assert by_element["H"]["kpts"] == [1, 1, 1]
    from matsim_agents.backends.dft.qe_relax import _fortran

    assert _fortran("plain") == "'plain'" and _fortran("'all'") == "'all'"
    helium = propose_reference_phases({"He"}, backend="qe", stress_tolerance_eV_per_A3=0.01)
    assert helium.phases[0]["phase_id"] == "he-atom" and not helium.unresolved
    assert min(by_element["W"]["kpts"]) >= 24
    with pytest.raises(ValueError, match="not both"):
        propose_reference_phases(
            {"W"},
            backend="qe",
            stress_tolerance_eV_per_A3=0.01,
            reference_kpts=(2, 2, 2),
            reference_kspacing_inv_A=spacing,
        )


def test_dft_settings_merge_into_the_generated_block(tmp_path):
    write_campaign(
        tmp_path / "c",
        mlip_config=UMA,
        dft_backend="qe",
        compositions=["W15Ta"],
        ecutwfc_ry=80,
        ecutrho_ry=640,
        kpts=(3, 3, 3),
        relaxation_force_tolerance_eV_per_A=0.005,
        dft_settings={"degauss_ry": 0.02, "extra_electrons": {"conv_thr": 1e-9}},
        preset="frontier",
    )
    qe = yaml.safe_load((tmp_path / "c/al.yaml").read_text())["dft"]["qe"]
    assert qe["degauss_ry"] == 0.02 and qe["smearing"] == "gaussian"
    assert qe["extra_electrons"] == {"conv_thr": 1e-9}
    plan = yaml.safe_load((tmp_path / "c/elemental-phases.yaml").read_text())
    assert {p["relaxation_force_tolerance_eV_per_A"] for p in plan["phases"]} == {0.005}


def test_coverage_check_rejects_missing_element(elemental_manifest):
    manifest = elemental_manifest({"W": -10.0})
    trainer = TrainerConfig(enabled=True, validation_reference_set=manifest)
    check_reference_coverage(trainer, {"W"})
    with pytest.raises(ValueError, match=r"\['Ta'\]"):
        check_reference_coverage(trainer, {"W", "Ta"})


def test_al_run_refuses_repository_examples():
    result = CliRunner().invoke(
        app, ["al", "run", str(ROOT / "examples/active_learning/al_config.example.yaml")]
    )
    assert result.exit_code == 2
    output = " ".join(result.output.split())
    assert "init-campaign" in output
    assert "--allow-template" in output


def test_deployment_jobs_opt_in_to_example_configs():
    jobs = sorted(ROOT.glob("deployments/*/jobs/job-a*-*.sh"))
    for job in jobs:
        text = job.read_text()
        if "examples/paper_cases/" in text and "al run" in text:
            assert "export MATSIM_ALLOW_TEMPLATE_CONFIG=1" in text, job
