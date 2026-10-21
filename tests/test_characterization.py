"""Characterization contracts, deterministic recipes, and stability-fidelity guards."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from ase import Atoms
from ase.io import write
from pydantic import ValidationError

from matsim_agents.backends.dft.qe_relax import QESettings
from matsim_agents.campaign.characterization import (
    CharacterizationContract,
    outstanding_steps,
    plan_recipe,
    run_characterization,
)
from matsim_agents.discovery.stability import ReferenceEnergySet
from matsim_agents.execution.contracts import (
    CharacterizationReport,
    CharacterizationStatus,
    EvidenceLevel,
    ObservableRequirement,
)
from matsim_agents.workflows.relaxation import (
    RelaxationMode,
    RelaxationStageResult,
    ScientificRelaxationResult,
)

pytest.importorskip("pymatgen")
SIGNATURE = "qe-pbe-test"


def _structure(path: Path, symbols: str, shift: float = 0.0) -> Path:
    atoms = Atoms(symbols)
    atoms.positions = [[i * 2.0 + shift, 0, 0] for i in range(len(atoms))]
    atoms.cell = [8, 8, 8]
    atoms.pbc = True
    write(path, atoms)
    return path


class FakeRelax:
    """Stands in for run_relaxation: energies per structure, and optional failures."""

    def __init__(self, energies, *, unconverged=(), broken=(), silent=(), level=None):
        self.energies = energies
        self.level = level
        self.silent = set(silent)
        self.unconverged = set(unconverged)
        self.broken = set(broken)
        self.calls = []
        self.geometries = []

    def __call__(self, cfg):
        name = Path(cfg.structure_path).name
        self.calls.append((name, cfg.mode))
        self.geometries.append(cfg.geometry)
        if name in self.broken:
            raise FileNotFoundError("pw.x not found")
        out = Path(cfg.output_root)
        out.mkdir(parents=True, exist_ok=True)
        optimized = out / f"relaxed-{name}"
        optimized.write_text(Path(cfg.structure_path).read_text())
        dft = cfg.mode == RelaxationMode.DFT
        converged = name not in self.unconverged
        level = (
            (EvidenceLevel.CONVERGED_DFT if converged else EvidenceLevel.LOW_FIDELITY_DFT)
            if dft
            else EvidenceLevel.MLIP_RELAXATION
        )
        level = self.level or level
        stage = RelaxationStageResult(
            stage="dft" if dft else "mlip",
            backend="qe" if dft else "uma",
            evidence_level=level,
            input_structure_path=cfg.structure_path,
            optimized_structure_path=str(optimized),
            energy_eV=None if name in self.silent else self.energies[name],
            max_force_eV_per_A=None if name in self.silent else (0.001 if converged else 0.5),
            steps=10,
            converged=converged,
        )
        return ScientificRelaxationResult(
            run_id=f"run-{name}",
            run_directory=str(out),
            mode=cfg.mode,
            status="complete" if converged else "partial",
            stages=[stage],
        )


@pytest.fixture
def system(tmp_path):
    a = _structure(tmp_path / "NbO2-a.extxyz", "NbOO")
    b = _structure(tmp_path / "NbO2-b.extxyz", "NbOO", shift=0.3)
    references = ReferenceEnergySet(
        identifier="nb-o",
        method_signature=SIGNATURE,
        elemental_energies_eV_per_atom={"Nb": -10.0, "O": -5.0},
        competing_phases={"NbO": -0.5},
    )
    path = tmp_path / "references.json"
    path.write_text(references.model_dump_json())
    return tmp_path, [a, b], path


def _contract(tmp_path, structures, recipe, **extra):
    return CharacterizationContract.model_validate(
        {
            "contract_id": f"{recipe}-test",
            "objective": "Is NbO2 thermodynamically stable?",
            "formula": "NbO2",
            "structures": [str(p) for p in structures],
            "recipe": recipe,
            "output_root": str(tmp_path / "runs"),
            **extra,
        }
    )


def _dft(references, **extra):
    return {
        "methods": {
            "dft": {
                "backend": "qe",
                "pseudo_dir": "/pseudo",
                "settings": {
                    "ecutwfc_ry": 60,
                    "ecutrho_ry": 480,
                    "occupations": "smearing",
                    "pseudopotentials": {"Nb": "Nb.UPF", "O": "O.UPF"},
                },
            },
            "dft_method_signature": SIGNATURE,
        },
        "reference_energies": str(references),
        "dft_approved": True,
        **extra,
    }


def test_complete_references_give_a_justified_stability_answer(system):
    tmp_path, structures, references = system
    # NbO2 at -20.3 eV (3 atoms): formation -0.433 eV/atom, above the NbO line? The
    # Nb-O hull through NbO (-0.5) puts NbO2 (x_O = 2/3) above it.
    relax = FakeRelax({"NbO2-a.extxyz": -21.3, "NbO2-b.extxyz": -21.0})
    contract = _contract(tmp_path, structures, "thermodynamic_stability", **_dft(references))
    report = run_characterization(contract, relax=relax)
    assert report.status == CharacterizationStatus.SUFFICIENTLY_CHARACTERIZED
    assert all(mode == RelaxationMode.DFT for _name, mode in relax.calls)
    hull = next(v for v in report.observables if v.name == "energy_above_hull")
    assert hull.unit == "eV/atom" and hull.evidence_level == EvidenceLevel.CONVERGED_DFT
    assert hull.structure == str(structures[0])
    assert hull.value == pytest.approx(0.0)
    assert "is on the convex hull of the declared reference phases" in report.conclusion
    assert {item.evidence_id for item in report.evidence} >= set(hull.evidence_ids)
    saved = CharacterizationReport.model_validate_json(
        (
            Path(contract.output_root)
            / "characterization-thermodynamic_stability-test"
            / "characterization-report.json"
        ).read_text()
    )
    assert saved == report


def test_a_phase_above_the_hull_reports_its_decomposition(system):
    tmp_path, structures, references = system
    # -20.5 eV: formation -0.167 eV/atom, above the NbO-O tie line (-0.333 at x_O = 2/3).
    relax = FakeRelax({"NbO2-a.extxyz": -20.5, "NbO2-b.extxyz": -20.4})
    contract = _contract(tmp_path, structures, "thermodynamic_stability", **_dft(references))
    report = run_characterization(contract, relax=relax)
    assert report.status == CharacterizationStatus.SUFFICIENTLY_CHARACTERIZED
    hull = next(v for v in report.observables if v.name == "energy_above_hull")
    assert hull.value == pytest.approx(1 / 6, abs=1e-6)
    assert "decomposes to NbO, O" in report.conclusion or "decomposes to O, NbO" in (
        report.conclusion
    )


def test_missing_competing_phases_are_inconclusive_without_spending_dft(system):
    tmp_path, structures, _references = system
    sparse = tmp_path / "sparse.json"
    sparse.write_text(
        ReferenceEnergySet(
            identifier="nb-o-sparse",
            method_signature=SIGNATURE,
            elemental_energies_eV_per_atom={"Nb": -10.0, "O": -5.0},
        ).model_dump_json()
    )
    relax = FakeRelax({})
    report = run_characterization(
        _contract(tmp_path, structures, "thermodynamic_stability", **_dft(sparse)), relax=relax
    )
    assert report.status == CharacterizationStatus.INCONCLUSIVE
    assert report.conclusion is None
    assert report.missing == ["competing phase in the Nb-O subsystem"]
    assert relax.calls == []


def test_incompatible_references_and_wrong_units_are_rejected(system):
    tmp_path, structures, references = system
    relax = FakeRelax({})
    other = _dft(references)
    other["methods"]["dft_method_signature"] = "vasp-pbe-other"
    report = run_characterization(
        _contract(tmp_path, structures, "thermodynamic_stability", **other), relax=relax
    )
    assert report.status == CharacterizationStatus.INVALID_PROTOCOL and relax.calls == []
    with pytest.raises(ValidationError, match="reported in 'eV/atom', not 'meV/atom'"):
        ObservableRequirement(name="energy_above_hull", unit="meV/atom")
    wrong = _structure(tmp_path / "TaO2.extxyz", "TaOO")
    report = run_characterization(
        _contract(tmp_path, [wrong], "relaxation", contract_id="wrong"), relax=relax
    )
    assert report.status == CharacterizationStatus.INVALID_PROTOCOL
    assert "is not NbO2" in report.missing[0]


def test_mlip_evidence_can_never_establish_thermodynamic_stability(system):
    tmp_path, structures, references = system
    weak = [
        {
            "name": "energy_above_hull",
            "unit": "eV/atom",
            "minimum_evidence_level": "mlip_relaxation",
        }
    ]
    with pytest.raises(ValidationError, match="needs converged DFT"):
        _contract(
            tmp_path, structures, "thermodynamic_stability", observables=weak, **_dft(references)
        )
    # Unconverged DFT is numerical evidence of nothing: inconclusive, never "unstable".
    relax = FakeRelax(
        {"NbO2-a.extxyz": -21.3, "NbO2-b.extxyz": -21.0}, unconverged={"NbO2-a.extxyz"}
    )
    report = run_characterization(
        _contract(tmp_path, structures, "thermodynamic_stability", **_dft(references)),
        relax=relax,
    )
    assert report.status == CharacterizationStatus.INCONCLUSIVE
    assert "not evidence against the material" in report.missing[0]


def test_dft_is_planned_only_when_an_observable_needs_it(system):
    tmp_path, structures, references = system
    ranking = _contract(tmp_path, structures, "energy_ranking")
    assert {step.fidelity for step in plan_recipe(ranking) if step.operator == "relax"} == {"mlip"}
    relax = FakeRelax({"NbO2-a.extxyz": -21.3, "NbO2-b.extxyz": -21.0})
    report = run_characterization(ranking, relax=relax)
    assert report.status == CharacterizationStatus.SUFFICIENTLY_CHARACTERIZED
    assert {mode for _name, mode in relax.calls} == {RelaxationMode.MLIP}
    assert "not a" not in report.conclusion and "NbO2-a" in report.conclusion
    assert any("does not establish formation energy" in item for item in report.limitations)
    stability = _contract(tmp_path, structures, "thermodynamic_stability", **_dft(references))
    assert {step.fidelity for step in plan_recipe(stability) if step.operator == "relax"} == {"dft"}


def test_near_degenerate_ranking_is_inconclusive(system):
    tmp_path, structures, _references = system
    relax = FakeRelax({"NbO2-a.extxyz": -21.0, "NbO2-b.extxyz": -21.003})
    report = run_characterization(_contract(tmp_path, structures, "energy_ranking"), relax=relax)
    assert report.status == CharacterizationStatus.INCONCLUSIVE
    assert report.recommended_next_test.startswith("Rank the near-degenerate")


def test_resume_runs_only_the_outstanding_steps(system):
    tmp_path, structures, references = system
    contract = _contract(tmp_path, structures, "thermodynamic_stability", **_dft(references))
    energies = {"NbO2-a.extxyz": -21.3, "NbO2-b.extxyz": -21.0}
    first = FakeRelax(energies, broken={"NbO2-b.extxyz"})
    report = run_characterization(contract, relax=first)
    assert report.status == CharacterizationStatus.FAILED_INFRASTRUCTURE
    assert "not evidence about the material" in report.limitations[0]
    assert [step.step_id for step in outstanding_steps(contract)] == ["relax-dft-001", "hull"]
    second = FakeRelax(energies)
    report = run_characterization(contract, relax=second)
    assert second.calls == [("NbO2-b.extxyz", RelaxationMode.DFT)]
    assert report.status == CharacterizationStatus.SUFFICIENTLY_CHARACTERIZED
    changed = contract.model_copy(update={"objective": "something else"})
    with pytest.raises(ValueError, match="holds a different contract"):
        run_characterization(changed, relax=second)


def test_dft_waits_for_approval_and_budget(system):
    tmp_path, structures, references = system
    relax = FakeRelax({"NbO2-a.extxyz": -21.3, "NbO2-b.extxyz": -21.0})
    unapproved = _contract(
        tmp_path,
        structures,
        "thermodynamic_stability",
        **_dft(references, dft_approved=False, contract_id="unapproved"),
    )
    report = run_characterization(unapproved, relax=relax)
    assert report.status == CharacterizationStatus.INCONCLUSIVE and relax.calls == []
    assert report.recommended_next_test == "Approve the required DFT calculations"
    limited = _contract(
        tmp_path,
        structures,
        "thermodynamic_stability",
        **_dft(references, budget={"max_dft_calculations": 1}, contract_id="limited"),
    )
    report = run_characterization(limited, relax=relax)
    assert len(relax.calls) == 1 and report.status == CharacterizationStatus.INCONCLUSIVE
    assert "beyond the dft relaxation budget" in report.missing[0]


def test_a_material_can_be_characterized_without_an_optimization_objective(system):
    tmp_path, structures, _references = system
    contract = _contract(tmp_path, structures[:1], "relaxation")
    report = run_characterization(contract, relax=FakeRelax({"NbO2-a.extxyz": -21.0}))
    assert report.status == CharacterizationStatus.SUFFICIENTLY_CHARACTERIZED
    names = {value.name for value in report.observables}
    assert names == {"relaxed_structure", "residual_max_force", "energy_per_atom"}


def test_reports_only_conclude_when_characterized_and_cite_known_evidence():
    with pytest.raises(ValidationError, match="draws no conclusion"):
        CharacterizationReport(
            contract_id="c",
            contract_sha256="x",
            recipe="relaxation",
            status=CharacterizationStatus.INCONCLUSIVE,
            conclusion="stable",
            missing=["x"],
        )
    with pytest.raises(ValidationError, match="cite unknown evidence"):
        CharacterizationReport(
            contract_id="c",
            contract_sha256="x",
            recipe="relaxation",
            status=CharacterizationStatus.INCONCLUSIVE,
            missing=["x"],
            observables=[
                {
                    "name": "energy_per_atom",
                    "value": -1.0,
                    "unit": "eV/atom",
                    "evidence_level": "mlip_relaxation",
                    "evidence_ids": ["nowhere"],
                }
            ],
        )


def test_cli_reports_an_invalid_contract_without_calculating(system, tmp_path):
    from typer.testing import CliRunner

    from matsim_agents.cli import app

    _tmp, structures, references = system
    contract = _dft(references)
    contract["methods"]["dft_method_signature"] = "other"
    path = tmp_path / "contract.json"
    path.write_text(
        json.dumps(
            {
                "contract_id": "cli",
                "objective": "Is NbO2 stable?",
                "formula": "NbO2",
                "structures": [str(p) for p in structures],
                "recipe": "thermodynamic_stability",
                "output_root": str(tmp_path / "runs"),
                **contract,
            }
        )
    )
    result = CliRunner().invoke(app, ["characterize", str(path)])
    assert result.exit_code == 1
    assert '"invalid_protocol"' in result.output


EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "characterization"


def test_examples_are_valid_contracts(tmp_path):
    import shutil

    from matsim_agents.campaign.characterization import load_contract

    for name in ("relaxation", "energy_ranking"):
        contract = load_contract(EXAMPLES / f"{name}.yaml")
        assert contract.recipe == name and not contract.needs_dft()
        assert all(path.is_file() for path in contract.structures)
    # The stability example ships no reference energies: the user supplies them.
    with pytest.raises(ValidationError, match="needs a reference_energies file"):
        load_contract(EXAMPLES / "thermodynamic_stability.yaml")
    shutil.copytree(EXAMPLES, tmp_path / "example")
    (tmp_path / "example" / "references.json").write_text(
        ReferenceEnergySet(
            identifier="user-supplied",
            method_signature="qe-pbe-pslibrary-ecut60-smearing",
            elemental_energies_eV_per_atom={"Mg": -1.0, "O": -1.0},
        ).model_dump_json()
    )
    contract = load_contract(tmp_path / "example" / "thermodynamic_stability.yaml")
    assert contract.needs_dft() and not contract.dft_approved
    QESettings(pseudo_dir="/pseudo", **contract.methods.dft.settings)
    assert contract.reference_energies == tmp_path / "example" / "references.json"


def test_stability_needs_one_fully_relaxed_dft_method(system):
    tmp_path, structures, references = system
    unpinned = _dft(references)
    del unpinned["methods"]["dft"]["settings"]["ecutwfc_ry"]
    with pytest.raises(ValidationError, match=r"to fix \['ecutwfc_ry'\]"):
        _contract(tmp_path, structures, "thermodynamic_stability", **unpinned)
    typo = _dft(references)
    typo["methods"]["dft"]["settings"]["ecutwfc"] = 60
    with pytest.raises(ValidationError, match=r"\['ecutwfc'\] are not qe settings"):
        _contract(tmp_path, structures, "thermodynamic_stability", **typo)
    with pytest.raises(ValidationError, match="must relax atoms and cell"):
        _contract(
            tmp_path,
            structures,
            "thermodynamic_stability",
            geometry={"relax_atoms": True, "relax_cell": False},
            **_dft(references),
        )
    relax = FakeRelax({"NbO2-a.extxyz": -21.3, "NbO2-b.extxyz": -21.0})
    run_characterization(
        _contract(tmp_path, structures, "thermodynamic_stability", **_dft(references)),
        relax=relax,
    )
    assert all(g.relax_atoms and g.relax_cell for g in relax.geometries)


def test_a_relaxation_without_an_energy_is_an_infrastructure_failure(system):
    tmp_path, structures, _references = system
    relax = FakeRelax(
        {"NbO2-a.extxyz": -21.0}, unconverged={"NbO2-a.extxyz"}, silent={"NbO2-a.extxyz"}
    )
    contract = _contract(tmp_path, structures[:1], "relaxation")
    report = run_characterization(contract, relax=relax)
    assert report.status == CharacterizationStatus.FAILED_INFRASTRUCTURE
    assert [step.step_id for step in outstanding_steps(contract)] == ["relax-mlip-000"]


def test_changed_input_files_invalidate_the_saved_run(system):
    tmp_path, structures, references = system
    contract = _contract(tmp_path, structures, "thermodynamic_stability", **_dft(references))
    relax = FakeRelax({"NbO2-a.extxyz": -21.3, "NbO2-b.extxyz": -21.0})
    run_characterization(contract, relax=relax)
    _structure(structures[1], "NbOO", shift=0.7)
    with pytest.raises(ValueError, match="different input files"):
        run_characterization(contract, relax=relax)
    _structure(structures[1], "NbOO", shift=0.3)
    run_characterization(contract, relax=relax)
    edited = ReferenceEnergySet.model_validate_json(references.read_text())
    edited.competing_phases["NbO"] = -0.6
    references.write_text(edited.model_dump_json())
    with pytest.raises(ValueError, match="different input files"):
        run_characterization(contract, relax=relax)


def test_approval_and_budget_changes_resume_the_same_run(system):
    tmp_path, structures, references = system
    energies = {"NbO2-a.extxyz": -21.3, "NbO2-b.extxyz": -21.0}
    data = _dft(references, dft_approved=False, budget={"max_dft_calculations": 1})
    first = run_characterization(
        _contract(tmp_path, structures, "thermodynamic_stability", **data), relax=FakeRelax({})
    )
    assert first.status == CharacterizationStatus.INCONCLUSIVE
    broken = FakeRelax(energies, broken={"NbO2-a.extxyz"})
    approved = _contract(
        tmp_path, structures, "thermodynamic_stability", **{**data, "dft_approved": True}
    )
    report = run_characterization(approved, relax=broken)
    assert broken.calls == [("NbO2-a.extxyz", RelaxationMode.DFT)]
    assert report.status == CharacterizationStatus.FAILED_INFRASTRUCTURE
    # The failed attempt consumed the budget: a resume may not run it again.
    again = FakeRelax(energies)
    report = run_characterization(approved, relax=again)
    assert again.calls == [] and report.status == CharacterizationStatus.FAILED_INFRASTRUCTURE
    assert any("beyond the dft relaxation budget (1 attempted)" in m for m in report.missing)
    raised = _contract(
        tmp_path,
        structures,
        "thermodynamic_stability",
        **{**data, "dft_approved": True, "budget": {"max_dft_calculations": 3}},
    )
    report = run_characterization(raised, relax=again)
    assert len(again.calls) == 2
    assert report.status == CharacterizationStatus.SUFFICIENTLY_CHARACTERIZED


def test_the_assessment_is_saved_and_not_repeated(system, monkeypatch):
    tmp_path, structures, references = system
    contract = _contract(tmp_path, structures, "thermodynamic_stability", **_dft(references))
    report = run_characterization(
        contract, relax=FakeRelax({"NbO2-a.extxyz": -21.3, "NbO2-b.extxyz": -21.0})
    )
    assert outstanding_steps(contract) == []
    import matsim_agents.campaign.characterization as module

    def fail(*_args, **_kwargs):
        raise AssertionError("the hull was recomputed")

    monkeypatch.setattr(module, "score_stability", fail)
    assert run_characterization(contract, relax=FakeRelax({})) == report


def test_fidelity_requirements_are_enforced(system):
    tmp_path, structures, _references = system
    with pytest.raises(ValidationError, match="the highest fidelity a characterization"):
        _contract(
            tmp_path,
            structures[:1],
            "relaxation",
            observables=[
                {
                    "name": "energy_per_atom",
                    "unit": "eV/atom",
                    "minimum_evidence_level": "higher_accuracy_dft",
                }
            ],
            methods={"dft": {"backend": "qe", "pseudo_dir": "/pseudo"}},
        )
    contract = _contract(
        tmp_path,
        structures[:1],
        "relaxation",
        observables=[
            {
                "name": "energy_per_atom",
                "unit": "eV/atom",
                "minimum_evidence_level": "converged_dft",
            }
        ],
        methods={"dft": {"backend": "qe", "pseudo_dir": "/pseudo"}},
        dft_approved=True,
    )
    weak = FakeRelax({"NbO2-a.extxyz": -21.0}, level=EvidenceLevel.LOW_FIDELITY_DFT)
    report = run_characterization(contract, relax=weak)
    assert report.status == CharacterizationStatus.INCONCLUSIVE
    assert report.missing == ["energy_per_atom at converged_dft (the evidence is low_fidelity_dft)"]


@pytest.mark.parametrize(
    ("backend", "override", "conflict"),
    [
        ("qe", {"extra_control": {"calculation": "relax"}}, "extra_control.calculation"),
        ("qe", {"extra_cell": {"cell_dofree": "shape"}}, "extra_cell.cell_dofree"),
        ("vasp", {"extra_incar": {"ISIF": 2}}, "extra_incar.ISIF"),
        ("vasp", {"extra_incar": {"encut": 300}}, "extra_incar.encut"),
    ],
)
def test_raw_overrides_cannot_replace_recipe_controls(system, backend, override, conflict):
    tmp_path, structures, references = system
    data = _dft(references)
    if backend == "vasp":
        data["methods"]["dft"] = {
            "backend": "vasp",
            "potcar_dir": "/potcars",
            "settings": {"encut_ev": 520, "ismear": 0, "sigma": 0.05, "kspacing": 0.2},
        }
    data["methods"]["dft"]["settings"].update(override)
    with pytest.raises(ValidationError, match=rf"overrides \['{conflict}'\]"):
        _contract(tmp_path, structures, "thermodynamic_stability", **data)
    harmless = _dft(references)
    harmless["methods"]["dft"]["settings"]["extra_system"] = {"nspin": 1}
    _contract(tmp_path, structures, "thermodynamic_stability", **harmless)


@pytest.mark.parametrize("content", ["{not json", '{"identifier": "x"}'])
def test_an_invalid_reference_file_is_an_invalid_protocol(system, content):
    tmp_path, structures, references = system
    references.write_text(content)
    relax = FakeRelax({})
    report = run_characterization(
        _contract(tmp_path, structures, "thermodynamic_stability", **_dft(references)),
        relax=relax,
    )
    assert report.status == CharacterizationStatus.INVALID_PROTOCOL and relax.calls == []
    assert "is not one" in report.missing[0]


def test_pseudopotential_directories_are_relative_to_the_contract(system):
    import yaml

    from matsim_agents.campaign.characterization import load_contract

    tmp_path, structures, references = system
    data = {
        "contract_id": "relative",
        "objective": "Relax NbO2 with DFT.",
        "formula": "NbO2",
        "recipe": "relaxation",
        "structures": [structures[0].name],
        "observables": [
            {
                "name": "energy_per_atom",
                "unit": "eV/atom",
                "minimum_evidence_level": "converged_dft",
            }
        ],
        "methods": {"dft": {"backend": "qe", "pseudo_dir": "pseudos"}},
    }
    path = tmp_path / "contract.yaml"
    path.write_text(yaml.safe_dump(data))
    contract = load_contract(path)
    assert contract.methods.dft.pseudo_dir == str(tmp_path / "pseudos")
    assert contract.structures == [structures[0]]


def test_a_second_execution_of_the_same_run_is_refused(system):
    from matsim_agents.execution.locking import exclusive_lock

    tmp_path, structures, _references = system
    contract = _contract(tmp_path, structures[:1], "relaxation")
    run_dir = Path(contract.output_root) / "characterization-relaxation-test"
    with exclusive_lock(run_dir / ".lock"):
        relax = FakeRelax({"NbO2-a.extxyz": -21.0})
        with pytest.raises(RuntimeError, match="in use by another execution"):
            run_characterization(contract, relax=relax)
        assert relax.calls == []
    report = run_characterization(contract, relax=relax)
    assert report.status == CharacterizationStatus.SUFFICIENTLY_CHARACTERIZED


def test_unreadable_structures_and_uncovered_phases_are_invalid_protocols(system):
    tmp_path, structures, references = system
    empty = tmp_path / "empty.vasp"
    empty.write_text("")
    relax = FakeRelax({})
    report = run_characterization(
        _contract(tmp_path, [empty], "relaxation", contract_id="empty"), relax=relax
    )
    assert report.status == CharacterizationStatus.INVALID_PROTOCOL
    assert "is not a readable structure" in report.missing[0] and relax.calls == []
    broken = ReferenceEnergySet.model_validate_json(references.read_text())
    broken.competing_phases["TaO"] = -0.4
    references.write_text(broken.model_dump_json())
    report = run_characterization(
        _contract(tmp_path, structures, "thermodynamic_stability", **_dft(references)),
        relax=relax,
    )
    assert report.status == CharacterizationStatus.INVALID_PROTOCOL and relax.calls == []
    assert report.missing == [
        "elemental energies for ['Ta'], needed by competing phase formula:TaO"
    ]


def test_relative_and_absolute_contract_paths_resume_the_same_run(system, monkeypatch):
    import yaml

    from matsim_agents.campaign.characterization import load_contract

    tmp_path, structures, _references = system
    path = tmp_path / "contract.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "contract_id": "paths",
                "objective": "Relax NbO2.",
                "formula": "NbO2",
                "recipe": "relaxation",
                "structures": [structures[0].name],
                "output_root": "runs",
            }
        )
    )
    monkeypatch.chdir(tmp_path)
    relative = load_contract("contract.yaml")
    absolute = load_contract(path)
    assert relative.digest() == absolute.digest()
    run_characterization(relative, relax=FakeRelax({"NbO2-a.extxyz": -21.0}))
    again = FakeRelax({})
    run_characterization(absolute, relax=again)
    assert again.calls == []


@pytest.mark.parametrize("where", ["set", "phase", "element"])
def test_references_from_another_dft_backend_are_invalid_protocols(system, where):
    from matsim_agents.discovery.stability import ElementalReferenceEntry, ReferencePhaseEntry

    tmp_path, structures, references = system
    foreign = ReferenceEnergySet.model_validate_json(references.read_text())
    if where == "set":
        foreign.backend = "vasp"
    elif where == "phase":
        foreign.phase_entries = [
            ReferencePhaseEntry(
                phase_id="NbO-vasp",
                formula="NbO",
                formation_energy_eV_per_atom=-0.5,
                method_signature=SIGNATURE,
                backend="vasp",
            )
        ]
    else:
        foreign.elemental_entries = {
            "Nb": ElementalReferenceEntry(
                element="Nb",
                phase_id="Nb-bcc",
                reference_formula="Nb",
                energy_eV_per_atom=-10.0,
                method_signature=SIGNATURE,
                backend="vasp",
            )
        }
    references.write_text(foreign.model_dump_json())
    relax = FakeRelax({})
    report = run_characterization(
        _contract(tmp_path, structures, "thermodynamic_stability", **_dft(references)),
        relax=relax,
    )
    assert report.status == CharacterizationStatus.INVALID_PROTOCOL and relax.calls == []
    assert "uses 'vasp', the relaxations 'qe'" in report.missing[0]


def test_fractional_and_grouped_reference_formulas_are_accepted(system):
    tmp_path, structures, references = system
    extended = ReferenceEnergySet.model_validate_json(references.read_text())
    extended.competing_phases.update({"Nb0.5O": -0.3, "Nb3(O2)2": -0.2})
    references.write_text(extended.model_dump_json())
    relax = FakeRelax({"NbO2-a.extxyz": -21.3, "NbO2-b.extxyz": -21.0})
    report = run_characterization(
        _contract(tmp_path, structures, "thermodynamic_stability", **_dft(references)),
        relax=relax,
    )
    assert report.status == CharacterizationStatus.SUFFICIENTLY_CHARACTERIZED


def test_an_edited_relaxation_output_is_not_reused_on_resume(system):
    tmp_path, structures, references = system
    contract = _contract(tmp_path, structures, "thermodynamic_stability", **_dft(references))
    energies = {"NbO2-a.extxyz": -21.3, "NbO2-b.extxyz": -21.0}
    run_characterization(contract, relax=FakeRelax(energies, broken={"NbO2-b.extxyz"}))
    state = json.loads(
        (
            Path(contract.output_root)
            / "characterization-thermodynamic_stability-test"
            / "characterization-state.json"
        ).read_text()
    )
    done = state["steps"]["relax-dft-000"]
    _structure(Path(done["optimized_structure_path"]), "NbOO", shift=0.7)
    retry = FakeRelax(energies)
    with pytest.raises(ValueError, match="differs from the relaxation's output"):
        run_characterization(contract, relax=retry)
    assert retry.calls == []


def test_every_attempt_and_the_answer_enter_the_evidence_ledger(system):
    from matsim_agents.execution.ledger import (
        EvidenceLedger,
        EvidenceOutcome,
        HypothesisRecord,
        LedgerValidationError,
    )

    tmp_path, structures, references = system
    contract = _contract(tmp_path, structures, "thermodynamic_stability", **_dft(references))
    energies = {"NbO2-a.extxyz": -21.3, "NbO2-b.extxyz": -21.0}
    run_characterization(contract, relax=FakeRelax(energies, broken={"NbO2-b.extxyz"}))
    report = run_characterization(contract, relax=FakeRelax(energies))
    ledger = EvidenceLedger(
        Path(contract.output_root)
        / "characterization-thermodynamic_stability-test"
        / "evidence-ledger.jsonl"
    )
    relaxations = [e for e in ledger.evidence() if e.kind == "relaxation"]
    assert [(e.calculation_id, e.task_id, e.outcome) for e in relaxations] == [
        ("relax-dft-000", "attempt-1", EvidenceOutcome.MEASURED),
        ("relax-dft-001", "attempt-2", EvidenceOutcome.FAILED_TO_COMPUTE),
        ("relax-dft-001", "attempt-3", EvidenceOutcome.MEASURED),
    ]
    assert relaxations[0].method.method_signature == SIGNATURE
    assert relaxations[0].output_artifact_sha256 and relaxations[0].input_sha256
    [answer] = [e for e in ledger.evidence() if e.kind == "stability"]
    assert answer.parent_record_ids == [relaxations[0].evidence_id, relaxations[2].evidence_id]
    assert answer.reference_complete and answer.evidence_level == EvidenceLevel.CONVERGED_DFT
    assert {item.evidence_id for item in report.evidence} == {
        relaxations[0].evidence_id,
        relaxations[2].evidence_id,
    }
    claim = {
        "hypothesis_id": "hyp-nbo2",
        "claim_type": "thermodynamic_stability",
        "scientific_claim": "NbO2 is on the Nb-O hull",
        "status": "supported",
    }
    with pytest.raises(LedgerValidationError, match="needs evidence of kind"):
        ledger.append(HypothesisRecord(**claim, evidence_ids=[relaxations[0].evidence_id]))
    ledger.append(HypothesisRecord(**claim, evidence_ids=[answer.evidence_id]))
