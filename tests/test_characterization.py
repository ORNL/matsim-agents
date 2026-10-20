"""Characterization contracts, deterministic recipes, and stability-fidelity guards."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from ase import Atoms
from ase.io import write
from pydantic import ValidationError

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

    def __init__(self, energies, *, unconverged=(), broken=()):
        self.energies = energies
        self.unconverged = set(unconverged)
        self.broken = set(broken)
        self.calls = []

    def __call__(self, cfg):
        name = Path(cfg.structure_path).name
        self.calls.append((name, cfg.mode))
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
        stage = RelaxationStageResult(
            stage="dft" if dft else "mlip",
            backend="qe" if dft else "uma",
            evidence_level=level,
            input_structure_path=cfg.structure_path,
            optimized_structure_path=str(optimized),
            energy_eV=self.energies[name],
            max_force_eV_per_A=0.001 if converged else 0.5,
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
            "dft": {"backend": "qe", "pseudo_dir": "/pseudo"},
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
            method_signature="qe-pbe-sssp-efficiency-ecut60",
            elemental_energies_eV_per_atom={"Mg": -1.0, "O": -1.0},
        ).model_dump_json()
    )
    contract = load_contract(tmp_path / "example" / "thermodynamic_stability.yaml")
    assert contract.needs_dft() and not contract.dft_approved
    assert contract.reference_energies == tmp_path / "example" / "references.json"
