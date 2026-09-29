from pathlib import Path

import pytest
from ase import Atoms
from ase.io import write

from matsim_agents.discovery.stability import (
    RankingMode,
    ReferenceCompletenessPolicy,
    ReferenceEnergySet,
    ReferencePhaseEntry,
    score_stability,
)
from matsim_agents.orchestration.state import RelaxationResult


def _phase(phase_id: str, formula: str, energy: float) -> ReferencePhaseEntry:
    return ReferencePhaseEntry(
        phase_id=phase_id,
        formula=formula,
        formation_energy_eV_per_atom=energy,
        method_signature="pbe-v1",
        backend="qe",
    )


def test_reference_registry_preserves_polymorphs_and_audits_complete_coverage() -> None:
    references = ReferenceEnergySet(
        identifier="nb-ta-o-pbe-v1",
        method_signature="pbe-v1",
        backend="qe",
        elemental_energies_eV_per_atom={"Nb": 0.0, "Ta": 0.0, "O": 0.0},
        phase_entries=[
            _phase("NbO2-rutile", "NbO2", -2.0),
            _phase("NbO2-distorted", "NbO2", -2.1),
            _phase("TaO2-rutile", "TaO2", -1.8),
            _phase("NbTa-b2", "NbTa", -0.1),
            _phase("NbTaO4-mixed", "NbTaO4", -1.5),
        ],
        completeness_policy=ReferenceCompletenessPolicy(
            required_formulas=["NbO2", "TaO2", "NbTaO4"],
            require_binary_subsystems=True,
            require_ternary_competitor=True,
        ),
    )

    assert [record[0] for record in references.competing_phase_records()][0:2] == [
        "NbO2-rutile",
        "NbO2-distorted",
    ]
    audit = references.audit_completeness(["Nb", "Ta", "O"])
    assert audit.provisional is False
    assert audit.missing_binary_subsystems == []


def test_reference_registry_rejects_incompatible_phase_method() -> None:
    with pytest.raises(ValueError, match="method signature"):
        ReferenceEnergySet(
            identifier="nb-o-pbe-v1",
            method_signature="pbe-v1",
            backend="qe",
            elemental_energies_eV_per_atom={"Nb": 0.0, "O": 0.0},
            phase_entries=[
                ReferencePhaseEntry(
                    phase_id="NbO2-other-method",
                    formula="NbO2",
                    formation_energy_eV_per_atom=-2.0,
                    method_signature="scan-v1",
                    backend="qe",
                )
            ],
        )


def test_reference_audit_identifies_missing_subsystems_and_required_phases() -> None:
    references = ReferenceEnergySet(
        identifier="nb-ta-o-pbe-v1",
        method_signature="pbe-v1",
        elemental_energies_eV_per_atom={"Nb": 0.0, "Ta": 0.0, "O": 0.0},
        phase_entries=[_phase("NbO2-rutile", "NbO2", -2.0)],
        completeness_policy=ReferenceCompletenessPolicy(
            required_formulas=["NbO2", "Ta2O5"],
            require_ternary_competitor=True,
        ),
    )

    audit = references.audit_completeness(["Nb", "Ta", "O"])
    assert audit.provisional is True
    assert audit.missing_required_formulas == ["Ta2O5"]
    assert audit.missing_binary_subsystems == ["Nb-Ta", "O-Ta"]
    assert audit.missing_ternary_competitor is True


def test_hull_decomposition_retains_reference_polymorph_identity(tmp_path: Path) -> None:
    structure = tmp_path / "NbO2.extxyz"
    write(structure, Atoms(["Nb", "O", "O"], positions=[[0, 0, 0], [1, 0, 0], [2, 0, 0]]))
    relaxation = RelaxationResult(
        structure_path=str(structure),
        optimized_structure_path=str(structure),
        trajectory_path="",
        log_csv_path="",
        final_energy_eV=-5.7,
        final_max_force_eV_per_A=0.01,
        num_steps=1,
        converged=True,
    )
    references = ReferenceEnergySet(
        identifier="nb-o-pbe-v1",
        method_signature="pbe-v1",
        backend="qe",
        elemental_energies_eV_per_atom={"Nb": 0.0, "O": 0.0},
        phase_entries=[
            _phase("NbO2-ground", "NbO2", -2.0),
            _phase("NbO2-metastable", "NbO2", -1.8),
        ],
    )

    report = score_stability(
        "NbO2",
        [relaxation],
        ranking_mode=RankingMode.CONVEX_HULL,
        reference_energies=references,
        method_signature="pbe-v1",
    )

    assert report.ground_state.energy_above_hull_eV_per_atom == pytest.approx(0.1)
    assert report.ground_state.decomposition == {"NbO2-ground": pytest.approx(1.0)}


def test_degeneracy_tolerance_is_user_configurable(tmp_path: Path) -> None:
    structures = [tmp_path / "phase-a.extxyz", tmp_path / "phase-b.extxyz"]
    for structure in structures:
        write(structure, Atoms(["Nb"], positions=[[0, 0, 0]]))
    relaxations = [
        RelaxationResult(
            structure_path=str(structure),
            optimized_structure_path=str(structure),
            trajectory_path="",
            log_csv_path="",
            final_energy_eV=energy,
            final_max_force_eV_per_A=0.01,
            num_steps=1,
            converged=True,
        )
        for structure, energy in zip(structures, [-1.0, -0.985], strict=True)
    ]

    default_report = score_stability("Nb", relaxations)
    wider_report = score_stability(
        "Nb",
        relaxations,
        degeneracy_tol_eV_per_atom=0.02,
    )

    assert default_report.chemically_stable_proxy is True
    assert wider_report.chemically_stable_proxy is False
    assert "**Energetically near-degenerate polymorphs within 20 meV/atom**" in (
        wider_report.summary
    )
    assert wider_report.degeneracy_tolerance_eV_per_atom == 0.02
    assert wider_report.degeneracy_reference_structure_path == str(structures[0])
    assert wider_report.near_degenerate_structure_paths == [str(structures[1])]
    assert f"relative to the lowest-energy eligible polymorph, {structures[0]}" in (
        wider_report.summary
    )
