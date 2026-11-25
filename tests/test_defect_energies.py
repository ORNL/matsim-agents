import pytest

from matsim_agents.discovery.alloys import AlloySpec, generate_alloy
from matsim_agents.discovery.bcc_defects import generate_bcc_interstitial, generate_frenkel_pair
from matsim_agents.discovery.defect_classification import classify_defect_outcome
from matsim_agents.discovery.defect_energies import classified_defect_formation_energy
from matsim_agents.discovery.point_defects import generate_vacancy

SPEC = AlloySpec(counts={"Mo": 18, "Ta": 18, "W": 18}, repeats=(3, 3, 3), seed=2)
A = SPEC.lattice_parameter
PARENT = generate_alloy(SPEC)
MU = {
    "v1": {"Mo": -10.0, "Ta": -11.0, "W": -12.0},
    "v2": {"Mo": -10.5, "Ta": -11.5, "W": -12.5},
}


def _prepare(relaxed, config, converged=True, **kwargs):
    outcome = classify_defect_outcome(
        PARENT, relaxed, config, A, relaxation_converged=converged, **kwargs
    )
    return outcome


def _energy(relaxed, config, outcome, **overrides):
    arguments = {
        "host_energy_eV": -600.0,
        "defect_energy_eV": -585.0,
        "host_method": "mlip-x",
        "defect_method": "mlip-x",
        "chemical_potentials": MU,
    } | overrides
    return classified_defect_formation_energy(PARENT, relaxed, config, outcome, **arguments)


def test_vacancy_energy_subtracts_the_removed_atoms_chemical_potential_at_each_vertex():
    relaxed, config = generate_vacancy(PARENT, "p", 10)
    species = PARENT.symbols[10]
    outcome = _prepare(relaxed, config)
    result = _energy(relaxed, config, outcome)
    # E_f = E_d - E_h - (-1) * mu = 15 + mu  (units eV)
    expected = {v: 15.0 + mu[species] for v, mu in MU.items()}
    assert dict(result.formation_energy_eV_by_vertex) == pytest.approx(expected)
    assert result.stoichiometry_change == ((species, -1),)
    assert not result.stoichiometry_conserving
    assert result.formation_energy_min_eV == pytest.approx(min(expected.values()))
    assert result.formation_energy_max_eV == pytest.approx(max(expected.values()))
    assert result.classification == "persisted" and result.config_id == config.config_id
    assert any("Magnetism" in text for text in result.limitations)
    assert any("neutral" in text for text in result.limitations)


def test_interstitial_energy_adds_the_inserted_atoms_chemical_potential():
    relaxed, config = generate_bcc_interstitial(PARENT, "p", 0, "Ta", "tetrahedral", A)
    result = _energy(relaxed, config, _prepare(relaxed, config), defect_energy_eV=-590.0)
    expected = {v: 10.0 - mu["Ta"] for v, mu in MU.items()}
    assert dict(result.formation_energy_eV_by_vertex) == pytest.approx(expected)
    assert result.stoichiometry_change == (("Ta", 1),)


def test_frenkel_pair_is_stoichiometry_conserving_and_vertex_independent():
    relaxed, config = generate_frenkel_pair(PARENT, "p", 5, 0, A)
    result = _energy(relaxed, config, _prepare(relaxed, config), defect_energy_eV=-596.5)
    assert result.stoichiometry_conserving and result.stoichiometry_change == ()
    values = [value for _, value in result.formation_energy_eV_by_vertex]
    assert values == pytest.approx([3.5, 3.5])
    assert result.formation_energy_min_eV == result.formation_energy_max_eV == pytest.approx(3.5)


def test_unconverged_unresolved_or_restored_outcomes_are_rejected():
    relaxed, config = generate_vacancy(PARENT, "p", 10)
    for converged, match in ((False, "converge"), (None, "converge")):
        outcome = _prepare(relaxed, config, converged=converged)
        with pytest.raises(ValueError, match=match):
            _energy(relaxed, config, outcome)
    clash = relaxed.copy()
    clash.positions[1] = PARENT.positions[0] + 0.05
    outcome = _prepare(clash, config)
    assert outcome.classification == "unresolved"
    with pytest.raises(ValueError, match="unresolved"):
        _energy(clash, config, outcome)
    frenkel, fconfig = generate_frenkel_pair(PARENT, "p", 5, 0, A)
    restored = PARENT.copy()
    outcome = _prepare(restored, fconfig)
    assert outcome.classification == "recombined_restored"
    with pytest.raises(ValueError, match="no defect to score"):
        _energy(restored, fconfig, outcome)


def test_evidence_from_another_configuration_structure_or_method_is_rejected():
    relaxed, config = generate_vacancy(PARENT, "p", 10)
    other_relaxed, other_config = generate_vacancy(PARENT, "p", 11)
    outcome = _prepare(relaxed, config)
    with pytest.raises(ValueError, match="different defect configuration"):
        _energy(relaxed, other_config, outcome)
    with pytest.raises(ValueError, match="does not match the classified outcome"):
        _energy(other_relaxed, config, outcome)
    with pytest.raises(ValueError, match="different methods"):
        _energy(relaxed, config, outcome, defect_method="dft-y")
    with pytest.raises(ValueError, match="method"):
        _energy(relaxed, config, outcome, host_method="", defect_method="")


def test_missing_or_nonfinite_references_and_energies_are_rejected():
    relaxed, config = generate_vacancy(PARENT, "p", 10)
    outcome = _prepare(relaxed, config)
    species = PARENT.symbols[10]
    incomplete = {"v1": {k: v for k, v in MU["v1"].items() if k != species}}
    with pytest.raises(ValueError, match="no chemical potential"):
        _energy(relaxed, config, outcome, chemical_potentials=incomplete)
    with pytest.raises(ValueError, match="at least one"):
        _energy(relaxed, config, outcome, chemical_potentials={})
    for bad in (float("nan"), float("inf"), True):
        with pytest.raises(ValueError, match="finite"):
            _energy(relaxed, config, outcome, defect_energy_eV=bad)
        with pytest.raises(ValueError, match="finite"):
            _energy(relaxed, config, outcome, host_energy_eV=bad)
    with pytest.raises(ValueError, match="finite"):
        _energy(
            relaxed,
            config,
            outcome,
            chemical_potentials={"v1": {**MU["v1"], species: float("nan")}},
        )


def test_energy_evaluation_is_deterministic():
    relaxed, config = generate_vacancy(PARENT, "p", 10)
    outcome = _prepare(relaxed, config)
    assert _energy(relaxed, config, outcome) == _energy(relaxed.copy(), config, outcome)
