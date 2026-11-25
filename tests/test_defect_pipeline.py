"""A3 integration: generate, describe, map, classify and score defects end to end.

Synthetic fixtures only (unrelaxed structures with a declared converged flag, made-up energies).
These tests check the software contracts; they say nothing about DFT or MLIP qualification.
"""

import pytest

from matsim_agents.science.defects.bcc import (
    generate_bcc_interstitial,
    generate_crowdion,
    generate_divacancy,
    generate_dumbbell_interstitial,
    generate_frenkel_pair,
)
from matsim_agents.science.defects.classification import classify_defect_outcome
from matsim_agents.science.defects.energies import classified_defect_formation_energy
from matsim_agents.science.defects.environments import defect_environment
from matsim_agents.science.defects.mapping import (
    assign_lattice_sites,
    identify_defect_sites,
)
from matsim_agents.science.defects.point_defects import generate_vacancy
from matsim_agents.science.structures.alloys import AlloySpec, generate_alloy

REFERENCE = AlloySpec(
    counts={"W": 640, "Ta": 128, "Cr": 64, "V": 96, "Hf": 32, "Mo": 64}, repeats=(8, 8, 8), seed=3
)
SMALL = AlloySpec(counts={"Mo": 18, "Ta": 18, "W": 18}, repeats=(3, 3, 3), seed=2)
MU = {"v": {"W": -12.0, "Ta": -11.0, "Cr": -9.0, "V": -9.5, "Hf": -10.0, "Mo": -10.5}}

# name -> (builder, expected classification of the unrelaxed structure)
CASES = {
    "vacancy": (lambda p, a: generate_vacancy(p, "i", 7), "persisted"),
    "divacancy-1": (
        lambda p, a: generate_divacancy(p, "i", 7, shell=1, lattice_parameter=a),
        "persisted",
    ),
    "divacancy-2": (
        lambda p, a: generate_divacancy(p, "i", 7, shell=2, lattice_parameter=a),
        "persisted",
    ),
    "tetrahedral": (
        lambda p, a: generate_bcc_interstitial(p, "i", 0, "W", "tetrahedral", a),
        "persisted",
    ),
    "octahedral": (
        lambda p, a: generate_bcc_interstitial(p, "i", 0, "W", "octahedral", a),
        "persisted",
    ),
    "frenkel": (lambda p, a: generate_frenkel_pair(p, "i", 7, 0, a), "persisted"),
    "dumbbell": (
        lambda p, a: generate_dumbbell_interstitial(
            p, "i", 0, p.get_chemical_symbols()[0], "110", a
        ),
        "persisted",
    ),
    "crowdion": (
        lambda p, a: generate_crowdion(p, "i", 0, p.get_chemical_symbols()[0], a),
        "persisted",
    ),
}


@pytest.mark.parametrize("spec", [SMALL, REFERENCE], ids=["54-atom", "1024-atom"])
@pytest.mark.parametrize("name", CASES)
def test_every_generator_flows_through_environment_mapping_classification(spec, name):
    parent = generate_alloy(spec)
    a = spec.lattice_parameter
    builder, expected = CASES[name]
    defective, config = builder(parent, a)
    assert config.n_parent_atoms == len(parent) and config.n_atoms == len(defective)
    assert config.geometry_fingerprint

    environment = defect_environment(parent, defective, config, a)
    assert environment.config_id == config.config_id
    assert len(environment.vacancies) == sum(s.role == "vacancy" for s in config.sites)
    assert len(environment.interstitials) == sum(s.role == "interstitial" for s in config.sites)

    assignment = assign_lattice_sites(parent, defective, a)
    found = identify_defect_sites(parent, defective, assignment)
    outcome = classify_defect_outcome(parent, defective, config, a, relaxation_converged=True)
    assert outcome.classification == expected
    assert found.resolved and not found.antisites
    assert {s.lattice_site for s in found.vacancies} == {
        s.lattice_site for s in config.sites if s.role == "vacancy"
    }
    assert len(found.interstitials) == sum(s.role == "interstitial" for s in config.sites)

    again = classify_defect_outcome(parent, defective, config, a, relaxation_converged=True)
    assert again == outcome
    assert (
        classify_defect_outcome(
            parent, defective, config, a, relaxation_converged=False
        ).classification
        == "unresolved"
    )


@pytest.mark.parametrize("name", ["vacancy", "divacancy-1", "tetrahedral", "frenkel"])
def test_energy_interface_uses_the_classified_outcome_on_the_reference_cell(name):
    parent = generate_alloy(REFERENCE)
    a = REFERENCE.lattice_parameter
    defective, config = CASES[name][0](parent, a)
    outcome = classify_defect_outcome(parent, defective, config, a, relaxation_converged=True)
    result = classified_defect_formation_energy(
        parent,
        defective,
        config,
        outcome,
        host_energy_eV=-1000.0,
        defect_energy_eV=-990.0,
        host_method="synthetic",
        defect_method="synthetic",
        chemical_potentials=MU,
    )
    change = dict(result.stoichiometry_change)
    expected = 10.0 - sum(n * MU["v"][element] for element, n in change.items())
    assert result.formation_energy_min_eV == pytest.approx(expected)
    assert result.stoichiometry_conserving == (name == "frenkel")


def test_dumbbell_with_foreign_species_is_unresolved_and_never_scored():
    parent = generate_alloy(SMALL)
    a = SMALL.lattice_parameter
    host = parent.get_chemical_symbols()[0]
    foreign = next(sym for sym in parent.get_chemical_symbols() if sym != host)
    defective, config = generate_dumbbell_interstitial(parent, "i", 0, foreign, "110", a)
    outcome = classify_defect_outcome(parent, defective, config, a, relaxation_converged=True)
    assert outcome.classification == "unresolved"
    with pytest.raises(ValueError, match="unresolved"):
        classified_defect_formation_energy(
            parent,
            defective,
            config,
            outcome,
            host_energy_eV=-600.0,
            defect_energy_eV=-590.0,
            host_method="synthetic",
            defect_method="synthetic",
            chemical_potentials={"v": {"Mo": -10.0, "Ta": -11.0, "W": -12.0}},
        )
