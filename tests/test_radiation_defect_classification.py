import numpy as np
import pytest

from matsim_agents.discovery.alloys import AlloySpec, generate_alloy
from matsim_agents.discovery.radiation_defect_classification import classify_defect_outcome
from matsim_agents.discovery.radiation_defects import (
    generate_bcc_interstitial,
    generate_dumbbell_interstitial,
    generate_frenkel_pair,
    generate_vacancy,
)

SPEC = AlloySpec(counts={"Mo": 18, "Ta": 18, "W": 18}, repeats=(3, 3, 3), seed=2)
A = SPEC.lattice_parameter
PARENT = generate_alloy(SPEC)
N = len(PARENT)


def _classify(relaxed, config, converged=True, **kwargs):
    return classify_defect_outcome(
        PARENT, relaxed, config, A, relaxation_converged=converged, **kwargs
    )


def test_unrelaxed_vacancy_and_interstitial_persist():
    defective, config = generate_vacancy(PARENT, "p", 10)
    outcome = _classify(defective, config)
    assert outcome.classification == "persisted" and outcome.relaxation_converged is True
    assert [s.lattice_site for s in outcome.final_sites] == [10]
    assert outcome.config_id == config.config_id
    inter, config = generate_bcc_interstitial(PARENT, "p", 0, "Ta", "tetrahedral", A)
    assert _classify(inter, config).classification == "persisted"


def test_frenkel_pair_persists_when_separated_and_recombines_when_restored():
    frenkel, config = generate_frenkel_pair(PARENT, "p", 5, 0, A)
    outcome = _classify(frenkel, config)
    assert outcome.classification == "persisted"
    assert sorted(s.role for s in outcome.final_sites) == ["interstitial", "vacancy"]
    restored = PARENT.copy()
    restored.set_array("atom_id", np.arange(N))
    outcome = _classify(restored, config)
    assert outcome.classification == "recombined_restored" and outcome.final_sites == ()


def test_recombination_onto_a_different_species_is_an_antisite():
    frenkel, config = generate_frenkel_pair(PARENT, "p", 5, 0, A)
    other = next(k for k, s in enumerate(PARENT.symbols) if s != PARENT.symbols[5])
    swapped = PARENT.copy()
    symbols = list(PARENT.symbols)
    symbols[5], symbols[other] = symbols[other], symbols[5]
    swapped.set_chemical_symbols(symbols)
    # The moved atom's lattice site is refilled by another species in this synthetic structure.
    outcome = _classify(swapped, config)
    assert outcome.classification == "recombined_antisite"
    assert outcome.notes and "antisites" in outcome.notes[0]


def test_migrated_vacancy_is_transformed():
    _, config = generate_vacancy(PARENT, "p", 10)
    same = next(k for k, s in enumerate(PARENT.symbols) if s == PARENT.symbols[10] and k != 10)
    migrated = PARENT.copy()
    del migrated[same]
    # The vacancy now sits at another site of the same species; the composition is unchanged.
    outcome = _classify(migrated, config)
    assert outcome.classification == "transformed"
    assert [s.lattice_site for s in outcome.final_sites] == [same]


def test_extra_defect_pair_makes_a_vacancy_configuration_transformed():
    _, config = generate_vacancy(PARENT, "p", 10)
    relaxed, _ = generate_frenkel_pair(PARENT, "p", 20, 0, A)
    del relaxed[10]
    outcome = _classify(relaxed, config)
    assert outcome.classification == "transformed"
    assert sorted(s.role for s in outcome.final_sites) == ["interstitial", "vacancy", "vacancy"]


@pytest.mark.parametrize("converged", [False, None])
def test_unconverged_or_unknown_relaxation_is_always_unresolved(converged):
    defective, config = generate_vacancy(PARENT, "p", 10)
    outcome = _classify(defective, config, converged=converged)
    assert outcome.classification == "unresolved" and outcome.final_sites == ()
    assert outcome.relaxation_converged is converged and "relaxation" in outcome.notes[0]


def test_inconsistent_evidence_is_unresolved_with_a_reason():
    defective, config = generate_vacancy(PARENT, "p", 10)
    cases = {
        "atoms": PARENT.copy(),
        "composition": defective.copy(),
        "non-finite": defective.copy(),
        "cell changed": defective.copy(),
    }
    cases["composition"].set_chemical_symbols(
        ["Mo" if s != "Mo" else "W" for s in defective.symbols]
    )
    cases["non-finite"].positions[0, 0] = np.nan
    cases["cell changed"].set_cell(defective.cell.array * 1.02, scale_atoms=True)
    for reason, relaxed in cases.items():
        outcome = _classify(relaxed, config)
        assert outcome.classification == "unresolved", reason
        assert reason in outcome.notes[-1], reason


def test_ambiguous_and_conflicting_atoms_are_unresolved_not_forced():
    defective, config = generate_vacancy(PARENT, "p", 10)
    ambiguous = defective.copy()
    ambiguous.positions[0] = 0.5 * (PARENT.positions[0] + PARENT.positions[1])
    outcome = _classify(ambiguous, config, site_tolerance=0.5)
    assert outcome.classification == "unresolved" and "ambiguous" in outcome.notes[-1]
    clash = defective.copy()
    clash.positions[1] = PARENT.positions[0] + 0.05
    outcome = _classify(clash, config)
    assert outcome.classification == "unresolved" and "share lattice sites" in outcome.notes[-1]


def test_split_interstitial_is_conservatively_unresolved():
    dumbbell, config = generate_dumbbell_interstitial(PARENT, "p", 0, "Ta", "110", A)
    outcome = _classify(dumbbell, config)
    assert outcome.classification == "unresolved"


def test_classification_is_deterministic_and_rejects_foreign_inputs():
    defective, config = generate_vacancy(PARENT, "p", 10)
    assert _classify(defective, config) == _classify(defective.copy(), config)
    other = generate_alloy(
        AlloySpec(counts={"Mo": 18, "Ta": 18, "W": 18}, repeats=(3, 3, 3), seed=9)
    )
    with pytest.raises(ValueError, match="parent ordering"):
        classify_defect_outcome(other, defective, config, A, relaxation_converged=True)
    open_cell = defective.copy()
    open_cell.pbc = False
    with pytest.raises(ValueError, match="periodic"):
        _classify(open_cell, config)
