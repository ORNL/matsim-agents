from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk

from matsim_agents.discovery.magnetism import generators as gen
from matsim_agents.discovery.neb_contracts import (
    MigrationMechanism,
    NebTransition,
    build_correspondence,
    endpoint_from_atoms,
)
from matsim_agents.kinetics.contracts import NetworkProvenance
from matsim_agents.kinetics.gas_mechanisms import (
    GAS_MECHANISM_IDS,
    GasMechanismError,
    OpenSystemTerm,
    add_gas_transition,
    check_event_inventory,
    get_gas_mechanism,
    inventory_of,
    prepare_gas_neb,
)
from matsim_agents.kinetics.graph import TransitionNetwork
from matsim_agents.kinetics.transitions import get_mechanism

N_SITES = 16
SPEC_IDS = [
    "hydrogen_migration",
    "helium_migration",
    "hydrogen_trapping",
    "helium_trapping",
    "hydrogen_detrapping",
    "helium_detrapping",
    "gas_cluster_association",
    "gas_cluster_dissociation",
    "vacancy_cluster_growth",
    "vacancy_cluster_dissociation",
    "solute_vacancy_association",
    "solute_vacancy_dissociation",
    "trap_mutation",
]


def host() -> Atoms:
    atoms = bulk("W", "bcc", a=3.16, cubic=True).repeat((2, 2, 2))
    atoms.set_array("atom_id", np.arange(len(atoms)))
    return atoms


def with_he(position, species="He") -> Atoms:
    atoms = host()
    gas = Atoms(species, positions=[position])
    gas.set_array("atom_id", np.array([len(atoms)]))
    atoms += gas
    return atoms


def transition(initial: Atoms, final: Atoms, kind: str) -> NebTransition:
    common = {"parent_id": "p", "defect_type": "interstitial"}
    a = endpoint_from_atoms(initial, "s-init", "initial", **common)
    b = endpoint_from_atoms(final, "s-final", "final", **common)
    moving = (len(host()),)
    mechanism = MigrationMechanism(kind=kind, moving_atom_ids=moving)
    return NebTransition(
        transition_id=f"{kind}-t",
        initial=a,
        final=b,
        correspondence=build_correspondence(a, b),
        mechanism=mechanism,
    )


def hop(kind="helium_migration") -> NebTransition:
    return transition(with_he([1.5, 0.8, 0.0]), with_he([1.5, 0.0, 0.8]), kind)


def test_all_prompt_mechanisms_are_registered():
    assert sorted(GAS_MECHANISM_IDS) == sorted(SPEC_IDS)
    for name in SPEC_IDS:
        assert get_mechanism(name).name == name


def test_reversibility_flags():
    assert get_mechanism("hydrogen_migration").reversible
    assert not get_mechanism("helium_trapping").reversible


def test_inventory_counts_vacancies_and_gas():
    atoms = with_he([1.5, 0.8, 0.0])
    del atoms[3]
    assert dict(inventory_of(atoms.get_chemical_symbols(), N_SITES)) == {"W": 15, "He": 1, "V": 1}


def test_closed_event_conserves_inventory():
    result = check_event_inventory("helium_migration", {"W": 16, "He": 1}, {"W": 16, "He": 1})
    assert result.system == "closed" and result.delta == {}


def test_closed_event_changing_inventory_rejected():
    with pytest.raises(GasMechanismError, match="without explicit source/sink"):
        check_event_inventory("helium_migration", {"W": 16, "He": 1}, {"W": 16, "He": 2})


def test_open_event_needs_matching_terms():
    term = OpenSystemTerm(species="He", delta=1, reservoir="implantation", basis="flux=1e15")
    ok = check_event_inventory("gas_cluster_association", {"He": 2}, {"He": 3}, [term])
    assert ok.system == "open" and ok.delta == {"He": 1}
    wrong = OpenSystemTerm(species="He", delta=2, reservoir="implantation", basis="flux")
    with pytest.raises(GasMechanismError, match="do not equal"):
        check_event_inventory("gas_cluster_association", {"He": 2}, {"He": 3}, [wrong])


def test_zero_delta_term_rejected():
    with pytest.raises(ValueError):
        OpenSystemTerm(species="H", delta=0, reservoir="r", basis="b")


def test_source_state_prerequisites():
    with pytest.raises(GasMechanismError, match="needs at least 1 He"):
        check_event_inventory("helium_migration", {"W": 16}, {"W": 16})
    with pytest.raises(GasMechanismError, match="gas atoms"):
        check_event_inventory("gas_cluster_association", {"He": 1}, {"He": 1})
    with pytest.raises(GasMechanismError, match="needs at least 2 V"):
        check_event_inventory("vacancy_cluster_growth", {"V": 1}, {"V": 1})


def test_unsupported_mechanism_rejected():
    with pytest.raises(GasMechanismError, match="unsupported mechanism"):
        get_gas_mechanism("bubble_burst")
    with pytest.raises(GasMechanismError):
        check_event_inventory("bubble_burst", {}, {})


def test_add_gas_transition_is_candidate_and_reversible():
    pair = hop()
    network = TransitionNetwork(
        NetworkProvenance(network_id="n", parent_id="p", parent_ordering_id=None, synthetic=True)
    )
    forward, back = add_gas_transition(network, pair, N_SITES)
    assert forward.mechanism == "helium_migration"
    assert forward.qualification == "candidate" and forward.barrier_eV is None
    assert back.source_id == forward.target_id


def test_add_gas_transition_rejects_wrong_species_prerequisite():
    network = TransitionNetwork(
        NetworkProvenance(network_id="n", parent_id="p", parent_ordering_id=None, synthetic=True)
    )
    with pytest.raises(GasMechanismError, match="needs at least 1 H"):
        add_gas_transition(network, hop("hydrogen_migration"), N_SITES)


def test_prepare_neb_fixes_inventory():
    prep = prepare_gas_neb(hop(), N_SITES)
    assert prep.inventory == {"He": 1, "W": 16}
    assert prep.inventory_fixed and not prep.magnetic_provenance_checked
    assert any("not checked" in n for n in prep.notes)


def test_prepare_neb_rejects_open_terms():
    term = OpenSystemTerm(species="He", delta=1, reservoir="r", basis="b")
    with pytest.raises(GasMechanismError, match="fixed"):
        prepare_gas_neb(hop(), N_SITES, open_terms=[term])


def test_prepare_neb_checks_magnetic_provenance():
    atoms = with_he([1.5, 0.8, 0.0])
    config = gen.ferromagnetic(atoms, "s", {"W": 0.1})
    prep = prepare_gas_neb(hop(), N_SITES, magnetic_images=[config, config])
    assert prep.magnetic_provenance_checked and prep.magnetic_config_ids
    other = gen.ferromagnetic(atoms, "s", {"W": 0.2})
    with pytest.raises(GasMechanismError, match="magnetic provenance"):
        prepare_gas_neb(hop(), N_SITES, magnetic_images=[config, other])


def test_prepare_neb_rejects_unsupported_kind():
    with pytest.raises(GasMechanismError, match="unsupported"):
        prepare_gas_neb(hop("bubble_burst"), N_SITES)
