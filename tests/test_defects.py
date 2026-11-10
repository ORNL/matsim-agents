import pytest
from pymatgen.core import Composition
from pymatgen.entries.computed_entries import ComputedEntry

from matsim_agents.discovery.defects import (
    MAGNETISM_LIMITATION,
    defect_formation_energies,
    defect_report,
    stoichiometry_change,
)


def _nbo_entries():
    # mu(Nb)=-2, mu(O)=-1 eV/atom; NbO is 1 eV/formula below the elements.
    return [
        ComputedEntry(Composition("Nb"), -2.0),
        ComputedEntry(Composition("O2"), -2.0),
        ComputedEntry(Composition("NbO"), -5.0, entry_id="NbO"),
    ]


def test_stoichiometry_change_counts_removed_and_added_atoms():
    assert stoichiometry_change(["Nb", "Nb", "O", "O"], ["Nb", "Nb", "O"]) == {"O": -1}
    assert stoichiometry_change(["Nb", "O"], ["O", "Nb"]) == {}


def test_vacancy_energy_spans_the_host_chemical_potential_range():
    report = defect_report(
        _nbo_entries(),
        "NbO",
        ["Nb", "Nb", "O", "O"],
        -10.0,
        [{"defect_id": "V_O", "symbols": ["Nb", "Nb", "O"], "energy_eV": -8.5}],
    )
    assert report["status"] == "evaluated"
    assert MAGNETISM_LIMITATION in report["limitations"]
    vacancy = report["defects"][0]
    # Nb-rich: mu_O = -5 - (-2) = -3; O-rich: mu_O = -1.
    assert vacancy["formation_energy_min_eV"] == pytest.approx(-1.5)
    assert vacancy["formation_energy_max_eV"] == pytest.approx(0.5)


def test_stoichiometry_conserving_defect_is_independent_of_chemical_potentials():
    energies = defect_formation_energies(
        ["Nb", "O"],
        -5.0,
        ["O", "Nb"],
        -4.2,
        {"a": {"Nb": -2.0, "O": -3.0}, "b": {"Nb": -4.0, "O": -1.0}},
    )
    assert energies == {"a": pytest.approx(0.8), "b": pytest.approx(0.8)}


def test_host_above_hull_has_no_chemical_potentials():
    report = defect_report(
        _nbo_entries(),
        "NbO",
        ["Nb", "O"],
        -4.0,
        [{"defect_id": "V_O", "symbols": ["Nb"], "energy_eV": -2.0}],
    )
    assert report["status"] == "host_not_on_hull"
    assert report["defects"] == []
