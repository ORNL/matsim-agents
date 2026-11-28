"""Direct tests of the NM/FM/AFM survey states (science.magnetism.survey)."""

import numpy as np
import pytest
from ase.build import bulk

from matsim_agents.science.magnetism import generators as gen
from matsim_agents.science.magnetism.survey import MagneticSurvey, survey_configurations

A = 2.88
CR = MagneticSurvey.model_validate(
    {"elements": {"Cr": {"moment_muB": 1.0, "qe_starting_magnetization": 0.1}}}
)


def _cell(n=2):
    atoms = bulk("W", "bcc", a=A, cubic=True) * (n, n, n)
    atoms.set_array("atom_id", np.arange(len(atoms), dtype=np.int64))
    return atoms


def test_parent_without_surveyed_element_gets_only_the_nonmagnetic_state():
    configs = survey_configurations(_cell(), "s", CR, nn_cutoff_A=gen.bcc_nn_cutoff(A))
    assert [c.mode for c in configs] == ["nonmagnetic"]
    assert not configs[0].spin_polarized


def test_survey_builds_three_states_with_signed_polarizations():
    atoms = _cell()
    sublattices = gen.bcc_two_sublattices(atoms, A)
    # Cr on both sublattices, W elsewhere.
    on_a = next(i for i, s in sublattices.items() if s == "A")
    on_b = next(i for i, s in sublattices.items() if s == "B")
    atoms.symbols[[on_a, on_b]] = "Cr"
    configs = survey_configurations(atoms, "s", CR, nn_cutoff_A=gen.bcc_nn_cutoff(A))
    assert [c.mode for c in configs] == ["nonmagnetic", "ferromagnetic", "antiferromagnetic"]
    fm, afm = configs[1], configs[2]
    for config in (fm, afm):
        polar = dict(zip(config.atom_ids, config.qe_starting_magnetization, strict=True))
        species = dict(zip(config.atom_ids, config.species, strict=True))
        assert all(p == 0.0 for i, p in polar.items() if species[i] == "W")
    assert dict(zip(fm.atom_ids, fm.qe_starting_magnetization, strict=True))[on_a] == 0.1
    afm_polar = dict(zip(afm.atom_ids, afm.qe_starting_magnetization, strict=True))
    assert sorted((afm_polar[on_a], afm_polar[on_b])) == [-0.1, 0.1]


def test_magnetic_atoms_on_one_sublattice_cannot_be_antiferromagnetic():
    atoms = _cell()
    sublattices = gen.bcc_two_sublattices(atoms, A)
    centres = [i for i, s in sublattices.items() if s == "B"]
    atoms.symbols[centres] = "Cr"
    with pytest.raises(ValueError, match="both sublattices"):
        survey_configurations(atoms, "s", CR, nn_cutoff_A=gen.bcc_nn_cutoff(A))
