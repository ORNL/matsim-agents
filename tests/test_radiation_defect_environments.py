from collections import Counter

import numpy as np
import pytest

from matsim_agents.discovery.alloys import AlloySpec, generate_alloy
from matsim_agents.discovery.radiation_defect_environments import (
    defect_environment,
    interstitial_environments,
    vacancy_environments,
    vacancy_interstitial_separation,
)
from matsim_agents.discovery.radiation_defects import (
    generate_bcc_interstitial,
    generate_frenkel_pair,
    generate_vacancy,
)

SPEC = AlloySpec(counts={"Mo": 18, "Ta": 18, "W": 18}, repeats=(3, 3, 3), seed=2)
A = SPEC.lattice_parameter
PARENT = generate_alloy(SPEC)


def test_vacancy_environment_has_bcc_shell_sizes_and_matching_composition():
    defective, config = generate_vacancy(PARENT, "p", 10)
    env = defect_environment(PARENT, defective, config, A)
    (vac,) = env.vacancies
    symbols = PARENT.get_chemical_symbols()
    assert vac.lattice_site == 10 and vac.species == symbols[10]
    assert len(vac.first_shell) == 8 and len(vac.second_shell) == 6
    assert dict(vac.first_composition) == dict(Counter(symbols[j] for j in vac.first_shell))
    assert sum(n for _, n in vac.second_composition) == 6
    assert vac.vacant_first == () and vac.vacant_second == ()
    assert env.composition_change == ((symbols[10], -1),)
    assert env.interstitials == () and env.vacancy_interstitial_separation is None


def test_interstitial_environment_reports_nearest_atoms_in_distance_order():
    defective, config = generate_bcc_interstitial(PARENT, "p", 0, "Ta", "tetrahedral", A)
    env = defect_environment(PARENT, defective, config, A)
    (inter,) = env.interstitials
    assert inter.atom_id == len(PARENT) and inter.species == "Ta"
    assert len(inter.nearest_indices) == 8
    assert list(inter.nearest_distances) == sorted(inter.nearest_distances)
    assert inter.nearest_distance == inter.nearest_distances[0]
    assert inter.coordination == 4  # ideal BCC tetrahedral site
    assert sum(n for _, n in inter.coordination_composition) == 4
    assert env.composition_change == (("Ta", 1),)
    assert env.vacancies == ()


def test_frenkel_environment_has_zero_net_change_and_a_periodic_separation():
    defective, config = generate_frenkel_pair(PARENT, "p", 5, 0, A)
    env = defect_environment(PARENT, defective, config, A)
    assert env.composition_change == ()
    assert env.parent_composition == env.defective_composition
    assert env.vacancy_interstitial_separation == pytest.approx(
        config.spec.parameters["separation"]
    )
    assert len(env.vacancies) == 1 and len(env.interstitials) == 1
    assert env.interstitials[0].atom_id == 5


def test_separation_uses_the_minimum_image_across_the_boundary():
    defective, config = generate_frenkel_pair(PARENT, "p", 5, 0, A)
    shifted = config.model_copy(
        update={
            "sites": (
                config.sites[0].model_copy(update={"position": (0.1, 0.1, 0.1)}),
                config.sites[1].model_copy(
                    update={"position": (config.cell[0][0] - 0.1, 0.1, 0.1)}
                ),
            )
        }
    )
    assert vacancy_interstitial_separation(shifted) == pytest.approx(0.2)


def test_environment_checks_reject_mismatched_inputs_and_bad_options():
    defective, config = generate_vacancy(PARENT, "p", 10)
    other = generate_alloy(
        AlloySpec(counts={"Mo": 18, "Ta": 18, "W": 18}, repeats=(3, 3, 3), seed=9)
    )
    with pytest.raises(ValueError, match="parent ordering"):
        defect_environment(other, defective, config, A)
    with pytest.raises(ValueError, match="defective has"):
        defect_environment(PARENT, PARENT, config, A)
    with pytest.raises(ValueError, match="lattice_parameter"):
        vacancy_environments(PARENT, config, 0.0)
    inter, iconfig = generate_bcc_interstitial(PARENT, "p", 0, "Ta", "octahedral", A)
    for bad in (0, True, 1.5):
        with pytest.raises(ValueError, match="n_nearest"):
            interstitial_environments(inter, iconfig, n_nearest=bad)
    with pytest.raises(ValueError, match="shell_tolerance"):
        interstitial_environments(inter, iconfig, shell_tolerance=float("nan"))
    stripped = inter.copy()
    del stripped.arrays["atom_id"]
    with pytest.raises(ValueError, match="atom_id"):
        interstitial_environments(stripped, iconfig)


def test_environments_are_deterministic():
    defective, config = generate_frenkel_pair(PARENT, "p", 5, 0, A)
    first = defect_environment(PARENT, defective, config, A)
    second = defect_environment(PARENT.copy(), defective.copy(), config, A)
    assert first == second
    assert np.isfinite(first.interstitials[0].nearest_distance)
