"""End-to-end Phase A2 chemical-ordering workflow on small cells and a 1,024-atom check."""

from __future__ import annotations

from collections import Counter

import numpy as np

from matsim_agents.discovery.alloy_ensemble import (
    ensemble_sro_statistics,
    find_exact_duplicates,
    find_symmetry_equivalents,
    generate_alloy_ensemble,
)
from matsim_agents.discovery.alloys import (
    AlloySpec,
    SROTarget,
    first_neighbor_list,
    generate_alloy,
    generate_targeted_sro,
    second_neighbor_list,
    warren_cowley_sro,
)

SMALL = AlloySpec(counts={"Mo": 12, "Ta": 8, "W": 12}, repeats=(4, 2, 2), seed=7)
REFERENCE = AlloySpec(counts={"W": 640, "Ta": 128, "Cr": 64, "V": 96, "Hf": 32, "Mo": 64}, seed=3)
TARGETS = [SROTarget(shell=1, pair=("Mo", "W"), alpha=-0.4)]


def _workflow():
    atoms = generate_alloy(SMALL)
    sro = warren_cowley_sro(atoms, SMALL.lattice_parameter)
    optimized = generate_targeted_sro(
        atoms, SMALL.lattice_parameter, TARGETS, seed=13, max_attempts=400
    )
    ensemble = generate_alloy_ensemble(SMALL, n_configurations=4, mode="random", seed=17)
    return atoms, sro, optimized, ensemble


def test_phase_a2_chemical_ordering_workflow():
    atoms, sro, optimized, ensemble = _workflow()

    assert Counter(atoms.get_chemical_symbols()) == dict(SMALL.counts)
    assert sro.species == ("Mo", "Ta", "W")
    for shell in (sro.first_shell, sro.second_shell):
        assert shell.alpha.shape == (3, 3) and np.isfinite(shell.alpha).all()

    assert optimized.objective < optimized.initial_objective
    assert Counter(optimized.atoms.get_chemical_symbols()) == dict(SMALL.counts)
    assert np.isclose(
        warren_cowley_sro(optimized.atoms, SMALL.lattice_parameter).first_shell.alpha[0, 2],
        optimized.sro.first_shell.alpha[0, 2],
    )

    exact = find_exact_duplicates(ensemble)
    symmetry = find_symmetry_equivalents(ensemble)
    assert symmetry.status == "complete"
    assert {r.config_id for r in symmetry.unique} <= {r.config_id for r in exact.unique}

    stats = ensemble_sro_statistics(ensemble)
    alphas = {
        "first": np.stack([r.sro.first_shell.alpha for r in exact.unique]),
        "second": np.stack([r.sro.second_shell.alpha for r in exact.unique]),
    }
    remeasured = np.stack(
        [
            warren_cowley_sro(ensemble.structures[k], SMALL.lattice_parameter).first_shell.alpha
            for k, r in enumerate(ensemble.records)
            if r in exact.unique
        ]
    )
    np.testing.assert_allclose(remeasured, alphas["first"])
    assert stats.n_unique == len(exact.unique)
    np.testing.assert_allclose(stats.first_shell_mean, alphas["first"].mean(axis=0))
    np.testing.assert_allclose(stats.first_shell_std, alphas["first"].std(axis=0, ddof=1))
    np.testing.assert_allclose(stats.second_shell_mean, alphas["second"].mean(axis=0))
    np.testing.assert_allclose(stats.second_shell_std, alphas["second"].std(axis=0, ddof=1))

    again = _workflow()
    assert np.array_equal(again[0].numbers, atoms.numbers)
    assert np.array_equal(again[2].atoms.numbers, optimized.atoms.numbers)
    assert again[2].objective == optimized.objective
    assert all(
        np.array_equal(a.numbers, b.numbers)
        for a, b in zip(again[3].structures, ensemble.structures, strict=True)
    )
    assert again[3].metadata() == ensemble.metadata()


def test_reference_1024_atom_structure_and_sro():
    atoms = generate_alloy(REFERENCE)
    assert len(atoms) == 1024
    assert Counter(atoms.get_chemical_symbols()) == dict(REFERENCE.counts)

    a = REFERENCE.lattice_parameter
    first, _, _ = first_neighbor_list(atoms, a)
    second, _, _ = second_neighbor_list(atoms, a)
    assert (np.bincount(first, minlength=1024) == 8).all()
    assert (np.bincount(second, minlength=1024) == 6).all()

    sro = warren_cowley_sro(atoms, a)
    assert len(sro.species) == 6
    for shell in (sro.first_shell, sro.second_shell):
        assert shell.alpha.shape == (6, 6) and np.isfinite(shell.alpha).all()

    repeat = warren_cowley_sro(generate_alloy(REFERENCE), a)
    assert np.array_equal(generate_alloy(REFERENCE).numbers, atoms.numbers)
    assert np.array_equal(repeat.first_shell.alpha, sro.first_shell.alpha)
    assert np.array_equal(repeat.second_shell.alpha, sro.second_shell.alpha)
