"""Independent configurational ensembles (A2.7): random and targeted-SRO, small BCC cells."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace

import numpy as np
import pytest

from matsim_agents.science.structures.alloy_ensemble import (
    AlloyEnsemble,
    _point_permutations,
    ensemble_sro_statistics,
    find_cross_cell_equivalents,
    find_exact_duplicates,
    find_symmetry_equivalents,
    find_translation_equivalents,
    generate_alloy_ensemble,
    occupation_fingerprint,
)
from matsim_agents.science.structures.alloys import (
    AlloySpec,
    ShellSRO,
    SROTarget,
    generate_alloy,
    generate_targeted_sro,
    warren_cowley_sro,
)

A = 3.16
SMALL = AlloySpec(counts={"Mo": 16, "W": 16}, repeats=(4, 2, 2), seed=0)
TERNARY = AlloySpec(counts={"Mo": 12, "Ta": 8, "W": 12}, repeats=(4, 2, 2), seed=0)
REFERENCE = AlloySpec(counts={"W": 640, "Ta": 128, "Cr": 64, "V": 96, "Hf": 32, "Mo": 64}, seed=0)
TARGET = [SROTarget(shell=1, pair=("Mo", "W"), alpha=-0.5)]


@pytest.fixture(scope="module")
def random_ensemble():
    return generate_alloy_ensemble(SMALL, n_configurations=6, mode="random", seed=11)


@pytest.fixture(scope="module")
def targeted_ensemble():
    return generate_alloy_ensemble(
        SMALL,
        n_configurations=3,
        mode="targeted_sro",
        seed=5,
        sro_targets=TARGET,
        max_swap_attempts=1500,
    )


def test_requested_number_and_ids(random_ensemble):
    assert len(random_ensemble.records) == len(random_ensemble.structures) == 6
    assert [r.config_id for r in random_ensemble.records] == [f"random-{k:04d}" for k in range(6)]
    assert random_ensemble.spec.n_configurations == 6


def test_size_is_not_capped_at_five():
    out = generate_alloy_ensemble(SMALL, n_configurations=9, mode="random", seed=0)
    assert len(out.records) == 9


def test_composition_and_geometry_preserved(random_ensemble, targeted_ensemble):
    lattice = generate_alloy(SMALL)
    for ensemble in (random_ensemble, targeted_ensemble):
        for record, atoms in zip(ensemble.records, ensemble.structures, strict=True):
            assert Counter(atoms.get_chemical_symbols()) == Counter(SMALL.counts)
            assert record.counts == dict(SMALL.counts)
            assert np.array_equal(atoms.positions, lattice.positions)
            assert np.array_equal(atoms.cell, lattice.cell)
            assert atoms.pbc.all()


def test_structures_are_distinct_objects(random_ensemble):
    structures = random_ensemble.structures
    assert len({id(atoms) for atoms in structures}) == 6
    assert len({atoms.positions.ctypes.data for atoms in structures}) == 6
    assert len({atoms.numbers.ctypes.data for atoms in structures}) == 6


def test_deterministic_replay(random_ensemble):
    again = generate_alloy_ensemble(SMALL, n_configurations=6, mode="random", seed=11)
    assert [a.get_chemical_symbols() for a in again.structures] == [
        a.get_chemical_symbols() for a in random_ensemble.structures
    ]
    assert [r.seed for r in again.records] == [r.seed for r in random_ensemble.records]
    assert json.dumps(again.metadata()) == json.dumps(random_ensemble.metadata())


def test_replay_does_not_touch_global_numpy_state():
    np.random.seed(123)
    before = np.random.get_state()[1].copy()
    generate_alloy_ensemble(SMALL, n_configurations=3, mode="random", seed=1)
    assert np.array_equal(np.random.get_state()[1], before)


def test_configurations_use_different_streams(random_ensemble):
    seeds = [r.seed for r in random_ensemble.records]
    assert len(set(seeds)) == len(seeds)
    assert random_ensemble.n_unique > 1
    other = generate_alloy_ensemble(SMALL, n_configurations=6, mode="random", seed=12)
    assert [r.seed for r in other.records] != seeds


def test_fingerprints_follow_occupation(random_ensemble):
    for record, atoms in zip(random_ensemble.records, random_ensemble.structures, strict=True):
        assert record.fingerprint == occupation_fingerprint(atoms)
        assert len(record.fingerprint) == 64
    swapped = random_ensemble.structures[0].copy()
    symbols = swapped.get_chemical_symbols()
    a, b = symbols.index("Mo"), symbols.index("W")
    symbols[a], symbols[b] = symbols[b], symbols[a]
    swapped.set_chemical_symbols(symbols)
    assert occupation_fingerprint(swapped) != random_ensemble.records[0].fingerprint
    same = random_ensemble.structures[0].copy()
    assert occupation_fingerprint(same) == random_ensemble.records[0].fingerprint


def test_exact_duplicates_are_detected_and_reported():
    # 1 Mo + 1 W on a 2-site cell has only two occupations, so 8 configurations must repeat.
    tiny = AlloySpec(counts={"Mo": 1, "W": 1}, repeats=(1, 1, 1), seed=0)
    out = generate_alloy_ensemble(tiny, n_configurations=8, mode="random", seed=0)
    assert out.n_unique <= 2
    assert out.duplicate_groups
    grouped = {config_id for group in out.duplicate_groups for config_id in group}
    for group in out.duplicate_groups:
        fingerprints = {r.fingerprint for r in out.records if r.config_id in group}
        assert len(fingerprints) == 1
    assert grouped <= {r.config_id for r in out.records}
    assert len(out.records) == 8


def test_no_duplicates_reported_when_all_distinct(random_ensemble):
    assert random_ensemble.n_unique == len(random_ensemble.records)
    assert random_ensemble.duplicate_groups == ()


def test_find_exact_duplicates_keeps_first_occurrence_and_metadata():
    tiny = AlloySpec(counts={"Mo": 1, "W": 1}, repeats=(1, 1, 1), seed=0)
    out = generate_alloy_ensemble(tiny, n_configurations=8, mode="random", seed=0)
    before = json.dumps(out.metadata())
    result = find_exact_duplicates(out)
    ids = [r.config_id for r in out.records]
    assert len(result.unique) == out.n_unique <= 2
    assert len(result.unique) + len(result.duplicate_of) == len(ids)
    assert result.unique[0] is out.records[0]
    by_id = {r.config_id: r for r in out.records}
    for duplicate, representative in result.duplicate_of.items():
        assert ids.index(representative) < ids.index(duplicate)
        assert representative in {r.config_id for r in result.unique}
        assert by_id[duplicate].fingerprint == by_id[representative].fingerprint
        assert by_id[duplicate].seed != by_id[representative].seed or duplicate == representative
    assert json.dumps(out.metadata()) == before


def test_find_exact_duplicates_keeps_distinct_occupations(random_ensemble):
    result = find_exact_duplicates(random_ensemble)
    assert result.unique == random_ensemble.records
    assert result.duplicate_of == {}


def _translated_ensemble(base, *, shifts, extra=()):
    """Ensemble from one occupation shifted by integer half-lattice-parameter vectors."""
    atoms = base.structures[0]
    half = SMALL.lattice_parameter / 2
    grid = np.rint(atoms.positions / half).astype(int)
    shape = 2 * np.array(SMALL.repeats)
    lookup = {tuple(g): s for s, g in enumerate(grid)}
    structures = []
    for shift in shifts:
        moved = np.array([lookup[tuple((g + shift) % shape)] for g in grid])
        copy = atoms.copy()
        copy.set_chemical_symbols(np.array(atoms.get_chemical_symbols())[moved])
        structures.append(copy)
    structures.extend(extra)
    records = tuple(
        replace(
            base.records[0],
            config_id=f"t-{k}",
            fingerprint=occupation_fingerprint(s),
        )
        for k, s in enumerate(structures)
    )
    return AlloyEnsemble(base.spec, records, tuple(structures), ())


def test_translated_occupations_are_equivalent_but_not_exact(random_ensemble):
    ensemble = _translated_ensemble(random_ensemble, shifts=[(0, 0, 0), (2, 0, 0), (0, 4, 2)])
    result = find_translation_equivalents(ensemble)
    assert [r.config_id for r in result.unique] == ["t-0"]
    assert result.translation_equivalent_to == {"t-1": "t-0", "t-2": "t-0"}
    assert result.exact_duplicate_of == {}


def test_translation_does_not_merge_different_occupations(random_ensemble):
    other = random_ensemble.structures[1]
    assert Counter(other.get_chemical_symbols()) == Counter(SMALL.counts)
    ensemble = _translated_ensemble(random_ensemble, shifts=[(0, 0, 0)], extra=[other])
    # A repeated occupation is exact; the different arrangement stays separate.
    result = find_translation_equivalents(ensemble)
    assert [r.config_id for r in result.unique] == ["t-0", "t-1"]
    assert result.translation_equivalent_to == {}


def test_body_centre_translation_swaps_sublattices(random_ensemble):
    ensemble = _translated_ensemble(random_ensemble, shifts=[(0, 0, 0), (1, 1, 1)])
    base, shifted = ensemble.structures
    corner = np.all(np.rint(base.positions / (SMALL.lattice_parameter / 2)) % 2 == 0, axis=1)
    assert not np.array_equal(base.numbers[corner], shifted.numbers[corner])
    result = find_translation_equivalents(ensemble)
    assert result.translation_equivalent_to == {"t-1": "t-0"}


def _transformed_ensemble(base, transforms, extra=()):
    """Ensemble from one occupation moved by maps on half-lattice-parameter grid coordinates."""
    atoms = base.structures[0]
    grid = np.rint(atoms.positions / (SMALL.lattice_parameter / 2)).astype(int)
    shape = 2 * np.array(SMALL.repeats)
    lookup = {tuple(g): s for s, g in enumerate(grid)}
    symbols = np.array(atoms.get_chemical_symbols())
    structures = []
    for transform in transforms:
        moved = np.array([lookup[tuple(transform(g) % shape)] for g in grid])
        copy = atoms.copy()
        copy.set_chemical_symbols(symbols[moved])
        structures.append(copy)
    structures.extend(extra)
    records = tuple(
        replace(base.records[0], config_id=f"s-{k}", fingerprint=occupation_fingerprint(s))
        for k, s in enumerate(structures)
    )
    return AlloyEnsemble(base.spec, records, tuple(structures), ())


def test_reflection_and_rotation_are_symmetry_equivalent_but_not_translations(random_ensemble):
    identity = lambda g: g  # noqa: E731
    mirror_x = lambda g: g * np.array([-1, 1, 1])  # noqa: E731
    swap_yz = lambda g: g[[0, 2, 1]]  # noqa: E731
    # 90-degree rotation about x; valid because SMALL has equal y and z repeats.
    rotate_x = lambda g: np.array([g[0], -g[2], g[1]])  # noqa: E731
    ensemble = _transformed_ensemble(random_ensemble, [identity, mirror_x, swap_yz, rotate_x])
    assert find_translation_equivalents(ensemble).translation_equivalent_to == {}
    result = find_symmetry_equivalents(ensemble)
    assert [r.config_id for r in result.unique] == ["s-0"]
    assert result.point_group_equivalent_to == {"s-1": "s-0", "s-2": "s-0", "s-3": "s-0"}
    assert result.exact_duplicate_of == {} and result.translation_equivalent_to == {}


def test_symmetry_keeps_translations_and_exact_duplicates_distinct(random_ensemble):
    shift = lambda g: g + np.array([1, 1, 1])  # noqa: E731
    identity = lambda g: g  # noqa: E731
    ensemble = _transformed_ensemble(random_ensemble, [identity, identity, shift])
    result = find_symmetry_equivalents(ensemble)
    assert result.exact_duplicate_of == {"s-1": "s-0"}
    assert result.translation_equivalent_to == {"s-2": "s-0"}
    assert result.point_group_equivalent_to == {}


def test_symmetry_does_not_merge_different_occupations(random_ensemble):
    other = random_ensemble.structures[1]
    ensemble = _transformed_ensemble(random_ensemble, [lambda g: g], extra=[other])
    result = find_symmetry_equivalents(ensemble)
    assert [r.config_id for r in result.unique] == ["s-0", "s-1"]
    assert not (
        result.exact_duplicate_of
        or result.translation_equivalent_to
        or result.point_group_equivalent_to
    )


CUBIC = AlloySpec(counts={"Mo": 8, "Ta": 4, "W": 4}, repeats=(2, 2, 2), seed=0)


def _cubic_copies(transforms, seed=21):
    """First configuration of a cubic-cell ensemble mapped by grid transforms, plus the original."""
    base = generate_alloy_ensemble(CUBIC, n_configurations=2, mode="random", seed=seed)
    grid = np.rint(base.structures[0].positions / (CUBIC.lattice_parameter / 2)).astype(int)
    lookup = {tuple(g): s for s, g in enumerate(grid)}
    structures = [base.structures[0]]
    for transform in transforms:
        copy = base.structures[0].copy()
        copy.numbers = base.structures[0].numbers[[lookup[tuple(transform(g) % 4)] for g in grid]]
        structures.append(copy)
    records = (base.records[0],) + tuple(
        replace(base.records[0], config_id=f"c-{k}", fingerprint=occupation_fingerprint(s))
        for k, s in enumerate(structures[1:], start=1)
    )
    return base, AlloyEnsemble(base.spec, records, tuple(structures), ())


def test_a28d_rotated_configurations_are_equivalent_on_a_cubic_cell():
    rotations = [
        lambda g: np.array([g[0], -g[2], g[1]]),  # 90 deg about x
        lambda g: np.array([-g[0], -g[1], g[2]]),  # 180 deg about z
        lambda g: g[[1, 2, 0]],  # 120 deg about [111]
    ]
    base, ensemble = _cubic_copies(rotations)
    result = find_symmetry_equivalents(ensemble)
    assert result.n_point_operations == 48 and result.status == "complete"
    assert result.unique == (base.records[0],)
    assert result.point_group_equivalent_to == {"c-1": base.records[0].config_id} | {
        "c-2": base.records[0].config_id,
        "c-3": base.records[0].config_id,
    }


def test_a28d_reflected_configurations_are_equivalent_on_a_cubic_cell():
    reflections = [
        lambda g: g * np.array([1, -1, 1]),  # mirror y
        lambda g: -g,  # inversion
        lambda g: g[[1, 0, 2]],  # mirror x = y
    ]
    base, ensemble = _cubic_copies(reflections)
    result = find_symmetry_equivalents(ensemble)
    assert result.status == "complete" and result.unique == (base.records[0],)
    assert set(result.point_group_equivalent_to) == {"c-1", "c-2", "c-3"}
    assert result.exact_duplicate_of == {} and result.translation_equivalent_to == {}


def test_a28d_chemically_distinct_configurations_are_not_merged():
    base, _ = _cubic_copies([])
    swapped = base.structures[0].copy()
    mo, ta = np.flatnonzero(swapped.numbers == 42)[0], np.flatnonzero(swapped.numbers == 73)[0]
    swapped.numbers[[mo, ta]] = swapped.numbers[[ta, mo]]
    relabelled = base.structures[0].copy()
    relabelled.numbers = np.where(
        relabelled.numbers == 73, 74, np.where(relabelled.numbers == 74, 73, relabelled.numbers)
    )
    structures = (base.structures[0], swapped, relabelled)
    records = tuple(
        replace(base.records[0], config_id=f"d-{k}", fingerprint=occupation_fingerprint(s))
        for k, s in enumerate(structures)
    )
    result = find_symmetry_equivalents(AlloyEnsemble(base.spec, records, structures, ()))
    assert result.status == "complete"
    assert [r.config_id for r in result.unique] == ["d-0", "d-1", "d-2"]
    assert not (
        result.exact_duplicate_of
        or result.translation_equivalent_to
        or result.point_group_equivalent_to
    )


def test_symmetry_check_reports_incomplete_when_budget_is_exhausted(random_ensemble):
    identity = lambda g: g  # noqa: E731
    mirror_x = lambda g: g * np.array([-1, 1, 1])  # noqa: E731
    ensemble = _transformed_ensemble(random_ensemble, [identity, mirror_x])
    full = find_symmetry_equivalents(ensemble)
    assert full.status == "complete" and full.unchecked == ()
    assert full.n_canonicalized == 2
    limited = find_symmetry_equivalents(ensemble, max_canonicalizations=1)
    assert limited.status == "incomplete"
    assert limited.unchecked == ("s-1",)
    assert [r.config_id for r in limited.unique] == ["s-0", "s-1"]
    assert limited.point_group_equivalent_to == {}
    with pytest.raises(ValueError, match="nonnegative"):
        find_symmetry_equivalents(ensemble, max_canonicalizations=-1)


def test_distinct_pair_counts_skip_canonicalization(random_ensemble):
    result = find_symmetry_equivalents(random_ensemble, max_canonicalizations=0)
    signature = {
        r.config_id: (
            r.sro.first_shell.pair_counts.tobytes(),
            r.sro.second_shell.pair_counts.tobytes(),
        )
        for r in random_ensemble.records
    }
    shared = Counter(signature.values())
    expected = tuple(cid for cid, sig in signature.items() if shared[sig] > 1)
    assert result.n_canonicalized == 0
    assert result.unchecked == expected
    assert result.status == ("incomplete" if expected else "complete")
    assert len(result.unique) == len(random_ensemble.records)


def test_symmetry_check_on_1024_atom_cell():
    ensemble = generate_alloy_ensemble(REFERENCE, n_configurations=2, mode="random", seed=3)
    grid = np.rint(ensemble.structures[0].positions / (REFERENCE.lattice_parameter / 2))
    lookup = {tuple(g): s for s, g in enumerate(grid.astype(int))}
    shape = 2 * np.array(REFERENCE.repeats)
    # Rotate by 90 degrees about z and shift by the body-centre vector.
    moved = [lookup[tuple((np.array([-g[1], g[0], g[2]]) + 1) % shape)] for g in grid.astype(int)]
    rotated = ensemble.structures[0].copy()
    rotated.numbers = ensemble.structures[0].numbers[moved]
    records = (
        *ensemble.records,
        replace(
            ensemble.records[0], config_id="rotated", fingerprint=occupation_fingerprint(rotated)
        ),
    )
    # Pair counts are symmetry invariant, so the stored SRO of record 0 stays valid.
    extended = AlloyEnsemble(ensemble.spec, records, (*ensemble.structures, rotated), ())
    result = find_symmetry_equivalents(extended)
    assert _point_permutations(REFERENCE.repeats, REFERENCE.lattice_parameter).shape == (48, 1024)
    assert result.status == "complete"
    assert result.point_group_equivalent_to == {"rotated": "random-0000"}
    assert [r.config_id for r in result.unique] == ["random-0000", "random-0001"]


def test_point_operations_respect_the_supercell_shape():
    a = SMALL.lattice_parameter
    assert len(_point_permutations((2, 2, 2), a)) == 48
    assert len(_point_permutations((4, 2, 2), a)) == 16
    assert len(_point_permutations((4, 2, 1), a)) == 8
    for repeats in [(4, 2, 2), (3, 3, 2)]:
        perms = _point_permutations(repeats, a)
        assert np.array_equal(perms[0], np.arange(2 * np.prod(repeats)))
        assert all(sorted(row) == list(range(perms.shape[1])) for row in perms)


def _recast(atoms, *, cell_rows, shift=(0, 0, 0), rotate=None, order=np.flip):
    """Same crystal in another cell basis, translated, optionally rotated and reordered."""
    out = atoms.copy()
    out.set_cell(np.array(cell_rows) @ atoms.cell.array, scale_atoms=False)
    out.positions += np.array(shift) * A / 2
    if rotate is not None:
        out.rotate(rotate, "z", rotate_cell=True)
    out.wrap()
    return out[order(np.arange(len(out)))]


def test_cross_cell_equivalence_ignores_cell_choice_origin_order_and_rotation(random_ensemble):
    base, other = random_ensemble.structures[:2]
    shear = [[1, 0, 0], [1, 1, 0], [0, 1, 1]]
    structures = {
        "base": base,
        "sheared": _recast(base, cell_rows=shear, shift=(1, 1, 1)),
        "rotated": _recast(base, cell_rows=shear, shift=(2, 0, 2), rotate=90),
        "other": _recast(other, cell_rows=shear),
    }
    result = find_cross_cell_equivalents(structures, A)
    assert result.status == "complete" and result.unchecked == ()
    assert result.unique == ("base", "other")
    assert result.equivalent_to == {"sheared": "base", "rotated": "base"}


def test_cross_cell_skips_different_compositions_and_respects_budget(random_ensemble):
    base, other = random_ensemble.structures[:2]
    changed = base.copy()
    changed.numbers[0] = 74 if changed.numbers[0] != 74 else 42
    result = find_cross_cell_equivalents({"base": base, "changed": changed}, A)
    assert result.unique == ("base", "changed") and result.n_comparisons == 0
    copy = _recast(base, cell_rows=np.eye(3, dtype=int), shift=(1, 1, 1))
    limited = find_cross_cell_equivalents({"base": base, "copy": copy}, A, max_comparisons=0)
    assert limited.status == "incomplete" and limited.unchecked == ("copy",)
    with pytest.raises(ValueError, match="nonnegative"):
        find_cross_cell_equivalents({"base": base}, A, max_comparisons=-1)


def test_cross_cell_recognises_arbitrarily_oriented_crystals(random_ensemble):
    base = random_ensemble.structures[0]
    tilted = _recast(base, cell_rows=[[1, 0, 0], [1, 1, 0], [0, 1, 1]], shift=(1, 1, 1))
    tilted.rotate(37.0, (1, 2, 3), rotate_cell=True)
    result = find_cross_cell_equivalents({"base": base, "tilted": tilted}, A)
    assert result.equivalent_to == {"tilted": "base"} and result.status == "complete"


def test_cross_cell_compares_only_configurations_with_equal_environments(random_ensemble):
    structures = {
        r.config_id: s
        for r, s in zip(random_ensemble.records, random_ensemble.structures, strict=True)
    }
    distinct = find_cross_cell_equivalents(structures, A, max_comparisons=0)
    assert distinct.status == "complete" and distinct.n_comparisons == 0
    copies = {
        f"copy-{k}": _recast(s, cell_rows=np.eye(3, dtype=int), shift=(k, 0, 0))
        for k, s in enumerate(structures.values())
    }
    result = find_cross_cell_equivalents(structures | copies, A)
    assert result.n_comparisons == len(copies)
    assert result.unique == tuple(structures)
    assert result.equivalent_to == dict(zip(copies, structures, strict=True))


def test_cross_cell_rejects_structures_off_the_bcc_lattice(random_ensemble):
    moved = random_ensemble.structures[0].copy()
    moved.positions[1] += 0.3
    with pytest.raises(ValueError, match="not a periodic BCC supercell"):
        find_cross_cell_equivalents({"moved": moved}, A)


def _with_alpha(record, first, second):
    shell = lambda alpha: ShellSRO(  # noqa: E731
        record.sro.first_shell.pair_counts, record.sro.first_shell.conditional_probability, alpha
    )
    sro = replace(
        record.sro,
        first_shell=shell(np.array(first, dtype=float)),
        second_shell=shell(np.array(second, dtype=float)),
    )
    return replace(record, sro=sro)


def _known_ensemble(random_ensemble, indices, alphas):
    records = tuple(
        _with_alpha(random_ensemble.records[i], *alpha)
        for i, alpha in zip(indices, alphas, strict=True)
    )
    structures = tuple(random_ensemble.structures[i] for i in indices)
    return AlloyEnsemble(random_ensemble.spec, records, structures, ())


def test_sro_statistics_match_hand_computed_mean_and_sample_std(random_ensemble):
    a = ([[0.1, -0.1], [-0.1, 0.1]], [[0.0, 0.2], [0.2, 0.0]])
    b = ([[0.3, -0.3], [-0.3, 0.3]], [[0.4, 0.0], [0.0, 0.4]])
    stats = ensemble_sro_statistics(_known_ensemble(random_ensemble, [0, 1], [a, b]))
    assert stats.species == ("Mo", "W")
    assert stats.n_unique == 2
    assert np.allclose(stats.first_shell_mean, [[0.2, -0.2], [-0.2, 0.2]])
    assert np.allclose(stats.second_shell_mean, [[0.2, 0.1], [0.1, 0.2]])
    # Two samples: s = |x1 - x2| / sqrt(2).
    assert np.allclose(stats.first_shell_std, np.full((2, 2), 0.2 / np.sqrt(2)))
    assert np.allclose(stats.second_shell_std, [[0.4, 0.2], [0.2, 0.4]] / np.sqrt(2))


def test_sro_statistics_exclude_duplicates(random_ensemble):
    a = ([[0.1, 0.0], [0.0, 0.1]], [[0.0, 0.0], [0.0, 0.0]])
    b = ([[0.3, 0.0], [0.0, 0.3]], [[0.2, 0.0], [0.0, 0.2]])
    # Indices 0, 0, 1: the second entry repeats occupation 0 with a different SRO payload.
    ensemble = _known_ensemble(random_ensemble, [0, 0, 1], [a, ([[9.0] * 2] * 2, a[1]), b])
    stats = ensemble_sro_statistics(ensemble)
    assert stats.n_unique == 2
    assert np.allclose(stats.first_shell_mean, [[0.2, 0.0], [0.0, 0.2]])


def test_sro_statistics_single_configuration_has_undefined_std(random_ensemble):
    single = AlloyEnsemble(
        random_ensemble.spec, random_ensemble.records[:1], random_ensemble.structures[:1], ()
    )
    stats = ensemble_sro_statistics(single)
    assert stats.n_unique == 1
    assert stats.first_shell_std is None
    assert stats.second_shell_std is None
    assert np.allclose(stats.first_shell_mean, random_ensemble.records[0].sro.first_shell.alpha)


def test_sro_metadata_matches_remeasurement(random_ensemble, targeted_ensemble):
    for ensemble in (random_ensemble, targeted_ensemble):
        for record, atoms in zip(ensemble.records, ensemble.structures, strict=True):
            fresh = warren_cowley_sro(atoms, A, record.sro.species)
            for shell in ("first_shell", "second_shell"):
                got, want = getattr(record.sro, shell), getattr(fresh, shell)
                assert np.array_equal(got.pair_counts, want.pair_counts)
                assert np.allclose(got.alpha, want.alpha, atol=1e-12)
    data = random_ensemble.metadata()["configurations"][0]
    assert data["species"] == ["Mo", "W"]
    assert np.array(data["first_shell"]["alpha"]).shape == (2, 2)
    assert np.array(data["second_shell"]["pair_counts"]).shape == (2, 2)
    assert "positions" not in json.dumps(data)


def test_targeted_records_report_objective_and_status(targeted_ensemble):
    for record, atoms in zip(targeted_ensemble.records, targeted_ensemble.structures, strict=True):
        assert record.mode == "targeted_sro"
        assert record.optimizer_seed is not None
        alpha = warren_cowley_sro(atoms, A, record.sro.species).first_shell.alpha[0, 1]
        assert record.objective == pytest.approx((alpha + 0.5) ** 2, abs=1e-12)
        assert record.converged == (record.objective <= targeted_ensemble.spec.tolerance)
        assert 0 <= record.attempts <= 1500


def test_random_records_have_no_optimizer_fields(random_ensemble):
    for record in random_ensemble.records:
        assert record.converged is None
        assert record.objective is None
        assert record.attempts is None
        assert record.optimizer_seed is None


def test_targeted_replay_is_deterministic(targeted_ensemble):
    again = generate_alloy_ensemble(
        SMALL,
        n_configurations=3,
        mode="targeted_sro",
        seed=5,
        sro_targets=TARGET,
        max_swap_attempts=1500,
    )
    assert [r.fingerprint for r in again.records] == [
        r.fingerprint for r in targeted_ensemble.records
    ]
    assert json.dumps(again.metadata()) == json.dumps(targeted_ensemble.metadata())


def test_nonconvergence_is_reported_not_raised():
    out = generate_alloy_ensemble(
        SMALL,
        n_configurations=2,
        mode="targeted_sro",
        seed=3,
        sro_targets=[SROTarget(shell=1, pair=("Mo", "W"), alpha=-0.9)],
        max_swap_attempts=2,
        tolerance=1e-12,
    )
    assert all(r.converged is False for r in out.records)
    assert all(r.attempts == 2 for r in out.records)


def test_convergence_reported_when_target_already_met():
    out = generate_alloy_ensemble(
        SMALL,
        n_configurations=2,
        mode="targeted_sro",
        seed=3,
        sro_targets=TARGET,
        max_swap_attempts=0,
        tolerance=10.0,
    )
    assert all(r.converged is True and r.attempts == 0 for r in out.records)


def test_ternary_ensemble_composition_preserved():
    out = generate_alloy_ensemble(TERNARY, n_configurations=3, mode="random", seed=2)
    for atoms in out.structures:
        assert Counter(atoms.get_chemical_symbols()) == Counter(TERNARY.counts)
    assert out.records[0].sro.species == ("Mo", "Ta", "W")


@pytest.mark.parametrize("n", [0, -1, True, 2.0, "3"])
def test_invalid_ensemble_sizes_are_rejected(n):
    with pytest.raises(ValueError):
        generate_alloy_ensemble(SMALL, n_configurations=n, mode="random", seed=0)


@pytest.mark.parametrize("mode", ["sqs", "clustered", "", "Random"])
def test_unsupported_modes_are_rejected(mode):
    with pytest.raises(ValueError):
        generate_alloy_ensemble(SMALL, n_configurations=2, mode=mode, seed=0)


def test_mode_and_target_consistency_and_seed_validation():
    with pytest.raises(ValueError):
        generate_alloy_ensemble(SMALL, n_configurations=2, mode="targeted_sro", seed=0)
    with pytest.raises(ValueError):
        generate_alloy_ensemble(
            SMALL, n_configurations=2, mode="random", seed=0, sro_targets=TARGET
        )
    with pytest.raises(ValueError):
        generate_alloy_ensemble(SMALL, n_configurations=2, mode="random", seed=-1)
    with pytest.raises(ValueError):
        generate_alloy_ensemble(
            SMALL, n_configurations=2, mode="random", seed=0, max_swap_attempts=-1
        )


def test_invalid_targets_are_rejected_before_generation():
    with pytest.raises(ValueError):
        generate_alloy_ensemble(
            SMALL,
            n_configurations=2,
            mode="targeted_sro",
            seed=0,
            sro_targets=[SROTarget(shell=1, pair=("Mo", "W"), alpha=-1.5)],
        )


def test_overlap_matrix_is_symmetric_with_unit_diagonal(random_ensemble):
    matrix = random_ensemble.overlap_matrix()
    assert matrix.shape == (6, 6)
    assert np.allclose(np.diag(matrix), 1.0)
    assert np.array_equal(matrix, matrix.T)
    a, b = random_ensemble.structures[0], random_ensemble.structures[1]
    assert matrix[0, 1] == pytest.approx(float((a.numbers == b.numbers).mean()))


def test_overlap_of_duplicates_is_one_and_summary_is_serializable():
    tiny = AlloySpec(counts={"Mo": 1, "W": 1}, repeats=(1, 1, 1), seed=0)
    out = generate_alloy_ensemble(tiny, n_configurations=8, mode="random", seed=0)
    first_group = out.duplicate_groups[0]
    ids = [r.config_id for r in out.records]
    i, j = ids.index(first_group[0]), ids.index(first_group[1])
    assert out.overlap_matrix()[i, j] == 1.0
    assert json.loads(json.dumps(out.metadata()))["overlap"]["max"] == 1.0


def test_random_overlap_is_consistent_with_expectation_and_single_config_has_none():
    spec = AlloySpec(counts={"Mo": 128, "W": 128}, repeats=(8, 4, 4), seed=0)
    out = generate_alloy_ensemble(spec, n_configurations=6, mode="random", seed=3)
    summary = out.overlap_summary()
    assert out.expected_random_overlap == pytest.approx(0.5)
    # sd of one pair is about 0.03 for 256 sites; 6 sigma over 15 pairs is a safe bound.
    assert abs(summary["mean"] - 0.5) < 0.05
    assert 0.3 < summary["min"] <= summary["mean"] <= summary["max"] < 0.7
    single = generate_alloy_ensemble(spec, n_configurations=1, mode="random", seed=3)
    assert single.overlap_summary()["mean"] is None


def test_reference_1024_atom_random_ensemble():
    out = generate_alloy_ensemble(REFERENCE, n_configurations=3, mode="random", seed=0)
    assert len(out.structures) == 3
    for atoms in out.structures:
        assert len(atoms) == 1024
        assert Counter(atoms.get_chemical_symbols()) == Counter(REFERENCE.counts)
    assert out.n_unique == 3
    sro = out.records[0].sro
    assert sro.first_shell.pair_counts.shape == (6, 6)
    assert sro.first_shell.pair_counts.sum() == 1024 * 8
    assert sro.second_shell.pair_counts.sum() == 1024 * 6


def test_record_seeds_replay_through_the_public_generators(targeted_ensemble):
    for record, atoms in zip(targeted_ensemble.records, targeted_ensemble.structures, strict=True):
        start = generate_alloy(SMALL.model_copy(update={"seed": record.seed}))
        replay = generate_targeted_sro(
            start,
            A,
            TARGET,
            seed=record.optimizer_seed,
            tolerance=targeted_ensemble.spec.tolerance,
            max_attempts=targeted_ensemble.spec.max_swap_attempts,
        )
        assert replay.atoms.get_chemical_symbols() == atoms.get_chemical_symbols()


def test_existing_generation_is_unchanged_by_topology_reuse():
    atoms = generate_alloy(SMALL)
    assert warren_cowley_sro(atoms, A).first_shell.pair_counts.sum() == 32 * 8
