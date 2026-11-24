"""A3.1: defect data contracts."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
from ase.geometry import get_distances
from pydantic import ValidationError

from matsim_agents.discovery.alloy_ensemble import occupation_fingerprint
from matsim_agents.discovery.alloys import AlloySpec, generate_alloy
from matsim_agents.discovery.bcc_defects import (
    BCC_111_DIRECTIONS,
    BCC_INTERSTITIAL_OFFSETS,
    DUMBBELL_ORIENTATIONS,
    generate_bcc_interstitial,
    generate_crowdion,
    generate_divacancy,
    generate_dumbbell_interstitial,
    generate_frenkel_pair,
)
from matsim_agents.discovery.point_defects import (
    DefectConfiguration,
    DefectOutcome,
    DefectSite,
    DefectSpec,
    generate_vacancy,
    make_defect_config_id,
)

CELL = ((6.32, 0.0, 0.0), (0.0, 6.32, 0.0), (0.0, 0.0, 6.32))
N = 16


def _vacancy_site(site=3, atom_id=3, species="Ta"):
    return DefectSite(
        role="vacancy", species=species, atom_id=atom_id, lattice_site=site, position=(1.0, 2, 3)
    )


def _interstitial_site(atom_id, species="W"):
    return DefectSite(
        role="interstitial", species=species, atom_id=atom_id, position=(0.5, 0.5, 0.5)
    )


def _config(spec, sites, n_atoms):
    return DefectConfiguration(
        config_id=make_defect_config_id("parent-1", spec),
        parent_id="parent-1",
        parent_ordering_id="abc123",
        spec=spec,
        n_parent_atoms=N,
        n_atoms=n_atoms,
        cell=CELL,
        sites=sites,
        geometry_fingerprint="f" * 8,
    )


def test_valid_specs_for_each_defect_type():
    DefectSpec(defect_type="vacancy", site_indices=(4,))
    DefectSpec(defect_type="divacancy", site_indices=(4, 9), parameters={"shell": 1})
    DefectSpec(defect_type="frenkel_pair", site_indices=(2,), seed=5, parameters={"d": 2.5})
    spec = DefectSpec(
        defect_type="interstitial",
        site_indices=(7,),
        species=("W",),
        parameters={"orientation": "111", "separation": 0.7},
        seed=0,
    )
    assert spec.parameters["orientation"] == "111"
    with pytest.raises(TypeError):
        spec.parameters["orientation"] = "110"
    # An interstitial need not be centred on a lattice site (e.g. a tetrahedral candidate).
    DefectSpec(defect_type="interstitial", species=("Ta",))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"defect_type": "cluster", "site_indices": (1,)},
        {"defect_type": "vacancy", "site_indices": ()},
        {"defect_type": "vacancy", "site_indices": (1, 2)},
        {"defect_type": "divacancy", "site_indices": (3, 3)},
        {"defect_type": "vacancy", "site_indices": (-1,)},
        {"defect_type": "vacancy", "site_indices": (True,)},
        {"defect_type": "vacancy", "site_indices": (1,), "species": ("W",)},
        {"defect_type": "interstitial", "site_indices": (1,)},
        {"defect_type": "interstitial", "species": ("Xx",)},
        {"defect_type": "vacancy", "site_indices": (1,), "seed": -1},
        {"defect_type": "vacancy", "site_indices": (1,), "parameters": {"d": float("nan")}},
    ],
)
def test_invalid_specs_are_rejected(kwargs):
    with pytest.raises(ValidationError):
        DefectSpec(**kwargs)


def test_serialization_round_trip_preserves_identities_and_sites():
    # A Frenkel pair: atom 3 leaves site 3 and becomes an off-lattice atom with the same id.
    spec = DefectSpec(defect_type="frenkel_pair", site_indices=(3,), seed=11)
    sites = (_vacancy_site(), _interstitial_site(atom_id=3, species="Ta"))
    config = _config(spec, sites, N)
    restored = DefectConfiguration.model_validate_json(config.model_dump_json())
    assert restored == config
    assert restored.defect_type == "frenkel_pair"
    assert [(s.role, s.atom_id, s.lattice_site) for s in restored.sites] == [
        ("vacancy", 3, 3),
        ("interstitial", 3, None),
    ]
    assert restored.config_id == make_defect_config_id("parent-1", spec)
    assert make_defect_config_id("parent-1", spec) != make_defect_config_id("parent-2", spec)
    assert "positions" not in config.model_dump()


def test_type_specific_site_and_count_consistency():
    vacancy = DefectSpec(defect_type="vacancy", site_indices=(3,))
    assert _config(vacancy, (_vacancy_site(),), N - 1).validation_status == "unvalidated"
    interstitial = DefectSpec(defect_type="interstitial", site_indices=(3,), species=("W",))
    _config(interstitial, (_interstitial_site(atom_id=N),), N + 1)

    with pytest.raises(ValidationError, match="n_atoms"):
        _config(vacancy, (_vacancy_site(),), N)
    with pytest.raises(ValidationError, match="off-lattice"):
        DefectSite(role="interstitial", species="W", atom_id=N, lattice_site=2, position=(0, 0, 0))
    with pytest.raises(ValidationError, match="not be off-lattice"):
        DefectSite(role="vacancy", species="W", atom_id=1, position=(0, 0, 0))
    with pytest.raises(ValidationError, match="no matching vacancy"):
        _config(interstitial, (_interstitial_site(atom_id=5),), N + 1)
    with pytest.raises(ValidationError, match="outside the parent"):
        _config(vacancy, (_vacancy_site(site=N, atom_id=3),), N - 1)


def test_outcome_requires_convergence_for_a_scientific_classification():
    DefectOutcome(config_id="x", classification="unresolved")
    DefectOutcome(config_id="x", classification="persisted", relaxation_converged=True)
    for converged in (False, None):
        with pytest.raises(ValidationError, match="converged"):
            DefectOutcome(config_id="x", classification="persisted", relaxation_converged=converged)
    with pytest.raises(ValidationError):
        DefectOutcome(config_id="x", classification="melted", relaxation_converged=True)
    outcome = DefectOutcome(
        config_id="x",
        classification="recombined_antisite",
        relaxation_converged=True,
        final_sites=(_vacancy_site(),),
    )
    assert DefectOutcome.model_validate_json(outcome.model_dump_json()) == outcome


# --- A3.2 vacancy generation ---

SMALL = AlloySpec(counts={"Mo": 6, "Ta": 4, "W": 6}, repeats=(2, 2, 2), seed=1)


def test_vacancy_removes_exactly_the_requested_atom_and_leaves_the_rest_unchanged():
    parent = generate_alloy(SMALL)
    before = parent.copy()
    site = 5
    defective, config = generate_vacancy(parent, "small-1", site)

    assert parent == before  # parent untouched
    assert len(defective) == len(parent) - 1 == config.n_atoms
    keep = [k for k in range(len(parent)) if k != site]
    assert defective.get_chemical_symbols() == [parent[k].symbol for k in keep]
    assert np.array_equal(defective.positions, parent.positions[keep])
    assert np.array_equal(defective.cell.array, parent.cell.array)
    assert defective.pbc.all()
    assert defective.get_array("atom_id").tolist() == keep
    assert Counter(parent.get_chemical_symbols()) - Counter(defective.get_chemical_symbols()) == {
        parent[site].symbol: 1
    }


@pytest.mark.parametrize("site", [0, 7, 15])
def test_vacancy_metadata_identifies_the_original_site_for_any_species(site):
    parent = generate_alloy(SMALL)
    _, config = generate_vacancy(parent, "small-1", site)
    (vacancy,) = config.sites
    assert config.defect_type == "vacancy" and config.spec.site_indices == (site,)
    assert (vacancy.role, vacancy.lattice_site, vacancy.atom_id) == ("vacancy", site, site)
    assert vacancy.species == parent[site].symbol
    assert vacancy.position == tuple(parent.positions[site])
    assert config.parent_id == "small-1" and config.n_parent_atoms == len(parent)
    assert config.parent_ordering_id == occupation_fingerprint(parent)
    assert config.config_id == make_defect_config_id("small-1", config.spec)
    assert DefectConfiguration.model_validate_json(config.model_dump_json()) == config


@pytest.mark.parametrize("site", [-1, 16, 10**6, True, 2.0, "3", None])
def test_invalid_vacancy_sites_are_rejected(site):
    with pytest.raises(ValueError):
        generate_vacancy(generate_alloy(SMALL), "small-1", site)
    with pytest.raises(ValueError, match="parent_id"):
        generate_vacancy(generate_alloy(SMALL), "", 0)


def test_vacancy_is_reproducible_and_the_1024_atom_parent_gives_1023_atoms():
    parent = generate_alloy(SMALL)
    first = generate_vacancy(parent, "small-1", 3)
    second = generate_vacancy(parent, "small-1", 3)
    assert first[1] == second[1] and first[0] == second[0]
    assert generate_vacancy(parent, "small-1", 4)[1].geometry_fingerprint != (
        first[1].geometry_fingerprint
    )

    big = AlloySpec(counts={"W": 640, "Ta": 128, "Cr": 64, "V": 96, "Hf": 32, "Mo": 64}, seed=3)
    atoms = generate_alloy(big)
    defective, config = generate_vacancy(atoms, "ref-1024", 1000)
    assert len(atoms) == 1024 and len(defective) == 1023 == config.n_atoms
    counts = Counter(atoms.get_chemical_symbols())
    counts[atoms[1000].symbol] -= 1
    assert Counter(defective.get_chemical_symbols()) == counts


# --- A3.3 dumbbell interstitials ---

A = SMALL.lattice_parameter


def _mic(atoms, i, j):
    vec, _ = get_distances(atoms.positions[i], atoms.positions[j], cell=atoms.cell, pbc=True)
    return vec[0, 0]


@pytest.mark.parametrize("orientation", ["111", "110", "100"])
@pytest.mark.parametrize("rotated", [False, True], ids=["axis-aligned", "rotated-cell"])
def test_dumbbell_is_centred_on_the_site_and_oriented_along_the_cell_axes(orientation, rotated):
    parent = generate_alloy(SMALL)
    if rotated:
        parent.rotate(30, "z", rotate_cell=True)
    defective, config = generate_dumbbell_interstitial(
        parent, "small-1", 5, "W", orientation, A, separation=1.2
    )
    old, new = 5, len(parent)
    axes = parent.cell.array / np.linalg.norm(parent.cell.array, axis=1)[:, None]
    expected = np.array(DUMBBELL_ORIENTATIONS[orientation]) @ axes
    expected /= np.linalg.norm(expected)

    axis = _mic(defective, old, new)
    assert np.linalg.norm(axis) == pytest.approx(1.2)
    assert np.allclose(axis / 1.2, expected)
    midpoint = defective.positions[old] + axis / 2
    _, offset = get_distances(midpoint, parent.positions[5], cell=parent.cell, pbc=True)
    assert offset[0, 0] == pytest.approx(0, abs=1e-9)
    assert config.spec.parameters["orientation"] == orientation
    assert config.spec.site_indices == (5,) and config.n_atoms == len(parent) + 1


def test_dumbbell_preserves_original_identities_and_assigns_a_new_one():
    parent = generate_alloy(SMALL)
    defective, config = generate_dumbbell_interstitial(parent, "small-1", 2, "Ta", "111", A)
    n = len(parent)
    assert len(defective) == n + 1 == config.n_atoms
    assert defective.get_chemical_symbols() == [*parent.get_chemical_symbols(), "Ta"]
    assert defective.get_array("atom_id").tolist() == list(range(n + 1))
    others = [k for k in range(n) if k != 2]
    assert np.array_equal(defective.positions[others], parent.positions[others])
    displaced, inserted = config.sites
    assert (displaced.role, displaced.atom_id, displaced.lattice_site) == ("displaced", 2, 2)
    assert displaced.species == parent[2].symbol
    assert (inserted.role, inserted.atom_id, inserted.lattice_site) == ("interstitial", n, None)
    assert inserted.species == "Ta"
    assert DefectConfiguration.model_validate_json(config.model_dump_json()) == config
    assert generate_dumbbell_interstitial(parent, "small-1", 2, "Ta", "111", A)[1] == config


def test_dumbbell_on_a_boundary_site_wraps_periodically():
    parent = generate_alloy(SMALL)
    assert np.allclose(parent.positions[0], 0)
    defective, config = generate_dumbbell_interstitial(
        parent, "small-1", 0, "W", "100", A, separation=1.2
    )
    assert np.allclose(defective.positions[0], [2 * A - 0.6, 0, 0])
    assert np.allclose(defective.positions[-1], [0.6, 0, 0])
    scaled = defective.get_scaled_positions(wrap=False)[[0, -1]]
    assert ((scaled >= 0) & (scaled < 1)).all()
    assert np.linalg.norm(_mic(defective, 0, len(parent))) == pytest.approx(1.2)
    assert config.sites[0].position == tuple(defective.positions[0])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"orientation": "211"},
        {"orientation": "[111]"},
        {"species": "Xx"},
        {"site": 16},
        {"site": True},
        {"separation": 0.0},
        {"separation": float("nan")},
        {"separation": 0.5},
        {"min_distance": 1.7, "separation": 1.6},
        {"lattice_parameter": 3.0},
    ],
)
def test_invalid_or_overlapping_dumbbells_are_rejected(kwargs):
    arguments = {
        "site": 3,
        "species": "W",
        "orientation": "111",
        "lattice_parameter": A,
    } | kwargs
    with pytest.raises(ValueError):
        generate_dumbbell_interstitial(
            generate_alloy(SMALL),
            "small-1",
            arguments.pop("site"),
            arguments.pop("species"),
            arguments.pop("orientation"),
            arguments.pop("lattice_parameter"),
            **arguments,
        )


# --- A3.4a tetrahedral and octahedral interstitials ---


def _site_shell(atoms, position, n):
    _, dist = get_distances(position, atoms.positions[:-1], cell=atoms.cell, pbc=True)
    return np.sort(dist[0])[:n] / A


def test_tetrahedral_candidate_has_the_ideal_bcc_position_and_shell():
    parent = generate_alloy(SMALL)
    defective, config = generate_bcc_interstitial(parent, "small-1", 0, "Ta", "tetrahedral", A)
    # Variant 0 is the offset (-1/2, -1/4, 0) a from the origin atom, wrapped into the 2a cell.
    assert BCC_INTERSTITIAL_OFFSETS["tetrahedral"][0] == (-0.5, -0.25, 0.0)
    assert np.allclose(defective.positions[-1], np.array([1.5, 1.75, 0.0]) * A)
    assert np.allclose(_site_shell(defective, defective.positions[-1], 4), 5**0.5 / 4)
    assert config.spec.parameters["site_type"] == "tetrahedral"
    assert config.sites[0].position == tuple(defective.positions[-1])


def test_octahedral_candidate_has_the_ideal_bcc_position_and_shell():
    parent = generate_alloy(SMALL)
    defective, config = generate_bcc_interstitial(parent, "small-1", 0, "W", "octahedral", A)
    assert BCC_INTERSTITIAL_OFFSETS["octahedral"][0] == (-0.5, 0.0, 0.0)
    assert np.allclose(defective.positions[-1], np.array([1.5, 0.0, 0.0]) * A)
    assert np.allclose(
        _site_shell(defective, defective.positions[-1], 6), [0.5, 0.5, *[0.5**0.5] * 4]
    )


@pytest.mark.parametrize("site_type", ["tetrahedral", "octahedral"])
@pytest.mark.parametrize("rotated", [False, True], ids=["axis-aligned", "rotated-cell"])
def test_every_variant_is_a_valid_site_and_parent_atoms_are_untouched(site_type, rotated):
    parent = generate_alloy(SMALL)
    if rotated:
        parent.rotate(30, "z", rotate_cell=True)
    n = len(parent)
    for variant in range(len(BCC_INTERSTITIAL_OFFSETS[site_type])):
        defective, config = generate_bcc_interstitial(
            parent, "small-1", 7, "V", site_type, A, variant=variant
        )
        assert len(defective) == n + 1 == config.n_atoms
        assert defective.get_chemical_symbols() == [*parent.get_chemical_symbols(), "V"]
        assert np.array_equal(defective.positions[:n], parent.positions)
        assert defective.get_array("atom_id").tolist() == list(range(n + 1))
        inserted = config.sites[0]
        assert (inserted.role, inserted.atom_id, inserted.lattice_site) == ("interstitial", n, None)
        assert config.n_parent_atoms == n and config.parent_id == "small-1"
        assert DefectConfiguration.model_validate_json(config.model_dump_json()) == config
    assert (
        generate_bcc_interstitial(parent, "small-1", 7, "V", site_type, A, variant=0)[1]
        == (generate_bcc_interstitial(parent, "small-1", 7, "V", site_type, A, variant=0)[1])
    )


def test_interstitial_near_the_cell_boundary_wraps_into_the_cell():
    parent = generate_alloy(SMALL)
    defective, _ = generate_bcc_interstitial(parent, "small-1", 0, "W", "octahedral", A, variant=0)
    scaled = defective.get_scaled_positions(wrap=False)[-1]
    assert ((scaled >= 0) & (scaled < 1)).all()
    assert np.allclose(defective.positions[-1], [1.5 * A, 0, 0])
    shell = _site_shell(defective, defective.positions[-1], 6)
    assert np.allclose(shell, [0.5, 0.5, *[0.5**0.5] * 4])


def test_a_1024_atom_parent_gives_a_1025_atom_candidate():
    parent = generate_alloy(
        AlloySpec(counts={"W": 640, "Ta": 128, "Cr": 64, "V": 96, "Hf": 32, "Mo": 64}, seed=3)
    )
    defective, config = generate_bcc_interstitial(
        parent, "hea-1024", 100, "W", "tetrahedral", A, variant=5
    )
    assert len(parent) == 1024 and len(defective) == config.n_atoms == 1025


def test_invalid_site_type_variant_site_and_overlap_are_rejected():
    parent = generate_alloy(SMALL)

    def build(**kwargs):
        arguments = {"site": 3, "species": "W", "site_type": "tetrahedral", "a": A} | kwargs
        return generate_bcc_interstitial(
            parent,
            "small-1",
            arguments.pop("site"),
            arguments.pop("species"),
            arguments.pop("site_type"),
            arguments.pop("a"),
            **arguments,
        )

    for bad in (
        {"site_type": "hexahedral"},
        {"species": "Xx"},
        {"site": 16},
        {"site": True},
        {"variant": 24},
        {"variant": -1},
        {"min_distance": 0.0},
        {"min_distance": 2.5},
        {"a": 3.0},
    ):
        with pytest.raises(ValueError):
            build(**bad)


def test_a_parent_that_is_not_bcc_around_the_site_is_rejected():
    parent = generate_alloy(SMALL)
    parent.positions[3] += 0.5
    with pytest.raises(ValueError, match="not BCC"):
        generate_bcc_interstitial(parent, "small-1", 3, "W", "octahedral", A)


# --- A3.4b collective <111> crowdion ---

CROWD = AlloySpec(counts={"Mo": 18, "Ta": 18, "W": 18}, repeats=(3, 3, 3), seed=2)
STEP = A * 3**0.5 / 2


def test_crowdion_moves_the_expected_row_atoms_collectively_and_inserts_one_atom():
    parent = generate_alloy(CROWD)
    n = len(parent)
    defective, config = generate_crowdion(parent, "crowd-1", 0, "W", A, n_participating=4)
    assert len(defective) == n + 1 == config.n_atoms
    row = [int(i) for i in config.spec.parameters["row_atom_ids"].split(",")]
    assert len(row) == 4 and len(set(row)) == 4 and 0 in row
    moved = np.flatnonzero(np.abs(defective.positions[:n] - parent.positions).max(axis=1) > 1e-9)
    assert sorted(moved) == sorted(row)
    amp = 0.25 * STEP
    shifts = [float(x) for x in config.spec.parameters["row_shifts"].split(",")]
    assert shifts == pytest.approx([-amp / 2, -amp, amp, amp / 2], abs=1e-5)
    assert config.spec.parameters["crowdion_length"] == pytest.approx(3 * STEP + amp)
    inserted = config.sites[-1]
    assert (inserted.role, inserted.atom_id, inserted.lattice_site) == ("interstitial", n, None)
    assert [s.atom_id for s in config.sites[:-1]] == row
    assert all(s.initial_position is not None for s in config.sites[:-1])
    assert config.sites[-1].initial_position is None


def test_crowdion_row_is_a_periodic_111_chain_and_keeps_orientation_and_identities():
    parent = generate_alloy(CROWD)
    n = len(parent)
    for rotated in (False, True):
        cell_parent = parent.copy()
        if rotated:
            cell_parent.rotate(30, "z", rotate_cell=True)
        defective, config = generate_crowdion(cell_parent, "crowd-1", 3, "Ta", A)
        row = [int(i) for i in config.spec.parameters["row_atom_ids"].split(",")]
        axes = cell_parent.cell.array / np.linalg.norm(cell_parent.cell.array, axis=1)[:, None]
        direction = np.array(BCC_111_DIRECTIONS["111"]) @ axes
        direction /= np.linalg.norm(direction)
        for a, b in zip(row[:-1], row[1:], strict=True):
            vec, _ = get_distances(
                cell_parent.positions[a], cell_parent.positions[b], cell=cell_parent.cell, pbc=True
            )
            assert np.allclose(vec[0, 0], STEP * direction, atol=1e-9)
        assert defective.get_chemical_symbols() == [*cell_parent.get_chemical_symbols(), "Ta"]
        assert defective.get_array("atom_id").tolist() == list(range(n + 1))
        for a, b in zip(row[:-1], row[1:], strict=True):
            vec, _ = get_distances(
                defective.positions[a], defective.positions[b], cell=defective.cell, pbc=True
            )
            assert np.allclose(np.cross(vec[0, 0], direction), 0, atol=1e-6)


def test_crowdion_is_deterministic_and_handles_the_periodic_boundary():
    parent = generate_alloy(CROWD)
    first = generate_crowdion(parent, "crowd-1", 0, "W", A, direction="-1-1-1")
    second = generate_crowdion(parent, "crowd-1", 0, "W", A, direction="-1-1-1")
    assert first[1] == second[1] and np.array_equal(first[0].positions, second[0].positions)
    defective, config = first
    scaled = defective.get_scaled_positions(wrap=False)
    assert ((scaled >= 0) & (scaled < 1)).all()
    row = [int(i) for i in config.spec.parameters["row_atom_ids"].split(",")]
    assert len(set(row)) == 4


@pytest.mark.parametrize(
    "kwargs",
    [
        {"direction": "110"},
        {"n_participating": 3},
        {"n_participating": 0},
        {"n_participating": 6},
        {"n_participating": True},
        {"amplitude": 0.0},
        {"amplitude": STEP},
        {"amplitude": float("nan")},
        {"min_distance": 3.0},
        {"site": 54},
        {"species": "Xx"},
        {"lattice_parameter": 3.0},
    ],
)
def test_invalid_crowdion_parameters_are_rejected(kwargs):
    arguments = {"site": 0, "species": "W", "lattice_parameter": A} | kwargs
    with pytest.raises(ValueError):
        generate_crowdion(
            generate_alloy(CROWD),
            "crowd-1",
            arguments.pop("site"),
            arguments.pop("species"),
            arguments.pop("lattice_parameter"),
            **arguments,
        )


def test_a_site_whose_row_is_not_bcc_is_rejected():
    parent = generate_alloy(CROWD)
    parent.positions[1] += 0.6
    parent.wrap()
    with pytest.raises(ValueError):
        generate_crowdion(parent, "crowd-1", 0, "W", A, n_participating=4)


# --- A3.5 Frenkel pairs ---


def test_frenkel_pair_moves_one_atom_and_preserves_count_composition_and_identity():
    parent = generate_alloy(SMALL)
    n = len(parent)
    defective, config = generate_frenkel_pair(parent, "small-1", 5, 0, A)
    assert len(defective) == n == config.n_atoms == config.n_parent_atoms
    assert Counter(defective.get_chemical_symbols()) == Counter(parent.get_chemical_symbols())
    assert defective.get_chemical_symbols() == parent.get_chemical_symbols()
    assert defective.get_array("atom_id").tolist() == list(range(n))
    others = [k for k in range(n) if k != 5]
    assert np.array_equal(defective.positions[others], parent.positions[others])
    vacancy, interstitial = config.sites
    assert (vacancy.role, vacancy.atom_id, vacancy.lattice_site) == ("vacancy", 5, 5)
    assert (interstitial.role, interstitial.atom_id, interstitial.lattice_site) == (
        "interstitial",
        5,
        None,
    )
    assert vacancy.species == interstitial.species == parent[5].symbol
    assert vacancy.position == tuple(parent.positions[5])
    assert interstitial.initial_position == vacancy.position
    assert interstitial.position == tuple(defective.positions[5])
    assert DefectConfiguration.model_validate_json(config.model_dump_json()) == config


def test_frenkel_pair_records_the_minimum_image_separation_and_is_deterministic():
    parent = generate_alloy(SMALL)
    defective, config = generate_frenkel_pair(
        parent, "small-1", 5, 0, A, site_type="tetrahedral", variant=3
    )
    expected, _ = generate_bcc_interstitial(
        parent, "small-1", 0, parent[5].symbol, "tetrahedral", A, variant=3
    )
    assert np.array_equal(defective.positions[5], expected.positions[-1])
    _, dist = get_distances(defective.positions[5], parent.positions[5], cell=parent.cell, pbc=True)
    assert config.spec.parameters["separation"] == pytest.approx(dist[0, 0])
    again = generate_frenkel_pair(parent, "small-1", 5, 0, A, site_type="tetrahedral", variant=3)
    assert again[1] == config and np.array_equal(again[0].positions, defective.positions)


def test_frenkel_pair_enforces_the_configurable_minimum_separation():
    parent = generate_alloy(SMALL)
    _, config = generate_frenkel_pair(parent, "small-1", 5, 0, A)
    separation = config.spec.parameters["separation"]
    generate_frenkel_pair(parent, "small-1", 5, 0, A, min_separation=separation - 0.01)
    with pytest.raises(ValueError, match="separation"):
        generate_frenkel_pair(parent, "small-1", 5, 0, A, min_separation=separation + 0.01)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"site": 16},
        {"site": True},
        {"anchor": 16},
        {"site_type": "hexahedral"},
        {"variant": 99},
        {"min_separation": -1.0},
        {"min_separation": float("nan")},
        {"lattice_parameter": 3.0},
    ],
)
def test_invalid_frenkel_pairs_are_rejected(kwargs):
    arguments = {"site": 5, "anchor": 0, "lattice_parameter": A} | kwargs
    with pytest.raises(ValueError):
        generate_frenkel_pair(
            generate_alloy(SMALL),
            "small-1",
            arguments.pop("site"),
            arguments.pop("anchor"),
            arguments.pop("lattice_parameter"),
            **arguments,
        )


# --- A3.9 Divacancies ---


def test_first_and_second_neighbour_divacancies_remove_the_selected_pair():
    parent = generate_alloy(CROWD)
    n = len(parent)
    for shell, ratio in ((1, 3**0.5 / 2), (2, 1.0)):
        defective, config = generate_divacancy(parent, "d", 4, shell=shell, lattice_parameter=A)
        low, high = config.spec.site_indices
        assert 4 in (low, high) and low < high
        assert len(defective) == n - 2 == config.n_atoms
        assert config.spec.parameters["shell"] == shell
        assert config.spec.parameters["separation"] == pytest.approx(ratio * A)
        assert [s.role for s in config.sites] == ["vacancy", "vacancy"]
        assert [s.lattice_site for s in config.sites] == [low, high]
        kept = [k for k in range(n) if k not in (low, high)]
        assert defective.get_array("atom_id").tolist() == kept
        assert np.array_equal(defective.positions, parent.positions[kept])
        removed = ",".join(parent[k].symbol for k in (low, high))
        assert config.spec.parameters["removed_species"] == removed
        assert config.spec.parameters["selection"] == "shell"
        assert DefectConfiguration.model_validate_json(config.model_dump_json()) == config


def test_shell_variants_enumerate_distinct_neighbours_and_are_deterministic():
    parent = generate_alloy(CROWD)
    partners = set()
    for variant in range(8):
        _, config = generate_divacancy(
            parent, "d", 4, shell=1, variant=variant, lattice_parameter=A
        )
        partners.add(next(k for k in config.spec.site_indices if k != 4))
    assert len(partners) == 8
    again = generate_divacancy(parent, "d", 4, shell=2, variant=3, lattice_parameter=A)
    assert (
        again[1] == generate_divacancy(parent, "d", 4, shell=2, variant=3, lattice_parameter=A)[1]
    )
    with pytest.raises(ValueError, match="variant"):
        generate_divacancy(parent, "d", 4, shell=1, variant=8, lattice_parameter=A)
    with pytest.raises(ValueError, match="variant"):
        generate_divacancy(parent, "d", 4, shell=2, variant=6, lattice_parameter=A)


def test_explicit_divacancy_records_the_periodic_separation_and_shell():
    parent = generate_alloy(CROWD)
    corner = int(np.argmin(np.linalg.norm(parent.positions, axis=1)))
    far = int(np.argmax(parent.positions[:, 0] + parent.positions[:, 1] + parent.positions[:, 2]))
    _, config = generate_divacancy(parent, "d", far, partner=corner)
    expected = get_distances(
        parent.positions[far], parent.positions[corner], cell=parent.cell, pbc=True
    )[1][0, 0]
    assert config.spec.parameters["separation"] == pytest.approx(expected)
    assert config.spec.parameters["shell"] == 0
    assert config.spec.parameters["selection"] == "explicit"
    first_neighbour = next(
        k
        for k in range(1, len(parent))
        if np.linalg.norm(_mic(parent, 0, k)) == pytest.approx(A * 3**0.5 / 2)
    )
    _, config = generate_divacancy(parent, "d", 0, partner=first_neighbour, lattice_parameter=A)
    assert config.spec.parameters["shell"] == 1
    swapped = generate_divacancy(parent, "d", first_neighbour, partner=0, lattice_parameter=A)
    assert swapped[1] == config


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"partner": 1, "shell": 1},
        {"partner": 4},
        {"partner": 999},
        {"partner": True},
        {"partner": 1.5},
        {"shell": 3},
        {"shell": 1, "lattice_parameter": None},
        {"shell": 1, "variant": -1},
        {"shell": 1, "lattice_parameter": 0.0},
        {"shell": 1, "lattice_parameter": float("nan")},
        {"partner": 1, "variant": True},
        {"partner": 1, "lattice_parameter": 1.0, "shell": None, "variant": 0.5},
    ],
)
def test_invalid_divacancy_requests_are_rejected(kwargs):
    arguments = {"lattice_parameter": A} | kwargs
    with pytest.raises(ValueError):
        generate_divacancy(generate_alloy(CROWD), "d", 4, **arguments)


def test_divacancy_rejects_a_non_periodic_parent_and_empty_parent_id():
    parent = generate_alloy(CROWD)
    open_cell = parent.copy()
    open_cell.pbc = False
    with pytest.raises(ValueError, match="periodic"):
        generate_divacancy(open_cell, "d", 4, shell=1, lattice_parameter=A)
    with pytest.raises(ValueError, match="parent_id"):
        generate_divacancy(parent, "", 4, shell=1, lattice_parameter=A)
