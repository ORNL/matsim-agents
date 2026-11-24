"""A3.1: defect data contracts."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
from ase.geometry import get_distances
from pydantic import ValidationError

from matsim_agents.discovery.alloy_ensemble import occupation_fingerprint
from matsim_agents.discovery.alloys import AlloySpec, generate_alloy
from matsim_agents.discovery.radiation_defects import (
    DUMBBELL_ORIENTATIONS,
    DefectConfiguration,
    DefectOutcome,
    DefectSite,
    DefectSpec,
    generate_dumbbell_interstitial,
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
