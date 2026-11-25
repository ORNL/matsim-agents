"""Structure generation on any parent lattice and any integer supercell."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
from pydantic import ValidationError

from matsim_agents.science.structures.alloy_ensemble import (
    find_symmetry_equivalents,
    generate_alloy_ensemble,
)
from matsim_agents.science.structures.alloys import (
    AlloySpec,
    SROTarget,
    UnitCell,
    build_supercell,
    generate_alloy,
    generate_targeted_sro,
    shell_topology,
)

CANTOR = {"Co": 52, "Cr": 51, "Fe": 51, "Mn": 51, "Ni": 51}


def _coordination(atoms, a):
    topology = shell_topology(atoms, a)
    first = set(np.bincount(topology.first[0], minlength=len(atoms)))
    second = set(np.bincount(topology.second[0], minlength=len(atoms)))
    return first, second


def test_cantor_fcc_4x4x4_builds_256_sites_with_exact_counts():
    spec = AlloySpec(
        crystal_structure="fcc", counts=CANTOR, repeats=(4, 4, 4), lattice_parameter=3.6, seed=7
    )
    atoms = generate_alloy(spec)
    assert len(atoms) == spec.n_sites == 256
    assert Counter(atoms.get_chemical_symbols()) == CANTOR
    assert np.allclose(atoms.cell.array, np.diag([14.4] * 3))
    assert _coordination(atoms, 3.6) == ({12}, {6})
    again = generate_alloy(spec)
    assert atoms.get_chemical_symbols() == again.get_chemical_symbols()


@pytest.mark.parametrize(
    ("structure", "extra", "repeats", "sites", "shells"),
    [
        ("sc", {}, (3, 3, 3), 27, ({6}, {12})),
        ("bcc", {}, (8, 8, 8), 1024, ({8}, {6})),
        ("fcc", {}, (2, 2, 2), 32, ({12}, {6})),
        ("hcp", {}, (4, 4, 3), 96, ({12}, {6})),
        ("diamond", {}, (2, 2, 2), 64, ({4}, {12})),
        ("fcc", {"cell_choice": "primitive"}, (3, 3, 3), 27, ({12}, {6})),
        ("bcc", {"cell_choice": "primitive"}, (3, 3, 3), 27, ({8}, {6})),
    ],
)
def test_named_lattices_have_their_shells(structure, extra, repeats, sites, shells):
    spec = AlloySpec(
        crystal_structure=structure,
        counts={"Mo": sites // 2, "W": sites - sites // 2},
        repeats=repeats,
        seed=1,
        **extra,
    )
    atoms = generate_alloy(spec)
    assert len(atoms) == spec.n_sites == sites
    assert _coordination(atoms, spec.lattice_parameter) == shells


def test_matrix_repeats_build_non_diagonal_supercells():
    matrix = ((-1, 1, 1), (1, -1, 1), (1, 1, -1))
    spec = AlloySpec(crystal_structure="fcc", counts={"Ni": 16}, repeats=matrix, seed=0)
    atoms = build_supercell(spec)
    assert len(atoms) == spec.n_sites == 16
    assert _coordination(atoms, spec.lattice_parameter) == ({12}, {6})


def test_custom_unit_cell():
    rocksalt_cation = UnitCell(
        cell=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        scaled_positions=((0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)),
    )
    spec = AlloySpec(
        crystal_structure="custom",
        unit_cell=rocksalt_cation,
        counts={"Mg": 16, "Ni": 16},
        repeats=(2, 2, 2),
        lattice_parameter=4.2,
        seed=3,
    )
    atoms = generate_alloy(spec)
    assert len(atoms) == 32
    assert _coordination(atoms, 4.2) == ({12}, {6})


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"crystal_structure": "fcc", "counts": {"W": 31}, "repeats": (2, 2, 2)}, "32 sites"),
        ({"crystal_structure": "custom", "counts": {"W": 1}}, "unit_cell"),
        ({"crystal_structure": "bcc", "counts": {"W": 2}, "c_over_a": 1.6}, "hcp"),
        (
            {
                "crystal_structure": "fcc",
                "counts": {"W": 4},
                "repeats": ((1, 0, 0), (1, 0, 0), (0, 0, 1)),
            },
            "singular",
        ),
    ],
)
def test_invalid_specs_are_rejected(kwargs, message):
    with pytest.raises(ValidationError, match=message):
        AlloySpec(seed=0, **kwargs)


def test_unit_cell_rejects_coincident_sites():
    with pytest.raises(ValidationError):
        UnitCell(cell=((1, 0, 0), (0, 1, 0), (0, 0, 1)), scaled_positions=((0, 0, 0), (1, 0, 0)))


def test_targeted_sro_runs_on_fcc():
    spec = AlloySpec(
        crystal_structure="fcc", counts={"Cu": 54, "Au": 54}, repeats=(3, 3, 3), seed=2
    )
    atoms = generate_alloy(spec)
    target = SROTarget(shell=1, pair=("Au", "Cu"), alpha=-0.1)
    result = generate_targeted_sro(atoms, spec.lattice_parameter, [target], seed=2)
    assert Counter(result.atoms.get_chemical_symbols()) == {"Cu": 54, "Au": 54}
    assert result.objective < result.initial_objective


def test_fcc_ensemble_symmetry_deduplication_is_complete():
    alloy = AlloySpec(
        crystal_structure="fcc", counts={"Cu": 16, "Ni": 16}, repeats=(2, 2, 2), seed=0
    )
    ensemble = generate_alloy_ensemble(alloy, n_configurations=3, mode="random", seed=0)
    result = find_symmetry_equivalents(ensemble)
    assert result.status == "complete"
