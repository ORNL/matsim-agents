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
    assign_species,
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


# ── reference-structure input: geometry from any crystal, occupation separate ─────────────── #


def _reference_cases():
    from ase.build import bulk

    from matsim_agents.science.structures.alloys import AlloySpec  # noqa: F401

    w_bcc = bulk("W", "bcc", a=3.16, cubic=True)
    cantor_fcc = bulk("Ni", "fcc", a=3.6, cubic=True)
    hcp = bulk("Ti", "hcp", a=2.95, c=4.68)
    return w_bcc, cantor_fcc, hcp


def test_reference_bcc_tungsten_matches_named_bcc_geometry():
    w_bcc, _, _ = _reference_cases()
    counts = {"W": 120, "Ta": 4, "V": 4}
    from_ref = AlloySpec.from_reference(w_bcc, counts=counts, repeats=(4, 4, 4), seed=3)
    named = AlloySpec(
        crystal_structure="bcc", counts=counts, repeats=(4, 4, 4), lattice_parameter=3.16, seed=3
    )
    a, b = generate_alloy(from_ref), generate_alloy(named)
    assert from_ref.n_sites == named.n_sites == 128
    assert np.allclose(a.cell.array, b.cell.array)
    assert np.allclose(
        np.sort(a.get_scaled_positions(), axis=0), np.sort(b.get_scaled_positions(), axis=0)
    )


def test_reference_cantor_fcc_has_exact_counts_and_coordination():
    _, fcc, _ = _reference_cases()
    spec = AlloySpec.from_reference(fcc, counts=CANTOR, repeats=(4, 4, 4), seed=1)
    atoms = generate_alloy(spec)
    assert len(atoms) == 256
    assert Counter(atoms.get_chemical_symbols()) == CANTOR
    assert _coordination(atoms, spec.lattice_parameter) == ({12}, {6})


def test_reference_hcp_and_primitive_supercell_matrix():
    _, _, hcp = _reference_cases()
    spec = AlloySpec.from_reference(
        hcp, counts={"Ti": 20, "Zr": 16}, repeats=((3, 0, 0), (0, 3, 0), (0, 0, 2)), seed=2
    )
    atoms = generate_alloy(spec)
    assert len(atoms) == spec.n_sites == 36
    assert Counter(atoms.get_chemical_symbols()) == {"Ti": 20, "Zr": 16}


def test_reference_arbitrary_user_crystal_from_cif(tmp_path):
    from ase.io import write
    from ase.spacegroup import crystal

    # Rock-salt-like two-site basis plus a low-symmetry (monoclinic) cell: no named lattice.
    atoms = crystal(
        ["Mg", "O"],
        basis=[(0, 0, 0), (0.5, 0.5, 0.5)],
        spacegroup=12,
        cellpar=[5.0, 3.1, 6.0, 90, 105, 90],
    )
    path = tmp_path / "user.cif"
    write(path, atoms)
    n = len(atoms)
    spec = AlloySpec.from_reference(
        path, counts={"Fe": n * 2, "Co": n * 2}, repeats=(2, 2, 1), seed=4
    )
    built = generate_alloy(spec)
    assert len(built) == spec.n_sites == n * 4
    assert Counter(built.get_chemical_symbols()) == {"Fe": n * 2, "Co": n * 2}
    assert np.allclose(built.cell.lengths()[:2], atoms.cell.lengths()[:2] * 2)


def test_reference_accepts_pymatgen_structure():
    from pymatgen.core import Lattice, Structure

    structure = Structure(Lattice.cubic(3.0), ["Cu", "Cu"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    spec = AlloySpec.from_reference(structure, counts={"Cu": 8, "Ni": 8}, repeats=(2, 2, 2), seed=0)
    assert len(generate_alloy(spec)) == 16


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"counts": {"W": 100}, "repeats": (4, 4, 4)}, "sum to 100"),
        ({"counts": {"W": 64, "Xx": 64}, "repeats": (4, 4, 4)}, "unknown element"),
        ({"counts": {"W": 128}, "repeats": (0, 4, 4)}, "positive"),
    ],
)
def test_reference_rejects_inconsistent_configurations(kwargs, message):
    w_bcc, _, _ = _reference_cases()
    with pytest.raises(ValidationError, match=message):
        AlloySpec.from_reference(w_bcc, seed=0, **kwargs)


def test_reference_rejects_nonperiodic_and_unsupported_input():
    from ase import Atoms

    with pytest.raises(ValueError, match="periodic"):
        AlloySpec.from_reference(
            Atoms("H", cell=[3, 3, 3], pbc=False), counts={"H": 1}, repeats=(1, 1, 1), seed=0
        )
    with pytest.raises(TypeError, match="reference must be"):
        AlloySpec.from_reference(object(), counts={"H": 1}, repeats=(1, 1, 1), seed=0)


def test_occupation_is_independent_of_geometry_source():
    w_bcc, _, _ = _reference_cases()
    counts = {"W": 60, "Ta": 4}
    spec = AlloySpec.from_reference(w_bcc, counts=counts, repeats=(4, 4, 2), seed=9)
    assert assign_species(spec) == assign_species(spec)
    assert sorted(assign_species(spec)) == sorted(["W"] * 60 + ["Ta"] * 4)
