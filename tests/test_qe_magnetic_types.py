"""QE atomic-type mapping for collinear magnetic configurations."""

from __future__ import annotations

import numpy as np
import pytest
from ase.build import bulk

from matsim_agents.backends.dft.qe_magnetic import build_magnetic_types
from matsim_agents.discovery import magnetism_generators as gen
from matsim_agents.discovery.magnetism import MagneticConfiguration

A = 2.87
PSEUDOS = {"Fe": "Fe.pbe.UPF", "Cr": "Cr.pbe.UPF"}


def _cell(n=2):
    atoms = bulk("Fe", "bcc", a=A, cubic=True) * (n, n, n)
    atoms.set_array("atom_id", np.arange(len(atoms), dtype=np.int64))
    return atoms


def _with_polarization(cfg: MagneticConfiguration, scale: float = 0.25):
    polar = tuple(m / 8.0 * 4.0 * scale / 2.0 for m in cfg.initial_moments_muB)
    return cfg.model_copy(update={"qe_starting_magnetization": polar})


def test_nonmagnetic_fe_has_one_type_and_nspin_one():
    atoms = _cell()
    mapping = build_magnetic_types(atoms, None, PSEUDOS)
    assert mapping.ntyp == 1 and mapping.nspin == 1
    assert mapping.types[0].label == "Fe"
    assert mapping.system_entries() == {}


def test_nonmagnetic_configuration_matches_none():
    atoms = _cell()
    cfg = gen.nonmagnetic(atoms, "s")
    assert build_magnetic_types(atoms, cfg, PSEUDOS).ntyp == 1


def test_ferromagnetic_fe_single_type_with_polarization():
    atoms = _cell()
    cfg = _with_polarization(gen.ferromagnetic(atoms, "s", {"Fe": 2.2}))
    mapping = build_magnetic_types(atoms, cfg, PSEUDOS)
    assert mapping.ntyp == 1 and mapping.nspin == 2
    entries = mapping.system_entries()
    assert entries["nspin"] == 2
    assert entries["starting_magnetization(1)"] > 0


def test_afm_fe_uses_two_types_sharing_one_pseudopotential():
    atoms = _cell()
    cfg = _with_polarization(gen.bcc_afm(atoms, "s", A, {"Fe": 2.0}))
    mapping = build_magnetic_types(atoms, cfg, PSEUDOS)
    assert mapping.ntyp == 2
    assert [t.label for t in mapping.types] == ["Fe1", "Fe2"]
    assert {t.pseudopotential for t in mapping.types} == {"Fe.pbe.UPF"}
    assert {t.element for t in mapping.types} == {"Fe"}
    signs = [np.sign(t.starting_magnetization) for t in mapping.types]
    assert sorted(signs) == [-1.0, 1.0]
    assert sum(len(t.atom_ids) for t in mapping.types) == len(atoms)
    entries = mapping.system_entries()
    assert entries["starting_magnetization(1)"] == -entries["starting_magnetization(2)"]


def test_positions_use_type_labels_in_atom_order():
    atoms = _cell()
    cfg = _with_polarization(gen.bcc_afm(atoms, "s", A, {"Fe": 2.0}))
    mapping = build_magnetic_types(atoms, cfg, PSEUDOS)
    lines = mapping.position_lines(atoms)
    assert len(lines) == len(atoms)
    labels = [line.split()[0] for line in lines]
    assert set(labels) == {"Fe1", "Fe2"}
    reordered = atoms[::-1]
    assert mapping.labels_for_atoms(reordered) == labels[::-1]


def test_fe_cr_mixed_groups_label_per_element():
    atoms = _cell()
    atoms.symbols[[0, 1]] = "Cr"
    moments = {i: 2.2 for i in range(len(atoms))}
    moments[0], moments[1] = -1.0, -1.5
    moments[2] = -2.0
    cfg = _with_polarization(gen.custom_collinear(atoms, "s", moments))
    mapping = build_magnetic_types(atoms, cfg, PSEUDOS)
    by_element: dict[str, list[str]] = {}
    for t in mapping.types:
        by_element.setdefault(t.element, []).append(t.label)
    assert by_element["Cr"] == ["Cr1", "Cr2"]
    assert by_element["Fe"] == ["Fe1", "Fe2"]
    assert mapping.ntyp == 4
    assert all(t.pseudopotential == PSEUDOS[t.element] for t in mapping.types)


def test_distinct_initializations_are_never_merged():
    atoms = _cell()
    moments = {i: 2.0 + 0.01 * (i % 3) for i in range(len(atoms))}
    cfg = _with_polarization(gen.custom_collinear(atoms, "s", moments))
    assert build_magnetic_types(atoms, cfg, PSEUDOS).ntyp == 3


def test_labels_are_deterministic():
    atoms = _cell()
    cfg = _with_polarization(gen.bcc_afm(atoms, "s", A, {"Fe": 2.0}))
    first = build_magnetic_types(atoms, cfg, PSEUDOS)
    second = build_magnetic_types(atoms[::-1], cfg, PSEUDOS)
    assert [(t.label, t.atom_ids) for t in first.types] == [
        (t.label, t.atom_ids) for t in second.types
    ]


def test_too_many_types_for_label_syntax_is_rejected():
    atoms = _cell(3)
    moments = {i: 1.0 + 0.01 * i for i in range(len(atoms))}
    cfg = _with_polarization(gen.custom_collinear(atoms, "s", moments), 0.01)
    with pytest.raises(ValueError, match="at most 3"):
        build_magnetic_types(atoms, cfg, PSEUDOS)


def test_missing_pseudopotential_and_polarization_are_rejected():
    atoms = _cell()
    cfg = gen.bcc_afm(atoms, "s", A, {"Fe": 2.0})
    with pytest.raises(ValueError, match="no pseudopotential"):
        build_magnetic_types(atoms, cfg, {})
    with pytest.raises(ValueError, match="starting polarization"):
        build_magnetic_types(atoms, cfg, PSEUDOS)


def test_configuration_for_other_structure_is_rejected():
    atoms = _cell()
    cfg = _with_polarization(gen.bcc_afm(atoms, "s", A, {"Fe": 2.0}))
    other = _cell(3)
    with pytest.raises(ValueError, match="do not match"):
        build_magnetic_types(other, cfg, PSEUDOS)
