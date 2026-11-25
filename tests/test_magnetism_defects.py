"""Magnetic configuration provenance through defects and NEB images (M7)."""

from __future__ import annotations

import numpy as np
import pytest
from ase.build import bulk

from matsim_agents.science.magnetism import generators as gen
from matsim_agents.science.magnetism.defects import (
    InsertedMoment,
    check_neb_magnetic_compatibility,
    derive_defect_configuration,
    flag_magnetic_path,
)
from matsim_agents.science.magnetism.results import result_from_moments

A = 2.87


def _parent():
    atoms = bulk("Fe", "bcc", a=A, cubic=True) * (2, 2, 2)
    atoms.set_array("atom_id", np.arange(len(atoms), dtype=np.int64))
    return atoms, gen.bcc_afm(atoms, "s", A, {"Fe": 2.0})


def _remove(atoms, drop):
    keep = [i for i in range(len(atoms)) if i not in drop]
    return atoms[keep]


def test_vacancy_keeps_parent_moments_by_atom_id():
    atoms, cfg = _parent()
    vac = _remove(atoms, {3})
    derived = derive_defect_configuration(cfg, vac, defect_structure_id="v1", defect_kind="vacancy")
    assert derived.mode == "antiferromagnetic"
    assert 3 not in derived.atom_ids and len(derived.atom_ids) == len(atoms) - 1
    parent_map = dict(zip(cfg.atom_ids, cfg.initial_moments_muB, strict=True))
    for atom_id, moment in zip(derived.atom_ids, derived.initial_moments_muB, strict=True):
        assert moment == parent_map[atom_id]
    assert derived.provenance["removed_atom_ids"] == [3]
    assert derived.provenance["parent_fingerprint"] == cfg.fingerprint()
    derived.validate_against(vac)


def test_vacancy_is_independent_of_atom_order():
    atoms, cfg = _parent()
    vac = _remove(atoms, {3})
    a = derive_defect_configuration(cfg, vac, defect_structure_id="v", defect_kind="vacancy")
    b = derive_defect_configuration(cfg, vac[::-1], defect_structure_id="v", defect_kind="vacancy")
    assert a.fingerprint() == b.fingerprint()


def test_interstitial_without_moment_is_rejected():
    atoms, cfg = _parent()
    inter = atoms.copy()
    inter.append("C")
    ids = list(atoms.get_array("atom_id")) + [len(atoms)]
    inter.set_array("atom_id", np.array(ids, dtype=np.int64))
    with pytest.raises(ValueError, match="no parent moment"):
        derive_defect_configuration(cfg, inter, defect_structure_id="i", defect_kind="interstitial")


def test_interstitial_with_explicit_and_documented_policy():
    atoms, cfg = _parent()
    inter = atoms.copy()
    inter.append("C")
    ids = list(atoms.get_array("atom_id")) + [len(atoms)]
    inter.set_array("atom_id", np.array(ids, dtype=np.int64))
    new_id = len(atoms)
    explicit = derive_defect_configuration(
        cfg,
        inter,
        defect_structure_id="i",
        defect_kind="interstitial",
        inserted={new_id: InsertedMoment(-0.3)},
    )
    assert explicit.initial_moments_muB[explicit.atom_ids.index(new_id)] == -0.3
    assert any("explicit policy" in a for a in explicit.assumptions)
    zero = derive_defect_configuration(
        cfg,
        inter,
        defect_structure_id="i",
        defect_kind="interstitial",
        inserted_policy=InsertedMoment(policy="zero_documented"),
    )
    assert zero.initial_moments_muB[zero.atom_ids.index(new_id)] == 0.0
    assert any("zero_documented" in a for a in zero.assumptions)


def test_frenkel_and_divacancy_compose():
    atoms, cfg = _parent()
    div = derive_defect_configuration(
        cfg, _remove(atoms, {0, 1}), defect_structure_id="d", defect_kind="divacancy"
    )
    assert div.provenance["removed_atom_ids"] == [0, 1]
    frenkel = _remove(atoms, {2}).copy()
    frenkel.append("Fe")
    ids = [int(i) for i in _remove(atoms, {2}).get_array("atom_id")] + [len(atoms)]
    frenkel.set_array("atom_id", np.array(ids, dtype=np.int64))
    out = derive_defect_configuration(
        cfg,
        frenkel,
        defect_structure_id="f",
        defect_kind="frenkel_pair",
        inserted={len(atoms): InsertedMoment(2.0)},
    )
    assert out.provenance["removed_atom_ids"] == [2]
    assert out.provenance["new_atom_ids"] == [len(atoms)]


def test_mode_downgrades_to_custom_when_order_is_lost():
    atoms = bulk("Fe", "bcc", a=A, cubic=True)
    atoms.set_array("atom_id", np.arange(len(atoms), dtype=np.int64))
    cfg = gen.bcc_afm(atoms, "s", A, {"Fe": 2.0})
    derived = derive_defect_configuration(
        cfg, _remove(atoms, {1}), defect_structure_id="v", defect_kind="vacancy"
    )
    assert derived.mode == "custom_collinear"
    assert any("no longer holds" in a for a in derived.assumptions)


def test_neb_compatibility_detects_changed_initialization():
    atoms, cfg = _parent()
    ok = check_neb_magnetic_compatibility([cfg, cfg])
    assert ok.compatible and ok.magnetic_type_map
    changed = cfg.model_copy(
        update={
            "initial_moments_muB": tuple(m * 0.5 for m in cfg.initial_moments_muB),
            "qe_starting_magnetization": None,
        }
    )
    bad = check_neb_magnetic_compatibility([cfg, cfg, changed])
    assert not bad.compatible and "image 2" in bad.problems[0]
    short = check_neb_magnetic_compatibility([cfg])
    assert not short.compatible


def test_neb_compatibility_rejects_different_atom_sets():
    atoms, cfg = _parent()
    vac = derive_defect_configuration(
        cfg, _remove(atoms, {3}), defect_structure_id="v", defect_kind="vacancy"
    )
    assert not check_neb_magnetic_compatibility([cfg, vac]).compatible


def test_path_flags_switching_collapse_and_clean_path():
    ids = [0, 1]
    clean = [result_from_moments([2.0, -2.0], ids), result_from_moments([2.1, -1.9], ids)]
    assert flag_magnetic_path(clean).qualified_for_barrier
    switch = clean + [result_from_moments([-2.0, -2.0], ids)]
    report = flag_magnetic_path(switch)
    assert not report.qualified_for_barrier
    assert any("switched" in f for f in report.flags)
    collapse = [clean[0], result_from_moments([0.0, 0.0], ids)]
    assert any("collapse" in f for f in flag_magnetic_path(collapse).flags)
    bad = [clean[0], result_from_moments([2.0, -2.0], ids, converged=False)]
    assert any("not_converged" in f for f in flag_magnetic_path(bad).flags)
