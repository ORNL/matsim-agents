"""Synthetic-output tests for magnetic result parsing and classification (M5)."""

from __future__ import annotations

import numpy as np
import pytest
from ase.build import bulk

from matsim_agents.discovery.magnetism import generators as gen
from matsim_agents.discovery.magnetism.results import (
    ClassificationTolerances,
    classify_moments,
    compare_with_request,
    parse_qe_magnetic_output,
    result_from_moments,
    spin_reversed_equivalent,
)

A = 2.87


def _atoms(n=2):
    atoms = bulk("Fe", "bcc", a=A, cubic=True) * (n, 1, 1)
    atoms.set_array("atom_id", np.arange(len(atoms), dtype=np.int64))
    return atoms


def _out(moments, total=None, converged=True):
    absolute = sum(abs(m) for m in moments)
    lines = ["     iteration #  1", "     total magnetization       =     9.99 Bohr mag/cell"]
    lines += ["     Magnetic moment per site:"]
    lines += [
        f"     atom:    {i + 1}    charge:    7.0000    magn:    {m:.4f}    constr:    0.0000"
        for i, m in enumerate(moments)
    ]
    lines += [
        "     iteration #  2",
        f"     total magnetization       =     {sum(moments) if total is None else total:.2f}"
        " Bohr mag/cell",
        f"     absolute magnetization    =     {absolute:.2f} Bohr mag/cell",
        "     Magnetic moment per site:",
    ]
    lines += [
        f"     atom:    {i + 1}    charge:    7.0000    magn:    {m:.4f}    constr:    0.0000"
        for i, m in enumerate(moments)
    ]
    lines += [
        "     convergence has been achieved in   9 iterations"
        if converged
        else "     convergence NOT achieved after 100 iterations: stopping",
        "     JOB DONE.",
    ]
    return "\n".join(lines)


def test_parses_last_block_and_totals():
    res = parse_qe_magnetic_output(_out([2.2, 2.1, 2.3, 2.2]), _atoms(2))
    assert res.converged
    assert res.local_moments_muB == (2.2, 2.1, 2.3, 2.2)
    assert res.atom_ids == (0, 1, 2, 3)


def test_parses_two_atom_cell_exactly():
    atoms = _atoms(1)
    res = parse_qe_magnetic_output(_out([2.2, -2.1]), atoms)
    assert res.local_moments_muB == (2.2, -2.1)
    assert res.total_magnetization_muB == pytest.approx(0.1)
    assert res.absolute_magnetization_muB == pytest.approx(4.3)
    assert res.atom_ids == (0, 1)


def test_not_converged_and_missing_sites():
    atoms = _atoms(1)
    assert classify_moments(parse_qe_magnetic_output(_out([2, 2], converged=False), atoms)) == (
        "not_converged"
    )
    res = parse_qe_magnetic_output(
        "convergence has been achieved in 5 iterations\n     total magnetization = 0.0",
        atoms,
    )
    assert classify_moments(res) == "insufficient_evidence"
    assert res.notes


@pytest.mark.parametrize(
    ("moments", "expected"),
    [
        ([0.02, -0.03], "nonmagnetic"),
        ([2.2, 2.1], "fm_like"),
        ([2.2, -2.2], "afm_like"),
        ([2.2, -1.0], "ferrimagnetic_like"),
        ([2.2, 0.05], "fm_like"),
    ],
)
def test_classification(moments, expected):
    res = result_from_moments(moments, [0, 1])
    assert classify_moments(res) == expected


def test_zero_total_magnetization_is_not_afm_by_itself():
    zero = result_from_moments([0.0, 0.0], [0, 1])
    assert classify_moments(zero) == "nonmagnetic"
    res = parse_qe_magnetic_output(_out([0.01, -0.02]), _atoms(1))
    assert classify_moments(res) == "nonmagnetic"


def test_tolerances_are_configurable():
    res = result_from_moments([2.2, -1.5], [0, 1])  # net fraction about 0.19
    assert classify_moments(res) == "ferrimagnetic_like"
    loose = ClassificationTolerances(afm_net_fraction=0.2)
    assert classify_moments(res, loose) == "afm_like"


def test_comparison_collapse_reversal_and_change():
    atoms = _atoms(1)
    cfg = gen.bcc_afm(atoms, "s", A, {"Fe": 2.0})
    signs = np.sign(cfg.initial_moments_muB)
    same = result_from_moments(list(2.2 * signs), cfg.atom_ids)
    assert compare_with_request(cfg, same).relation == "same_arrangement"
    flipped = result_from_moments(list(-2.2 * signs), cfg.atom_ids)
    assert compare_with_request(cfg, flipped).relation == "same_up_to_global_reversal"
    nm = result_from_moments([0.0, 0.0], cfg.atom_ids)
    assert compare_with_request(cfg, nm).relation == "collapsed_to_nonmagnetic"
    fm = result_from_moments([2.0, 2.0], cfg.atom_ids)
    assert compare_with_request(cfg, fm).relation == "different_arrangement"
    bad = result_from_moments([2.0, 2.0], cfg.atom_ids, converged=False)
    assert compare_with_request(cfg, bad).relation == "indeterminate"


def test_spin_reversal_equivalence():
    a = result_from_moments([2.0, -2.0], [0, 1])
    assert spin_reversed_equivalent(a, result_from_moments([-2.0, 2.0], [0, 1]))
    assert spin_reversed_equivalent(a, result_from_moments([2.01, -2.0], [0, 1]))
    assert not spin_reversed_equivalent(a, result_from_moments([2.0, 2.0], [0, 1]))
    assert not spin_reversed_equivalent(a, result_from_moments([2.0, -2.0], [0, 2]))


def test_with_converged_does_not_change_mode():
    atoms = _atoms(1)
    cfg = gen.bcc_afm(atoms, "s", A, {"Fe": 2.0})
    res = result_from_moments([2.0, 2.0], cfg.atom_ids)
    cmp = compare_with_request(cfg, res)
    updated = cfg.with_converged(res.local_moments_muB, cmp.classification)
    assert updated.mode == "antiferromagnetic"
    assert updated.converged_classification == "fm_like"
