"""Magnetic consistency of formation energies, barriers and KMC admission (M8)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from ase.build import bulk

from matsim_agents.discovery.magnetism import generators as gen
from matsim_agents.discovery.magnetism.consistency import (
    ConsistencyReport,
    MagneticEvidence,
    arrhenius_rate,
    check_barrier,
    check_formation_energy,
    gate_transition_for_kmc,
    rate_sensitivity,
)
from matsim_agents.discovery.magnetism.results import result_from_moments

A = 2.87
IDS = [0, 1]


def _cfg(kind="afm"):
    atoms = bulk("Fe", "bcc", a=A, cubic=True)
    atoms.set_array("atom_id", np.arange(len(atoms), dtype=np.int64))
    if kind == "nm":
        return gen.nonmagnetic(atoms, "s")
    return gen.bcc_afm(atoms, "s", A, {"Fe": 2.0})


def _ev(label, moments=(2.0, -2.0), method="m1", cfg=None, qual="numerically_validated", conv=True):
    return MagneticEvidence(
        label,
        method,
        cfg if cfg is not None else _cfg(),
        result_from_moments(list(moments), IDS, converged=conv),
        qual,
    )


def test_formation_energy_consistent_pair():
    report = check_formation_energy(_ev("defect"), _ev("pristine"))
    assert report.verdict == "consistent" and report.usable


def test_formation_energy_rejects_method_spin_and_state_mismatch():
    assert check_formation_energy(_ev("d", method="m2"), _ev("p")).verdict == "inconsistent"
    nm = _ev("d", moments=(0.0, 0.0), cfg=_cfg("nm"))
    report = check_formation_energy(nm, _ev("p"))
    assert report.verdict == "inconsistent"
    assert any("spin polarization" in r for r in report.reasons)
    collapsed = _ev("d", moments=(0.0, 0.0))
    mixed = check_formation_energy(collapsed, _ev("p"))
    assert any("mixes magnetic states" in r for r in mixed.reasons)
    relaxed = check_formation_energy(collapsed, _ev("p"), require_same_class=False)
    assert relaxed.usable


def test_formation_energy_needs_evidence():
    missing = MagneticEvidence("d", "m1", None, None)
    report = check_formation_energy(missing, _ev("p"))
    assert report.verdict == "insufficient_evidence"
    unconverged = check_formation_energy(_ev("d", conv=False), _ev("p"))
    assert unconverged.verdict == "insufficient_evidence"


def test_barrier_consistency_and_path_flags():
    ok = check_barrier(_ev("i"), [_ev("saddle")], _ev("f"))
    assert ok.usable
    flagged = check_barrier(_ev("i"), [_ev("saddle")], _ev("f"), path_flags=["image 1: collapse"])
    assert flagged.verdict == "inconsistent" and "image 1: collapse" in flagged.reasons
    switched = check_barrier(_ev("i"), [_ev("s", moments=(0.0, 0.0))], _ev("f"))
    assert switched.verdict == "inconsistent"


def test_rate_sensitivity_matches_arrhenius():
    rate = arrhenius_rate(0.5, 600.0, 1e13)
    assert rate == pytest.approx(1e13 * math.exp(-0.5 / (8.617333262e-5 * 600.0)))
    low = rate_sensitivity({"fm": 0.50, "afm": 0.51}, 900.0)
    assert not low.sensitive
    high = rate_sensitivity({"fm": 0.50, "afm": 0.70}, 600.0)
    assert high.sensitive and high.max_ratio > 2
    with pytest.raises(ValueError):
        rate_sensitivity({"fm": 0.5}, 600.0)
    with pytest.raises(ValueError):
        arrhenius_rate(0.5, 0.0, 1e13)


def test_kmc_gate_rejects_inconsistent_and_unqualified_evidence():
    good = ConsistencyReport("consistent", ())
    evidence = [_ev("i"), _ev("f")]
    assert gate_transition_for_kmc(good, evidence).allowed
    bad = ConsistencyReport("inconsistent", ("different methods",))
    gate = gate_transition_for_kmc(bad, evidence)
    assert not gate.allowed and "different methods" in gate.reasons
    weak = [_ev("i", qual="software_tested"), _ev("f")]
    assert not gate_transition_for_kmc(good, weak).allowed
    admitted = gate_transition_for_kmc(good, weak, allow_unqualified=True)
    assert admitted.allowed and any("not numerically validated" in f for f in admitted.flags)


def test_kmc_gate_flags_magnetic_rate_sensitivity():
    good = ConsistencyReport("consistent", ())
    sens = rate_sensitivity({"fm": 0.5, "afm": 0.8}, 600.0)
    gate = gate_transition_for_kmc(good, [_ev("i")], sensitivity=sens)
    assert gate.allowed and any("depends on magnetic state" in f for f in gate.flags)
