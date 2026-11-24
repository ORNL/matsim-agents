"""Competing magnetic state planning and ranking (M6)."""

from __future__ import annotations

import numpy as np
import pytest
from ase.build import bulk

from matsim_agents.discovery import magnetism_generators as gen
from matsim_agents.discovery.magnetic_competition import (
    StateEvaluation,
    method_fingerprint,
    plan_competition,
    rank_states,
    run_pending,
)
from matsim_agents.discovery.magnetic_results import result_from_moments

A = 2.87
METHOD = {"code": "qe", "ecutwfc": 80, "kpts": [4, 4, 4], "functional": "pbe"}


def _atoms():
    atoms = bulk("Fe", "bcc", a=A, cubic=True)
    atoms.set_array("atom_id", np.arange(len(atoms), dtype=np.int64))
    return atoms


def _configs():
    atoms = _atoms()
    return atoms, [
        gen.nonmagnetic(atoms, "s"),
        gen.ferromagnetic(atoms, "s", {"Fe": 2.2}),
        gen.bcc_afm(atoms, "s", A, {"Fe": 2.0}),
    ]


def _eval(cfg, energy, moments, method=METHOD, converged=True, n_atoms=2):
    result = result_from_moments(moments, cfg.atom_ids, converged=converged)
    return StateEvaluation(cfg, method_fingerprint(method), energy, n_atoms, result, "Fe2")


def test_plan_removes_duplicates_and_spin_reversed_twins():
    atoms, (nm, fm, afm) = _configs()
    flipped = gen.bcc_afm(atoms, "s", A, {"Fe": 2.0})
    flipped = flipped.model_copy(
        update={
            "config_id": "flipped",
            "initial_moments_muB": tuple(-m for m in flipped.initial_moments_muB),
            "qe_starting_magnetization": None,
        }
    )
    plan = plan_competition([nm, fm, afm, fm, flipped], METHOD)
    assert [r.config.config_id for r in plan.requests] == [
        nm.config_id,
        fm.config_id,
        afm.config_id,
    ]
    reasons = dict(plan.skipped)
    assert "duplicate" in reasons[fm.config_id]
    assert "spin reversal" in reasons["flipped"]


def test_plan_skips_work_already_done_for_same_method_only():
    _, configs = _configs()
    first = plan_competition(configs, METHOD)
    done = [r.key for r in first.requests]
    assert plan_competition(configs, METHOD, already_done=done).requests == ()
    other = dict(METHOD, ecutwfc=100)
    assert len(plan_competition(configs, other, already_done=done).requests) == 3


def test_running_requires_authorization():
    _, configs = _configs()
    plan = plan_competition(configs, METHOD)
    with pytest.raises(PermissionError, match="authorization"):
        run_pending(plan, lambda r: None, authorized=False)
    calls = []
    run_pending(plan, lambda r: calls.append(r.key), authorized=True)
    assert len(calls) == 3


def test_ranking_orders_by_energy_and_reports_limits():
    _, (nm, fm, afm) = _configs()
    report = rank_states(
        [
            _eval(nm, -100.00, [0.0, 0.0]),
            _eval(fm, -100.20, [2.2, 2.2]),
            _eval(afm, -100.10, [2.0, -2.0]),
        ]
    )
    assert [r.config_id for r in report.ranking] == [fm.config_id, afm.config_id, nm.config_id]
    assert report.ranking[0].relative_energy_meV_per_atom == 0.0
    assert report.ranking[1].relative_energy_meV_per_atom == pytest.approx(50.0)
    assert "evaluated states only" in report.statement
    assert "ground state" not in report.statement


def test_incompatible_methods_and_unconverged_states_are_not_ranked():
    _, (nm, fm, afm) = _configs()
    report = rank_states(
        [
            _eval(fm, -100.2, [2.2, 2.2]),
            _eval(afm, -100.5, [2.0, -2.0], method=dict(METHOD, ecutwfc=100)),
            _eval(nm, -100.0, [0.0, 0.0], converged=False),
        ]
    )
    assert [r.config_id for r in report.ranking] == [fm.config_id]
    reasons = dict(report.excluded)
    assert reasons[afm.config_id] == "different numerical method"
    assert "not converged" in reasons[nm.config_id]
    assert report.missing_evidence


def test_ties_and_collapse_are_reported():
    _, (nm, fm, afm) = _configs()
    report = rank_states(
        [
            _eval(fm, -100.2000, [2.2, 2.2]),
            _eval(afm, -100.2005, [2.0, -2.0]),
            _eval(nm, -100.0, [0.0, 0.0]),
        ],
    )
    text = " ".join(report.missing_evidence)
    assert "not resolved" in text and "collapsed" in text


def test_converged_spin_reversed_duplicates_are_merged():
    atoms, (_, _, afm) = _configs()
    twin = afm.model_copy(update={"config_id": "twin"})
    report = rank_states(
        [_eval(afm, -100.1, [2.0, -2.0]), _eval(twin, -100.1, [-2.0, 2.0])],
    )
    assert len(report.ranking) == 1
    assert any("duplicates" in reason for _, reason in report.excluded)


def test_different_sizes_are_not_compared():
    _, (nm, fm, _) = _configs()
    report = rank_states([_eval(fm, -100.0, [2.2, 2.2]), _eval(nm, -200.0, [0.0, 0.0], n_atoms=4)])
    assert report.ranking == ()
    assert "not comparable" in " ".join(report.missing_evidence)


def test_empty_input():
    assert rank_states([]).missing_evidence == ("no evaluations supplied",)
