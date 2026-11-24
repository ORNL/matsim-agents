"""Material-specific magnetic initialization policies (M9)."""

from __future__ import annotations

import numpy as np
import pytest
from ase.build import bulk

from matsim_agents.discovery.magnetic_policies import (
    POLICIES,
    configurations_for_policy,
    policy_readiness,
    select_policy,
)

A = 2.87


def _cell(symbols_by_index=None, element="Fe"):
    atoms = bulk(element, "bcc", a=A, cubic=True) * (2, 2, 2)
    for index, symbol in (symbols_by_index or {}).items():
        atoms.symbols[index] = symbol
    atoms.set_array("atom_id", np.arange(len(atoms), dtype=np.int64))
    return atoms


def test_selection_by_composition():
    assert select_policy(_cell({0: "Cr"})).name == "fe_cr_rafm"
    assert select_policy(_cell({0: "Y", 1: "O"})).name == "ods_ferritic"
    assert select_policy(_cell(element="W")).name == "refractory_w_ta_cr_v_hf_mo"
    assert select_policy(_cell({0: "Cr", 1: "Ti"}, element="V")).name == "v_cr_ti"
    assert select_policy(_cell(element="Au")) is None


def test_fe_cr_policy_generates_fm_antiparallel_and_nm():
    atoms = _cell({0: "Cr", 1: "Cr"})
    policy = select_policy(atoms)
    configs = configurations_for_policy(atoms, "s", policy)
    modes = [c.mode for c in configs]
    assert modes == ["ferromagnetic", "custom_collinear", "nonmagnetic"]
    custom = configs[1]
    by_symbol = {}
    for sym, moment in zip(custom.species, custom.initial_moments_muB, strict=True):
        by_symbol.setdefault(sym, set()).add(moment)
    assert by_symbol["Fe"] == {2.2} and by_symbol["Cr"] == {-1.0}
    assert all(any("not a ground-state claim" in a for a in c.assumptions) for c in configs)
    assert len({c.fingerprint() for c in configs}) == 3


def test_refractory_w_rich_is_nonmagnetic_without_cr():
    atoms = _cell(element="W")
    configs = configurations_for_policy(atoms, "s", select_policy(atoms))
    assert [c.mode for c in configs] == ["nonmagnetic"]


def test_refractory_with_cr_adds_small_ferromagnetic_trial():
    atoms = _cell({0: "Cr"}, element="W")
    configs = configurations_for_policy(atoms, "s", select_policy(atoms))
    assert [c.mode for c in configs] == ["nonmagnetic", "ferromagnetic"]


@pytest.mark.parametrize(
    ("element", "swaps", "name"),
    [("V", {0: "Cr", 1: "Ti"}, "v_cr_ti")],
)
def test_nonmagnetic_only_policies(element, swaps, name):
    atoms = _cell(swaps, element=element)
    policy = select_policy(atoms)
    assert policy.name == name
    assert [c.mode for c in configurations_for_policy(atoms, "s", policy)] == ["nonmagnetic"]


def test_sic_policy_does_not_invent_moments():
    atoms = bulk("SiC", "zincblende", a=4.36, cubic=True)
    atoms.set_array("atom_id", np.arange(len(atoms), dtype=np.int64))
    policy = select_policy(atoms)
    assert policy.name == "sic_sic"
    configs = configurations_for_policy(atoms, "s", policy)
    assert [c.mode for c in configs] == ["nonmagnetic"]
    assert all(m == 0.0 for m in configs[0].initial_moments_muB)


def test_ods_policy_keeps_oxide_elements_unpolarized():
    atoms = _cell({0: "Y", 1: "O", 2: "Cr"})
    configs = configurations_for_policy(atoms, "s", select_policy(atoms))
    antiparallel = next(c for c in configs if c.mode == "custom_collinear")
    for sym, moment in zip(antiparallel.species, antiparallel.initial_moments_muB, strict=True):
        if sym in ("Y", "O"):
            assert moment == 0.0


def test_readiness_never_reports_qualification():
    report = policy_readiness()
    assert set(report) == {p.name for p in POLICIES}
    assert all(
        entry["software_implemented"]
        and not entry["numerically_validated"]
        and not entry["scientifically_qualified"]
        for entry in report.values()
    )
