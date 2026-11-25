"""Magnetic initialization generators on small BCC cells."""

from __future__ import annotations

import numpy as np
import pytest
from ase.build import bulk

from matsim_agents.science.magnetism import generators as gen

A = 2.87


def _cell(symbol="Fe", n=2):
    atoms = bulk(symbol, "bcc", a=A, cubic=True) * (n, n, n)
    atoms.set_array("atom_id", np.arange(len(atoms), dtype=np.int64))
    return atoms


def test_nonmagnetic_is_not_spin_polarized():
    cfg = gen.nonmagnetic(_cell(), "s")
    assert cfg.mode == "nonmagnetic" and not cfg.spin_polarized
    assert set(cfg.initial_moments_muB) == {0.0}


def test_ferromagnetic_fe_and_other_species_zero():
    atoms = _cell("Fe")
    atoms.symbols[[0, 1]] = "Cr"
    cfg = gen.ferromagnetic(atoms, "s", {"Fe": 2.2})
    moments = dict(zip(cfg.atom_ids, cfg.initial_moments_muB, strict=True))
    assert moments[0] == 0.0 and moments[1] == 0.0
    assert all(m == 2.2 for i, m in moments.items() if i > 1)
    with pytest.raises(ValueError, match="positive"):
        gen.ferromagnetic(atoms, "s", {"Fe": -1.0})


def test_bcc_sublattices_by_geometry_not_order():
    atoms = _cell()
    labels = gen.bcc_two_sublattices(atoms, A)
    assert sorted(set(labels.values())) == ["A", "B"]
    assert sum(v == "A" for v in labels.values()) == len(atoms) // 2
    shuffled = atoms[np.random.default_rng(1).permutation(len(atoms))]
    assert gen.bcc_two_sublattices(shuffled, A) == labels


def test_bcc_afm_has_antiparallel_nearest_neighbours():
    atoms = _cell()
    cfg = gen.bcc_afm(atoms, "s", A, {"Fe": 2.0})
    assert cfg.mode == "antiferromagnetic"
    assert sum(cfg.initial_moments_muB) == pytest.approx(0.0)
    moments = cfg.moments_for_atoms(atoms)
    distances = atoms.get_all_distances(mic=True)
    nearest = np.isclose(distances, A * np.sqrt(3) / 2, atol=1e-3)
    i, j = np.nonzero(nearest)
    assert len(i) > 0 and np.all(moments[i] * moments[j] < 0)


def test_afm_net_moment_need_not_be_zero_for_unequal_species():
    atoms = _cell("Fe")
    atoms.symbols[0] = "Cr"
    cfg = gen.bcc_afm(atoms, "s", A, {"Fe": 2.0, "Cr": 1.0})
    assert sum(cfg.initial_moments_muB) != pytest.approx(0.0)


def test_afm_rejects_incomplete_or_unsupported_patterns():
    atoms = _cell()
    labels = gen.bcc_two_sublattices(atoms, A)
    partial = {k: v for k, v in labels.items() if k != 0}
    with pytest.raises(ValueError, match="not given"):
        gen.antiferromagnetic(atoms, "s", partial, {"Fe": 2.0})
    three = dict(labels)
    three[0] = "C"
    with pytest.raises(ValueError, match="explicit signs"):
        gen.antiferromagnetic(atoms, "s", three, {"Fe": 2.0})
    with pytest.raises(ValueError, match="orthogonal integer"):
        gen.bcc_two_sublattices(atoms, 2.0)
    with pytest.raises(ValueError, match="same sign|antiparallel"):
        gen.antiferromagnetic(atoms, "s", {i: "A" for i in labels}, {"Fe": 2.0}, signs={"A": 1})


def test_ferrimagnetic_uses_unequal_magnitudes():
    atoms = _cell()
    labels = gen.bcc_two_sublattices(atoms, A)
    cfg = gen.ferrimagnetic(
        atoms,
        "s",
        labels,
        {"A": {"Fe": 3.0}, "B": {"Fe": 1.0}},
        signs={"A": 1, "B": -1},
    )
    assert cfg.mode == "ferrimagnetic"
    assert sum(cfg.initial_moments_muB) == pytest.approx(len(atoms) / 2 * 2.0)


def test_fe_cr_environment_dependent_moments_same_element():
    atoms = _cell("Fe")
    atoms.symbols[[0, 5]] = "Cr"
    moments = {i: 2.2 for i in range(len(atoms))}
    moments[0], moments[5] = -1.0, -1.5
    moments[1] = 1.8
    cfg = gen.custom_collinear(atoms, "s", moments)
    fe_groups = {g for g, s in zip(cfg.group_labels, cfg.species, strict=True) if s == "Fe"}
    assert len(fe_groups) == 2
    with pytest.raises(ValueError, match="not present"):
        gen.custom_collinear(atoms, "s", {999: 1.0})


def test_w_rich_bcc_nonmagnetic_and_fixed_ids():
    atoms = _cell("W")
    atoms.symbols[[3, 9]] = "Cr"
    cfg = gen.custom_collinear(atoms, "w-rich", {3: 1.0, 9: -1.0})
    assert cfg.atom_ids == tuple(range(len(atoms)))
    assert cfg.moments_for_atoms(atoms[::-1]).tolist()[::-1] == list(cfg.initial_moments_muB)


def test_generation_is_deterministic():
    atoms = _cell()
    one = gen.bcc_afm(atoms, "s", A, {"Fe": 2.0})
    two = gen.bcc_afm(atoms, "s", A, {"Fe": 2.0})
    assert one == two and one.config_id == two.config_id


def test_seeded_alternatives_are_reproducible_and_distinct():
    atoms = _cell()
    first = gen.seeded_collinear_alternatives(atoms, "s", {"Fe": 2.0}, n=5, seed=11)
    second = gen.seeded_collinear_alternatives(atoms, "s", {"Fe": 2.0}, n=5, seed=11)
    other = gen.seeded_collinear_alternatives(atoms, "s", {"Fe": 2.0}, n=5, seed=12)
    assert [c.fingerprint() for c in first] == [c.fingerprint() for c in second]
    assert len({c.fingerprint() for c in first}) == 5
    assert [c.fingerprint() for c in first] != [c.fingerprint() for c in other]
    assert all(c.seed == 11 for c in first)
    with pytest.raises(ValueError, match="two magnetic"):
        gen.seeded_collinear_alternatives(_cell("W"), "s", {"Fe": 2.0}, n=1, seed=0)
