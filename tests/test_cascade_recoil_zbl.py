import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.lj import LennardJones

from matsim_agents.science.cascade.recoil import (
    RecoilError,
    crystal_direction,
    cubic_family,
    initialize_recoil,
    isotropic_direction,
    recoil_speed,
    sample_family_direction,
)
from matsim_agents.science.cascade.zbl import (
    ZBLCalculator,
    ZBLError,
    build_short_range_calculator,
    check_pair_curve,
    pair_curve,
    switch,
    switched_derivative,
    switched_energy,
)


def _w():
    return bulk("W", "bcc", a=3.165, cubic=True).repeat((3, 3, 3))


def test_speed_formula_and_kinetic_energy():
    atoms = _w()
    info = initialize_recoil(atoms, 5, 1000.0, [1, 2, 3], expected_symbol="W")
    assert info.speed_ase == pytest.approx(recoil_speed(1000.0, atoms.get_masses()[5]))
    v = atoms.get_velocities()[5]
    assert 0.5 * atoms.get_masses()[5] * v @ v == pytest.approx(1000.0, rel=1e-12)
    assert info.realized_energy_eV == pytest.approx(1000.0, rel=1e-12)


def test_wrong_species_and_bad_inputs_are_rejected():
    atoms = _w()
    with pytest.raises(RecoilError, match="expects"):
        initialize_recoil(atoms, 0, 10.0, [1, 0, 0], expected_symbol="Ta")
    with pytest.raises(RecoilError):
        initialize_recoil(atoms, 999, 10.0, [1, 0, 0], expected_symbol="W")
    with pytest.raises(RecoilError):
        initialize_recoil(atoms, 0, -1.0, [1, 0, 0], expected_symbol="W")
    with pytest.raises(RecoilError):
        initialize_recoil(atoms, 0, 10.0, [0, 0, 0], expected_symbol="W")


def test_replace_and_add_conventions_differ_by_thermal_cross_term():
    base = _w()
    base.set_velocities(np.random.default_rng(0).normal(scale=0.01, size=(len(base), 3)))
    a, b = base.copy(), base.copy()
    ra = initialize_recoil(a, 0, 50.0, [1, 0, 0], expected_symbol="W", convention="replace")
    rb = initialize_recoil(b, 0, 50.0, [1, 0, 0], expected_symbol="W", convention="add")
    assert ra.pka_kinetic_energy_eV == pytest.approx(50.0)
    assert rb.pka_kinetic_energy_eV != pytest.approx(50.0, rel=1e-6)
    assert rb.realized_energy_eV == pytest.approx(50.0)
    assert ra.thermal_kinetic_energy_removed_eV > 0 and rb.thermal_kinetic_energy_removed_eV == 0


def test_isotropic_directions_reproducible_unit_and_unbiased():
    d1 = [isotropic_direction(np.random.default_rng(3)) for _ in range(2)]
    assert np.allclose(d1[0], d1[1])
    rng = np.random.default_rng(1)
    d = np.array([isotropic_direction(rng) for _ in range(4000)])
    assert np.allclose(np.linalg.norm(d, axis=1), 1.0)
    assert np.all(np.abs(d.mean(axis=0)) < 0.05)


def test_crystal_direction_and_cubic_family():
    atoms = _w()
    assert np.allclose(crystal_direction(atoms, [1, 1, 1]), np.ones(3) / np.sqrt(3))
    assert len(cubic_family((1, 1, 1))) == 8
    assert len(cubic_family((1, 0, 0))) == 6
    assert len(cubic_family((1, 1, 0))) == 12
    u = sample_family_direction(np.random.default_rng(0), atoms, cubic_family((1, 1, 0)))
    assert np.linalg.norm(u) == pytest.approx(1.0)


def test_switch_is_c2_at_both_ends():
    r_in, r_out = 0.8, 1.6
    assert switch(0.5, r_in, r_out) == 1.0 and switch(2.0, r_in, r_out) == 0.0
    eps = 1e-4
    for edge in (r_in, r_out):
        e0 = switched_energy(edge - eps, 74, 74, r_in, r_out)
        e1 = switched_energy(edge + eps, 74, 74, r_in, r_out)
        assert abs(e1 - e0) < 1e-2 * max(abs(e0), 1e-3) + 1e-3


def test_analytic_derivative_matches_finite_difference():
    r = np.linspace(0.3, 2.0, 40)
    h = 1e-6
    num = (
        switched_energy(r + h, 74, 73, 0.8, 1.6) - switched_energy(r - h, 74, 73, 0.8, 1.6)
    ) / (2 * h)
    assert np.allclose(num, switched_derivative(r, 74, 73, 0.8, 1.6), rtol=1e-5, atol=1e-8)


def test_calculator_forces_match_finite_difference_and_newton_third_law():
    atoms = _w()
    atoms.rattle(0.3, seed=1)
    atoms.positions[1] = atoms.positions[0] + [1.0, 0.2, 0.1]
    atoms.calc = ZBLCalculator(r_in=0.8, r_out=1.6)
    f = atoms.get_forces()
    assert np.allclose(f.sum(axis=0), 0.0, atol=1e-8)
    for atom, axis in ((0, 0), (1, 2)):
        h = 1e-5
        p = atoms.copy()
        p.calc = ZBLCalculator(r_in=0.8, r_out=1.6)
        p.positions[atom, axis] += h
        m = p.copy()
        m.calc = ZBLCalculator(r_in=0.8, r_out=1.6)
        m.positions[atom, axis] -= 2 * h
        num = -(p.get_potential_energy() - m.get_potential_energy()) / (2 * h)
        assert num == pytest.approx(f[atom, axis], rel=1e-4, abs=1e-6)


def test_double_counting_and_unjustified_use_are_refused():
    base = LennardJones()
    with pytest.raises(ZBLError, match="double counts"):
        build_short_range_calculator(base, native_short_range=True, additive_zbl=True)
    with pytest.raises(ZBLError, match="justification"):
        build_short_range_calculator(base, native_short_range=False, additive_zbl=True)
    calc, rec = build_short_range_calculator(
        base, native_short_range=False, additive_zbl=True, justification="test"
    )
    assert rec.policy == "additive_zbl" and rec.r_out == 1.6 and calc is not base
    _, rec2 = build_short_range_calculator(base, native_short_range=False, additive_zbl=False)
    assert rec2.policy == "none" and rec2.assumptions


def test_pair_curve_of_zbl_passes_and_attractive_curve_fails():
    r = np.linspace(0.4, 2.5, 800)
    e, f = pair_curve(ZBLCalculator(r_in=0.8, r_out=1.6), "W", "Ta", r)
    assert check_pair_curve(r, e, f, repulsive_below=1.5).ok
    e2, f2 = pair_curve(LennardJones(sigma=2.5, epsilon=0.5), "W", "W", r)
    rep = check_pair_curve(r, e2, f2, repulsive_below=2.5)
    assert rep.ok  # LJ is repulsive inside sigma
    bad_f = f.copy()
    bad_f[10] = -abs(bad_f[10])
    assert not check_pair_curve(r, e, bad_f, repulsive_below=1.5).ok


def test_discontinuous_curve_is_rejected():
    r = np.linspace(0.4, 2.0, 400)
    e, f = pair_curve(ZBLCalculator(r_in=0.8, r_out=1.6), "W", "W", r)
    e = e.copy()
    e[200:] += 0.5
    assert "inconsistent" in " ".join(check_pair_curve(r, e, f, repulsive_below=1.5).problems)
