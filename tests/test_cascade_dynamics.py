import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from ase.calculators.lj import LennardJones

from matsim_agents.science.cascade.dynamics import (
    Boundary,
    CascadeNumericsError,
    NoStopping,
    StepControl,
    TabulatedStopping,
    run_cascade,
)
from matsim_agents.science.cascade.recoil import initialize_recoil
from matsim_agents.science.cascade.zbl import ZBLCalculator


def _pair(energy_eV, gap=2.4):
    atoms = Atoms("W2", positions=[[0, 0, 0], [gap, 0, 0]], pbc=False)
    atoms.calc = ZBLCalculator(r_in=1.0, r_out=2.4 * 1.5)
    initialize_recoil(atoms, 0, energy_eV, [1, 0, 0], expected_symbol="W")
    return atoms


def test_two_atom_collision_conserves_energy_and_reverses_motion():
    atoms = _pair(80.0)
    ctl = StepControl(max_displacement_A=0.005, dt_min_fs=1e-4, dt_max_fs=0.5)
    run = run_cascade(atoms, 40.0, control=ctl)
    assert run.stable
    assert abs(run.energy_drift_eV) < 1e-3 * 80.0
    assert run.electronic_loss_eV == 0.0 and run.thermostat_loss_eV == 0.0
    v = atoms.get_velocities()
    assert v[0, 0] < 0.5 * abs(v[1, 0])  # the target has taken the momentum


def test_momentum_is_conserved_without_external_terms():
    atoms = _pair(60.0)
    p0 = (atoms.get_masses()[:, None] * atoms.get_velocities()).sum(axis=0)
    run_cascade(atoms, 30.0, control=StepControl(max_displacement_A=0.005, dt_max_fs=0.5))
    p1 = (atoms.get_masses()[:, None] * atoms.get_velocities()).sum(axis=0)
    assert np.allclose(p0, p1, rtol=1e-6, atol=1e-6)


def test_step_shrinks_during_the_collision_and_is_recorded():
    atoms = _pair(80.0)
    run = run_cascade(
        atoms, 30.0, control=StepControl(max_displacement_A=0.005, dt_max_fs=1.0, dt_min_fs=1e-4)
    )
    assert run.dt_min_used_fs < run.dt_max_used_fs
    assert len(run.trace.dt_fs) == len(run.trace.time_fs)
    assert run.trace.time_fs[-1] == pytest.approx(30.0)
    assert sum(run.trace.dt_fs) == pytest.approx(30.0)


def test_small_dt_floor_reports_criterion_violation():
    atoms = _pair(500.0, gap=1.5)
    run = run_cascade(
        atoms, 5.0, control=StepControl(max_displacement_A=0.001, dt_min_fs=0.5, dt_max_fs=1.0)
    )
    assert run.criterion_violated_at_dt_min > 0 and not run.stable


def test_electronic_stopping_energy_is_accounted():
    atoms = _pair(200.0, gap=6.0)
    table = {"W": ([1.0, 1e4], [0.5, 0.5])}
    stop = TabulatedStopping(table, source="unit-test constant table")
    run = run_cascade(
        atoms, 20.0, control=StepControl(max_displacement_A=0.01, dt_max_fs=0.5), stopping=stop
    )
    assert run.electronic_loss_eV > 0
    assert abs(run.energy_drift_eV) < 1e-3 * 200.0
    assert run.stopping_source == "unit-test constant table"


def test_stopping_outside_table_is_counted_or_fails():
    table = {"W": ([1e3, 1e4], [0.5, 0.5])}
    run = run_cascade(
        _pair(50.0, gap=6.0),
        5.0,
        stopping=TabulatedStopping(table, source="t"),
        control=StepControl(max_displacement_A=0.05, dt_max_fs=0.5),
    )
    assert run.stopping_outside_range_evaluations > 0 and run.electronic_loss_eV == 0.0
    with pytest.raises(CascadeNumericsError, match="outside"):
        run_cascade(
            _pair(50.0, gap=6.0),
            5.0,
            stopping=TabulatedStopping(table, source="t", outside="error"),
            control=StepControl(max_displacement_A=0.05, dt_max_fs=0.5),
        )


def test_stopping_needs_a_source_and_species_data():
    with pytest.raises(ValueError, match="source"):
        TabulatedStopping({"W": ([1, 2], [0, 0])}, source=" ")
    with pytest.raises(CascadeNumericsError, match="no stopping data"):
        run_cascade(
            _pair(50.0, gap=6.0),
            1.0,
            stopping=TabulatedStopping({"Ta": ([1, 2], [0, 0])}, source="t"),
        )


def test_default_is_disabled_stopping_and_says_so():
    atoms = _pair(10.0, gap=6.0)
    run = run_cascade(atoms, 1.0)
    assert "disabled" in run.stopping_source and NoStopping().force("W", 1.0, np.ones(3)).sum() == 0


def test_boundary_thermostat_acts_only_on_masked_atoms_and_is_accounted():
    atoms = bulk("W", "bcc", a=3.165, cubic=True).repeat((3, 3, 3))
    atoms.calc = LennardJones(sigma=2.7, epsilon=0.1, rc=6.0)
    rng = np.random.default_rng(0)
    atoms.set_velocities(rng.normal(scale=0.02, size=(len(atoms), 3)))
    mask = np.zeros(len(atoms), bool)
    mask[:10] = True
    core_v0 = atoms.get_velocities()[~mask].copy()
    run = run_cascade(
        atoms,
        20.0,
        control=StepControl(max_displacement_A=0.02, dt_max_fs=1.0),
        boundary=Boundary(mask=mask, target_K=10.0, tau_fs=20.0),
    )
    assert run.boundary_used and run.thermostat_loss_eV != 0.0
    assert abs(run.energy_drift_eV) < 0.05 * abs(run.initial_total_eV) + 0.05
    assert not np.allclose(atoms.get_velocities()[~mask], core_v0)


def test_invalid_setup_and_nonfinite_forces_are_detected():
    atoms = _pair(10.0)
    with pytest.raises(CascadeNumericsError):
        run_cascade(atoms, -1.0)
    with pytest.raises(CascadeNumericsError):
        run_cascade(atoms, 1.0, control=StepControl(dt_min_fs=2.0, dt_max_fs=1.0))
    bare = Atoms("W2", positions=[[0, 0, 0], [3, 0, 0]])
    with pytest.raises(CascadeNumericsError, match="calculator"):
        run_cascade(bare, 1.0)
    atoms = _pair(10.0)
    with pytest.raises(CascadeNumericsError, match="max_steps"):
        run_cascade(atoms, 100.0, control=StepControl(max_steps=3))


def test_speed_limit_triggers_numerical_failure():
    atoms = _pair(1000.0, gap=6.0)
    with pytest.raises(CascadeNumericsError, match="speed"):
        run_cascade(atoms, 5.0, max_atom_speed=1e-3)


def test_run_is_reproducible():
    a, b = _pair(60.0), _pair(60.0)
    ctl = StepControl(max_displacement_A=0.01, dt_max_fs=0.5)
    ra, rb = run_cascade(a, 10.0, control=ctl), run_cascade(b, 10.0, control=ctl)
    assert np.array_equal(a.get_positions(), b.get_positions())
    assert ra.trace.dt_fs == rb.trace.dt_fs
