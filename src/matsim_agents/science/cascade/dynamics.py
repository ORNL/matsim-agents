"""Adaptive velocity-Verlet cascade integrator with explicit energy accounting.

The step is ``dt = clip(min(x_max/v_max, sqrt(2 x_max/a_max)), dt_min, dt_max)``, chosen from
the state at the start of the step and recorded. A variable step makes the integrator only
approximately energy conserving; the drift is measured, not assumed. Electronic stopping is
off unless a model with a stated source and energy range is supplied, and the energy it
removes is accumulated. An optional Berendsen rescale acts only on a caller-chosen boundary
mask, never on the cascade core, and the energy it removes is recorded. ASE units throughout
(eV, angstrom, amu; time in ASE units, converted from fs at the boundary).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from ase import Atoms, units


class CascadeNumericsError(RuntimeError):
    """The integration became non-finite or exceeded a stated safety limit."""


@dataclass(frozen=True)
class StoppingRange:
    low_eV: float
    high_eV: float


class ElectronicStopping:
    """Interface: a friction force opposing velocity, with a stated source and range."""

    source = "none (electronic stopping disabled)"
    approximations: tuple[str, ...] = ("no energy loss to electronic degrees of freedom",)

    def force(self, symbol: str, kinetic_energy_eV: float, velocity: np.ndarray) -> np.ndarray:
        return np.zeros(3)


class NoStopping(ElectronicStopping):
    pass


class TabulatedStopping(ElectronicStopping):
    """Stopping power S_e(E) in eV/angstrom per species, from a caller-cited table.

    Outside the table the atom feels no stopping (``outside='zero'``) or the run fails
    (``outside='error'``); the number of such evaluations is counted by the integrator.
    """

    def __init__(
        self,
        table: dict[str, tuple[Sequence[float], Sequence[float]]],
        *,
        source: str,
        outside: str = "zero",
    ) -> None:
        if not source.strip():
            raise ValueError("a stopping model needs a source for its parameters")
        if outside not in ("zero", "error"):
            raise ValueError("outside must be 'zero' or 'error'")
        self._table = {}
        for sym, (e, s) in table.items():
            e_a, s_a = np.asarray(e, float), np.asarray(s, float)
            if len(e_a) < 2 or np.any(np.diff(e_a) <= 0) or np.any(s_a < 0):
                raise ValueError(f"{sym}: need increasing energies and non-negative stopping")
            self._table[sym] = (e_a, s_a)
        self.source = source
        self.outside = outside
        self.approximations = (
            "friction magnitude from a tabulated stopping power, applied along -v",
            "no dependence on local electron density",
        )
        self.n_outside = 0

    def ranges(self) -> dict[str, StoppingRange]:
        return {k: StoppingRange(float(e[0]), float(e[-1])) for k, (e, _) in self._table.items()}

    def force(self, symbol, kinetic_energy_eV, velocity):
        if symbol not in self._table:
            raise CascadeNumericsError(f"no stopping data for species {symbol!r}")
        e, s = self._table[symbol]
        speed = float(np.linalg.norm(velocity))
        if speed == 0.0:
            return np.zeros(3)
        if not (e[0] <= kinetic_energy_eV <= e[-1]):
            self.n_outside += 1
            if self.outside == "error":
                raise CascadeNumericsError(
                    f"{symbol} at {kinetic_energy_eV:.3g} eV is outside the stopping table"
                )
            return np.zeros(3)
        return -float(np.interp(kinetic_energy_eV, e, s)) * velocity / speed


@dataclass(frozen=True)
class StepControl:
    max_displacement_A: float = 0.05
    dt_min_fs: float = 1e-4
    dt_max_fs: float = 2.0
    dt_initial_fs: float | None = None
    max_steps: int = 1_000_000


@dataclass(frozen=True)
class Boundary:
    mask: np.ndarray  # boolean, atoms in the thermostatted region
    target_K: float
    tau_fs: float


@dataclass
class CascadeTrace:
    time_fs: list[float] = field(default_factory=list)
    dt_fs: list[float] = field(default_factory=list)
    potential_eV: list[float] = field(default_factory=list)
    kinetic_eV: list[float] = field(default_factory=list)
    temperature_K: list[float] = field(default_factory=list)
    max_force: list[float] = field(default_factory=list)
    electronic_eV: list[float] = field(default_factory=list)
    thermostat_eV: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class CascadeRun:
    steps: int
    duration_fs: float
    trace: CascadeTrace
    initial_total_eV: float
    final_total_eV: float
    electronic_loss_eV: float
    thermostat_loss_eV: float
    energy_drift_eV: float
    dt_min_used_fs: float
    dt_max_used_fs: float
    steps_at_dt_min: int
    steps_at_dt_max: int
    criterion_violated_at_dt_min: int
    stopping_source: str
    stopping_approximations: tuple[str, ...]
    stopping_outside_range_evaluations: int
    boundary_used: bool
    stable: bool


def _choose_dt(v, a, masses, control: StepControl) -> tuple[float, bool, str]:
    vmax = float(np.max(np.linalg.norm(v, axis=1)))
    amax = float(np.max(np.linalg.norm(a, axis=1)))
    x = control.max_displacement_A
    candidates = []
    if vmax > 0:
        candidates.append(x / vmax)
    if amax > 0:
        candidates.append(math.sqrt(2.0 * x / amax))
    dt = min(candidates) if candidates else control.dt_max_fs * units.fs
    lo, hi = control.dt_min_fs * units.fs, control.dt_max_fs * units.fs
    violated = dt < lo
    return min(max(dt, lo), hi), violated, ("v" if vmax > 0 else "a")


def run_cascade(
    atoms: Atoms,
    duration_fs: float,
    *,
    control: StepControl | None = None,
    stopping: ElectronicStopping | None = None,
    boundary: Boundary | None = None,
    record_every: int = 1,
    on_step: Callable[[Atoms, int, float], None] | None = None,
    max_atom_speed: float | None = None,
) -> CascadeRun:
    """Integrate ``atoms`` (calculator attached, velocities set) in place."""
    control = control or StepControl()
    stopping = stopping or NoStopping()
    if atoms.calc is None:
        raise CascadeNumericsError("atoms needs a calculator")
    if not (duration_fs > 0 and math.isfinite(duration_fs)):
        raise CascadeNumericsError("duration must be positive and finite")
    if control.dt_min_fs <= 0 or control.dt_min_fs > control.dt_max_fs:
        raise CascadeNumericsError("need 0 < dt_min <= dt_max")
    if boundary is not None and (boundary.tau_fs <= 0 or boundary.mask.shape != (len(atoms),)):
        raise CascadeNumericsError("boundary needs tau > 0 and a mask over all atoms")
    if atoms.get_velocities() is None:
        atoms.set_velocities(np.zeros((len(atoms), 3)))
    masses = atoms.get_masses()[:, None]
    symbols = atoms.get_chemical_symbols()
    trace = CascadeTrace()
    t = 0.0
    total_t = duration_fs * units.fs
    electronic = 0.0
    thermo = 0.0
    forces = atoms.get_forces()
    initial_total = atoms.get_potential_energy() + atoms.get_kinetic_energy()
    n_lo = n_hi = n_viol = 0
    dts: list[float] = []
    steps = 0

    def record() -> None:
        trace.time_fs.append(t / units.fs)
        trace.dt_fs.append(dts[-1] / units.fs if dts else 0.0)
        trace.potential_eV.append(float(atoms.get_potential_energy()))
        trace.kinetic_eV.append(float(atoms.get_kinetic_energy()))
        trace.temperature_K.append(float(atoms.get_temperature()))
        trace.max_force.append(float(np.max(np.linalg.norm(forces, axis=1))))
        trace.electronic_eV.append(electronic)
        trace.thermostat_eV.append(thermo)

    record()
    while t < total_t - 1e-15 * total_t:
        if steps >= control.max_steps:
            raise CascadeNumericsError(f"exceeded max_steps={control.max_steps} before duration")
        v = atoms.get_velocities()
        a = forces / masses
        dt, violated, _ = _choose_dt(v, a, masses, control)
        dt = min(dt, total_t - t)
        if violated:
            n_viol += 1
        lo, hi = control.dt_min_fs * units.fs, control.dt_max_fs * units.fs
        if dt <= lo * (1 + 1e-12):
            n_lo += 1
        if dt >= hi * (1 - 1e-12):
            n_hi += 1
        v_half = v + 0.5 * dt * a
        atoms.set_positions(atoms.get_positions() + dt * v_half)
        forces = atoms.get_forces()
        if not (np.all(np.isfinite(forces)) and np.all(np.isfinite(atoms.get_positions()))):
            raise CascadeNumericsError(f"non-finite forces or positions at step {steps}")
        a_new = forces / masses
        v_new = v_half + 0.5 * dt * a_new
        # Stopping is applied as an operator-split velocity update after the Verlet step.
        for i, sym in enumerate(symbols):
            ke = 0.5 * float(masses[i, 0]) * float(v_new[i] @ v_new[i])
            f = stopping.force(sym, ke, v_new[i])
            if np.any(f):
                dv = f / float(masses[i, 0]) * dt
                speed = float(np.linalg.norm(v_new[i]))
                dv_norm = float(np.linalg.norm(dv))
                if dv_norm > speed:
                    dv *= speed / dv_norm
                before = ke
                v_new[i] = v_new[i] + dv
                after = 0.5 * float(masses[i, 0]) * float(v_new[i] @ v_new[i])
                electronic += before - after
        if boundary is not None and np.any(boundary.mask):
            m = boundary.mask
            ke_m = 0.5 * float(np.sum(masses[m] * v_new[m] ** 2))
            temp = 2.0 * ke_m / (3.0 * int(m.sum()) * units.kB)
            if temp > 0:
                lam = math.sqrt(
                    max(
                        0.0,
                        1.0 + dt / (boundary.tau_fs * units.fs) * (boundary.target_K / temp - 1.0),
                    )
                )
                v_new[m] *= lam
                thermo += ke_m - 0.5 * float(np.sum(masses[m] * v_new[m] ** 2))
        atoms.set_velocities(v_new)
        speed_max = float(np.max(np.linalg.norm(v_new, axis=1)))
        if not math.isfinite(speed_max):
            raise CascadeNumericsError(f"non-finite velocity at step {steps}")
        if max_atom_speed is not None and speed_max > max_atom_speed:
            raise CascadeNumericsError(f"atom speed {speed_max:.3g} exceeds the stated limit")
        t += dt
        dts.append(dt)
        steps += 1
        if on_step is not None:
            on_step(atoms, steps, t / units.fs)
        if steps % record_every == 0:
            record()
    if steps % record_every:
        record()
    final_total = atoms.get_potential_energy() + atoms.get_kinetic_energy()
    drift = final_total + electronic + thermo - initial_total
    return CascadeRun(
        steps=steps,
        duration_fs=t / units.fs,
        trace=trace,
        initial_total_eV=float(initial_total),
        final_total_eV=float(final_total),
        electronic_loss_eV=float(electronic),
        thermostat_loss_eV=float(thermo),
        energy_drift_eV=float(drift),
        dt_min_used_fs=min(dts) / units.fs if dts else 0.0,
        dt_max_used_fs=max(dts) / units.fs if dts else 0.0,
        steps_at_dt_min=n_lo,
        steps_at_dt_max=n_hi,
        criterion_violated_at_dt_min=n_viol,
        stopping_source=stopping.source,
        stopping_approximations=stopping.approximations,
        stopping_outside_range_evaluations=int(getattr(stopping, "n_outside", 0)),
        boundary_used=boundary is not None,
        stable=n_viol == 0,
    )


__all__ = [
    "Boundary",
    "CascadeNumericsError",
    "CascadeRun",
    "CascadeTrace",
    "ElectronicStopping",
    "NoStopping",
    "StepControl",
    "StoppingRange",
    "TabulatedStopping",
    "run_cascade",
]
