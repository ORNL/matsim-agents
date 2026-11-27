"""Universal ZBL repulsion, a smooth switch to an MLIP, and pair-curve validation.

``E_pair = w(r) * V_ZBL(r)`` is added to the base calculator. ``w`` is 1 inside ``r_in``,
0 beyond ``r_out`` and C2 in between (quintic smootherstep). This is additive: the base
model's own energy at ``r < r_out`` is not removed, so it is only meaningful when the base
model is soft or meaningless there; it is refused for a model declared to carry its own
short-range repulsion, and it needs a written justification. Pair curves of the combined
model must still pass ``check_pair_curve`` before it is used. Only energy and forces are
provided (no stress). Not validated against published ZBL fits or radiation data.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.mixing import SumCalculator
from ase.data import atomic_numbers
from ase.neighborlist import neighbor_list

COULOMB_EV_A = 14.399645352
BOHR_A = 0.529177210903
_C = np.array([0.1818, 0.5099, 0.2802, 0.02817])
_E = np.array([3.2, 0.9423, 0.4029, 0.2016])


class ZBLError(ValueError):
    pass


def screening_length(z1, z2):
    return 0.8854 * BOHR_A / (np.power(z1, 0.23) + np.power(z2, 0.23))


def zbl_energy(r, z1, z2):
    r = np.asarray(r, dtype=float)
    x = r / screening_length(z1, z2)
    phi = sum(c * np.exp(-e * x) for c, e in zip(_C, _E, strict=True))
    return COULOMB_EV_A * z1 * z2 / r * phi


def zbl_derivative(r, z1, z2):
    r = np.asarray(r, dtype=float)
    a = screening_length(z1, z2)
    x = r / a
    phi = sum(c * np.exp(-e * x) for c, e in zip(_C, _E, strict=True))
    dphi = sum(-c * e * np.exp(-e * x) for c, e in zip(_C, _E, strict=True))
    return COULOMB_EV_A * z1 * z2 * (-phi / r**2 + dphi / (a * r))


def _smoother(t):
    return 6 * t**5 - 15 * t**4 + 10 * t**3


def _dsmoother(t):
    return 30 * t**4 - 60 * t**3 + 30 * t**2


def switch(r, r_in: float, r_out: float):
    t = np.clip((np.asarray(r, dtype=float) - r_in) / (r_out - r_in), 0.0, 1.0)
    return 1.0 - _smoother(t)


def switch_derivative(r, r_in: float, r_out: float):
    r = np.asarray(r, dtype=float)
    inside = (r > r_in) & (r < r_out)
    t = np.clip((r - r_in) / (r_out - r_in), 0.0, 1.0)
    return np.where(inside, -_dsmoother(t) / (r_out - r_in), 0.0)


def switched_energy(r, z1, z2, r_in, r_out):
    return switch(r, r_in, r_out) * zbl_energy(r, z1, z2)


def switched_derivative(r, z1, z2, r_in, r_out):
    return switch_derivative(r, r_in, r_out) * zbl_energy(r, z1, z2) + switch(
        r, r_in, r_out
    ) * zbl_derivative(r, z1, z2)


class ZBLCalculator(Calculator):
    implemented_properties = ["energy", "forces"]  # noqa: RUF012

    def __init__(self, r_in: float = 0.8, r_out: float = 1.6, **kwargs) -> None:
        if not (0 < r_in < r_out):
            raise ZBLError("need 0 < r_in < r_out")
        super().__init__(**kwargs)
        self.r_in = float(r_in)
        self.r_out = float(r_out)

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        atoms = self.atoms
        i, j, d, vec = neighbor_list("ijdD", atoms, self.r_out)
        z = atoms.numbers
        forces = np.zeros((len(atoms), 3))
        energy = 0.0
        if len(i):
            z1, z2 = z[i].astype(float), z[j].astype(float)
            energy = 0.5 * float(np.sum(switched_energy(d, z1, z2, self.r_in, self.r_out)))
            dv = switched_derivative(d, z1, z2, self.r_in, self.r_out)
            np.add.at(forces, i, (dv / d)[:, None] * vec)
        self.results = {"energy": energy, "forces": forces}


@dataclass(frozen=True)
class ShortRangeModel:
    policy: str
    r_in: float | None
    r_out: float | None
    native_short_range: bool
    justification: str
    assumptions: tuple[str, ...] = field(default_factory=tuple)


def build_short_range_calculator(
    base: Calculator,
    *,
    native_short_range: bool,
    additive_zbl: bool,
    r_in: float = 0.8,
    r_out: float = 1.6,
    justification: str = "",
) -> tuple[Calculator, ShortRangeModel]:
    """Return the calculator to use and a record of the short-range choice."""
    if not additive_zbl:
        return base, ShortRangeModel(
            "native" if native_short_range else "none",
            None,
            None,
            native_short_range,
            justification,
            ()
            if native_short_range
            else ("no short-range correction: unqualified in the collision regime",),
        )
    if native_short_range:
        raise ZBLError("the model declares native short-range repulsion; adding ZBL double counts")
    if not justification.strip():
        raise ZBLError("adding ZBL needs a written justification for this model and domain")
    calc = SumCalculator([base, ZBLCalculator(r_in=r_in, r_out=r_out)])
    return calc, ShortRangeModel(
        "additive_zbl",
        r_in,
        r_out,
        native_short_range,
        justification,
        (
            "the base model energy inside r_out is not removed",
            "ZBL is a two-body screened Coulomb fit, not a many-body core",
        ),
    )


def pair_curve(
    calculator: Calculator, symbol_a: str, symbol_b: str, r: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    """Energy (shifted to 0 at the largest r) and repulsive force on atom 2 along the axis."""
    for s in (symbol_a, symbol_b):
        if s not in atomic_numbers:
            raise ZBLError(f"unknown element {s!r}")
    radii = np.asarray(r, dtype=float)
    energy = np.empty(len(radii))
    force = np.empty(len(radii))
    for k, rk in enumerate(radii):
        atoms = Atoms([symbol_a, symbol_b], positions=[[0, 0, 0], [rk, 0, 0]], pbc=False)
        atoms.calc = calculator
        energy[k] = atoms.get_potential_energy()
        force[k] = atoms.get_forces()[1, 0]
    return energy - energy[np.argmax(radii)], force


@dataclass(frozen=True)
class PairCurveReport:
    ok: bool
    problems: tuple[str, ...]


def check_pair_curve(
    r: Sequence[float],
    energy: Sequence[float],
    force: Sequence[float],
    *,
    repulsive_below: float,
    path_rtol: float = 0.05,
    scale_floor: float = 1e-4,
    max_force_step_ratio: float = 0.5,
) -> PairCurveReport:
    """Reject non-finite, attractive, discontinuous or energy/force-inconsistent curves.

    ``r`` must be increasing and fine enough that the trapezoid rule resolves the curve.
    Absolute tolerances are ``scale_floor`` times the curve's own largest step, so the
    smooth tail near zero force is not judged on relative error alone.
    """
    r_a, e_a, f_a = (np.asarray(x, dtype=float) for x in (r, energy, force))
    problems: list[str] = []
    if not (len(r_a) == len(e_a) == len(f_a) >= 3) or np.any(np.diff(r_a) <= 0):
        raise ZBLError("need at least three increasing radii with matching energy and force")
    if not (np.all(np.isfinite(e_a)) and np.all(np.isfinite(f_a))):
        return PairCurveReport(False, ("non-finite energy or force",))
    low = r_a <= repulsive_below
    if np.any(f_a[low] <= 0):
        problems.append("attractive force in the collision regime")
    if np.any(np.diff(e_a[low]) >= 0):
        problems.append("energy does not decrease with distance in the collision regime")
    de = np.diff(e_a)
    work = -0.5 * (f_a[:-1] + f_a[1:]) * np.diff(r_a)
    de_floor = scale_floor * float(np.max(np.abs(de)))
    if np.any(np.abs(de - work) > de_floor + path_rtol * np.abs(de)):
        problems.append("energy and force are inconsistent or discontinuous")
    scale = np.maximum(np.abs(f_a[:-1]), np.abs(f_a[1:]))
    step = np.abs(np.diff(f_a))
    if np.any(step > max_force_step_ratio * scale + scale_floor * float(np.max(np.abs(f_a)))):
        problems.append("force jumps between neighbouring radii")
    return PairCurveReport(not problems, tuple(problems))


__all__ = [
    "PairCurveReport",
    "ShortRangeModel",
    "ZBLCalculator",
    "ZBLError",
    "build_short_range_calculator",
    "check_pair_curve",
    "pair_curve",
    "switch",
    "switch_derivative",
    "switched_derivative",
    "switched_energy",
    "zbl_derivative",
    "zbl_energy",
]
