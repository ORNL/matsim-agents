"""PKA recoil initialization: speed from energy, direction sampling, numerical validation.

Units are ASE units (eV, angstrom, amu), so ``v = sqrt(2 E / m)`` needs no conversion.
Two energy conventions are supported and recorded: ``replace`` sets the PKA velocity so its
kinetic energy equals the requested recoil energy (its thermal velocity is discarded and
reported); ``add`` adds a velocity increment carrying the requested energy to the thermal
velocity, so the PKA's total kinetic energy differs by the cross term, which is reported.
SPECTRA-PKA supplies no directions; every direction here comes from the caller or a named
sampler, and the source is recorded by the caller.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from ase import Atoms


class RecoilError(ValueError):
    """The requested recoil cannot be initialized consistently."""


Convention = Literal["replace", "add"]


def recoil_speed(energy_eV: float, mass_amu: float) -> float:
    if not (math.isfinite(energy_eV) and energy_eV > 0):
        raise RecoilError(f"recoil energy must be positive and finite, got {energy_eV!r}")
    if not (math.isfinite(mass_amu) and mass_amu > 0):
        raise RecoilError(f"mass must be positive and finite, got {mass_amu!r}")
    return math.sqrt(2.0 * energy_eV / mass_amu)


def _unit(vec: Sequence[float]) -> np.ndarray:
    v = np.asarray(vec, dtype=float)
    if v.shape != (3,) or not np.all(np.isfinite(v)):
        raise RecoilError("a direction must be three finite numbers")
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise RecoilError("a direction must be non-zero")
    return v / n


def isotropic_direction(rng: np.random.Generator) -> np.ndarray:
    return _unit(rng.normal(size=3))


def user_direction(vec: Sequence[float]) -> np.ndarray:
    return _unit(vec)


def crystal_direction(atoms: Atoms, uvw: Sequence[int]) -> np.ndarray:
    """Cartesian unit vector of the lattice direction ``[uvw]`` of the cell vectors."""
    cell = np.asarray(atoms.cell.array, dtype=float)
    if abs(np.linalg.det(cell)) < 1e-12:
        raise RecoilError("a crystallographic direction needs a non-degenerate cell")
    return _unit(np.asarray(uvw, dtype=float) @ cell)


def cubic_family(uvw: Sequence[int]) -> tuple[tuple[int, int, int], ...]:
    """All distinct permutations and sign changes of ``uvw`` (the cubic direction family)."""
    out: set[tuple[int, int, int]] = set()
    for perm in itertools.permutations(uvw):
        for signs in itertools.product((1, -1), repeat=3):
            out.add(tuple(int(s * p) for s, p in zip(signs, perm, strict=True)))  # type: ignore[arg-type]
    return tuple(sorted(out))


def sample_family_direction(
    rng: np.random.Generator, atoms: Atoms, family: Sequence[Sequence[int]]
) -> np.ndarray:
    if not family:
        raise RecoilError("a direction family must not be empty")
    return crystal_direction(atoms, family[int(rng.integers(len(family)))])


def external_direction(
    rng: np.random.Generator, sampler: Callable[[np.random.Generator], Sequence[float]]
) -> np.ndarray:
    return _unit(sampler(rng))


@dataclass(frozen=True)
class RecoilInit:
    index: int
    symbol: str
    mass_amu: float
    requested_energy_eV: float
    realized_energy_eV: float
    pka_kinetic_energy_eV: float
    thermal_kinetic_energy_removed_eV: float
    direction: tuple[float, float, float]
    speed_ase: float
    convention: str
    temperature_before_K: float
    total_kinetic_energy_eV: float
    total_momentum: tuple[float, float, float]
    cell_lengths_A: tuple[float, float, float]
    n_atoms: int


def initialize_recoil(
    atoms: Atoms,
    index: int,
    energy_eV: float,
    direction: Sequence[float],
    *,
    expected_symbol: str,
    mass_amu: float | None = None,
    convention: Convention = "replace",
    rel_tol: float = 1e-9,
) -> RecoilInit:
    """Give atom ``index`` the recoil velocity in place and check the energy numerically."""
    if not 0 <= index < len(atoms):
        raise RecoilError(f"atom index {index} is outside 0..{len(atoms) - 1}")
    symbol = atoms.get_chemical_symbols()[index]
    if symbol != expected_symbol:
        raise RecoilError(f"atom {index} is {symbol}, but the event expects {expected_symbol}")
    if convention not in ("replace", "add"):
        raise RecoilError(f"unknown convention {convention!r}")
    m = float(atoms.get_masses()[index]) if mass_amu is None else float(mass_amu)
    u = _unit(direction)
    speed = recoil_speed(energy_eV, m)
    vel = atoms.get_velocities()
    if vel is None:
        vel = np.zeros((len(atoms), 3))
    v0 = vel[index].copy()
    temperature = float(atoms.get_temperature()) if np.any(vel) else 0.0
    ke_thermal = 0.5 * m * float(v0 @ v0)
    dv = speed * u
    if convention == "replace":
        vel[index] = dv
        realized = 0.5 * m * float(vel[index] @ vel[index])
    else:
        vel[index] = v0 + dv
        realized = 0.5 * m * float(dv @ dv)
    pka_ke = 0.5 * m * float(vel[index] @ vel[index])
    if abs(realized - energy_eV) > rel_tol * energy_eV:
        raise RecoilError(f"realized recoil energy {realized!r} differs from {energy_eV!r}")
    if mass_amu is not None:
        masses = atoms.get_masses()
        masses[index] = m
        atoms.set_masses(masses)
    atoms.set_velocities(vel)
    masses = atoms.get_masses()
    momentum = (masses[:, None] * vel).sum(axis=0)
    return RecoilInit(
        index=index,
        symbol=symbol,
        mass_amu=m,
        requested_energy_eV=float(energy_eV),
        realized_energy_eV=realized,
        pka_kinetic_energy_eV=pka_ke,
        thermal_kinetic_energy_removed_eV=ke_thermal if convention == "replace" else 0.0,
        direction=(float(u[0]), float(u[1]), float(u[2])),
        speed_ase=speed,
        convention=convention,
        temperature_before_K=temperature,
        total_kinetic_energy_eV=float(atoms.get_kinetic_energy()),
        total_momentum=(float(momentum[0]), float(momentum[1]), float(momentum[2])),
        cell_lengths_A=tuple(float(x) for x in atoms.cell.lengths()),  # type: ignore[arg-type]
        n_atoms=len(atoms),
    )


__all__ = [
    "RecoilError",
    "RecoilInit",
    "crystal_direction",
    "cubic_family",
    "external_direction",
    "initialize_recoil",
    "isotropic_direction",
    "recoil_speed",
    "sample_family_direction",
    "user_direction",
]
