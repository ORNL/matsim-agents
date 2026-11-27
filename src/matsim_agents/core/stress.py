"""Finite-difference stress for ASE calculators that only provide energy and forces."""

from __future__ import annotations

import numpy as np


class NumericalStressCalculator:
    """Wraps a forces-only ASE calculator to add stress via numerical finite differences.

    Computes the 6-component Voigt stress tensor by applying central finite
    differences to each of the 6 independent cell-strain components and measuring
    the energy response.  The overhead is 12 extra energy evaluations per
    optimizer step, which is negligible for a fast ML potential.

    The wrapper transparently forwards all other attribute access to the inner
    calculator, so its custom attributes (for example ``model``) remain accessible.

    Args:
        inner:  An ASE calculator that implements energy and forces.
        dx:     Strain step size (dimensionless).  Default 1e-3.
    """

    implemented_properties = ["energy", "forces", "stress", "stresses"]

    def __init__(self, inner, dx: float = 1e-3, *, require_native: bool = False):
        self._inner = inner
        self.dx = dx
        self._require_native = require_native
        self.results: dict = {}
        self._state: tuple | None = None
        # An inner calculator that predicts stress is used for it directly.
        self._native_stress = "stress" in getattr(inner, "implemented_properties", ())
        # Patch the inner calculator's implemented_properties so ASE doesn't
        # short-circuit to raising NotImplementedError before our wrapper runs.
        if hasattr(inner, "implemented_properties"):
            orig = list(inner.implemented_properties)
            if "stress" not in orig:
                orig.append("stress")
            if "stresses" not in orig:
                orig.append("stresses")
            inner.implemented_properties = orig

    def __getattr__(self, name: str):
        # Forward attribute reads (e.g. model) to the inner calc.
        if name.startswith("_") or name in ("results", "implemented_properties"):
            raise AttributeError(name)
        # Block methods that would raise PropertyNotImplementedError on the inner
        # calc — return a no-op lambda so ASE internals get None instead of crashing.
        _unsupported = {
            "get_dipole_moment",
            "get_magnetic_moment",
            "get_magnetic_moments",
            "get_charges",
        }
        if name in _unsupported:
            return lambda *a, **kw: None
        return getattr(self._inner, name)

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=None):
        """Compute energy, forces, and (numerical) stress for *atoms*."""
        from ase.calculators.calculator import all_changes

        # Energy + forces from the inner calculator on the original geometry. ASE
        # calculators expect a list of changes; None means "unknown", i.e. all.
        changes = all_changes if system_changes is None else system_changes
        properties = ["energy", "forces"] + (["stress"] if self._native_stress else [])
        self._inner.calculate(atoms, properties, changes)
        self.results = {
            "energy": self._inner.results["energy"],
            "forces": self._inner.results["forces"],
        }
        native = self._inner.results.get("stress") if self._native_stress else None
        if self._require_native and native is None:
            raise ValueError("Calculator did not provide the required native stress.")
        # Always provide stress so ASE's property cache stays consistent.
        self.results["stress"] = native if native is not None else self._numerical_stress(atoms)
        self._state = self._snapshot(atoms)

    @staticmethod
    def _snapshot(atoms) -> tuple:
        return (
            atoms.get_positions().copy(),
            atoms.cell.array.copy(),
            atoms.numbers.copy(),
            atoms.pbc.copy(),
        )

    def _current(self, atoms, key: str):
        """Results for ``atoms``, recalculated when the geometry changed."""
        import numpy as np

        if atoms is None:
            atoms = getattr(self._inner, "atoms", None)
        stale = (
            key not in self.results
            or self._state is None
            or (
                atoms is not None
                and not all(
                    np.array_equal(old, new)
                    for old, new in zip(self._state, self._snapshot(atoms), strict=True)
                )
            )
        )
        if stale:
            self.calculate(atoms)
        return self.results[key]

    def get_potential_energy(self, atoms=None, force_consistent=False):
        return self._current(atoms, "energy")

    def get_forces(self, atoms=None):
        return self._current(atoms, "forces")

    def get_stress(self, atoms=None):
        return self._current(atoms, "stress")

    def get_property(self, name, atoms=None, allow_calculation=True):
        # 'stresses' (per-atom) is requested by some ASE filters; map to 'stress'.
        key = "stress" if name == "stresses" else name
        if key not in self.implemented_properties:
            # Return None gracefully for unsupported properties (e.g. 'dipole')
            # rather than raising, so ASE internals don't crash.
            return None
        if not allow_calculation:
            return self.results.get(key)
        return self._current(atoms, key)

    def _numerical_stress(self, atoms) -> np.ndarray:
        """Central-difference stress, ASE convention: Voigt (xx, yy, zz, yz, xz, xy), eV/Å³.

        sigma_ij = (1/V) dE/d(epsilon_ij) for the symmetric strain epsilon applied
        to the Cartesian coordinates (cell -> cell @ (I + epsilon)), as in
        ase.calculators.calculator.Calculator.calculate_numerical_stress.
        """
        import numpy as np

        cell = atoms.cell.array.copy()
        volume = atoms.get_volume()
        stress = np.zeros((3, 3))

        def energy(strain: np.ndarray) -> float:
            deformed = atoms.copy()
            deformed.set_cell(cell @ (np.eye(3) + strain), scale_atoms=True)
            deformed.calc = self._inner
            return float(deformed.get_potential_energy())

        for i in range(3):
            for j in range(i, 3):
                strain = np.zeros((3, 3))
                strain[i, j] = strain[j, i] = self.dx
                difference = energy(strain) - energy(-strain)
                # A shear strain moves both epsilon_ij and epsilon_ji.
                scale = 2.0 if i == j else 4.0
                stress[i, j] = stress[j, i] = difference / (scale * self.dx * volume)

        # Restore inner calc's atom reference so its cache is consistent.
        if hasattr(self._inner, "atoms"):
            self._inner.atoms = atoms

        return stress.flat[[0, 4, 8, 5, 2, 1]]
