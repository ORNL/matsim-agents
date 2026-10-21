"""Cell relaxation with the stress wrapper follows the geometry it is asked about."""

from __future__ import annotations

import pytest
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.filters import ExpCellFilter
from ase.optimize import FIRE

from matsim_agents.backends.mlip.relaxation import _NumericalStressCalculator


class _ForcesOnlyEMT(EMT):
    implemented_properties = ["energy", "forces"]


def _relaxed_lattice(calculator) -> float:
    atoms = bulk("Cu", "fcc", a=3.8, cubic=True)
    atoms.calc = calculator
    optimizer = FIRE(ExpCellFilter(atoms), logfile=None)
    assert optimizer.run(fmax=0.01, steps=300)
    return float(atoms.cell.lengths()[0])


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize("inner", [EMT, _ForcesOnlyEMT], ids=["native-stress", "numerical"])
def test_cell_relaxation_matches_the_reference_calculator(inner):
    reference = _relaxed_lattice(EMT())
    assert _relaxed_lattice(_NumericalStressCalculator(inner())) == pytest.approx(
        reference, abs=1e-3
    )


def test_results_follow_geometry_changes():
    atoms = bulk("Cu", "fcc", a=3.8, cubic=True)
    atoms.calc = _NumericalStressCalculator(_ForcesOnlyEMT())
    energy = atoms.get_potential_energy()
    stress = atoms.get_stress().copy()
    atoms.positions[0] += 0.1
    assert atoms.get_potential_energy() != energy
    atoms.set_cell(atoms.cell * 1.02, scale_atoms=True)
    assert not (atoms.get_stress() == stress).all()


def test_numerical_stress_matches_the_analytical_stress():
    import numpy as np

    atoms = bulk("Cu", "fcc", a=3.7, cubic=True)
    atoms.rattle(0.05, seed=1)
    atoms.set_cell(atoms.cell @ np.array([[1.0, 0.02, 0.0], [0.0, 1.0, 0.01], [0.0, 0.0, 1.03]]))
    atoms.calc = EMT()
    analytical = atoms.get_stress()
    atoms.calc = _NumericalStressCalculator(_ForcesOnlyEMT())
    assert atoms.get_stress() == pytest.approx(analytical, abs=1e-5)
