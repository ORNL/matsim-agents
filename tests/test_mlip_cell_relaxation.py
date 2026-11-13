"""Cell relaxation with the stress wrapper follows the geometry it is asked about."""

from __future__ import annotations

import pytest
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.filters import ExpCellFilter
from ase.io import read, write
from ase.optimize import FIRE

from matsim_agents.backends.mlip.relaxation import (
    RelaxStructureInput,
    _NumericalStressCalculator,
    _RelaxCalculator,
    relax_with_mlip,
)


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


def _relax_emt(tmp_path, monkeypatch, **options):
    source = tmp_path / "input.extxyz"
    atoms = bulk("Cu", "fcc", a=3.8, cubic=True)
    write(source, atoms)
    monkeypatch.setattr(
        "matsim_agents.backends.mlip.relaxation._relaxation_calculator",
        lambda args: _RelaxCalculator(EMT(), 0, 0, None),
    )
    guard = options.pop("step_guard", None)
    result = relax_with_mlip(
        RelaxStructureInput(
            structure_path=str(source),
            mlip_backend="mace",
            mlp_device="cpu",
            relax_cell=True,
            maxiter=2,
            relative_increase_threshold=100.0,
            output_dir=str(tmp_path),
            **options,
        ),
        step_guard=guard,
    )
    return atoms, read(result.optimized_structure_path), result


def test_in_plane_mask_preserves_out_of_plane_cell(tmp_path, monkeypatch):
    initial, final, _ = _relax_emt(
        tmp_path, monkeypatch, cell_mask=(True, True, False, False, False, True)
    )
    assert final.cell[2] == pytest.approx(initial.cell[2])


def test_stress_tolerance_prevents_false_convergence(tmp_path, monkeypatch):
    _, _, result = _relax_emt(
        tmp_path,
        monkeypatch,
        fmax=100.0,
        stress_tolerance_eV_per_A3=1e-10,
        require_native_stress=True,
    )
    assert not result.converged
    assert result.termination_reason == "max_steps"


def test_convergence_and_guard_take_precedence_over_time_limit(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import matsim_agents.backends.mlip.relaxation as relaxation

    clock = [0.0]
    original_step = FIRE.step

    def slow_step(optimizer, *args, **kwargs):
        result = original_step(optimizer, *args, **kwargs)
        clock[0] += 1
        return result

    monkeypatch.setattr(FIRE, "step", slow_step)
    monkeypatch.setattr(relaxation, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    _, _, result = _relax_emt(tmp_path, monkeypatch, fmax=100.0, max_optimization_seconds=0.01)
    assert result.converged
    assert result.termination_reason == "force_threshold"

    _, _, result = _relax_emt(
        tmp_path,
        monkeypatch,
        max_optimization_seconds=0.01,
        step_guard=lambda atoms: "mof_connectivity_changed",
    )
    assert not result.converged
    assert result.termination_reason == "mof_connectivity_changed"


def test_in_plane_convergence_ignores_out_of_plane_stress(tmp_path, monkeypatch):
    import numpy as np
    from ase.calculators.calculator import Calculator, all_changes

    class OutOfPlaneStress(Calculator):
        implemented_properties = ["energy", "forces", "stress"]

        def calculate(self, atoms=None, properties=None, system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            self.results = {
                "energy": 0.0,
                "forces": np.zeros((len(atoms), 3)),
                "stress": np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
            }

    source = tmp_path / "input.extxyz"
    write(source, bulk("Cu", "fcc", a=3.8, cubic=True))
    monkeypatch.setattr(
        "matsim_agents.backends.mlip.relaxation._relaxation_calculator",
        lambda args: _RelaxCalculator(OutOfPlaneStress(), 0, 0, None),
    )
    args = RelaxStructureInput(
        structure_path=str(source),
        mlip_backend="mace",
        relax_cell=True,
        cell_mask=(True, True, False, False, False, True),
        require_native_stress=True,
        stress_tolerance_eV_per_A3=0.01,
        maxiter=2,
        output_dir=str(tmp_path),
    )
    assert relax_with_mlip(args).converged


def test_connectivity_guard_rolls_back_rejected_step(tmp_path, monkeypatch):
    initial, final, result = _relax_emt(
        tmp_path, monkeypatch, step_guard=lambda atoms: "mof_connectivity_changed"
    )
    assert result.termination_reason == "mof_connectivity_changed"
    assert not result.converged
    assert final.cell.array == pytest.approx(initial.cell.array)
    assert final.positions == pytest.approx(initial.positions)


def test_cell_strain_guard_rolls_back_rejected_step(tmp_path, monkeypatch):
    initial, final, result = _relax_emt(tmp_path, monkeypatch, max_cell_step_strain=1e-10)
    assert result.termination_reason == "cell_step_strain_limit"
    assert not result.converged
    assert final.cell.array == pytest.approx(initial.cell.array)
    assert final.positions == pytest.approx(initial.positions)


@pytest.mark.parametrize(
    ("scale", "expected_reason"), [(1.099, "max_steps"), (1.101, "cell_total_strain_limit")]
)
def test_total_strain_limit_at_ten_percent(tmp_path, monkeypatch, scale, expected_reason):
    initial_cell = bulk("Cu", "fcc", a=3.8, cubic=True).cell.array

    def set_cell_step(optimizer, *args, **kwargs):
        optimizer.atoms.atoms.set_cell(initial_cell * scale, scale_atoms=True)

    monkeypatch.setattr(FIRE, "step", set_cell_step)
    initial, final, result = _relax_emt(
        tmp_path, monkeypatch, fmax=1e-10, max_cell_total_strain=0.1
    )
    assert result.termination_reason == expected_reason
    assert final.cell.array == pytest.approx(
        initial.cell.array
        if expected_reason == "cell_total_strain_limit"
        else initial.cell.array * scale
    )


def test_native_stress_requirement_rejects_forces_only(tmp_path, monkeypatch):
    source = tmp_path / "input.extxyz"
    write(source, bulk("Cu", "fcc", a=3.8))
    monkeypatch.setattr(
        "matsim_agents.backends.mlip.relaxation._relaxation_calculator",
        lambda args: _RelaxCalculator(_ForcesOnlyEMT(), 0, 0, None),
    )
    args = RelaxStructureInput(
        structure_path=str(source),
        mlip_backend="mace",
        relax_cell=True,
        require_native_stress=True,
        output_dir=str(tmp_path),
    )
    with pytest.raises(ValueError, match="native calculator stress"):
        relax_with_mlip(args)


@pytest.mark.parametrize(
    ("threshold", "steps", "reason"),
    [(0.05, 2, "force_growth"), (0.2, 3, "force_growth"), (None, 3, "max_steps")],
)
def test_optional_force_growth_threshold(tmp_path, monkeypatch, threshold, steps, reason):
    import numpy as np
    from ase.calculators.calculator import Calculator, all_changes

    class GrowingForce(Calculator):
        implemented_properties = ["energy", "forces"]

        def calculate(self, atoms=None, properties=None, system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            magnitude = (1.0, 1.19, 1.44)[int(round(atoms.positions[0, 0])) - 1]
            self.results = {
                "energy": 0.0,
                "forces": np.tile([magnitude, 0.0, 0.0], (len(atoms), 1)),
            }

    source = tmp_path / "input.extxyz"
    atoms = bulk("Cu", "fcc", a=3.8, cubic=True)
    write(source, atoms)
    monkeypatch.setattr(
        "matsim_agents.backends.mlip.relaxation._relaxation_calculator",
        lambda args: _RelaxCalculator(GrowingForce(), 0, 0, None),
    )

    def step(optimizer):
        positions = optimizer.atoms.get_positions()
        positions[:, 0] += 1
        optimizer.atoms.set_positions(positions)

    monkeypatch.setattr(FIRE, "step", step)
    result = relax_with_mlip(
        RelaxStructureInput(
            structure_path=str(source),
            output_dir=str(tmp_path),
            mlip_backend="mace",
            maxiter=3,
            relative_increase_threshold=threshold,
        )
    )
    assert result.num_steps == steps
    assert result.termination_reason == reason


@pytest.mark.parametrize("threshold", [-0.1, float("inf"), float("nan")])
def test_force_growth_rejects_invalid_threshold(threshold):
    with pytest.raises(ValueError):
        RelaxStructureInput(
            structure_path="input.extxyz",
            mlip_backend="mace",
            relative_increase_threshold=threshold,
        )


def test_force_growth_default():
    args = RelaxStructureInput(structure_path="input.extxyz", mlip_backend="mace")
    assert args.relative_increase_threshold == 0.2
