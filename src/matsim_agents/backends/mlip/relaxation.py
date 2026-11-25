"""ASE structure relaxation tool with selectable surrogate backend.

This module wraps the workflow implemented in
``HydraGNN/examples/multidataset_hpo_sc26/structure_optimization_ASE.py``
as a LangGraph-compatible tool. The HydraGNN-specific code (model loading,
fused inference, ASE calculator) is imported lazily so that the rest of the
agent framework can be developed and tested without a HydraGNN install.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal, NamedTuple

import numpy as np
from langchain_core.tools import tool
from pydantic import BaseModel, Field, model_validator

from matsim_agents.backends.mlip.config import HydraGNNConfig, MLIPBackendConfig, MLIPConfig
from matsim_agents.core.relaxation import RelaxationResult
from matsim_agents.core.seeding import numpy_rng
from matsim_agents.core.stress import NumericalStressCalculator


class RelaxationSettings(BaseModel):
    """Optimizer and cell controls of an MLIP relaxation, independent of the backend."""

    optimizer: Literal["FIRE", "BFGS", "BFGSLineSearch"] = "FIRE"
    maxiter: int = 200
    max_optimization_seconds: float | None = Field(
        None,
        gt=0,
        allow_inf_nan=False,
        description=(
            "Soft wall-clock limit checked after completed optimizer steps. Excludes "
            "model loading and initialization; an in-progress step is not interrupted."
        ),
    )
    maxstep: float = 1e-2
    fmax: float = Field(0.02, description="Stop when max force drops below this value (eV/Å).")
    relative_increase_threshold: float | None = Field(
        0.2,
        ge=0,
        allow_inf_nan=False,
        description="Abort on fractional optimizer-force growth between steps; None disables.",
    )
    relax_cell: bool = Field(
        False,
        description=(
            "If True, relax the unit cell (volume + shape) in addition to atomic positions "
            "using ASE ExpCellFilter. Requires the calculator to predict stress. "
            "Use for vc-relax benchmarks."
        ),
    )
    cell_mask: tuple[bool, bool, bool, bool, bool, bool] | None = None
    stress_tolerance_eV_per_A3: float | None = Field(None, gt=0, allow_inf_nan=False)
    max_cell_step_strain: float | None = Field(None, gt=0, allow_inf_nan=False)
    max_cell_total_strain: float | None = Field(None, gt=0, lt=1, allow_inf_nan=False)
    require_native_stress: bool = False

    @model_validator(mode="after")
    def _validate_cell_options(self):
        if not self.relax_cell and (
            self.cell_mask is not None
            or self.stress_tolerance_eV_per_A3 is not None
            or self.max_cell_step_strain is not None
            or self.max_cell_total_strain is not None
            or self.require_native_stress
        ):
            raise ValueError("Cell-relaxation options require relax_cell=True.")
        if self.cell_mask is not None and not any(self.cell_mask):
            raise ValueError("cell_mask must enable at least one strain component.")
        return self


class RelaxStructureInput(RelaxationSettings):
    """Inputs for :func:`relax_structure`."""

    structure_path: str = Field(
        ..., description="Path to the input structure file (e.g. .vasp, .cif, .xyz)."
    )
    mlip: MLIPConfig = Field(
        ...,
        description="MLIP backend and its model settings (backend plus its own config block).",
    )
    charge: float | None = Field(
        None, description="System charge; unset uses the mlip block's configured charge."
    )
    spin: float | None = Field(
        None, description="System spin; unset uses the mlip block's configured spin."
    )
    random_displacement: bool = False
    random_displacement_scale: float = 0.1
    displacement_seed: int | None = Field(
        None, description="Seed of the random displacement; required with random_displacement."
    )
    output_dir: str | None = Field(
        None,
        description="Where to write the optimized structure, trajectory, and CSV log. "
        "Defaults to the structure's parent directory.",
    )

    @model_validator(mode="after")
    def _validate_model_inputs(self):
        if self.random_displacement and self.displacement_seed is None:
            raise ValueError("random_displacement requires displacement_seed")
        self.mlip.check_loadable()
        self.conditioned_mlip()
        return self

    def conditioned_mlip(self) -> MLIPBackendConfig:
        """The model to relax with: the mlip block with this request's charge and spin."""
        charge, spin = self.mlip.conditioning()
        return self.mlip.with_conditioning(
            charge if self.charge is None else self.charge,
            spin if self.spin is None else self.spin,
        )


def atoms_to_graph(atoms, graph_attr, radius: float, max_neighbours: int):
    """Mirror of the helper in the upstream ASE optimization script."""
    import torch
    from hydragnn.preprocess.graph_samples_checks_and_updates import get_radius_graph_pbc
    from torch_geometric.data import Data

    atomic_numbers = np.asarray(atoms.get_atomic_numbers(), dtype=np.int64)
    positions = np.asarray(atoms.get_positions(), dtype=np.float64)
    cell = np.asarray(atoms.cell.array, dtype=np.float64)
    pbc = np.asarray(atoms.get_pbc(), dtype=bool)

    hist, _ = np.histogram(atomic_numbers, bins=range(1, 118 + 2))
    data = Data(
        x=torch.tensor(atomic_numbers, dtype=torch.get_default_dtype()).unsqueeze(1),
        atomic_numbers=torch.tensor(atomic_numbers, dtype=torch.long),
        pos=torch.tensor(positions, dtype=torch.get_default_dtype()),
        chemical_composition=torch.tensor(hist, dtype=torch.float32).unsqueeze(1),
        graph_attr=graph_attr.clone(),
        natoms=torch.tensor([len(atomic_numbers)], dtype=torch.long),
        cell=torch.tensor(cell, dtype=torch.get_default_dtype()),
        pbc=torch.tensor(pbc, dtype=torch.bool),
    )
    add_edges_pbc = get_radius_graph_pbc(radius=radius, max_neighbours=max_neighbours)
    return add_edges_pbc(data)


def _load_fused_stack(logdir, checkpoint, mlp_checkpoint, precision, mlp_precision, mlp_device):
    """Load upstream models while enforcing explicit XPU co-location."""
    from inference_fused import load_fused_stack

    if mlp_device == "xpu":
        import torch

        if not torch.xpu.is_available():
            raise RuntimeError("XPU unavailable; HydraGNN CPU fallback is forbidden")
    stack = load_fused_stack(
        logdir, checkpoint, mlp_checkpoint, precision, mlp_precision, mlp_device
    )
    if mlp_device == "xpu" and stack[3].type != "xpu":
        raise RuntimeError(f"HydraGNN GNN is on {stack[3]}, but XPU was requested")
    from matsim_agents.backends.mlip.hydragnn_checkpoint import restore_hydragnn_checkpoint

    if checkpoint is None:
        from inference_random_structures import find_checkpoint

        checkpoint = find_checkpoint(logdir, checkpoint)
    model = stack[0]
    model._hydragnn_checkpoint_state_keys = restore_hydragnn_checkpoint(
        model, checkpoint, device=stack[3]
    )
    if mlp_device != "xpu":
        return stack
    from contextlib import nullcontext

    (
        model,
        mlp,
        config,
        device,
        autocast_ctx,
        param_dtype,
        num_branches,
        _mlp_device,
        _mlp_autocast_ctx,
        _unified,
        gnn_prec,
        mlp_prec,
    ) = stack
    mlp = mlp.to(device=device)
    unified = mlp_prec == gnn_prec
    mlp_autocast_ctx = (
        torch.autocast("xpu", dtype=torch.bfloat16)
        if not unified and mlp_prec == "bf16"
        else nullcontext()
    )
    return (
        model,
        mlp,
        config,
        device,
        autocast_ctx,
        param_dtype,
        num_branches,
        device,
        mlp_autocast_ctx,
        unified,
        gnn_prec,
        mlp_prec,
    )


def _build_calculator(
    model,
    mlp,
    radius,
    max_neighbours,
    param_dtype,
    autocast_ctx,
    device,
    num_branches,
    mlp_device,
    mlp_autocast_ctx,
    unified_mlp_gnn_stack,
    charge,
    spin,
):
    """Construct the FusedHydraGNNCalculator class lazily."""
    import torch
    from ase.calculators.calculator import Calculator, all_changes
    from inference_fused import run_fused_inference  # provided alongside the HydraGNN example

    class FusedHydraGNNCalculator(Calculator):
        implemented_properties = ["energy", "forces"]

        def __init__(self):
            super().__init__()
            self.model = model  # expose for score_mc_dropout (uncertainty.py)
            self.mlp = mlp
            self.graph_attr = torch.tensor([charge, spin], dtype=torch.float32)
            self.last_branch_weights = None

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            structure = atoms_to_graph(atoms, self.graph_attr, radius, max_neighbours)
            (
                all_energies,
                all_forces,
                _all_natoms,
                all_weights,
                _batch_latencies_ms,
                _total_timed_structures,
                _stage_stats,
            ) = run_fused_inference(
                model,
                mlp,
                [structure],
                batch_size=1,
                param_dtype=param_dtype,
                autocast_ctx=autocast_ctx,
                device=device,
                num_branches=num_branches,
                num_warmup=0,
                mlp_device=mlp_device,
                mlp_autocast_ctx=mlp_autocast_ctx,
                unified_mlp_gnn_stack=unified_mlp_gnn_stack,
                profile_stages=False,
            )
            self.last_branch_weights = all_weights[0].numpy()
            self.results["energy"] = float(all_energies[0])
            self.results["forces"] = all_forces[0].numpy()

    return FusedHydraGNNCalculator()


def _build_optimizer(name: str, atoms, maxstep: float):
    from ase.optimize import BFGS, FIRE
    from ase.optimize.bfgslinesearch import BFGSLineSearch

    if name == "FIRE":
        return FIRE(atoms, maxstep=maxstep)
    if name == "BFGS":
        return BFGS(atoms, maxstep=maxstep)
    return BFGSLineSearch(atoms, maxstep=maxstep)


class _RelaxPaths(NamedTuple):
    structure: str
    trajectory: str
    log_csv: str
    optimized: str


def _relaxation_paths(args: RelaxStructureInput) -> _RelaxPaths:
    structure_path = os.path.abspath(args.structure_path)
    out_dir = (
        os.path.abspath(args.output_dir) if args.output_dir else os.path.dirname(structure_path)
    )
    os.makedirs(out_dir, exist_ok=True)

    stem, ext = os.path.splitext(os.path.basename(structure_path))
    return _RelaxPaths(
        structure=structure_path,
        trajectory=os.path.join(out_dir, f"{stem}_optimization.traj"),
        log_csv=os.path.join(out_dir, f"{stem}_optimization.csv"),
        optimized=os.path.join(
            out_dir,
            f"{stem}_optimized_structure"
            f"{'_from_initial_randomly_perturbed_structure' if args.random_displacement else ''}"
            f"{ext}",
        ),
    )


class RelaxCalculator(NamedTuple):
    calculator: Any
    num_branches: int
    active_branches: int
    uq_note: str | None


def hydragnn_relaxation_calculator(cfg: HydraGNNConfig) -> RelaxCalculator:
    """HydraGNN calculator with its branch-weight UQ description."""
    declared_checkpoint = Path(cfg.logdir) / "energy-convention.json"
    if cfg.inference_head is not None or declared_checkpoint.exists():
        return _declared_hydragnn_calculator(cfg)
    return _fused_hydragnn_calculator(cfg)


def _declared_hydragnn_calculator(cfg: HydraGNNConfig) -> RelaxCalculator:
    """HydraGNN with a declared energy convention or an explicitly selected head."""
    from matsim_agents.backends.mlip.calculator import build_hydragnn_calculator

    calculator = build_hydragnn_calculator(cfg)
    num_branches = (
        1 if cfg.inference_head is not None else int(getattr(calculator.model, "num_branches", 1))
    )
    active_branches = int(getattr(calculator, "active_branch_count", num_branches))
    uq_note = (
        "HydraGNN has fewer than two active branches; branch-weight UQ disabled"
        if active_branches < 2
        else None
    )
    return RelaxCalculator(calculator, num_branches, active_branches, uq_note)


def _fused_hydragnn_calculator(cfg: HydraGNNConfig) -> RelaxCalculator:
    """Pretrained multi-branch HydraGNN weighted by its BranchWeightMLP."""
    from ase.calculators.calculator import Calculator

    from matsim_agents.backends.mlip.hydragnn_energy import restore_hydragnn_total_energy

    restore_hydragnn_total_energy(Calculator(), Path(cfg.logdir), cfg.checkpoint)

    (
        model,
        mlp,
        config,
        device,
        autocast_ctx,
        param_dtype,
        num_branches,
        mlp_device,
        mlp_autocast_ctx,
        unified_mlp_gnn_stack,
        _gnn_prec,
        _mlp_prec,
    ) = _load_fused_stack(
        str(cfg.logdir),
        cfg.checkpoint,
        str(cfg.mlp_checkpoint),
        cfg.precision,
        cfg.mlp_precision,
        cfg.device,
    )
    arch = config["NeuralNetwork"]["Architecture"]
    calculator = _build_calculator(
        model,
        mlp,
        float(arch.get("radius", 5.0)),
        int(arch.get("max_neighbours", 20)),
        param_dtype,
        autocast_ctx,
        device,
        num_branches,
        mlp_device,
        mlp_autocast_ctx,
        unified_mlp_gnn_stack,
        cfg.charge,
        cfg.spin,
    )
    return RelaxCalculator(calculator, num_branches, num_branches, None)


class CalculatorCache:
    """Loaded MLIP calculators reused across relaxations with identical model settings.

    Keeps the ``capacity`` most recently used calculators so alternating
    between, for example, an incumbent and a candidate model does not reload
    either; older entries are dropped to bound device memory.
    """

    def __init__(self, capacity: int = 2) -> None:
        if capacity < 1:
            raise ValueError("calculator cache capacity must be positive")
        self.capacity = capacity
        self._entries: dict[tuple, RelaxCalculator] = {}
        self.loads = 0

    @staticmethod
    def key(args: RelaxStructureInput) -> tuple:
        return args.conditioned_mlip().model_dump_json()

    def get(self, args: RelaxStructureInput) -> RelaxCalculator:
        key = self.key(args)
        entry = self._entries.pop(key, None)
        if entry is None:
            if len(self._entries) >= self.capacity:
                from matsim_agents.backends.mlip.pool import release_device_memory

                while len(self._entries) >= self.capacity:
                    self._entries.pop(next(iter(self._entries)))
                # Free the evicted model before loading its replacement.
                release_device_memory()
            entry = _relaxation_calculator(args)
            self.loads += 1
        elif hasattr(entry.calculator, "reset"):
            entry.calculator.reset()
        self._entries[key] = entry
        return entry


def _relaxation_calculator(args: RelaxStructureInput) -> RelaxCalculator:
    """Build the selected MLIP's relaxation calculator."""
    return args.conditioned_mlip().relaxation_calculator()


class _OptimizationOutcome(NamedTuple):
    energy: float
    max_force: float
    top_branch: int
    top_weight: float
    converged: bool
    steps_taken: int
    termination_reason: str


def _optimize_with_log(
    args: RelaxStructureInput,
    atoms,
    optimizable,
    optimizer,
    calculator,
    traj_writer,
    *,
    log_csv_path: str,
    num_branches: int,
    active_branches: int,
    cell_relax_active: bool,
    step_guard: Callable[[Any], str | None] | None = None,
) -> _OptimizationOutcome:
    """Step the optimizer, logging per-step energy, force, and branch weights to CSV.

    Stops at convergence, after ``maxiter`` steps, or when the maximum force grows
    by more than ``relative_increase_threshold`` in one step (positions are then
    rolled back unless the cell is relaxing). An optional wall-clock limit stops
    after a completed step without discarding the calculator or its model.
    """
    csv_header = ["step", "energy_eV", "max_force_eV_per_A", "top_branch", "top_weight"] + [
        f"w_branch_{i}" for i in range(int(num_branches))
    ]
    if cell_relax_active:
        csv_header += ["max_atomic_force_eV_per_A", "max_active_stress_eV_per_A3", "cell_volume_A3"]
    prev_max_force: float | None = None
    prev_positions = None
    energy = float("nan")
    max_force = float("nan")
    top_branch = -1
    top_weight = float("nan")
    converged = False
    steps_taken = 0
    termination_reason = "max_steps"
    initial_cell = atoms.cell.array.copy()
    previous_cell = initial_cell.copy()
    initial_determinant = np.linalg.det(initial_cell)
    if cell_relax_active and (
        not np.isfinite(initial_determinant) or abs(initial_determinant) < 1e-12
    ):
        raise ValueError("Cell relaxation requires a finite nonsingular initial cell.")

    with open(log_csv_path, "w") as csv_file:
        csv_file.write(",".join(csv_header) + "\n")
        try:
            started = time.monotonic() if args.max_optimization_seconds is not None else None
            for step in range(args.maxiter):
                previous_positions = atoms.get_positions().copy()
                optimizer.step()
                steps_taken = step + 1
                if cell_relax_active:
                    current_cell = atoms.cell.array.copy()
                    if (
                        not np.isfinite(current_cell).all()
                        or np.linalg.det(current_cell) * initial_determinant <= 0
                    ):
                        reason = "invalid_cell"
                    else:
                        step_strain = np.max(
                            np.abs(
                                np.linalg.svd(
                                    np.linalg.solve(previous_cell, current_cell), compute_uv=False
                                )
                                - 1
                            )
                        )
                        total_strain = np.max(
                            np.abs(
                                np.linalg.svd(
                                    np.linalg.solve(initial_cell, current_cell), compute_uv=False
                                )
                                - 1
                            )
                        )
                        reason = None
                        if (
                            args.max_cell_step_strain is not None
                            and step_strain > args.max_cell_step_strain
                        ):
                            reason = "cell_step_strain_limit"
                        elif (
                            args.max_cell_total_strain is not None
                            and total_strain > args.max_cell_total_strain
                        ):
                            reason = "cell_total_strain_limit"
                        elif step_guard is not None:
                            reason = step_guard(atoms)
                    if reason is not None:
                        atoms.set_cell(previous_cell, scale_atoms=False)
                        atoms.set_positions(previous_positions)
                        energy = float(atoms.get_potential_energy())
                        max_force = float(np.linalg.norm(atoms.get_forces(), axis=1).max())
                        termination_reason = reason
                        traj_writer.write()
                        break
                    previous_cell = current_cell

                energy = atoms.get_potential_energy()
                atomic_forces = atoms.get_forces()
                # For convergence we check the optimizable's forces (which
                # includes stress pseudo-forces when relax_cell=True).
                opt_forces = optimizable.get_forces()
                max_force = float(np.sqrt((opt_forces**2).sum(axis=1).max()))
                atomic_max_force = float(np.linalg.norm(atomic_forces, axis=1).max())
                if not np.isfinite(energy) or not np.isfinite(opt_forces).all():
                    raise ValueError("Non-finite energy or forces during relaxation.")
                active_stress = None
                if cell_relax_active:
                    stress = np.asarray(atoms.get_stress(), dtype=float)
                    if not np.isfinite(stress).all():
                        raise ValueError("Non-finite stress during relaxation.")
                    active = np.asarray(args.cell_mask or (True,) * 6, dtype=bool)
                    active_stress = float(np.max(np.abs(stress[active])))
                weights = getattr(calculator, "last_branch_weights", None)
                top_branch = int(np.argmax(weights)) if weights is not None else -1
                top_weight = (
                    float(weights[top_branch])
                    if weights is not None and active_branches >= 2
                    else float("nan")
                )

                traj_writer.write()

                row = [
                    str(steps_taken),
                    f"{energy:.8e}",
                    f"{max_force:.8e}",
                    str(top_branch),
                    f"{top_weight:.6f}",
                ]
                row += (
                    [f"{float(w):.6f}" for w in weights]
                    if weights is not None
                    else ["nan"] * int(num_branches)
                )
                if cell_relax_active:
                    row += [
                        f"{atomic_max_force:.8e}",
                        f"{active_stress:.8e}",
                        f"{atoms.get_volume():.8e}",
                    ]
                csv_file.write(",".join(row) + "\n")
                csv_file.flush()

                stress_ok = True
                if args.stress_tolerance_eV_per_A3 is not None:
                    assert active_stress is not None
                    stress_ok = active_stress <= args.stress_tolerance_eV_per_A3
                if max_force < args.fmax and atomic_max_force < args.fmax and stress_ok:
                    converged = True
                    termination_reason = "force_threshold"
                    break

                if (
                    args.relative_increase_threshold is not None
                    and prev_max_force is not None
                    and prev_max_force > 0.0
                ):
                    relative_increase = (max_force - prev_max_force) / prev_max_force
                    if relative_increase > args.relative_increase_threshold:
                        if not cell_relax_active:
                            atoms.set_positions(prev_positions)
                            energy = float(atoms.get_potential_energy())
                            max_force = float(np.linalg.norm(atoms.get_forces(), axis=1).max())
                        termination_reason = "force_growth"
                        break

                prev_max_force = max_force
                prev_positions = deepcopy(atoms.get_positions())
                if (
                    started is not None
                    and time.monotonic() - started >= args.max_optimization_seconds
                ):
                    termination_reason = "optimization_time_limit"
                    break
            else:
                converged = False
        finally:
            traj_writer.close()

    return _OptimizationOutcome(
        energy,
        max_force,
        top_branch,
        top_weight,
        converged,
        steps_taken,
        termination_reason,
    )


def relax_with_mlip(
    args: RelaxStructureInput,
    *,
    calculators: CalculatorCache | None = None,
    step_guard: Callable[[Any], str | None] | None = None,
) -> RelaxationResult:
    """Relax one structure with the selected MLIP backend and ASE optimizer.

    With *calculators*, a model already loaded for the same settings is reused
    instead of being read from disk again.
    """
    from ase.io import read, write
    from ase.io.trajectory import Trajectory

    paths = _relaxation_paths(args)
    built = calculators.get(args) if calculators is not None else _relaxation_calculator(args)
    calculator, num_branches, active_branches, uq_note = built

    atoms = read(paths.structure)
    atoms.calc = calculator

    if args.random_displacement:
        assert args.displacement_seed is not None
        rng = numpy_rng(args.displacement_seed)
        atoms.set_positions(
            atoms.get_positions()
            + rng.uniform(
                -args.random_displacement_scale,
                args.random_displacement_scale,
                size=atoms.get_positions().shape,
            )
        )

    atoms.get_potential_energy()
    atoms.get_forces()
    traj_writer = Trajectory(paths.trajectory, mode="w", atoms=atoms)
    traj_writer.write()

    cell_relax_active = False
    if args.relax_cell:
        from ase.filters import ExpCellFilter

        if args.require_native_stress and "stress" not in getattr(
            calculator, "implemented_properties", ()
        ):
            raise ValueError("Cell relaxation requires native calculator stress.")
        if args.cell_mask == (True, True, False, False, False, True):
            cell = atoms.cell.array
            if not np.allclose(cell[:2, 2], 0, atol=1e-8) or not np.allclose(
                cell[2, :2], 0, atol=1e-8
            ):
                raise ValueError("The in-plane cell mask requires the vacuum vector along z.")
        # Wrap with numerical stress unconditionally when relax_cell=True;
        # NumericalStressCalculator.calculate() will try the inner calc first
        # and only use FD for stress (which the inner calc won't provide).
        # This also patches implemented_properties on the inner calc so ASE's
        # property cache doesn't raise before reaching our wrapper.
        calculator = NumericalStressCalculator(
            calculator, require_native=args.require_native_stress
        )
        atoms.calc = calculator
        optimizable = ExpCellFilter(atoms, mask=args.cell_mask)
        cell_relax_active = True
    else:
        optimizable = atoms

    optimizer = _build_optimizer(args.optimizer, optimizable, args.maxstep)
    outcome = _optimize_with_log(
        args,
        atoms,
        optimizable,
        optimizer,
        calculator,
        traj_writer,
        log_csv_path=paths.log_csv,
        num_branches=num_branches,
        active_branches=active_branches,
        cell_relax_active=cell_relax_active,
        step_guard=step_guard,
    )

    write(paths.optimized, atoms)

    return RelaxationResult(
        structure_path=paths.structure,
        optimized_structure_path=paths.optimized,
        trajectory_path=paths.trajectory,
        log_csv_path=paths.log_csv,
        final_energy_eV=float(outcome.energy),
        final_max_force_eV_per_A=(
            float(np.linalg.norm(atoms.get_forces(), axis=1).max())
            if cell_relax_active
            else float(outcome.max_force)
        ),
        num_steps=outcome.steps_taken,
        converged=outcome.converged,
        termination_reason=outcome.termination_reason,
        top_branch=outcome.top_branch if outcome.top_branch >= 0 else None,
        top_branch_weight=outcome.top_weight if np.isfinite(outcome.top_weight) else None,
        notes=uq_note,
    )


@tool("relax_structure", args_schema=RelaxStructureInput)
def relax_structure(**kwargs) -> dict:
    """Relax an atomistic structure with a selected MLP backend + ASE optimizer.

    Returns the path of the optimized structure, trajectory, per-step CSV log,
    and the final energy and maximum force.
    """
    result = relax_with_mlip(RelaxStructureInput(**kwargs))
    return result.model_dump()
