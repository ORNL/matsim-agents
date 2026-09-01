"""ASE structure relaxation tool with selectable surrogate backend.

This module wraps the workflow implemented in
``HydraGNN/examples/multidataset_hpo_sc26/structure_optimization_ASE.py``
as a LangGraph-compatible tool. The HydraGNN-specific code (model loading,
fused inference, ASE calculator) is imported lazily so that the rest of the
agent framework can be developed and tested without a HydraGNN install.
"""

from __future__ import annotations

import os
from copy import deepcopy
from typing import Literal

import numpy as np
from langchain_core.tools import tool
from pydantic import BaseModel, Field, model_validator

from matsim_agents.orchestration.state import RelaxationResult


class RelaxStructureInput(BaseModel):
    """Inputs for :func:`relax_structure`."""

    structure_path: str = Field(
        ..., description="Path to the input structure file (e.g. .vasp, .cif, .xyz)."
    )
    mlip_backend: Literal["hydragnn", "uma"] = Field(
        "hydragnn",
        description="Surrogate backend used by the relaxation tool.",
    )
    logdir: str | None = Field(
        None,
        description="HydraGNN logdir containing config.json + checkpoint (required for mlip_backend='hydragnn').",
    )
    hydragnn_branch_mlp_checkpoint: str | None = Field(
        None,
        description="HydraGNN BranchWeightMLP checkpoint (.pt), required for mlip_backend='hydragnn'.",
    )
    checkpoint: str | None = Field(
        None, description="Optional HydraGNN checkpoint filename or absolute path."
    )
    optimizer: Literal["FIRE", "BFGS", "BFGSLineSearch"] = "FIRE"
    maxiter: int = 200
    maxstep: float = 1e-2
    fmax: float = Field(0.02, description="Stop when max force drops below this value (eV/Å).")
    relative_increase_threshold: float = 0.05
    charge: float = 0.0
    spin: float = 0.0
    precision: str | None = None
    mlp_precision: str | None = None
    mlp_device: Literal["cuda", "cpu"] = "cuda"
    uma_model_name: str = Field(
        "uma-s-1p1",
        description="UMA pretrained model name/checkpoint when mlip_backend='uma'.",
    )
    uma_task: Literal["omat", "omol"] = Field(
        "omat",
        description="UMA task head when mlip_backend='uma'.",
    )
    relax_cell: bool = Field(
        False,
        description=(
            "If True, relax the unit cell (volume + shape) in addition to atomic positions "
            "using ASE ExpCellFilter. Requires the calculator to predict stress. "
            "Use for vc-relax benchmarks."
        ),
    )
    random_displacement: bool = False
    random_displacement_scale: float = 0.1
    seed: int = 42
    output_dir: str | None = Field(
        None,
        description="Where to write the optimized structure, trajectory, and CSV log. "
        "Defaults to the structure's parent directory.",
    )

    @model_validator(mode="after")
    def _validate_backend_inputs(self):
        if self.mlip_backend == "hydragnn":
            if not self.logdir:
                raise ValueError("mlip_backend='hydragnn' requires logdir.")
            if not self.hydragnn_branch_mlp_checkpoint:
                raise ValueError("mlip_backend='hydragnn' requires hydragnn_branch_mlp_checkpoint.")
        return self


def _atoms_to_graph(atoms, graph_attr, radius: float, max_neighbours: int):
    """Mirror of the helper in the upstream ASE optimization script."""
    import torch
    from torch_geometric.data import Data

    from hydragnn.preprocess.graph_samples_checks_and_updates import get_radius_graph_pbc

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
            self.graph_attr = torch.tensor([charge, spin], dtype=torch.float32)
            self.last_branch_weights = None

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            structure = _atoms_to_graph(atoms, self.graph_attr, radius, max_neighbours)
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


class _NumericalStressCalculator:
    """Wraps a forces-only ASE calculator to add stress via numerical finite differences.

    Computes the 6-component Voigt stress tensor by applying central finite
    differences to each of the 6 independent cell-strain components and measuring
    the energy response.  The overhead is 12 extra energy evaluations per
    optimizer step, which is negligible for a fast ML potential.

    The wrapper transparently forwards all other attribute access to the inner
    calculator, so ``last_branch_weights`` and other custom attributes remain
    accessible to the benchmark logging code.

    Args:
        inner:  An ASE calculator that implements energy and forces.
        dx:     Strain step size (dimensionless).  Default 1e-3.
    """

    implemented_properties = ["energy", "forces", "stress", "stresses"]

    def __init__(self, inner, dx: float = 1e-3):
        self._inner = inner
        self.dx = dx
        self.results: dict = {}
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
        # Forward attribute reads (e.g. last_branch_weights) to the inner calc.
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
        # Energy + forces from the inner calculator on the original geometry.
        self._inner.calculate(atoms, ["energy", "forces"], system_changes)
        self.results["energy"] = self._inner.results["energy"]
        self.results["forces"] = self._inner.results["forces"]
        # Copy last_branch_weights *before* perturbations overwrite them.
        if hasattr(self._inner, "last_branch_weights"):
            self.last_branch_weights = self._inner.last_branch_weights

        # Always compute numerical stress so ASE's property cache stays consistent.
        self.results["stress"] = self._numerical_stress(atoms)

    def get_potential_energy(self, atoms=None, force_consistent=False):
        if "energy" not in self.results:
            self.calculate(atoms)
        return self.results["energy"]

    def get_forces(self, atoms=None):
        if "forces" not in self.results:
            self.calculate(atoms)
        return self.results["forces"]

    def get_stress(self, atoms=None):
        if "stress" not in self.results:
            self.calculate(atoms)
        return self.results["stress"]

    def get_property(self, name, atoms=None, allow_calculation=True):
        # 'stresses' (per-atom) is requested by some ASE filters; map to 'stress'.
        key = "stress" if name == "stresses" else name
        if key not in self.implemented_properties:
            # Return None gracefully for unsupported properties (e.g. 'dipole')
            # rather than raising, so ASE internals don't crash.
            return None
        if key not in self.results:
            if not allow_calculation:
                return None
            self.calculate(atoms)
        return self.results[key]

    def _numerical_stress(self, atoms) -> "np.ndarray":
        """Central FD stress in ASE Voigt 6-vector convention (eV/Å³)."""
        import numpy as np

        cell = atoms.cell.array.copy()
        vol = atoms.get_volume()
        stress = np.zeros(6)

        # Voigt order: xx=0, yy=1, zz=2, yz=3, xz=4, xy=5
        # Corresponding 3×3 indices:
        voigt = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]

        for k, (i, j) in enumerate(voigt):
            energies = {}
            for sign in (+1, -1):
                F = np.eye(3)
                F[i, j] += sign * self.dx
                a = atoms.copy()
                a.set_cell(F @ cell, scale_atoms=True)
                a.calc = self._inner
                energies[sign] = a.get_potential_energy()

            # σ_k = −(1/V) · dE/dε_k  (central difference)
            stress[k] = -(energies[+1] - energies[-1]) / (2.0 * self.dx * vol)

        # Restore inner calc's atom reference so its cache is consistent.
        if hasattr(self._inner, "atoms"):
            self._inner.atoms = atoms

        return stress


def _run(args: RelaxStructureInput) -> RelaxationResult:
    """Pure-python core of the tool (kept separate for unit testing)."""
    from ase.io import read, write
    from ase.io.trajectory import Trajectory

    structure_path = os.path.abspath(args.structure_path)
    out_dir = (
        os.path.abspath(args.output_dir) if args.output_dir else os.path.dirname(structure_path)
    )
    os.makedirs(out_dir, exist_ok=True)

    stem, ext = os.path.splitext(os.path.basename(structure_path))
    trajectory_path = os.path.join(out_dir, f"{stem}_optimization.traj")
    log_csv_path = os.path.join(out_dir, f"{stem}_optimization.csv")
    optimized_path = os.path.join(
        out_dir,
        f"{stem}_optimized_structure"
        f"{'_from_initial_randomly_perturbed_structure' if args.random_displacement else ''}{ext}",
    )

    if args.mlip_backend == "hydragnn":
        from inference_fused import load_fused_stack

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
        ) = load_fused_stack(
            args.logdir,
            args.checkpoint,
            args.hydragnn_branch_mlp_checkpoint,
            args.precision,
            args.mlp_precision,
            args.mlp_device,
        )

        arch = config["NeuralNetwork"]["Architecture"]
        radius = float(arch.get("radius", 5.0))
        max_neighbours = int(arch.get("max_neighbours", 20))

        calculator = _build_calculator(
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
            args.charge,
            args.spin,
        )
        uq_note = None
    else:
        from matsim_agents.active_learning.calculator import build_uma_calculator
        from matsim_agents.active_learning.config import UMAConfig

        calculator = build_uma_calculator(
            UMAConfig(
                model_name=args.uma_model_name,
                task=args.uma_task,
                device=args.mlp_device,
            ),
            enable_mc_dropout=False,
        )
        num_branches = 0
        uq_note = "branch-weight UQ unavailable for UMA backend"

    atoms = read(structure_path)
    atoms.calc = calculator

    if args.random_displacement:
        rng = np.random.default_rng(args.seed)
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
    traj_writer = Trajectory(trajectory_path, mode="w", atoms=atoms)
    traj_writer.write()

    cell_relax_active = False
    if args.relax_cell:
        from ase.constraints import ExpCellFilter

        # Wrap with numerical stress unconditionally when relax_cell=True;
        # _NumericalStressCalculator.calculate() will try the inner calc first
        # and only use FD for stress (which the inner calc won't provide).
        # This also patches implemented_properties on the inner calc so ASE's
        # property cache doesn't raise before reaching our wrapper.
        calculator = _NumericalStressCalculator(calculator)
        atoms.calc = calculator
        optimizable = ExpCellFilter(atoms)
        cell_relax_active = True
    else:
        optimizable = atoms

    optimizer = _build_optimizer(args.optimizer, optimizable, args.maxstep)

    csv_header = ["step", "energy_eV", "max_force_eV_per_A", "top_branch", "top_weight"] + [
        f"w_branch_{i}" for i in range(int(num_branches))
    ]
    prev_max_force: float | None = None
    prev_positions = None
    energy = float("nan")
    max_force = float("nan")
    top_branch = -1
    top_weight = float("nan")
    converged = False
    steps_taken = 0

    with open(log_csv_path, "w") as csv_file:
        csv_file.write(",".join(csv_header) + "\n")
        try:
            for step in range(args.maxiter):
                optimizer.step()
                steps_taken = step + 1

                energy = atoms.get_potential_energy()
                forces = atoms.get_forces()
                # For convergence we check the optimizable's forces (which
                # includes stress pseudo-forces when relax_cell=True).
                opt_forces = optimizable.get_forces()
                max_force = float(np.sqrt((opt_forces**2).sum(axis=1).max()))
                weights = getattr(calculator, "last_branch_weights", None)
                top_branch = int(np.argmax(weights)) if weights is not None else -1
                top_weight = float(weights[top_branch]) if weights is not None else float("nan")

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
                csv_file.write(",".join(row) + "\n")
                csv_file.flush()

                if max_force < args.fmax:
                    converged = True
                    break

                if prev_max_force is not None and prev_max_force > 0.0:
                    relative_increase = (max_force - prev_max_force) / prev_max_force
                    if relative_increase > args.relative_increase_threshold:
                        if not cell_relax_active:
                            atoms.set_positions(prev_positions)
                        break

                prev_max_force = max_force
                prev_positions = deepcopy(atoms.get_positions())
            else:
                converged = False
        finally:
            traj_writer.close()

    write(optimized_path, atoms)

    return RelaxationResult(
        structure_path=structure_path,
        optimized_structure_path=optimized_path,
        trajectory_path=trajectory_path,
        log_csv_path=log_csv_path,
        final_energy_eV=float(energy),
        final_max_force_eV_per_A=float(max_force),
        num_steps=steps_taken,
        converged=converged,
        top_branch=top_branch if top_branch >= 0 else None,
        top_branch_weight=top_weight if np.isfinite(top_weight) else None,
        notes=uq_note,
    )


@tool("relax_structure", args_schema=RelaxStructureInput)
def relax_structure(**kwargs) -> dict:
    """Relax an atomistic structure with a selected MLP backend + ASE optimizer.

    Returns the path of the optimized structure, trajectory, per-step CSV log,
    and the final energy and maximum force.
    """
    result = _run(RelaxStructureInput(**kwargs))
    return result.model_dump()
