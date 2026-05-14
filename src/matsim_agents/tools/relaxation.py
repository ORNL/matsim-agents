"""HydraGNN + ASE structure relaxation tool.

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
from pydantic import BaseModel, Field

from matsim_agents.state import RelaxationResult


class RelaxStructureInput(BaseModel):
    """Inputs for :func:`relax_structure`."""

    structure_path: str = Field(..., description="Path to the input structure file (e.g. .vasp, .cif, .xyz).")
    logdir: str = Field(..., description="Directory containing the HydraGNN config.json and checkpoint.")
    mlp_checkpoint: str = Field(..., description="Path to the auxiliary BranchWeightMLP checkpoint (.pt).")
    checkpoint: str | None = Field(None, description="Optional HydraGNN checkpoint filename or absolute path.")
    optimizer: Literal["FIRE", "BFGS", "BFGSLineSearch"] = "FIRE"
    maxiter: int = 200
    maxstep: float = 1e-2
    relative_increase_threshold: float = 0.05
    charge: float = 0.0
    spin: float = 0.0
    precision: str | None = None
    mlp_precision: str | None = None
    mlp_device: Literal["cuda", "cpu"] = "cuda"
    random_displacement: bool = False
    random_displacement_scale: float = 0.1
    seed: int = 42
    output_dir: str | None = Field(
        None,
        description="Where to write the optimized structure, trajectory, and CSV log. "
        "Defaults to the structure's parent directory.",
    )


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


def _build_calculator(model, mlp, radius, max_neighbours, param_dtype, autocast_ctx,
                     device, num_branches, mlp_device, mlp_autocast_ctx,
                     unified_mlp_gnn_stack, charge, spin):
    """Construct the FusedHydraGNNCalculator class lazily."""
    import torch
    from ase.calculators.calculator import Calculator, all_changes

    from inference_fused import run_fused_inference  # provided alongside the HydraGNN example

    class FusedHydraGNNCalculator(Calculator):
        implemented_properties = ["energy", "forces"]

        def __init__(self):
            super().__init__()
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
                model, mlp, [structure],
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


def _run(args: RelaxStructureInput) -> RelaxationResult:
    """Pure-python core of the tool (kept separate for unit testing)."""
    from ase.io import read, write
    from ase.io.trajectory import Trajectory

    from inference_fused import load_fused_stack

    structure_path = os.path.abspath(args.structure_path)
    out_dir = os.path.abspath(args.output_dir) if args.output_dir else os.path.dirname(structure_path)
    os.makedirs(out_dir, exist_ok=True)

    stem, ext = os.path.splitext(os.path.basename(structure_path))
    trajectory_path = os.path.join(out_dir, f"{stem}_optimization.traj")
    log_csv_path = os.path.join(out_dir, f"{stem}_optimization.csv")
    optimized_path = os.path.join(
        out_dir,
        f"{stem}_optimized_structure"
        f"{'_from_initial_randomly_perturbed_structure' if args.random_displacement else ''}{ext}",
    )

    (model, mlp, config, device, autocast_ctx, param_dtype, num_branches,
     mlp_device, mlp_autocast_ctx, unified_mlp_gnn_stack, _gnn_prec, _mlp_prec) = load_fused_stack(
        args.logdir, args.checkpoint, args.mlp_checkpoint,
        args.precision, args.mlp_precision, args.mlp_device,
    )

    arch = config["NeuralNetwork"]["Architecture"]
    radius = float(arch.get("radius", 5.0))
    max_neighbours = int(arch.get("max_neighbours", 20))

    calculator = _build_calculator(
        model, mlp, radius, max_neighbours, param_dtype, autocast_ctx,
        device, num_branches, mlp_device, mlp_autocast_ctx,
        unified_mlp_gnn_stack, args.charge, args.spin,
    )

    atoms = read(structure_path)
    atoms.calc = calculator

    if args.random_displacement:
        rng = np.random.default_rng(args.seed)
        atoms.set_positions(
            atoms.get_positions()
            + rng.uniform(-args.random_displacement_scale, args.random_displacement_scale,
                          size=atoms.get_positions().shape)
        )

    atoms.get_potential_energy()
    atoms.get_forces()
    traj_writer = Trajectory(trajectory_path, mode="w", atoms=atoms)
    traj_writer.write()

    optimizer = _build_optimizer(args.optimizer, atoms, args.maxstep)

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
                max_force = float(np.sqrt((forces ** 2).sum(axis=1).max()))
                weights = calculator.last_branch_weights
                top_branch = int(np.argmax(weights)) if weights is not None else -1
                top_weight = float(weights[top_branch]) if weights is not None else float("nan")

                traj_writer.write()

                row = [str(steps_taken), f"{energy:.8e}", f"{max_force:.8e}",
                       str(top_branch), f"{top_weight:.6f}"]
                row += ([f"{float(w):.6f}" for w in weights]
                        if weights is not None else ["nan"] * int(num_branches))
                csv_file.write(",".join(row) + "\n")
                csv_file.flush()

                if prev_max_force is not None and prev_max_force > 0.0:
                    relative_increase = (max_force - prev_max_force) / prev_max_force
                    if relative_increase > args.relative_increase_threshold:
                        atoms.set_positions(prev_positions)
                        converged = True  # treat revert as a soft-converged stop
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
        top_branch=top_branch,
        top_branch_weight=top_weight,
    )


@tool("relax_structure", args_schema=RelaxStructureInput)
def relax_structure(**kwargs) -> dict:
    """Relax an atomistic structure with HydraGNN + an ASE optimizer.

    Returns the path of the optimized structure, trajectory, per-step CSV log,
    and the final energy and maximum force.
    """
    result = _run(RelaxStructureInput(**kwargs))
    return result.model_dump()
