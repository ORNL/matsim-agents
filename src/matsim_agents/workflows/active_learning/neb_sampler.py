"""ASE NEB sampling that emits the shared active-learning candidate contract."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.io import read as ase_read
from ase.io import write as ase_write

from matsim_agents.science.defects.point_defects import geometry_fingerprint
from matsim_agents.science.neb.contracts import NebResult
from matsim_agents.science.neb.images import generate_linear_images
from matsim_agents.science.neb.paths import correspond_atoms
from matsim_agents.workflows.active_learning.candidates import Candidate
from matsim_agents.workflows.active_learning.config import NebSamplingConfig

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class NebSamplingOutcome:
    candidates: list[Candidate]
    result: NebResult
    images: tuple[Atoms, ...]
    pathway_uncertainty_eV: float | None = None


class _UncertaintyLimitReached(Exception):
    def __init__(self, value: float):
        self.value = value


def _optimizer(name: str):
    from ase.optimize import BFGS, FIRE, LBFGS

    return {"FIRE": FIRE, "BFGS": BFGS, "LBFGS": LBFGS}[name]


def _max_force(atoms: Atoms) -> float:
    forces = np.asarray(atoms.get_forces(), dtype=float)
    if forces.shape != (len(atoms), 3) or not np.isfinite(forces).all():
        raise ValueError("NEB calculator returned non-finite or malformed forces")
    return float(np.linalg.norm(forces, axis=1).max(initial=0.0))


def _relax_endpoints(
    initial_path: Path,
    final_path: Path,
    out_dir: Path,
    *,
    mlip_config: Any,
    config: NebSamplingConfig,
) -> tuple[Atoms, Atoms]:
    from matsim_agents.backends.mlip.relaxation import (
        CalculatorCache,
        RelaxStructureInput,
        relax_with_mlip,
    )

    cache = CalculatorCache()
    relaxed = []
    for role, path in (("initial", initial_path), ("final", final_path)):
        endpoint_dir = out_dir / "endpoints" / role
        args = RelaxStructureInput(
            structure_path=str(path),
            output_dir=str(endpoint_dir),
            mlip=mlip_config,
            optimizer="FIRE",
            maxiter=config.endpoint_max_steps,
            fmax=config.endpoint_fmax_eV_per_A,
        )
        outcome = relax_with_mlip(args, calculators=cache)
        if not outcome.converged:
            raise RuntimeError(f"{role} endpoint relaxation did not converge")
        atoms = ase_read(outcome.optimized_structure_path)
        relaxed.append(atoms)
    return relaxed[0], relaxed[1]


def _load_restart(path: Path, expected_count: int, reference_images):
    if not path.is_file():
        return None
    images = list(ase_read(path, index=":"))
    if len(images) != expected_count:
        raise ValueError(
            f"NEB checkpoint {path} has {len(images)} images; expected {expected_count}"
        )
    correspond_atoms(reference_images[0], images[0])
    correspond_atoms(reference_images[-1], images[-1])
    if not np.allclose(images[0].positions, reference_images[0].positions, atol=1e-8, rtol=0.0):
        raise ValueError("NEB checkpoint initial endpoint differs from the requested endpoint")
    if not np.allclose(images[-1].positions, reference_images[-1].positions, atol=1e-8, rtol=0.0):
        raise ValueError("NEB checkpoint final endpoint differs from the requested endpoint")
    return images


def _pathway_barrier_uncertainty(images: tuple[Atoms, ...] | list[Atoms], calculators):
    if len(calculators) < 2:
        return None
    barriers = []
    for calculator in calculators:
        energies = []
        for image in images:
            atoms = image.copy()
            atoms.calc = calculator
            energies.append(float(atoms.get_potential_energy()))
        if not np.isfinite(energies).all():
            raise ValueError("ensemble returned non-finite NEB image energies")
        barriers.append(max(energies) - energies[0])
    return float(np.std(barriers))


def sample_neb_candidates(
    config: NebSamplingConfig,
    calculator,
    out_dir: str | Path,
    *,
    iteration: int = 0,
    model_id: str | None = None,
    mlip_config: Any = None,
    uncertainty_calculators=(),
) -> NebSamplingOutcome:
    """Optimize one MLIP NEB pathway and return its interior images as candidates."""
    from ase.io.trajectory import Trajectory
    from ase.mep import NEB

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if config.max_pathway_uncertainty_eV is not None and len(uncertainty_calculators) < 2:
        raise ValueError("max_pathway_uncertainty_eV requires at least two uncertainty calculators")
    initial = ase_read(config.initial_path)
    final = ase_read(config.final_path)
    correspond_atoms(initial, final)
    if config.relax_endpoints:
        if mlip_config is None:
            raise ValueError("endpoint relaxation requires the configured MLIP block")
        initial, final = _relax_endpoints(
            config.initial_path, config.final_path, out_dir, mlip_config=mlip_config, config=config
        )
        correspond_atoms(initial, final)

    generated = generate_linear_images(
        initial,
        final,
        config.n_images,
        min_distance=config.min_distance_A,
        max_step=config.max_step_A,
    )
    checkpoint_path = out_dir / "neb-images.extxyz"
    images = (
        _load_restart(checkpoint_path, config.n_images, generated.images) if config.resume else None
    )
    is_restart = images is not None
    images = images or list(generated.images)
    for image in images:
        image.calc = calculator
    neb = NEB(
        images,
        k=config.spring_constant_eV_per_A2,
        climb=config.climbing_image,
        allow_shared_calculator=True,
    )
    if config.interpolation == "idpp" and not is_restart:
        neb.interpolate(method="idpp")

    trajectory_path = out_dir / "neb-optimization.traj"
    state_path = out_dir / "neb-optimization-state.json"
    prior_steps = 0
    if config.resume and state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        prior_steps = state.get("optimization_steps")
        if not isinstance(prior_steps, int) or prior_steps < 0:
            raise ValueError(f"Invalid optimization step count in NEB state {state_path}")
    optimizer = _optimizer(config.optimizer)(
        neb,
        restart=str(out_dir / "neb-optimizer.json") if config.resume else None,
        logfile=str(out_dir / "neb-optimization.log"),
    )
    starting_steps = optimizer.get_number_of_steps()
    trajectory = Trajectory(
        trajectory_path,
        mode="a" if config.resume and trajectory_path.exists() else "w",
    )

    def save_checkpoint():
        ase_write(checkpoint_path, images, format="extxyz")

    def record_trajectory():
        for image in images:
            trajectory.write(image)

    optimizer.attach(save_checkpoint, interval=1)
    optimizer.attach(record_trajectory, interval=config.record_every)
    uncertainty_evaluations = 0
    latest_interim_uncertainty = None

    def assess_pathway():
        nonlocal uncertainty_evaluations, latest_interim_uncertainty
        latest_interim_uncertainty = _pathway_barrier_uncertainty(images, uncertainty_calculators)
        uncertainty_evaluations += 1
        if (
            latest_interim_uncertainty is not None
            and config.max_pathway_uncertainty_eV is not None
            and latest_interim_uncertainty > config.max_pathway_uncertainty_eV
        ):
            raise _UncertaintyLimitReached(latest_interim_uncertainty)

    if config.uncertainty_interval_steps and uncertainty_calculators:
        optimizer.attach(assess_pathway, interval=config.uncertainty_interval_steps)
    uncertainty_stop = None
    try:
        converged = bool(optimizer.run(fmax=config.fmax_eV_per_A, steps=config.max_steps))
    except _UncertaintyLimitReached as exc:
        converged = False
        uncertainty_stop = exc.value
    finally:
        trajectory.close()
    save_checkpoint()
    optimization_steps = prior_steps + max(0, optimizer.get_number_of_steps() - starting_steps)
    state_path.write_text(
        json.dumps({"optimization_steps": optimization_steps}, indent=2) + "\n",
        encoding="utf-8",
    )

    energies = np.asarray([float(image.get_potential_energy()) for image in images])
    if not np.isfinite(energies).all():
        raise ValueError("NEB calculator returned non-finite image energies")
    relative = energies - energies[0]
    saddle_index = int(np.argmax(energies[1:-1])) + 1
    transition_id = config.transition_id or (
        f"{config.initial_path.stem}-to-{config.final_path.stem}"
    )
    barrier_uncertainty = _pathway_barrier_uncertainty(images, uncertainty_calculators)
    max_path_force = max(_max_force(image) for image in images[1:-1])
    result = NebResult(
        state="converged_unqualified" if converged else "unconverged",
        method_id=f"ase-neb-{model_id or 'mlip'}",
        n_images=len(images),
        relative_energies_eV=tuple(float(value) for value in relative),
        forward_barrier_eV=float(energies.max() - energies[0]),
        reverse_barrier_eV=float(energies.max() - energies[-1]),
        saddle_image_index=saddle_index,
        optimization_steps=optimization_steps,
        max_force_eV_per_A=max_path_force,
        pathway_uncertainty_eV=barrier_uncertainty,
        climbing_image_used=config.climbing_image,
        provenance={
            "backend": "mlip",
            "model_id": model_id or "unspecified",
            "transition_id": transition_id,
            "iteration": str(iteration),
            "optimizer": config.optimizer,
            "interpolation": config.interpolation,
            "numerically_converged": str(converged).lower(),
            "uncertainty_evaluations": str(uncertainty_evaluations),
            "uncertainty_stop_eV": ("none" if uncertainty_stop is None else str(uncertainty_stop)),
            "max_force_eV_per_A": str(max_path_force),
            "optimization_steps": str(optimization_steps),
        },
    )

    candidates = []
    for index, image in enumerate(images[1:-1], start=1):
        energy = float(image.get_potential_energy())
        forces = np.asarray(image.get_forces(), dtype=float)
        force_max = _max_force(image)
        candidate_atoms = image.copy()
        candidate_atoms.calc = None
        candidate_atoms.info.update(
            sampling_method="neb",
            transition_id=transition_id,
            image_index=index,
            saddle_point_candidate=index == saddle_index,
            neb_converged=converged,
        )
        candidates.append(
            Candidate(
                candidate_id=f"{transition_id}_image{index:03d}",
                atoms=candidate_atoms,
                predicted_energy_eV=energy,
                predicted_forces_eV_per_A=forces.tolist(),
                predicted_max_force_eV_per_A=force_max,
                sampling_method="neb",
                parent_id=config.initial_path.stem,
                sampling_run_id=f"{transition_id}-iteration-{iteration}",
                geometry_fingerprint=geometry_fingerprint(candidate_atoms),
                model_id=model_id,
                uncertainty=barrier_uncertainty,
                provenance={
                    "transition_id": transition_id,
                    "neb_state": result.state,
                    "pathway_uncertainty_eV": barrier_uncertainty,
                },
                metadata={
                    "image_index": index,
                    "reaction_coordinate": index / (len(images) - 1),
                    "saddle_point_candidate": index == saddle_index,
                    "optimization_steps": optimizer.get_number_of_steps(),
                },
            )
        )

    result_path = out_dir / f"{transition_id}-result.json"
    result_path.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    ase_write(out_dir / f"{transition_id}-pathway.extxyz", images, format="extxyz")
    log.info(
        "NEB transition=%s: converged=%s, images=%d, forward barrier=%.6g eV",
        transition_id,
        converged,
        len(images),
        result.forward_barrier_eV,
    )
    return NebSamplingOutcome(
        candidates=candidates,
        result=result,
        images=tuple(images),
        pathway_uncertainty_eV=barrier_uncertainty,
    )


__all__ = ["NebSamplingOutcome", "sample_neb_candidates"]
