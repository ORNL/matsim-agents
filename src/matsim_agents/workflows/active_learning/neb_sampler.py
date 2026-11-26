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
from matsim_agents.science.neb.contracts import (
    MLIP_ENDPOINT_PREFIX,
    NebEndpoint,
    NebResult,
    NebTransition,
    build_correspondence,
)
from matsim_agents.science.neb.images import generate_linear_images, validate_images
from matsim_agents.science.neb.paths import correspond_atoms, minimum_image
from matsim_agents.workflows.active_learning.candidates import Candidate
from matsim_agents.workflows.active_learning.config import NebSamplingConfig
from matsim_agents.workflows.active_learning.uncertainty import (
    PathwayUncertainty,
    score_ensemble_pathway,
    score_mc_dropout_pathway,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class NebSamplingOutcome:
    candidates: list[Candidate]
    result: NebResult
    images: tuple[Atoms, ...]
    pathway_uncertainty_eV: float | None = None
    transition: NebTransition | None = None
    # The validated (or relaxed) endpoint structures the pathway was built from;
    # the first and last images are reordered and unwrapped, so they can differ.
    initial: Atoms | None = None
    final: Atoms | None = None


@dataclass(frozen=True)
class NebSamplingTask:
    config_json: str
    mlip_config_json: str
    out_dir: str
    transition_label: str
    iteration: int
    model_id: str | None
    enable_mc_dropout: bool
    uncertainty_mode: str | None
    dropout_passes: int
    dropout_probability: float
    uncertainty_seed: int


@dataclass(frozen=True)
class NebSamplingWorkerResult:
    candidates: list[Candidate]
    result_json: str
    images: tuple[Atoms, ...]
    pathway_uncertainty_eV: float | None
    transition_json: str | None
    initial: Atoms | None = None
    final: Atoms | None = None


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


def _validate_transition_endpoints(transition: NebTransition, initial: Atoms, final: Atoms) -> None:
    for endpoint, atoms in ((transition.initial, initial), (transition.final, final)):
        if "atom_id" not in atoms.arrays:
            raise ValueError("typed NEB transitions require stable atom_id arrays in endpoints")
        species_by_id = {
            int(atom_id): symbol
            for atom_id, symbol in zip(
                atoms.get_array("atom_id"), atoms.get_chemical_symbols(), strict=True
            )
        }
        expected_species = dict(zip(endpoint.atom_ids, endpoint.species, strict=True))
        if species_by_id != expected_species:
            raise ValueError(f"{endpoint.role} endpoint identities differ from typed transition")
        if geometry_fingerprint(atoms) != endpoint.geometry_fingerprint:
            raise ValueError(f"{endpoint.role} endpoint geometry differs from typed transition")
        if tuple(bool(value) for value in atoms.pbc) != endpoint.pbc:
            raise ValueError(f"{endpoint.role} endpoint PBC differs from typed transition")
        if not np.allclose(atoms.cell.array, np.asarray(endpoint.cell), atol=1e-6, rtol=0.0):
            raise ValueError(f"{endpoint.role} endpoint cell differs from typed transition")


def _constraint_signature(atoms: Atoms) -> str:
    """Constraints keyed by stable atom identity, not by (reorderable) ASE index."""
    from ase.constraints import FixAtoms, FixCartesian

    atom_ids = [int(atom_id) for atom_id in atoms.get_array("atom_id")]
    fixed_atoms: set[int] = set()
    fixed_axes: dict[int, tuple[bool, bool, bool]] = {}
    for constraint in atoms.constraints:
        if isinstance(constraint, FixAtoms):
            fixed_atoms.update(atom_ids[int(index)] for index in constraint.index)
        elif isinstance(constraint, FixCartesian):
            mask = tuple(bool(value) for value in np.asarray(constraint.mask).reshape(-1)[:3])
            for index in np.atleast_1d(constraint.index):
                atom_id = atom_ids[int(index)]
                previous = fixed_axes.get(atom_id, (False, False, False))
                fixed_axes[atom_id] = tuple(a or b for a, b in zip(previous, mask, strict=True))
        else:
            raise ValueError(
                f"NEB endpoint constraint {type(constraint).__name__} cannot be validated"
            )
    return json.dumps(
        {
            "fix_atoms": sorted(fixed_atoms),
            "fix_cartesian": {str(key): list(value) for key, value in sorted(fixed_axes.items())},
        },
        sort_keys=True,
    )


def _endpoint_settings(atoms: Atoms) -> tuple[str, dict[int, tuple[float, ...]]]:
    if "atom_id" not in atoms.arrays:
        raise ValueError("NEB endpoint magnetic validation requires stable atom_id arrays")
    constraint_signature = _constraint_signature(atoms)
    moments = np.asarray(atoms.get_initial_magnetic_moments(), dtype=float)
    if moments.shape not in ((len(atoms),), (len(atoms), 3)) or not np.isfinite(moments).all():
        raise ValueError("NEB endpoint has malformed or non-finite initial magnetic moments")
    moment_by_id = {
        int(atom_id): tuple(float(value) for value in np.atleast_1d(moment))
        for atom_id, moment in zip(atoms.get_array("atom_id"), moments, strict=True)
    }
    return constraint_signature, moment_by_id


def _validate_endpoint_settings_match(initial: Atoms, final: Atoms) -> None:
    initial_constraints, initial_moments = _endpoint_settings(initial)
    final_constraints, final_moments = _endpoint_settings(final)
    if initial_constraints != final_constraints:
        raise ValueError("initial and final NEB endpoint constraints differ")
    if initial_moments.keys() != final_moments.keys() or any(
        not np.allclose(initial_moments[atom_id], final_moments[atom_id], atol=1e-8, rtol=0.0)
        for atom_id in initial_moments
    ):
        raise ValueError("initial and final NEB endpoint magnetic configurations differ")


def _validate_endpoint_settings_preserved(
    before: tuple[str, dict[int, tuple[float, ...]]],
    after_atoms: Atoms,
    role: str,
) -> None:
    after = _endpoint_settings(after_atoms)
    if before[0] != after[0]:
        raise ValueError(f"{role} endpoint relaxation changed the atomic constraints")
    if before[1].keys() != after[1].keys() or any(
        not np.allclose(before[1][atom_id], after[1][atom_id], atol=1e-8, rtol=0.0)
        for atom_id in before[1]
    ):
        raise ValueError(f"{role} endpoint relaxation changed the magnetic configuration")


def _validate_moving_atoms_survive_relaxation(
    transition: NebTransition | None,
    initial: Atoms,
    final: Atoms,
) -> None:
    if transition is None:
        return
    initial_index = {
        int(atom_id): index for index, atom_id in enumerate(initial.get_array("atom_id"))
    }
    final_index = {int(atom_id): index for index, atom_id in enumerate(final.get_array("atom_id"))}
    displacements = []
    for atom_id in transition.mechanism.moving_atom_ids:
        vector = final.positions[final_index[atom_id]] - initial.positions[initial_index[atom_id]]
        _, lengths, _ = minimum_image(vector.reshape(1, 3), initial.cell.array)
        displacements.append(float(lengths[0]))
    if max(displacements, default=0.0) <= 1e-6:
        raise ValueError(
            "endpoint relaxation collapsed the declared moving-atom displacement; "
            "the intended transition is no longer represented"
        )


def _validate_relaxed_endpoint_provenance(
    transition: NebTransition | None,
    model_id: str | None,
    *,
    relax_endpoints: bool,
) -> None:
    if transition is None or relax_endpoints:
        return
    expected_method = f"{MLIP_ENDPOINT_PREFIX}{model_id or 'unspecified'}"
    for endpoint in (transition.initial, transition.final):
        if endpoint.status != "generated" and endpoint.method != expected_method:
            raise ValueError(
                f"{endpoint.role} endpoint was relaxed with {endpoint.method!r}; "
                f"the active MLIP is {expected_method!r}. Set relax_endpoints=true "
                "to relax it with the active model."
            )


def _transition_for_images(
    transition: NebTransition | None,
    initial: Atoms,
    final: Atoms,
    *,
    model_id: str | None,
    endpoints_relaxed: bool,
) -> NebTransition | None:
    if transition is None or not endpoints_relaxed:
        return transition

    def relaxed_endpoint(endpoint, atoms):
        data = endpoint.model_dump(mode="python")
        data.update(
            geometry_fingerprint=geometry_fingerprint(atoms),
            cell=tuple(tuple(float(value) for value in row) for row in atoms.cell.array),
            pbc=tuple(bool(value) for value in atoms.pbc),
            status="relaxed",
            relaxation_converged=True,
            method=f"{MLIP_ENDPOINT_PREFIX}{model_id or 'unspecified'}",
            qualification_evidence=(),
        )
        return NebEndpoint.model_validate(data)

    updated_initial = relaxed_endpoint(transition.initial, initial)
    updated_final = relaxed_endpoint(transition.final, final)
    data = transition.model_dump(mode="python")
    data.update(
        initial=updated_initial,
        final=updated_final,
        correspondence=build_correspondence(updated_initial, updated_final),
    )
    return NebTransition.model_validate(data)


def sample_neb_candidates(
    config: NebSamplingConfig,
    calculator,
    out_dir: str | Path,
    *,
    iteration: int = 0,
    model_id: str | None = None,
    mlip_config: Any = None,
    uncertainty_calculators=(),
    uncertainty_mode: str | None = None,
    dropout_passes: int = 0,
    dropout_probability: float = 0.1,
    uncertainty_seed: int | None = None,
) -> NebSamplingOutcome:
    """Optimize one MLIP NEB pathway and return its interior images as candidates."""
    from ase.io.trajectory import Trajectory
    from ase.mep import NEB

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if uncertainty_mode is None and len(uncertainty_calculators) >= 2:
        uncertainty_mode = "ensemble"
    if uncertainty_mode == "ensemble" and len(uncertainty_calculators) < 2:
        raise ValueError("ensemble pathway uncertainty requires at least two calculators")
    if uncertainty_mode == "mc_dropout" and dropout_passes < 2:
        raise ValueError("MC-dropout pathway uncertainty requires at least two passes")
    if uncertainty_mode not in (None, "ensemble", "mc_dropout"):
        raise ValueError(f"unknown pathway uncertainty mode {uncertainty_mode!r}")
    if config.max_pathway_uncertainty_eV is not None and uncertainty_mode is None:
        raise ValueError("max_pathway_uncertainty_eV requires configured pathway uncertainty")
    initial = ase_read(config.initial_path)
    final = ase_read(config.final_path)
    correspond_atoms(initial, final)
    _validate_endpoint_settings_match(initial, final)
    if config.transition is not None:
        _validate_transition_endpoints(config.transition, initial, final)
        _validate_relaxed_endpoint_provenance(
            config.transition,
            model_id,
            relax_endpoints=config.relax_endpoints,
        )
    if config.relax_endpoints:
        initial_settings = _endpoint_settings(initial)
        final_settings = _endpoint_settings(final)
        if mlip_config is None:
            raise ValueError("endpoint relaxation requires the configured MLIP block")
        initial, final = _relax_endpoints(
            config.initial_path, config.final_path, out_dir, mlip_config=mlip_config, config=config
        )
        correspond_atoms(initial, final)
        _validate_endpoint_settings_preserved(initial_settings, initial, "initial")
        _validate_endpoint_settings_preserved(final_settings, final, "final")
        _validate_endpoint_settings_match(initial, final)
        _validate_moving_atoms_survive_relaxation(config.transition, initial, final)
    transition = _transition_for_images(
        config.transition,
        initial,
        final,
        model_id=model_id,
        endpoints_relaxed=config.relax_endpoints,
    )

    generated = generate_linear_images(
        initial,
        final,
        config.n_images,
        min_distance=config.min_distance_A,
        max_step=config.max_step_A,
        check_intermediate_overlaps=config.interpolation != "idpp",
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
    if config.interpolation == "idpp":
        validate_images(images, config.min_distance_A)

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
    latest_interim_uncertainty: PathwayUncertainty | None = None

    def pathway_uncertainty() -> PathwayUncertainty | None:
        if uncertainty_mode == "ensemble":
            return score_ensemble_pathway(images, uncertainty_calculators)
        if uncertainty_mode == "mc_dropout":
            if uncertainty_seed is None:
                raise ValueError("mc_dropout pathway uncertainty requires uncertainty_seed")
            return score_mc_dropout_pathway(
                images,
                calculator,
                dropout_passes,
                dropout_probability,
                seed=uncertainty_seed,
            )
        return None

    def assess_pathway():
        nonlocal uncertainty_evaluations, latest_interim_uncertainty
        latest_interim_uncertainty = pathway_uncertainty()
        uncertainty_evaluations += 1
        if (
            latest_interim_uncertainty is not None
            and config.max_pathway_uncertainty_eV is not None
            and latest_interim_uncertainty.pathway_eV > config.max_pathway_uncertainty_eV
        ):
            raise _UncertaintyLimitReached(latest_interim_uncertainty.pathway_eV)

    if config.uncertainty_interval_steps and uncertainty_mode is not None:
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
    refinement_stage_used = False
    refinement_steps = 0
    if config.refinement_stage is not None and converged:
        refinement = config.refinement_stage
        refinement_checkpoint = out_dir / "neb-refinement-images.extxyz"
        refinement_state = out_dir / "neb-refinement-state.json"
        prior_refinement_steps = 0
        if config.resume and refinement_state.is_file():
            saved_state = json.loads(refinement_state.read_text(encoding="utf-8"))
            prior_refinement_steps = saved_state.get("optimization_steps")
            if not isinstance(prior_refinement_steps, int) or prior_refinement_steps < 0:
                raise ValueError(f"Invalid refinement step count in NEB state {refinement_state}")
        refinement_images = (
            _load_restart(refinement_checkpoint, config.n_images, images) if config.resume else None
        )
        images = refinement_images or images
        for image in images:
            image.calc = calculator
        refinement_neb = NEB(
            images,
            k=config.spring_constant_eV_per_A2,
            climb=refinement.climbing_image,
            allow_shared_calculator=True,
        )
        refinement_optimizer = _optimizer(refinement.optimizer)(
            refinement_neb,
            restart=str(out_dir / "neb-refinement-optimizer.json") if config.resume else None,
            logfile=str(out_dir / "neb-refinement.log"),
        )
        starting_refinement_steps = refinement_optimizer.get_number_of_steps()
        refinement_trajectory = Trajectory(
            out_dir / "neb-refinement.traj",
            mode="a" if config.resume and (out_dir / "neb-refinement.traj").exists() else "w",
        )

        def save_refinement_checkpoint():
            ase_write(refinement_checkpoint, images, format="extxyz")
            save_checkpoint()

        def record_refinement_trajectory():
            for image in images:
                refinement_trajectory.write(image)

        refinement_optimizer.attach(save_refinement_checkpoint, interval=1)
        refinement_optimizer.attach(record_refinement_trajectory, interval=config.record_every)
        if config.uncertainty_interval_steps and uncertainty_mode is not None:
            refinement_optimizer.attach(assess_pathway, interval=config.uncertainty_interval_steps)
        try:
            converged = bool(
                refinement_optimizer.run(
                    fmax=refinement.fmax_eV_per_A,
                    steps=refinement.max_steps,
                )
            )
        except _UncertaintyLimitReached as exc:
            converged = False
            uncertainty_stop = exc.value
        finally:
            refinement_trajectory.close()
        save_refinement_checkpoint()
        refinement_steps = prior_refinement_steps + max(
            0, refinement_optimizer.get_number_of_steps() - starting_refinement_steps
        )
        optimization_steps += refinement_steps
        refinement_state.write_text(
            json.dumps({"optimization_steps": refinement_steps}, indent=2) + "\n",
            encoding="utf-8",
        )
        optimizer = refinement_optimizer
        refinement_stage_used = True

    energies = np.asarray([float(image.get_potential_energy()) for image in images])
    if not np.isfinite(energies).all():
        raise ValueError("NEB calculator returned non-finite image energies")
    relative = energies - energies[0]
    saddle_index = int(np.argmax(energies[1:-1])) + 1
    transition_id = (
        transition.transition_id
        if transition is not None
        else config.transition_id or f"{config.initial_path.stem}-to-{config.final_path.stem}"
    )
    barrier_uncertainty = pathway_uncertainty()
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
        pathway_uncertainty_eV=(
            None if barrier_uncertainty is None else barrier_uncertainty.pathway_eV
        ),
        forward_barrier_uncertainty_eV=(
            None if barrier_uncertainty is None else barrier_uncertainty.forward_barrier_eV
        ),
        reverse_barrier_uncertainty_eV=(
            None if barrier_uncertainty is None else barrier_uncertainty.reverse_barrier_eV
        ),
        climbing_image_used=config.climbing_image or refinement_stage_used,
        provenance={
            "backend": "mlip",
            "model_id": model_id or "unspecified",
            "transition_id": transition_id,
            "initial_endpoint_method": (
                "unrecorded" if transition is None else transition.initial.method or "generated"
            ),
            "final_endpoint_method": (
                "unrecorded" if transition is None else transition.final.method or "generated"
            ),
            "initial_endpoint_fingerprint": geometry_fingerprint(images[0]),
            "final_endpoint_fingerprint": geometry_fingerprint(images[-1]),
            "endpoint_settings_validated": "true",
            "mechanism_after_relaxation": (
                "validated" if config.relax_endpoints and transition is not None else "not_assessed"
            ),
            "iteration": str(iteration),
            "optimizer": config.optimizer,
            "interpolation": config.interpolation,
            "refinement_stage_used": str(refinement_stage_used).lower(),
            "refinement_steps": str(refinement_steps),
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
                uncertainty=None if barrier_uncertainty is None else barrier_uncertainty.pathway_eV,
                provenance={
                    "transition_id": transition_id,
                    "neb_state": result.state,
                    "pathway_uncertainty_eV": result.pathway_uncertainty_eV,
                    "forward_barrier_uncertainty_eV": result.forward_barrier_uncertainty_eV,
                    "reverse_barrier_uncertainty_eV": result.reverse_barrier_uncertainty_eV,
                },
                metadata={
                    "image_index": index,
                    "reaction_coordinate": index / (len(images) - 1),
                    "saddle_point_candidate": index == saddle_index,
                    "optimization_steps": optimization_steps,
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
    for image in images:
        image.calc = None
    return NebSamplingOutcome(
        candidates=candidates,
        result=result,
        images=tuple(images),
        pathway_uncertainty_eV=(
            None if barrier_uncertainty is None else barrier_uncertainty.pathway_eV
        ),
        transition=transition,
        initial=initial.copy(),
        final=final.copy(),
    )


def _sample_neb_task(task: NebSamplingTask) -> NebSamplingWorkerResult:
    """Worker entry point for one independent pathway; calculators are worker-cached."""
    from matsim_agents.backends.mlip.config import parse_mlip_config
    from matsim_agents.backends.mlip.pool import prepare_worker_device
    from matsim_agents.workflows.active_learning.md_sampler import _worker_mlip_calculators

    prepare_worker_device()
    config = NebSamplingConfig.model_validate_json(task.config_json)
    mlip_config = parse_mlip_config(json.loads(task.mlip_config_json))
    calculator, ensemble = _worker_mlip_calculators(
        task.mlip_config_json,
        task.enable_mc_dropout,
        include_ensemble=task.uncertainty_mode == "ensemble",
    )
    outcome = sample_neb_candidates(
        config,
        calculator,
        task.out_dir,
        iteration=task.iteration,
        model_id=task.model_id,
        mlip_config=mlip_config,
        uncertainty_calculators=ensemble,
        uncertainty_mode=task.uncertainty_mode,
        dropout_passes=task.dropout_passes,
        dropout_probability=task.dropout_probability,
        uncertainty_seed=task.uncertainty_seed,
    )
    for image in outcome.images:
        image.calc = None
    return NebSamplingWorkerResult(
        candidates=outcome.candidates,
        result_json=outcome.result.model_dump_json(),
        images=outcome.images,
        pathway_uncertainty_eV=outcome.pathway_uncertainty_eV,
        transition_json=(
            None if outcome.transition is None else outcome.transition.model_dump_json()
        ),
        initial=outcome.initial,
        final=outcome.final,
    )


def sample_neb_candidates_parallel(tasks: list[NebSamplingTask], pool) -> list[NebSamplingOutcome]:
    """Run independent pathways on MLIP pool workers, preserving submission order."""
    results = pool.map(_sample_neb_task, tasks)
    if len(results) != len(tasks):
        raise RuntimeError(
            f"MLIP worker pool returned {len(results)} outcomes for {len(tasks)} NEB pathways"
        )
    outcomes = []
    for task, result in zip(tasks, results, strict=True):
        if result.error is not None:
            raise RuntimeError(f"NEB pathway {task.transition_label} failed: {result.error}") from (
                result.error
            )
        worker_result = result.value
        outcomes.append(
            NebSamplingOutcome(
                candidates=worker_result.candidates,
                result=NebResult.model_validate_json(worker_result.result_json),
                images=worker_result.images,
                pathway_uncertainty_eV=worker_result.pathway_uncertainty_eV,
                transition=(
                    None
                    if worker_result.transition_json is None
                    else NebTransition.model_validate_json(worker_result.transition_json)
                ),
                initial=worker_result.initial,
                final=worker_result.final,
            )
        )
    return outcomes


__all__ = [
    "NebSamplingOutcome",
    "NebSamplingTask",
    "NebSamplingWorkerResult",
    "sample_neb_candidates",
    "sample_neb_candidates_parallel",
]
