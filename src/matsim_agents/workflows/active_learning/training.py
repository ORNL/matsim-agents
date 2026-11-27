"""Training and validation data assembly and candidate-model retraining."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from matsim_agents.backends.mlip.config import MLIPConfig
from matsim_agents.core.energy_references import (
    load_elemental_reference_manifest,
    validate_dataset_reference_method,
)
from matsim_agents.core.seeding import numpy_rng, stream_seed
from matsim_agents.data.governance import (
    DatasetManifest,
    structure_identity,
    validate_labelled_frames,
    write_dataset_manifest,
)
from matsim_agents.execution.placement import Placement
from matsim_agents.workflows.active_learning.config import (
    ALConfig,
)
from matsim_agents.workflows.active_learning.evaluate import (
    _reference_energy,
    _reference_forces,
)
from matsim_agents.workflows.active_learning.iteration_state import IterationState
from matsim_agents.workflows.active_learning.labelling import (
    _dft_method_signature,
    _record_provenance,
)
from matsim_agents.workflows.active_learning.trainer import (
    append_frames_to_extxyz,
    dft_results_to_frames,
    retrain,
)

log = logging.getLogger(__name__)


def _split_training_validation_frames(
    frames: list[Any],
    *,
    validation_fraction: float,
    seed: int,
    existing_training_count: int = 0,
    existing_validation_count: int = 0,
    existing_training_trajectories: frozenset[str] | set[str] = frozenset(),
    existing_validation_trajectories: frozenset[str] | set[str] = frozenset(),
) -> tuple[list[Any], list[Any]]:
    """Assign only new frames to meet the cumulative held-out target.

    Frames of one MD trajectory are correlated, so a trajectory is held out whole
    or not at all. New frames of a trajectory already in a persisted partition
    (e.g. on a retried iteration) join that partition. Any other trajectory is
    held out only if it fits the remaining target, so a long trajectory waits
    for the cumulative target to grow.
    """
    if shared := existing_training_trajectories & existing_validation_trajectories:
        raise ValueError(
            f"persisted training and validation datasets share trajectories {sorted(shared)}; "
            "repartition them by trajectory_id before resuming"
        )
    if validation_fraction <= 0.0:
        return frames, []
    total = existing_training_count + existing_validation_count + len(frames)
    target = max(1, int(round(total * validation_fraction)))
    groups: dict[Any, list[int]] = {}
    for index, frame in enumerate(frames):
        info = getattr(getattr(frame, "atoms", None), "info", {})
        groups.setdefault(info.get("trajectory_id", ("frame", index)), []).append(index)
    validation_indices = {
        index
        for key, indices in groups.items()
        if key in existing_validation_trajectories
        for index in indices
    }
    pinned = existing_training_trajectories | existing_validation_trajectories
    members = [indices for key, indices in groups.items() if key not in pinned]
    n_validation = max(len(validation_indices), target - existing_validation_count)
    for group in numpy_rng(seed).permutation(len(members)):
        if len(validation_indices) + len(members[group]) <= n_validation:
            validation_indices.update(members[group])
    training = [frame for index, frame in enumerate(frames) if index not in validation_indices]
    validation = [frame for index, frame in enumerate(frames) if index in validation_indices]
    return training, validation


def _trajectory_ids(frames: list[Any]) -> set[str]:
    return {str(f.info["trajectory_id"]) for f in frames if "trajectory_id" in f.info}


def _append_labelled_frames(
    cfg: ALConfig,
    results: list,
    selected: list,
    i: int,
    dataset_path: Path,
    validation_dataset_path: Path,
    backend: Any,
    state: IterationState,
) -> tuple[set[str], str, int]:
    """Validate new labels against the persisted datasets, split, append, and write manifests."""
    t0 = time.time()
    frames = dft_results_to_frames(results, iteration=i)
    _record_provenance(cfg, frames, selected, i)
    existing_training_frames = []
    existing_validation_frames = []
    manifest_path = dataset_path.with_suffix(dataset_path.suffix + ".manifest.json")
    parent_dataset_id = None
    if dataset_path.exists():
        from ase.io import read as ase_read

        existing_training_frames = list(ase_read(dataset_path, index=":"))
    if validation_dataset_path.exists():
        from ase.io import read as ase_read

        existing_validation_frames = list(ase_read(validation_dataset_path, index=":"))
    existing_frames = existing_training_frames + existing_validation_frames
    if {structure_identity(frame) for frame in existing_training_frames} & {
        structure_identity(frame) for frame in existing_validation_frames
    }:
        raise ValueError("persisted training and validation datasets overlap")
    existing_elements = {
        symbol for frame in existing_frames for symbol in frame.get_chemical_symbols()
    }
    if existing_training_frames and not manifest_path.exists():
        raise ValueError(
            "cannot append to a non-empty dataset without a manifest containing "
            "its DFT method signature; migrate the dataset explicitly first"
        )
    if manifest_path.exists():
        previous_manifest = DatasetManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
        if previous_manifest.dft_backend != backend.name:
            raise ValueError("cannot append labels from a different DFT backend")
        if existing_frames and previous_manifest.method_signature is None:
            raise ValueError(
                "cannot append to a non-empty dataset without a DFT method signature; "
                "migrate the dataset explicitly first"
            )
        existing_signature = _dft_method_signature(cfg, existing_elements)
        if (
            previous_manifest.method_signature is not None
            and previous_manifest.method_signature != existing_signature
        ):
            raise ValueError("cannot append labels with a different DFT method signature")
        validate_dataset_reference_method(
            dataset_path,
            reference_backend=backend.name,
            reference_method_signature=existing_signature,
            require_sidecar=True,
        )
        parent_dataset_id = previous_manifest.dataset_id
    validation_parent_dataset_id = None
    validation_manifest_path = validation_dataset_path.with_suffix(
        validation_dataset_path.suffix + ".manifest.json"
    )
    if validation_dataset_path.exists():
        validate_dataset_reference_method(
            validation_dataset_path,
            reference_backend=backend.name,
            reference_method_signature=_dft_method_signature(cfg, existing_elements),
            require_sidecar=True,
        )
        validation_parent_dataset_id = DatasetManifest.model_validate_json(
            validation_manifest_path.read_text(encoding="utf-8")
        ).dataset_id
    expected_atomic_numbers = {
        int(number) for candidate in selected for number in candidate.atoms.get_atomic_numbers()
    }
    # Frames of an external validation set never enter training.
    external_validation_frames = []
    if cfg.trainer.validation_set is not None and cfg.trainer.validation_fraction <= 0:
        from ase.io import read as ase_read

        external_validation_frames = list(ase_read(cfg.trainer.validation_set, index=":"))
    frames, validation = validate_labelled_frames(
        frames,
        existing_frames=existing_frames + external_validation_frames,
        expected_atomic_numbers=expected_atomic_numbers,
    )
    training_frames, validation_frames = _split_training_validation_frames(
        frames,
        validation_fraction=cfg.trainer.validation_fraction,
        seed=stream_seed(cfg.random_seed, "validation-split", i),
        existing_training_count=len(existing_training_frames),
        existing_validation_count=len(existing_validation_frames),
        existing_training_trajectories=_trajectory_ids(existing_training_frames),
        existing_validation_trajectories=_trajectory_ids(existing_validation_frames),
    )
    labelled_elements = existing_elements | {
        symbol for frame in frames for symbol in frame.atoms.get_chemical_symbols()
    }
    method_signature = _dft_method_signature(cfg, labelled_elements)
    n_appended = append_frames_to_extxyz(training_frames, dataset_path)
    n_validation_appended = append_frames_to_extxyz(validation_frames, validation_dataset_path)
    state.dataset_path = str(dataset_path)
    state.n_training_frames = n_appended
    state.n_validation_frames = n_validation_appended
    state.n_training_frames_total = len(existing_training_frames) + n_appended
    state.n_validation_frames_total = len(existing_validation_frames) + n_validation_appended
    if validation_dataset_path.exists():
        state.validation_dataset_path = str(validation_dataset_path)
    if dataset_path.exists():
        write_dataset_manifest(
            dataset_path,
            dft_backend=backend.name,
            energy_reference=f"{backend.name}:native_total_energy",
            validation=validation.model_copy(update={"accepted": len(training_frames)}),
            parent_dataset_id=parent_dataset_id,
            method_signature=method_signature,
            method_signature_elements=labelled_elements,
            base_method_signature=_dft_method_signature(cfg, set()),
            element_method_signatures={
                element: _dft_method_signature(cfg, {element})
                for element in sorted(labelled_elements)
            },
        )
    if validation_dataset_path.exists():
        write_dataset_manifest(
            validation_dataset_path,
            dft_backend=backend.name,
            energy_reference=f"{backend.name}:native_total_energy",
            validation=validation.model_copy(update={"accepted": len(validation_frames)}),
            parent_dataset_id=validation_parent_dataset_id,
            method_signature=method_signature,
            method_signature_elements=labelled_elements,
            base_method_signature=_dft_method_signature(cfg, set()),
            element_method_signatures={
                element: _dft_method_signature(cfg, {element})
                for element in sorted(labelled_elements)
            },
            split_role="validation",
        )
    state.timings_sec["append_dataset"] = time.time() - t0
    log.info(
        "Iter %d: appended %d training and %d validation frames",
        i,
        n_appended,
        n_validation_appended,
    )
    return labelled_elements, method_signature, n_appended


def _check_reference_method(
    cfg: ALConfig, labelled_elements: set[str], backend: Any, method_signature: str
) -> None:
    """Require elemental references computed with the same DFT method as the labels."""
    if cfg.trainer.validation_reference_set is None:
        raise ValueError("model comparison requires a DFT-labelled elemental reference manifest")
    reference_manifest, _, _ = load_elemental_reference_manifest(
        cfg.trainer.validation_reference_set,
        required_elements=labelled_elements,
    )
    reference_element_signatures = reference_manifest.get("element_method_signatures")
    scoped_method_matches = (
        reference_manifest.get("base_method_signature") == _dft_method_signature(cfg, set())
        and reference_element_signatures is not None
        and all(
            reference_element_signatures.get(element) == _dft_method_signature(cfg, {element})
            for element in labelled_elements
        )
    )
    if reference_manifest["backend"] != backend.name or (
        not scoped_method_matches and reference_manifest["method_signature"] != method_signature
    ):
        raise ValueError("elemental references and campaign labels use different DFT methods")


def _collect_training_deferrals(
    cfg: ALConfig,
    state: IterationState,
    validation_dataset_path: Path,
    comparison_enabled: bool,
) -> None:
    """Record why training/comparison must wait for more cumulative data."""
    if state.n_training_frames_total < 2:
        state.training_deferred_reasons.append(
            f"cumulative training set has {state.n_training_frames_total} frames; "
            "at least 2 required"
        )
    if comparison_enabled:
        from ase.io import read as ase_read

        if cfg.trainer.validation_fraction > 0:
            held_out_frames = (
                list(ase_read(validation_dataset_path, index=":"))
                if validation_dataset_path.exists()
                else []
            )
        elif cfg.trainer.validation_set is not None:
            held_out_frames = list(ase_read(cfg.trainer.validation_set, index=":"))
        else:
            raise ValueError("model comparison requires held-out validation data")
        for label, reader in (
            ("energy", _reference_energy),
            ("force", _reference_forces),
        ):
            count = sum(reader(frame) is not None for frame in held_out_frames)
            minimum = cfg.trainer.promotion_min_evaluated_frames
            if count < minimum:
                state.training_deferred_reasons.append(
                    f"cumulative validation set has {count} {label}-labelled frames; "
                    f"at least {minimum} required"
                )


def _retrain_candidate(
    cfg: ALConfig,
    dataset_path: Path,
    i: int,
    it_dir: Path,
    root: Path,
    state: IterationState,
    n_appended: int,
    placement: Placement,
    base_mlip: MLIPConfig,
) -> None:
    """Fine-tune a candidate model for the configured backend, if training is enabled.

    ``base_mlip`` is the model the fine-tune starts from (``trainer.start_from``).
    """
    if not cfg.trainer.enabled:
        log.info(
            "Skipping retraining for backend=%s; %d labelled frames accumulated in %s.",
            cfg.mlip.backend,
            n_appended,
            dataset_path,
        )
        return
    common = {
        "dataset_path": dataset_path,
        "iteration": i,
        "seed": stream_seed(cfg.random_seed, "training", i),
        "launch": placement.pool("training").launches[0],
        "dft_config": cfg.dft,
        "dft_launch": placement.pool("dft").launches[0],
        "reference_root": root / "elemental-training-reference",
    }
    new_model = retrain(base_mlip, cfg.trainer, out_dir=it_dir / "model", **common)
    state.candidate_model_path = str(new_model)
