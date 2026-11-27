"""Top-level active-learning driver.

State persistence
-----------------
Each iteration writes a JSON ``state.json`` under
``loop.out_dir/iteration_{i:04d}/`` containing:

* number of candidates produced & selected
* per-candidate uncertainty stats
* number of converged VASP jobs
* path to the appended dataset
* timing breakdown

Resume logic: on restart, the driver scans ``loop.out_dir`` for the highest
existing ``iteration_*/state.json`` with ``status == "complete"`` and starts
the next iteration from there. Partial iterations are wiped (so we never
double-count VASP results in the dataset), but each converged DFT label is
also saved to ``dft_label_cache/iteration_{i:04d}.jsonl`` as it finishes; the
rerun of that iteration reuses a saved label instead of repeating the
calculation when the candidate ID, geometry and DFT method all match.

Usage (Python)
--------------
    from matsim_agents.workflows.active_learning import ALConfig
    from matsim_agents.workflows.active_learning.loop import run_active_learning

    cfg = ALConfig.from_yaml("al.yaml")
    run_active_learning(cfg)
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from matsim_agents.backends.dft.base import DFTJobSpec, make_backend
from matsim_agents.backends.dft.runner import run_dft_batch
from matsim_agents.backends.mlip.calculator import build_ensemble, make_mlip_calculator
from matsim_agents.backends.mlip.config import MLIPConfig
from matsim_agents.backends.mlip.pool import (
    MLIPWorkerPool,
    driver_on_mlip_device,
    release_device_memory,
)
from matsim_agents.core.energy_references import (
    load_elemental_reference_manifest,
    validate_dataset_reference_method,
)
from matsim_agents.core.fileio import append_line
from matsim_agents.core.seeding import numpy_rng, stream_seed
from matsim_agents.data.governance import (
    DatasetManifest,
    structure_identity,
    validate_labelled_frames,
    write_dataset_manifest,
)
from matsim_agents.execution.placement import Placement
from matsim_agents.science.neb.contracts import NebResult, NebTransition
from matsim_agents.workflows.active_learning.campaign_init import (
    check_reference_coverage,
    elements_of,
)
from matsim_agents.workflows.active_learning.candidates import Candidate
from matsim_agents.workflows.active_learning.config import (
    ALConfig,
    MDSamplingStrategy,
    NebEscalationConfig,
    NebSamplingConfig,
    NEBSamplingStrategy,
    SamplingStrategy,
)
from matsim_agents.workflows.active_learning.dft_protocol import (
    dft_method_signature,
    scientific_dft_payload,
)
from matsim_agents.workflows.active_learning.eval_report import METRIC_DEFINITIONS
from matsim_agents.workflows.active_learning.evaluate import (
    _apply_model_override,
    _reference_energy,
    _reference_forces,
    evaluate_frames,
    evaluate_promotion_candidate,
)
from matsim_agents.workflows.active_learning.md_sampler import sample_md_candidates
from matsim_agents.workflows.active_learning.seed_sources import resolve_seed_structures
from matsim_agents.workflows.active_learning.trainer import (
    append_frames_to_extxyz,
    dft_results_to_frames,
    retrain,
)
from matsim_agents.workflows.active_learning.uncertainty import select_candidates

log = logging.getLogger(__name__)

CandidateSampler = Callable[[ALConfig, int, Path], list[Candidate]]


def _scientific_dft_payload(cfg: ALConfig, elements: set[str]) -> dict[str, Any]:
    return scientific_dft_payload(cfg.dft, elements)


def _dft_method_signature(cfg: ALConfig, elements: set[str]) -> str:
    return dft_method_signature(cfg.dft, elements)


# --------------------------------------------------------------------------- #
# Per-iteration state                                                         #
# --------------------------------------------------------------------------- #


@dataclass
class IterationState:
    iteration: int
    status: str = "running"  # running | complete | failed
    n_candidates: int = 0
    n_selected: int = 0
    n_dft_converged: int = 0
    n_dft_failed: int = 0
    n_dft_reused: int = 0
    n_dft_neb_runs: int = 0
    neb_pathways: list[dict[str, Any]] = field(default_factory=list)
    dft_backend: str | None = None
    score_min: float | None = None
    score_max: float | None = None
    score_mean: float | None = None
    selected_candidate_ids: list[str] = field(default_factory=list)
    candidate_uncertainty: dict[str, float] = field(default_factory=dict)
    dft_force_errors_eV_per_A: dict[str, float] = field(default_factory=dict)
    accuracy_triggered: bool | None = None
    dataset_path: str | None = None
    validation_dataset_path: str | None = None
    n_training_frames: int = 0
    n_validation_frames: int = 0
    n_training_frames_total: int = 0
    n_validation_frames_total: int = 0
    training_status: str = "disabled"
    training_deferred_reasons: list[str] = field(default_factory=list)
    candidate_model_path: str | None = None
    model_comparison: dict[str, Any] | None = None
    model_promoted: bool = False
    promotion_validation: dict[str, Any] | None = None
    lammps_export: dict[str, Any] | None = None
    new_logdir: str | None = None
    timings_sec: dict[str, float] = field(default_factory=dict)
    notes: str | None = None
    # Scores of the model in use on loop.benchmark_sets, after this iteration's decisions.
    benchmark: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


IterationAssessor = Callable[[ALConfig, int, Path, IterationState], None]


# --------------------------------------------------------------------------- #
# Resume                                                                      #
# --------------------------------------------------------------------------- #


def _iter_dir(root: Path, i: int) -> Path:
    return root / f"iteration_{i:04d}"


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


_ROOT_SEED_FILE = "random-seed.json"


def _prior_run_artifacts(root: Path) -> list[str]:
    """Evidence a previous run left in *root*: iterations, datasets, and seed structures."""
    names = ["dataset.extxyz", "dataset.db", "validation.extxyz", "seeds", "dft_label_cache"]
    found = [name for name in names if (root / name).exists()]
    return found + sorted(path.name for path in root.glob("iteration_*"))


def _bind_root_seed(cfg: ALConfig, root: Path) -> None:
    """Record the run's root seed; existing evidence must already carry the same one.

    Checked before seed structures are resolved or written. A resume must match the
    recorded seed. Artifacts without a recorded seed (from before seed recording) and,
    for a fresh run, any prior artifacts are rejected, so retained datasets are never
    reused under a seed that did not produce them.
    """
    path = root / _ROOT_SEED_FILE
    artifacts = _prior_run_artifacts(root)
    if path.exists():
        recorded = int(json.loads(path.read_text())["random_seed"])
        if not cfg.loop.resume:
            raise ValueError(
                f"{root} already holds a run (random_seed {recorded}); set loop.resume to "
                "continue it or choose a new out_dir"
            )
        if recorded != cfg.random_seed:
            raise ValueError(
                f"{root} was started with random_seed {recorded}; resuming it with "
                f"{cfg.random_seed} would mix two seed streams in one run"
            )
        return
    if artifacts:
        raise ValueError(
            f"{root} holds artifacts {artifacts} without a recorded random_seed; they "
            "cannot be attributed to a seed; choose a new out_dir"
        )
    path.write_text(json.dumps({"random_seed": cfg.random_seed}) + "\n")


def _scan_resume(root: Path) -> tuple[int, Path | None]:
    """Return (start_iteration, current_logdir_or_None) based on existing state.

    The current_logdir is taken from the most recent completed iteration's
    ``state.json::new_logdir`` if present.
    """
    if not root.exists():
        return 0, None
    completed: list[tuple[int, Path | None]] = []
    for d in sorted(root.glob("iteration_*")):
        sf = d / "state.json"
        if not sf.exists():
            # Partial — nuke it so we don't double-count.
            log.warning("Removing incomplete iteration dir %s", d)
            shutil.rmtree(d, ignore_errors=True)
            continue
        try:
            data = json.loads(sf.read_text())
            if data.get("status") == "complete":
                new_logdir = data.get("new_logdir")
                completed.append((int(data["iteration"]), Path(new_logdir) if new_logdir else None))
            else:
                log.warning("Removing failed/partial iteration dir %s", d)
                shutil.rmtree(d, ignore_errors=True)
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not parse %s (%s); removing", sf, exc)
            shutil.rmtree(d, ignore_errors=True)
    if not completed:
        return 0, None
    last_i = max(iteration for iteration, _ in completed)
    promoted = [entry for entry in completed if entry[1] is not None]
    last_logdir = max(promoted, key=lambda entry: entry[0])[1] if promoted else None
    return last_i + 1, last_logdir


# --------------------------------------------------------------------------- #
# Main driver                                                                 #
# --------------------------------------------------------------------------- #


def _validate_comparison_inputs(cfg: ALConfig) -> bool:
    """Return whether candidate comparison is enabled, validating its inputs up front."""
    comparison_enabled = cfg.trainer.enabled and (
        cfg.trainer.compare_after_training or cfg.trainer.promote_model
    )
    if comparison_enabled:
        if cfg.trainer.validation_reference_set is None:
            raise ValueError(
                "model comparison requires a DFT-labelled elemental reference manifest"
            )
        reference_manifest, _, _ = load_elemental_reference_manifest(
            cfg.trainer.validation_reference_set, required_elements=set()
        )
        if cfg.trainer.validation_set is not None:
            validate_dataset_reference_method(
                cfg.trainer.validation_set,
                reference_backend=reference_manifest["backend"],
                reference_method_signature=reference_manifest["method_signature"],
                reference_base_method_signature=reference_manifest.get("base_method_signature"),
                reference_element_method_signatures=reference_manifest.get(
                    "element_method_signatures"
                ),
                require_sidecar=True,
            )
    return comparison_enabled


def _completed_dft_count(cfg: ALConfig, root: Path) -> int:
    """DFT calculations already spent by completed iterations when resuming."""
    completed_dft = 0
    if cfg.loop.resume:
        for state_path in root.glob("iteration_*/state.json"):
            state_data = json.loads(state_path.read_text(encoding="utf-8"))
            if state_data.get("status") == "complete":
                completed_dft += int(state_data.get("n_dft_converged", 0)) + int(
                    state_data.get("n_dft_failed", 0)
                )
    return completed_dft


def _select_for_labelling(
    cfg: ALConfig,
    candidates: list,
    primary_calc: Any,
    ensemble_calcs: list,
    i: int,
    completed_dft: int,
    state: IterationState,
) -> list:
    """Score sampling candidates and pick those to label within the remaining DFT budget."""
    t0 = time.time()
    selected, scores = select_candidates(
        candidates,
        cfg.acquisition,
        primary_calculator=primary_calc,
        ensemble_calculators=ensemble_calcs or None,
        seed=stream_seed(cfg.random_seed, "acquisition", i),
    )
    if cfg.loop.max_dft_calculations is not None:
        remaining_dft = cfg.loop.max_dft_calculations - completed_dft
        selected = selected[:remaining_dft]
    state.n_selected = len(selected)
    state.selected_candidate_ids = [candidate.candidate_id for candidate in selected]
    selected_ids = set(state.selected_candidate_ids)
    if cfg.acquisition.strategy != "random":
        for candidate, score in zip(candidates, scores, strict=True):
            candidate.uncertainty = float(score) if np.isfinite(score) else None
    state.candidate_uncertainty = {
        candidate.candidate_id: candidate.uncertainty
        for candidate in candidates
        if candidate.candidate_id in selected_ids and candidate.uncertainty is not None
    }
    finite_scores = scores[np.isfinite(scores)]
    if finite_scores.size:
        state.score_min = float(np.min(finite_scores))
        state.score_max = float(np.max(finite_scores))
        state.score_mean = float(np.mean(finite_scores))
    state.timings_sec["acquisition"] = time.time() - t0
    log.info(
        "Iter %d: selected %d/%d candidates (score min/mean/max = %s/%s/%s)",
        i,
        len(selected),
        len(candidates),
        state.score_min,
        state.score_mean,
        state.score_max,
    )
    return selected


def _label_with_dft(
    cfg: ALConfig,
    selected: list,
    it_dir: Path,
    i: int,
    state: IterationState,
    placement: Placement,
) -> tuple[list, Any]:
    """Run single-point DFT for the selected candidates and record outcome counts."""
    t0 = time.time()
    backend = make_backend(cfg.dft)
    state.dft_backend = backend.name
    dft_dir = it_dir / "dft"
    specs = [
        DFTJobSpec(
            job_id=cand.candidate_id,
            atoms=cand.atoms,
            work_dir=str(dft_dir / cand.candidate_id),
        )
        for cand in selected
    ]
    ledger = it_dir.parent / _DFT_LABEL_CACHE / f"{it_dir.name}.jsonl"
    cached = _load_dft_labels(ledger)
    reused: dict[str, Any] = {}
    pending = []
    for spec in specs:
        result = cached.get(_dft_label_key(cfg, spec))
        if result is None or result.backend != backend.name:
            pending.append(spec)
            continue
        result.work_dir = spec.work_dir
        reused[spec.job_id] = result
    if reused:
        log.info("Iter %d: reusing %d saved DFT labels from %s", i, len(reused), ledger)
    state.n_dft_reused = len(reused)

    by_job = dict(reused)

    def save(spec: DFTJobSpec, result: Any) -> None:
        by_job[spec.job_id] = result
        if result.converged:
            _save_dft_label(ledger, _dft_label_key(cfg, spec), result)

    run_dft_batch(pending, backend, placement.pool("dft"), on_complete=save)
    results = [by_job[spec.job_id] for spec in specs]
    n_ok = sum(1 for r in results if r.converged)
    state.n_dft_converged = n_ok
    state.n_dft_failed = len(results) - n_ok
    state.timings_sec["dft"] = time.time() - t0
    log.info(
        "Iter %d: %s converged=%d failed=%d",
        i,
        backend.name.upper(),
        n_ok,
        len(results) - n_ok,
    )

    if cfg.loop.fail_fast and state.n_dft_failed:
        raise RuntimeError(f"{state.n_dft_failed} {backend.name} jobs failed and fail_fast=True")
    return results, backend


_DFT_LABEL_CACHE = "dft_label_cache"


def _dft_label_key(cfg: ALConfig, spec: DFTJobSpec) -> str:
    elements = set(spec.atoms.get_chemical_symbols())
    return "|".join(
        (spec.job_id, structure_identity(spec.atoms), _dft_method_signature(cfg, elements))
    )


def _load_dft_labels(ledger: Path) -> dict[str, Any]:
    from ase.io.jsonio import decode

    from matsim_agents.backends.dft.base import DFTResult

    labels: dict[str, Any] = {}
    if not ledger.is_file():
        return labels
    for line in ledger.read_text(encoding="utf-8").splitlines():
        try:
            entry = decode(line)
            labels[entry["key"]] = DFTResult(**entry["result"])
        except Exception as exc:  # noqa: BLE001
            # A write cut short by the interruption; that job simply reruns.
            log.warning("Ignoring unreadable DFT label in %s: %s", ledger, exc)
    return labels


def _save_dft_label(ledger: Path, key: str, result: Any) -> None:
    from ase.io.jsonio import encode

    ledger.parent.mkdir(parents=True, exist_ok=True)
    line = encode({"key": key, "result": asdict(result)})
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write(line.replace("\n", " ") + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _record_provenance(cfg: ALConfig, frames: list, selected: list, i: int) -> None:
    """Record where each label came from, for grouped and chronological dataset splits.

    Method-independent identity and sampling provenance are shared. MD steps remain
    MD-only; NEB image indices are recorded as optimization coordinates, not time.
    """
    candidates = {candidate.candidate_id: candidate for candidate in selected}
    for frame in frames:
        candidate = candidates.get(Path(frame.source_work_dir).name)
        info = frame.atoms.info
        if candidate is not None:
            frame_info = dict(info)
            info.update(candidate.atoms.info)
            info.update(frame_info)
            method = candidate.atoms.info.get("sampling_method", candidate.sampling_method)
            info["sampling_method"] = method
            info["candidate_id"] = candidate.candidate_id
            if candidate.parent_id is not None:
                info["parent_structure_id"] = candidate.parent_id
            else:
                info.pop("parent_structure_id", None)
            if candidate.sampling_run_id is not None:
                info["sampling_run_id"] = candidate.sampling_run_id
            else:
                info.pop("sampling_run_id", None)
            if candidate.model_id is not None:
                info["mlip_model_id"] = candidate.model_id
            else:
                info.pop("mlip_model_id", None)
            if candidate.geometry_fingerprint is not None:
                info["geometry_fingerprint"] = candidate.geometry_fingerprint
            else:
                info.pop("geometry_fingerprint", None)
            if candidate.uncertainty is not None:
                info["configuration_uncertainty"] = candidate.uncertainty
            else:
                info.pop("configuration_uncertainty", None)
            if method == "md":
                temperature = candidate.metadata.get("temperature_K")
                md_config = cfg.md
                if md_config is None:
                    md_strategies = [
                        strategy.parameters
                        for strategy in _sampling_strategies(cfg)
                        if isinstance(strategy, MDSamplingStrategy)
                    ]
                    strategy_index = candidate.provenance.get("sampling_strategy_index", 0)
                    if isinstance(strategy_index, int) and strategy_index < len(md_strategies):
                        md_config = md_strategies[strategy_index]
                if temperature is None and md_config is not None:
                    temperature = md_config.temperature_K
                if temperature is not None:
                    info["temperature_K"] = temperature
                thermostat = candidate.metadata.get(
                    "thermostat", None if md_config is None else md_config.thermostat
                )
                if thermostat is not None:
                    info["thermostat"] = thermostat
                info["md_step"] = candidate.md_step
            elif method == "neb":
                info["transition_id"] = candidate.provenance.get("transition_id", "")
                info["neb_state"] = candidate.provenance.get("neb_state", "")
                for key, value in candidate.provenance.items():
                    if value is not None:
                        info[key] = value
                info.update(candidate.metadata)
                # Set last so no candidate field can relabel a fixed-geometry image as a DFT MEP.
                info["label_kind"] = "dft_single_point"
                info["label_is_dft_mep"] = False
            for key in ("e3_action_id", "e3_geometry_stage", "incumbent_checkpoint_sha256"):
                if key in candidate.atoms.info:
                    info[key] = candidate.atoms.info[key]
        info["source_campaign"] = str(cfg.loop.out_dir)
        if candidate is None:
            continue
        info["config_id"] = f"iteration-{i:04d}/{candidate.candidate_id}"
        method = candidate.atoms.info.get("sampling_method", candidate.sampling_method)
        if method == "md":
            seed = Path(candidate.seed_path).stem
            run_id = candidate.sampling_run_id or seed
            info["trajectory_id"] = f"iteration-{i:04d}/{run_id}"
            info["time_index"] = int(candidate.md_step or 0)
            info["initial_structure_id"] = seed
        elif method == "neb":
            run_id = candidate.sampling_run_id or candidate.parent_id or candidate.candidate_id
            info["trajectory_id"] = f"iteration-{i:04d}/{run_id}"
            info["initial_structure_id"] = candidate.parent_id or run_id
        else:
            seed = (
                candidate.candidate_id
                if method == "e3_geometry_optimization"
                else Path(candidate.seed_path).stem
            )
            info["trajectory_id"] = f"iteration-{i:04d}/{seed}"
            info["time_index"] = int(candidate.md_step or 0)
            info["initial_structure_id"] = seed


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


def _compare_and_promote(
    cfg: ALConfig,
    state: IterationState,
    i: int,
    dataset_path: Path,
    validation_dataset_path: Path,
    comparison_enabled: bool,
    incumbent_calculator: Any = None,
) -> None:
    """Compare the candidate against the incumbent and promote it when approved.

    ``incumbent_calculator`` is the incumbent model when it is still loaded.
    """
    compare_candidate = comparison_enabled and state.candidate_model_path is not None
    if compare_candidate:
        try:
            evaluation_cfg = cfg.model_copy(deep=True)
            if cfg.trainer.validation_fraction > 0:
                evaluation_cfg.trainer.validation_set = validation_dataset_path
                evaluation_cfg.trainer.validation_fraction = 0.0
            decision = evaluate_promotion_candidate(
                evaluation_cfg,
                state.candidate_model_path,
                iteration=i,
                training_set=dataset_path,
                incumbent_calculator=incumbent_calculator,
            )
            state.model_comparison = asdict(decision)
        except Exception as exc:  # noqa: BLE001
            state.model_comparison = {
                "approved": False,
                "reasons": [f"model comparison failed: {exc}"],
                "candidate_metrics": {},
                "incumbent_metrics": {},
            }
            log.exception("Iteration %d candidate model comparison failed", i)
        if cfg.trainer.promote_model:
            state.promotion_validation = state.model_comparison
    if (
        cfg.trainer.promote_model
        and state.promotion_validation is not None
        and state.promotion_validation["approved"]
    ):
        state.new_logdir = state.candidate_model_path
        state.model_promoted = True
        _apply_model_override(cfg, state.candidate_model_path)
        _export_for_lammps(cfg, state)


def _export_for_lammps(cfg: ALConfig, state: IterationState) -> None:
    """Opt-in: validate the promoted model under the LAMMPS ML-IAP bridge and record the result."""
    species = list(cfg.trainer.lammps_export_species)
    if not species or state.candidate_model_path is None:
        return
    try:
        from matsim_agents.backends.external.lammps.mliap.export import (
            export_checkpoint_for_lammps,
        )

        state.lammps_export = export_checkpoint_for_lammps(
            cfg.mlip,
            state.candidate_model_path,
            Path(state.candidate_model_path)
            if Path(state.candidate_model_path).is_dir()
            else Path(state.candidate_model_path).parent,
            species=species,
            seed=cfg.random_seed,
        )
    except Exception as exc:  # noqa: BLE001 - recorded, never blocks the promotion
        log.exception("LAMMPS export validation failed for %s", state.candidate_model_path)
        state.lammps_export = {"approved": False, "error": str(exc)}


class _SurrogateModels:
    """The surrogate's calculators and MD worker pool across AL iterations.

    When no other pool uses the mlip devices (``keep``), the calculators and the
    MD workers stay loaded between iterations and are rebuilt only when the
    model changes (a promotion). Otherwise they are built each iteration and
    released before DFT and training, which use the same devices.
    """

    def __init__(self, placement: Placement, *, keep: bool) -> None:
        self.placement = placement
        self.keep = keep
        self._key: tuple[str, bool] | None = None
        self._primary: Any = None
        self._ensemble: list = []
        self._pool: MLIPWorkerPool | None = None

    def calculators(self, mlip_cfg: MLIPConfig, enable_drop: bool) -> tuple[Any, list]:
        key = (mlip_cfg.model_dump_json(), enable_drop)
        if key != self._key:
            self._drop()
            self._primary = make_mlip_calculator(mlip_cfg, enable_mc_dropout=enable_drop)
            self._ensemble = (
                build_ensemble(mlip_cfg, enable_mc_dropout=enable_drop)
                if mlip_cfg.ensemble_models
                else []
            )
            self._key = key
        return self._primary, self._ensemble

    @contextmanager
    def md_pool(self) -> Iterator[MLIPWorkerPool]:
        """Workers running the MD seeds concurrently, one per mlip device."""
        if not self.keep:
            # The workers exit afterwards so their memory is free for DFT.
            with MLIPWorkerPool.from_placement(self.placement) as pool:
                yield pool
            return
        if self._pool is None:
            self._pool = MLIPWorkerPool.from_placement(self.placement)
        yield self._pool

    def after_scoring(self) -> None:
        """Free the mlip devices for DFT and training unless the models stay loaded."""
        if not self.keep:
            self._drop()

    def incumbent_calculator(self, mlip_cfg: MLIPConfig) -> Any:
        """The loaded deterministic incumbent model, for the promotion comparison."""
        if self.keep and self._key == (mlip_cfg.model_dump_json(), False):
            return self._primary
        return None

    def _drop(self) -> None:
        if self._key is not None:
            self._primary, self._ensemble, self._key = None, [], None
            release_device_memory()

    def close(self) -> None:
        self._drop()
        if self._pool is not None:
            self._pool.close()
            self._pool = None


def run_active_learning(
    cfg: ALConfig,
    *,
    candidate_sampler: CandidateSampler | None = None,
    iteration_assessor: IterationAssessor | None = None,
) -> None:
    """Run AL; an optional sampler replaces MD, including MD seed resolution.

    The sampler receives the current (possibly promoted) config, zero-based
    iteration and iteration directory, and returns exact geometries to label.
    An assessor must persist scientific revisions before an iteration is committed,
    including final iterations and iterations where training is deferred.
    """
    if (
        cfg.trainer.enabled
        and cfg.trainer.training_references is None
        and cfg.trainer.validation_reference_set is None
    ):
        # Callers that edit a loaded configuration skip TrainerConfig validation;
        # fail before any MD or DFT rather than at the first retrain.
        raise ValueError(
            "trainer.enabled=True requires trainer.training_references (or "
            "trainer.validation_reference_set): every fine-tune trains against "
            "elemental DFT references"
        )
    comparison_enabled = _validate_comparison_inputs(cfg)
    root = Path(cfg.loop.out_dir)
    root.mkdir(parents=True, exist_ok=True)
    _bind_root_seed(cfg, root)
    placement = cfg.resources.place(root)
    # Driver-side MLIP work (model loading, single-seed MD, scoring, model
    # comparison) runs on the mlip pool's device, not on device 0.
    with driver_on_mlip_device(placement):
        dataset_path = root / (
            "dataset.extxyz" if cfg.loop.dataset_format == "extxyz" else "dataset.db"
        )
        validation_dataset_path = root / "validation.extxyz"

        # Resolve MD seed structures once per run. For ``kind='prompt'`` this
        # invokes each configured seed source once and keeps strategy artifacts separate.
        seeds_dir = root / "seeds"
        strategies = _sampling_strategies(cfg)
        md_strategies = [
            (index, strategy)
            for index, strategy in enumerate(strategies)
            if isinstance(strategy, MDSamplingStrategy)
        ]
        seed_paths_by_strategy: dict[int, list[Path]] = {}
        if candidate_sampler is None:
            for strategy_index, strategy in md_strategies:
                strategy_seeds_dir = (
                    seeds_dir
                    if len(md_strategies) == 1
                    else seeds_dir / f"strategy-{strategy_index:03d}"
                )
                seed_stream = (
                    stream_seed(cfg.random_seed, "seed-source")
                    if len(md_strategies) == 1
                    else stream_seed(cfg.random_seed, "seed-source", strategy_index)
                )
                paths = resolve_seed_structures(
                    strategy.parameters.seed_source,
                    strategy_seeds_dir,
                    random_seed=seed_stream,
                )
                seed_paths_by_strategy[strategy_index] = paths
                log.info(
                    "Resolved %d seed structure(s) for strategy %d under %s",
                    len(paths),
                    strategy_index,
                    strategy_seeds_dir,
                )
        training_inputs = [path for paths in seed_paths_by_strategy.values() for path in paths]
        training_inputs.extend(
            path
            for strategy in strategies
            if isinstance(strategy, NEBSamplingStrategy)
            for path in (strategy.parameters.initial_path, strategy.parameters.final_path)
        )
        check_reference_coverage(cfg.trainer, elements_of(training_inputs))

        # The configured model, before a resume points cfg.mlip at the last promoted one.
        foundation_mlip = cfg.mlip.model_copy(deep=True)
        start_iter, resumed_logdir = (0, None)
        if cfg.loop.resume:
            start_iter, resumed_logdir = _scan_resume(root)
            if start_iter > 0:
                log.info("Resuming AL loop at iteration %d (logdir=%s)", start_iter, resumed_logdir)
                if resumed_logdir is not None and resumed_logdir.exists():
                    _apply_model_override(cfg, str(resumed_logdir))

        completed_dft = _completed_dft_count(cfg, root)
        completed_dft_neb = _completed_dft_neb_count(root)
        models = _SurrogateModels(
            placement, keep=not placement.shares_devices("mlip", ("dft", "training"))
        )
        log.info(
            "Surrogate models %s between iterations",
            "stay loaded" if models.keep else "are reloaded (their devices are shared)",
        )
        try:
            _run_iterations(
                cfg,
                placement,
                models,
                root=root,
                start_iter=start_iter,
                completed_dft=completed_dft,
                completed_dft_neb=completed_dft_neb,
                seed_paths=seed_paths_by_strategy,
                dataset_path=dataset_path,
                validation_dataset_path=validation_dataset_path,
                comparison_enabled=comparison_enabled,
                foundation_mlip=foundation_mlip,
                candidate_sampler=candidate_sampler,
                iteration_assessor=iteration_assessor,
            )
        finally:
            models.close()


def _score_benchmarks(
    cfg: ALConfig,
    state: IterationState,
    models: _SurrogateModels,
    dataset_path: Path,
    validation_dataset_path: Path,
) -> None:
    """Score the model in use on the frozen benchmark sets, for learning curves.

    Runs after the iteration's training and promotion decisions and feeds none
    of them; frames that were also acquired as labels are counted and logged.
    """
    if not cfg.loop.benchmark_sets:
        return
    from ase.io import read as ase_read

    labelled = {
        structure_identity(frame)
        for path in (dataset_path, validation_dataset_path)
        if path.exists()
        for frame in ase_read(path, index=":")
    }
    for benchmark in cfg.loop.benchmark_sets:
        frames = list(ase_read(benchmark.dataset, index=":"))
        overlap = sum(structure_identity(frame) in labelled for frame in frames)
        if overlap:
            log.warning(
                "Benchmark %s: %d frame(s) were also acquired as labels", benchmark.name, overlap
            )
        metrics, _ = evaluate_frames(
            cfg.mlip,
            frames,
            iteration=state.iteration,
            model_path=cfg.mlip.model_reference,
            test_set_label=str(benchmark.dataset),
            elemental_reference_manifest=cfg.benchmark_reference_manifest(),
            calculator=models.incumbent_calculator(cfg.mlip),
        )
        values = asdict(metrics)
        state.benchmark[benchmark.name] = {
            "model": cfg.mlip.model_reference,
            "n_dft_labels_total": state.n_training_frames_total + state.n_validation_frames_total,
            "frames_also_labelled": overlap,
            "metrics": {key: values[key] for key in METRIC_DEFINITIONS},
        }


def _sampling_strategies(cfg: ALConfig) -> list[SamplingStrategy]:
    if cfg.sampling is not None:
        strategies = list(cfg.sampling.strategies)
        if cfg.md is not None and not any(
            isinstance(strategy, MDSamplingStrategy) for strategy in strategies
        ):
            strategies.insert(0, MDSamplingStrategy(method="md", parameters=cfg.md))
        return strategies
    if cfg.md is None:
        raise ValueError("legacy MD sampling requires md configuration")
    return [MDSamplingStrategy(method="md", parameters=cfg.md)]


_DFT_NEB_RESERVATIONS = "dft_neb_reservations.jsonl"


def _completed_dft_neb_count(root: Path) -> int:
    """Count reserved full DFT NEB runs so the configured cap survives resume.

    Reservations live in a root-level ledger because ``_scan_resume`` deletes an
    interrupted iteration directory, and a reserved QE run may already have cost
    its allocation. Runs recorded in an iteration state but not in the ledger (from before
    the ledger) are added; ledger-backed runs are never counted twice.
    """
    reserved_by_iteration: dict[int, int] = {}
    ledger = root / _DFT_NEB_RESERVATIONS
    if ledger.exists():
        for line in ledger.read_text(encoding="utf-8").splitlines():
            if line:
                iteration = int(json.loads(line)["iteration"])
                reserved_by_iteration[iteration] = reserved_by_iteration.get(iteration, 0) + 1
    legacy = 0
    for path in root.glob("iteration_*/state.json"):
        state = json.loads(path.read_text(encoding="utf-8"))
        count = state.get("n_dft_neb_runs", 0)
        if not isinstance(count, int) or count < 0:
            raise ValueError(f"Invalid n_dft_neb_runs in {path}")
        legacy += max(0, count - reserved_by_iteration.get(int(state.get("iteration", -1)), 0))
    return sum(reserved_by_iteration.values()) + legacy


def _reserve_dft_neb(root: Path, iteration: int, transition_id: str) -> None:
    entry = json.dumps({"iteration": iteration, "transition_id": transition_id}, sort_keys=True)
    append_line(root / _DFT_NEB_RESERVATIONS, entry)


def _apply_neb_escalation(
    policy: NebEscalationConfig | None,
    outcome,
    iteration_dir: Path,
    pathway_directory: Path,
    state: IterationState,
    completed_dft_neb: int,
    env: Mapping[str, str] | None = None,
    timeout_sec: float | None = None,
) -> tuple[dict[str, Any], bool]:
    result = outcome.result
    uncertainty = result.pathway_uncertainty_eV
    record: dict[str, Any] = {
        "transition_id": result.provenance.get("transition_id"),
        "mlip_result": result.model_dump(mode="json"),
        "decision": "not_configured",
    }
    if policy is None:
        return record, False

    acceptable = (
        result.state == "converged_unqualified"
        and uncertainty is not None
        and uncertainty <= (policy.max_pathway_uncertainty_eV or -1.0)
    )
    run_dft = policy.policy == "always_dft_neb" or (
        policy.policy == "uncertainty_gated" and not acceptable
    )
    if acceptable:
        reason = None
    elif result.state != "converged_unqualified":
        reason = f"mlip_state_{result.state}"
    elif uncertainty is None:
        reason = "uncertainty_unavailable"  # unavailable is never read as low
    else:
        reason = "uncertainty_above_limit"
    record["escalation"] = {
        "policy": policy.policy,
        "mlip_model_id": result.provenance.get("model_id"),
        "reason": "always_dft_neb" if policy.policy == "always_dft_neb" else reason,
    }
    if not run_dft:
        record["decision"] = "accepted_mlip" if acceptable else "not_accepted"
        return record, False

    if completed_dft_neb >= policy.max_dft_neb_runs:
        raise RuntimeError(
            "full DFT NEB escalation required, but escalation.max_dft_neb_runs is exhausted"
        )
    transition = outcome.transition
    if transition is None:
        raise ValueError("full DFT NEB escalation requires typed transition metadata")
    if policy.dft_neb is None:
        raise ValueError("full DFT NEB escalation has no QE NEB configuration")
    if any(len(image.constraints) for image in outcome.images):
        # The QE NEB input carries no ASE constraints; running it would relax atoms the
        # surrogate pathway held fixed. Refuse before spending any of the DFT budget.
        raise ValueError(
            f"transition {transition.transition_id!r} has constrained images; full DFT NEB "
            "escalation does not preserve ASE constraints"
        )

    _reserve_dft_neb(iteration_dir.parent, state.iteration, transition.transition_id)
    state.n_dft_neb_runs += 1
    record["decision"] = "dispatching_dft_neb"
    state.neb_pathways.append(record)
    _write_state(iteration_dir, state)

    from matsim_agents.science.neb.endpoints import MigrationPair
    from matsim_agents.workflows.neb import run_neb_workflow

    # The optimized MLIP path and the endpoints it was actually built from (relaxed or
    # reordered), not the transition's generated endpoints or a fresh linear guess.
    initial = getattr(outcome, "initial", None) or outcome.images[0]
    final = getattr(outcome, "final", None) or outcome.images[-1]
    pair = MigrationPair(
        initial=initial.copy(),
        final=final.copy(),
        transition=transition,
        final_config=None,
    )
    start_images = tuple(outcome.images) if len(outcome.images) == policy.dft_neb.n_images else None
    dft_outcome = run_neb_workflow(
        pair,
        policy.dft_neb,
        pathway_directory / f"{transition.transition_id}-dft-neb",
        dispatch=True,
        allow_unrelaxed_endpoints=policy.allow_unrelaxed_endpoints,
        timeout_sec=timeout_sec,
        env=env,
        images=start_images,
    )
    record["dft_start_path"] = (
        "mlip_optimized" if start_images is not None else "linear_regenerated"
    )
    if dft_outcome.plan.status != "planned" or dft_outcome.result is None:
        raise RuntimeError(
            "QE NEB escalation did not produce a result: " + "; ".join(dft_outcome.plan.reasons)
        )
    record["decision"] = "dft_neb_complete"
    record["dft_result"] = dft_outcome.result.model_dump(mode="json")
    record["dft_run_directory"] = dft_outcome.plan.directory
    if dft_outcome.result.state not in ("converged_unqualified", "qualified"):
        record["decision"] = "dft_neb_unconverged"
    return record, True


def _escalate_neb_pathways(
    cfg: ALConfig,
    pathways,
    it_dir: Path,
    root: Path,
    state: IterationState,
    completed_dft_neb: int,
    placement: Placement,
) -> int:
    """Apply the escalation policy and kinetics output to each sampled NEB pathway."""
    env = None
    if cfg.escalation is not None:
        # An empty launcher runs neb.x on this node, so it needs the local device mask and threads.
        local = cfg.escalation.dft_neb is not None and not cfg.escalation.dft_neb.launcher
        env = placement.pool("dft").launches[0].environment(local=local)
    for pathway in pathways:
        outcome = pathway.outcome
        pathway_record, dft_neb_dispatched = _apply_neb_escalation(
            cfg.escalation,
            outcome,
            it_dir,
            pathway.directory,
            state,
            completed_dft_neb,
            env=env,
        )
        if pathway.parameters.kinetics is not None:
            kinetics_result = None
            if pathway_record["decision"] == "accepted_mlip":
                kinetics_result = outcome.result
            elif pathway_record["decision"] == "dft_neb_complete":
                kinetics_result = NebResult.model_validate(pathway_record["dft_result"])
            if kinetics_result is not None:
                kinetics_artifact = _write_neb_kinetics(
                    pathway.parameters,
                    outcome.transition,
                    kinetics_result,
                    pathway.directory / str(pathway_record["transition_id"]),
                )
                pathway_record["kinetics_artifact"] = str(kinetics_artifact)
        if dft_neb_dispatched:
            completed_dft_neb += 1
            state.neb_pathways[-1] = pathway_record
        else:
            state.neb_pathways.append(pathway_record)
    return completed_dft_neb


def _write_neb_kinetics(
    parameters: NebSamplingConfig,
    transition: NebTransition | None,
    result,
    directory: Path,
) -> Path:
    """Attach an accepted barrier to the existing exploratory-kinetics contracts."""
    from matsim_agents.science.kinetics.contracts import NetworkProvenance
    from matsim_agents.science.kinetics.evidence import attach_neb_result
    from matsim_agents.science.kinetics.graph import TransitionNetwork
    from matsim_agents.science.kinetics.rates import event_rate
    from matsim_agents.science.kinetics.transitions import add_candidate_transition
    from matsim_agents.science.kinetics.validation import validate_network

    kinetics = parameters.kinetics
    if kinetics is None or transition is None:
        raise ValueError("NEB kinetics output requires typed transition and rate assumptions")
    network = TransitionNetwork(
        NetworkProvenance(
            network_id=f"al-{transition.transition_id}",
            parent_id=transition.initial.parent_id,
            parent_ordering_id=transition.initial.parent_ordering_id,
            notes=("NEB barrier attached by active learning",),
        )
    )
    events = add_candidate_transition(network, transition)
    declared = transition.calculation
    evidence_transition = transition
    if declared is not None and (
        declared.method_id != result.method_id or declared.n_images != result.n_images
    ):
        # The accepted result came from another calculation than the one declared (an MLIP
        # NEB, or a QE escalation of the same transition); describe the one that ran.
        from matsim_agents.science.neb.contracts import NebCalculationSpec

        evidence_transition = transition.model_copy(
            update={
                "calculation": NebCalculationSpec(
                    n_images=result.n_images,
                    engine=(
                        "mlip" if result.provenance.get("backend") == "mlip" else declared.engine
                    ),
                    method_id=result.method_id,
                    ci_scheme=declared.ci_scheme,
                )
            }
        )
    attach_neb_result(
        network,
        evidence_transition,
        result,
        attempt_frequency_hz=kinetics.attempt_frequency_hz,
        prefactor_note=kinetics.prefactor_note,
    )
    rates = [
        event_rate(event, kinetics.temperature_K, require="exploratory").model_dump(mode="json")
        for event in network.events
    ]
    report = validate_network(network, events[0].source_id)
    target = directory / "kinetics-evidence.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "network_provenance": network.provenance.model_dump(mode="json"),
        "states": [state.model_dump(mode="json") for state in network.states],
        "events": [event.model_dump(mode="json") for event in network.events],
        "rates": rates,
        "report": report.model_dump(mode="json"),
        "neb_result": result.model_dump(mode="json"),
        "declared_calculation": (None if declared is None else declared.model_dump(mode="json")),
    }
    target.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return target


def _run_iterations(
    cfg: ALConfig,
    placement: Placement,
    models: _SurrogateModels,
    *,
    root: Path,
    start_iter: int,
    completed_dft: int,
    completed_dft_neb: int,
    seed_paths: dict[int, list],
    dataset_path: Path,
    validation_dataset_path: Path,
    comparison_enabled: bool,
    foundation_mlip: MLIPConfig,
    candidate_sampler: CandidateSampler | None = None,
    iteration_assessor: IterationAssessor | None = None,
) -> None:
    """Iterations ``start_iter`` .. ``cfg.loop.n_iterations - 1`` of the AL loop."""
    for i in range(start_iter, cfg.loop.n_iterations):
        if (
            cfg.loop.max_dft_calculations is not None
            and completed_dft >= cfg.loop.max_dft_calculations
        ):
            log.info("DFT calculation cap reached; ending loop.")
            break
        it_dir = _iter_dir(root, i)
        it_dir.mkdir(parents=True, exist_ok=True)
        state = IterationState(iteration=i)
        t_iter0 = time.time()

        try:
            # --- 1. Build calculator(s) for this iteration --------------------
            t0 = time.time()
            enable_drop = cfg.acquisition.strategy in {"mc_dropout", "ensemble_then_dropout"}
            primary_calc, ensemble_calcs = models.calculators(cfg.mlip, enable_drop)
            state.timings_sec["build_calculators"] = time.time() - t0

            # --- 2. Generate candidates with the configured numerical methods -
            t0 = time.time()
            neb_pathways: list = []
            if candidate_sampler is not None:
                candidates = candidate_sampler(cfg, i, it_dir)
            else:
                from matsim_agents.workflows.active_learning.sampling import sample_strategies

                batch = sample_strategies(
                    _sampling_strategies(cfg),
                    primary_calc,
                    it_dir,
                    random_seed=cfg.random_seed,
                    iteration=i,
                    seed_paths_by_strategy=seed_paths,
                    mlip_config=cfg.mlip,
                    acquisition=cfg.acquisition,
                    ensemble_calculators=ensemble_calcs or (),
                    md_pool_factory=models.md_pool,
                    md_sampler=sample_md_candidates,
                )
                candidates = batch.candidates
                neb_pathways = batch.neb_pathways
            state.n_candidates = len(candidates)
            state.timings_sec["sampling"] = time.time() - t0
            log.info("Iter %d: produced %d candidates", i, len(candidates))

            if not candidates:
                del primary_calc, ensemble_calcs
                models.after_scoring()
                completed_dft_neb = _escalate_neb_pathways(
                    cfg, neb_pathways, it_dir, root, state, completed_dft_neb, placement
                )
                state.notes = "No candidates produced; ending loop."
                state.status = "complete"
                _write_state(it_dir, state)
                break

            # --- 3. Score & select --------------------------------------------
            selected = _select_for_labelling(
                cfg, candidates, primary_calc, ensemble_calcs, i, completed_dft, state
            )
            predicted_forces = {}
            if cfg.trainer.force_error_trigger_eV_per_A is not None:
                for candidate in selected:
                    atoms = candidate.atoms.copy()
                    atoms.calc = primary_calc
                    try:
                        forces = np.asarray(atoms.get_forces(), dtype=float)
                    finally:
                        atoms.calc = None
                    if forces.shape != (len(atoms), 3) or not np.isfinite(forces).all():
                        raise ValueError(f"Invalid incumbent forces for {candidate.candidate_id}")
                    predicted_forces[candidate.candidate_id] = forces.copy()
            # The surrogate is not needed again this iteration; free its device
            # memory before DFT and training, unless they use other devices.
            del primary_calc, ensemble_calcs
            models.after_scoring()

            # Full DFT NEB escalation can run for a long time on the DFT devices, so it
            # starts only once the surrogate has released its memory.
            completed_dft_neb = _escalate_neb_pathways(
                cfg, neb_pathways, it_dir, root, state, completed_dft_neb, placement
            )

            # --- 4. DFT labelling (VASP or QE) --------------------------------
            results, backend = _label_with_dft(cfg, selected, it_dir, i, state, placement)
            completed_dft += len(results)
            if cfg.trainer.force_error_trigger_eV_per_A is not None:
                state.dft_force_errors_eV_per_A = _audit_force_errors(
                    selected, results, predicted_forces
                )
                state.accuracy_triggered = any(
                    error > cfg.trainer.force_error_trigger_eV_per_A
                    for error in state.dft_force_errors_eV_per_A.values()
                )

            # --- 5. Append to dataset -----------------------------------------
            labelled_elements, method_signature, n_appended = _append_labelled_frames(
                cfg,
                results,
                selected,
                i,
                dataset_path,
                validation_dataset_path,
                backend,
                state,
            )

            # --- 6. (Optional) retrain the surrogate --------------------------
            if comparison_enabled:
                _check_reference_method(cfg, labelled_elements, backend, method_signature)
            t0 = time.time()
            if cfg.trainer.enabled and state.accuracy_triggered is False:
                state.training_status = "not_triggered"
                state.notes = (
                    "No converged QE/DFT audit exceeded the force-error tolerance."
                    if state.dft_force_errors_eV_per_A
                    else "No usable DFT force audits; accuracy is unknown, not validated."
                )
                log.warning("Iteration %d: %s", i, state.notes)
                _score_benchmarks(cfg, state, models, dataset_path, validation_dataset_path)
                state.status = "complete"
                continue
            if cfg.trainer.enabled:
                _collect_training_deferrals(cfg, state, validation_dataset_path, comparison_enabled)
                if state.training_deferred_reasons:
                    state.training_status = "deferred"
                    log.warning(
                        "Iteration %d training/comparison deferred: %s",
                        i,
                        "; ".join(state.training_deferred_reasons),
                    )
                    state.timings_sec["retrain"] = time.time() - t0
                    _score_benchmarks(cfg, state, models, dataset_path, validation_dataset_path)
                    state.status = "complete"
                    continue
                state.training_status = "running"
            base_mlip = foundation_mlip if cfg.trainer.start_from == "foundation" else cfg.mlip
            _retrain_candidate(
                cfg, dataset_path, i, it_dir, root, state, n_appended, placement, base_mlip
            )
            _compare_and_promote(
                cfg,
                state,
                i,
                dataset_path,
                validation_dataset_path,
                comparison_enabled,
                incumbent_calculator=models.incumbent_calculator(cfg.mlip),
            )
            state.timings_sec["retrain"] = time.time() - t0
            if cfg.trainer.enabled:
                state.training_status = "completed"
            _score_benchmarks(cfg, state, models, dataset_path, validation_dataset_path)

            state.status = "complete"
        except Exception as exc:  # noqa: BLE001
            state.status = "failed"
            if state.training_status == "running":
                state.training_status = "failed"
            state.notes = repr(exc)
            log.exception("Iteration %d failed", i)
            _write_state(it_dir, state)
            raise
        finally:
            state.timings_sec["total"] = time.time() - t_iter0
            if state.status == "complete" and iteration_assessor is not None:
                try:
                    iteration_assessor(cfg, i, it_dir, state)
                except Exception as exc:
                    state.status = "failed"
                    state.notes = f"Scientific hypothesis assessment failed: {exc!r}"
                    _write_state(it_dir, state)
                    log.exception("Iteration %d hypothesis assessment failed", i)
                    raise
            _write_state(it_dir, state)


def _write_state(it_dir: Path, state: IterationState) -> None:
    (it_dir / "state.json").write_text(json.dumps(state.to_dict(), indent=2))


def _audit_force_errors(
    selected: list[Candidate], results: list, predictions: dict[str, np.ndarray]
) -> dict[str, float]:
    """Compare force components only at the identical labelled geometry."""
    candidates = {candidate.candidate_id: candidate for candidate in selected}
    errors = {}
    for result in results:
        if not result.converged:
            continue
        key = Path(result.work_dir).name
        if key not in candidates:
            raise ValueError(f"DFT audit result has unknown candidate ID: {key}")
        source = candidates[key].atoms
        labelled = result.final_atoms
        # QE's SCF output rounds cell vectors and positions more than its input.
        tolerance = 1e-5 if result.backend == "qe" else 1e-8
        if (
            labelled is None
            or not np.array_equal(source.numbers, labelled.numbers)
            or not np.allclose(source.cell, labelled.cell, rtol=0, atol=tolerance)
            or not np.allclose(source.positions, labelled.positions, rtol=0, atol=tolerance)
        ):
            raise ValueError(f"DFT audit changed the geometry of {key}")
        forces = np.asarray(result.forces_eV_per_A, dtype=float)
        if forces.shape != (len(source), 3) or not np.isfinite(forces).all():
            raise ValueError(f"Missing or non-finite DFT audit forces for {key}")
        errors[key] = float(np.mean(np.abs(forces - predictions[key])))
        if result.backend == "qe":
            result.final_atoms = source.copy()
            result.final_atoms.info["qe_output_geometry_tolerance_A"] = tolerance
    return errors
