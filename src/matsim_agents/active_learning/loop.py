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
double-count VASP results in the dataset).

Usage (Python)
--------------
    from matsim_agents.active_learning import ALConfig
    from matsim_agents.active_learning.loop import run_active_learning

    cfg = ALConfig.from_yaml("al.yaml")
    run_active_learning(cfg)
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from matsim_agents.active_learning.calculator import build_ensemble, make_mlip_calculator
from matsim_agents.active_learning.candidates import sample_md_candidates
from matsim_agents.active_learning.config import ALConfig
from matsim_agents.active_learning.dataset_governance import (
    DatasetManifest,
    structure_identity,
    validate_labelled_frames,
    write_dataset_manifest,
)
from matsim_agents.active_learning.dft_backend import DFTJobSpec, make_backend
from matsim_agents.active_learning.dft_protocol import (
    dft_method_signature,
    scientific_dft_payload,
)
from matsim_agents.active_learning.dft_runner import run_dft_batch
from matsim_agents.active_learning.evaluate import (
    _apply_model_override,
    _reference_energy,
    _reference_forces,
    evaluate_promotion_candidate,
)
from matsim_agents.active_learning.seeds import resolve_seed_structures
from matsim_agents.active_learning.trainer import (
    append_frames_to_extxyz,
    dft_results_to_frames,
    retrain_hydragnn,
    retrain_mace,
    retrain_uma,
)
from matsim_agents.active_learning.uncertainty import select_candidates
from matsim_agents.discovery.energy_references import (
    load_elemental_reference_manifest,
    validate_dataset_reference_method,
)

log = logging.getLogger(__name__)


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
    dft_backend: str | None = None
    score_min: float | None = None
    score_max: float | None = None
    score_mean: float | None = None
    selected_candidate_ids: list[str] = field(default_factory=list)
    candidate_uncertainty: dict[str, float] = field(default_factory=dict)
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
    new_logdir: str | None = None
    timings_sec: dict[str, float] = field(default_factory=dict)
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
) -> tuple[list[Any], list[Any]]:
    """Assign only new frames to meet the cumulative held-out target."""
    if validation_fraction <= 0.0:
        return frames, []
    total = existing_training_count + existing_validation_count + len(frames)
    target = max(1, int(round(total * validation_fraction)))
    n_validation = min(len(frames), max(0, target - existing_validation_count))
    validation_indices = set(
        np.random.default_rng(seed).permutation(len(frames))[:n_validation].tolist()
    )
    training = [frame for index, frame in enumerate(frames) if index not in validation_indices]
    validation = [frame for index, frame in enumerate(frames) if index in validation_indices]
    return training, validation


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


def run_active_learning(cfg: ALConfig) -> None:
    """Run the full AL loop. Idempotent: safe to re-invoke after a job restart."""
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
                require_sidecar=True,
            )
    root = Path(cfg.loop.out_dir)
    root.mkdir(parents=True, exist_ok=True)

    dataset_path = root / (
        "dataset.extxyz" if cfg.loop.dataset_format == "extxyz" else "dataset.db"
    )
    validation_dataset_path = root / "validation.extxyz"

    # Resolve MD seed structures once per run. For ``kind='prompt'`` this
    # invokes the LLM exactly once and caches the resulting JSON under
    # ``out_dir/seeds/llm_proposed_compositions.json`` for reproducibility.
    seeds_dir = root / "seeds"
    seed_paths = resolve_seed_structures(cfg.md.seed_source, seeds_dir)
    log.info("Resolved %d seed structure(s) under %s", len(seed_paths), seeds_dir)

    start_iter, resumed_logdir = (0, None)
    if cfg.loop.resume:
        start_iter, resumed_logdir = _scan_resume(root)
        if start_iter > 0:
            log.info("Resuming AL loop at iteration %d (logdir=%s)", start_iter, resumed_logdir)
            if resumed_logdir is not None and resumed_logdir.exists():
                _apply_model_override(cfg, str(resumed_logdir))

    completed_dft = 0
    if cfg.loop.resume:
        for state_path in root.glob("iteration_*/state.json"):
            state_data = json.loads(state_path.read_text(encoding="utf-8"))
            if state_data.get("status") == "complete":
                completed_dft += int(state_data.get("n_dft_converged", 0)) + int(
                    state_data.get("n_dft_failed", 0)
                )
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
            primary_calc = make_mlip_calculator(cfg.mlip, enable_mc_dropout=enable_drop)
            ensemble_calcs: list = []
            if cfg.mlip.ensemble_paths:
                ensemble_calcs = build_ensemble(cfg.mlip, enable_mc_dropout=enable_drop)
            state.timings_sec["build_calculators"] = time.time() - t0

            # --- 2. Generate candidates via MD --------------------------------
            t0 = time.time()
            md_dir = it_dir / "md"
            candidates = sample_md_candidates(cfg.md, primary_calc, md_dir, seed_paths=seed_paths)
            state.n_candidates = len(candidates)
            state.timings_sec["md_sampling"] = time.time() - t0
            log.info("Iter %d: produced %d MD candidates", i, len(candidates))

            if not candidates:
                state.notes = "No candidates produced; ending loop."
                state.status = "complete"
                _write_state(it_dir, state)
                break

            # --- 3. Score & select --------------------------------------------
            t0 = time.time()
            selected, scores = select_candidates(
                candidates,
                cfg.acquisition,
                primary_calculator=primary_calc,
                ensemble_calculators=ensemble_calcs or None,
                seed=42 + i,
            )
            if cfg.loop.max_dft_calculations is not None:
                remaining_dft = cfg.loop.max_dft_calculations - completed_dft
                selected = selected[:remaining_dft]
            state.n_selected = len(selected)
            state.selected_candidate_ids = [candidate.candidate_id for candidate in selected]
            selected_ids = set(state.selected_candidate_ids)
            state.candidate_uncertainty = {
                candidate.candidate_id: float(score)
                for candidate, score in zip(candidates, scores, strict=True)
                if candidate.candidate_id in selected_ids and np.isfinite(score)
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

            # --- 4. DFT labelling (VASP or QE) --------------------------------
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
            results = run_dft_batch(
                specs,
                backend,
                max_workers=cfg.dft.max_concurrent_jobs,
            )
            n_ok = sum(1 for r in results if r.converged)
            state.n_dft_converged = n_ok
            state.n_dft_failed = len(results) - n_ok
            completed_dft += len(results)
            state.timings_sec["dft"] = time.time() - t0
            log.info(
                "Iter %d: %s converged=%d failed=%d",
                i,
                backend.name.upper(),
                n_ok,
                len(results) - n_ok,
            )

            if cfg.loop.fail_fast and state.n_dft_failed:
                raise RuntimeError(
                    f"{state.n_dft_failed} {backend.name} jobs failed and fail_fast=True"
                )

            # --- 5. Append to dataset -----------------------------------------
            t0 = time.time()
            frames = dft_results_to_frames(results, iteration=i)
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
                int(number)
                for candidate in selected
                for number in candidate.atoms.get_atomic_numbers()
            }
            frames, validation = validate_labelled_frames(
                frames,
                existing_frames=existing_frames,
                expected_atomic_numbers=expected_atomic_numbers,
            )
            training_frames, validation_frames = _split_training_validation_frames(
                frames,
                validation_fraction=cfg.trainer.validation_fraction,
                seed=cfg.trainer.validation_split_seed + i,
                existing_training_count=len(existing_training_frames),
                existing_validation_count=len(existing_validation_frames),
            )
            labelled_elements = existing_elements | {
                symbol for frame in frames for symbol in frame.atoms.get_chemical_symbols()
            }
            method_signature = _dft_method_signature(cfg, labelled_elements)
            n_appended = append_frames_to_extxyz(training_frames, dataset_path)
            n_validation_appended = append_frames_to_extxyz(
                validation_frames, validation_dataset_path
            )
            state.dataset_path = str(dataset_path)
            state.n_training_frames = n_appended
            state.n_validation_frames = n_validation_appended
            state.n_training_frames_total = len(existing_training_frames) + n_appended
            state.n_validation_frames_total = (
                len(existing_validation_frames) + n_validation_appended
            )
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
                )
            if validation_dataset_path.exists():
                write_dataset_manifest(
                    validation_dataset_path,
                    dft_backend=backend.name,
                    energy_reference=f"{backend.name}:native_total_energy",
                    validation=validation.model_copy(update={"accepted": len(validation_frames)}),
                    parent_dataset_id=validation_parent_dataset_id,
                    method_signature=method_signature,
                    split_role="validation",
                )
            state.timings_sec["append_dataset"] = time.time() - t0
            log.info(
                "Iter %d: appended %d training and %d validation frames",
                i,
                n_appended,
                n_validation_appended,
            )

            # --- 6. (Optional) retrain the surrogate --------------------------
            if comparison_enabled:
                if cfg.trainer.validation_reference_set is None:
                    raise ValueError(
                        "model comparison requires a DFT-labelled elemental reference manifest"
                    )
                reference_manifest, _, _ = load_elemental_reference_manifest(
                    cfg.trainer.validation_reference_set,
                    required_elements=labelled_elements,
                )
                if (
                    reference_manifest["backend"] != backend.name
                    or reference_manifest["method_signature"] != method_signature
                ):
                    raise ValueError(
                        "elemental references and campaign labels use different DFT methods"
                    )
            t0 = time.time()
            if cfg.trainer.enabled:
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
                if state.training_deferred_reasons:
                    state.training_status = "deferred"
                    log.warning(
                        "Iteration %d training/comparison deferred: %s",
                        i,
                        "; ".join(state.training_deferred_reasons),
                    )
                    state.timings_sec["retrain"] = time.time() - t0
                    state.status = "complete"
                    continue
                state.training_status = "running"
            if (
                cfg.trainer.enabled
                and cfg.mlip.backend == "hydragnn"
                and cfg.mlip.hydragnn is not None
            ):
                new_logdir = retrain_hydragnn(
                    cfg.trainer,
                    cfg.mlip.hydragnn,
                    dataset_path=dataset_path,
                    iteration=i,
                    out_logdir=it_dir / "model",
                    dft_config=cfg.dft,
                    reference_root=root / "elemental-training-reference",
                )
                state.candidate_model_path = str(new_logdir)
            elif cfg.mlip.backend == "uma" and cfg.mlip.uma is not None and cfg.trainer.enabled:
                new_model = retrain_uma(
                    cfg.trainer,
                    cfg.mlip.uma,
                    dataset_path=dataset_path,
                    iteration=i,
                    out_model_dir=it_dir / "model",
                )
                state.candidate_model_path = str(new_model)
            elif cfg.mlip.backend == "mace" and cfg.mlip.mace is not None and cfg.trainer.enabled:
                new_model = retrain_mace(
                    cfg.trainer,
                    cfg.mlip.mace,
                    dataset_path=dataset_path,
                    iteration=i,
                    out_model_dir=it_dir / "model",
                )
                state.candidate_model_path = str(new_model)
            else:
                # Frozen foundation model / disabled trainer: keep accumulating labels.
                log.info(
                    "Skipping retraining for backend=%s; %d labelled frames accumulated in %s.",
                    cfg.mlip.backend,
                    n_appended,
                    dataset_path,
                )
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
            state.timings_sec["retrain"] = time.time() - t0
            if cfg.trainer.enabled:
                state.training_status = "completed"

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
            _write_state(it_dir, state)


def _write_state(it_dir: Path, state: IterationState) -> None:
    (it_dir / "state.json").write_text(json.dumps(state.to_dict(), indent=2))
