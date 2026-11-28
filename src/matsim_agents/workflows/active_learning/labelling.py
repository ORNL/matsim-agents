"""Selection of candidates for DFT labelling, the DFT label cache and provenance."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from matsim_agents.backends.dft.base import DFTJobSpec, make_backend
from matsim_agents.backends.dft.runner import run_dft_batch
from matsim_agents.core.fileio import atomic_write_text
from matsim_agents.core.hashing import sha256_file
from matsim_agents.core.restart import KeyedResultLog
from matsim_agents.core.seeding import stream_seed
from matsim_agents.data.governance import (
    structure_identity,
)
from matsim_agents.execution.placement import Placement
from matsim_agents.workflows.active_learning.candidates import Candidate
from matsim_agents.workflows.active_learning.config import (
    ALConfig,
    MDSamplingStrategy,
    SamplingStrategy,
)
from matsim_agents.workflows.active_learning.dft_attempts import (
    ATTEMPTS_DIR,
    create_attempt,
    update_attempt,
)
from matsim_agents.workflows.active_learning.dft_protocol import (
    dft_method_signature,
    scientific_dft_payload,
)
from matsim_agents.workflows.active_learning.iteration_state import IterationState
from matsim_agents.workflows.active_learning.uncertainty import select_candidates

log = logging.getLogger(__name__)


def _scientific_dft_payload(cfg: ALConfig, elements: set[str]) -> dict[str, Any]:
    return scientific_dft_payload(cfg.dft, elements)


def _dft_method_signature(cfg: ALConfig, elements: set[str]) -> str:
    return dft_method_signature(cfg.dft, elements)


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
    _validate_candidate_ids(candidates)
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
    _validate_candidate_ids(selected)
    backend = make_backend(cfg.dft)
    state.dft_backend = backend.name
    specs = [
        DFTJobSpec(
            job_id=cand.candidate_id,
            atoms=cand.atoms,
            work_dir="",
        )
        for cand in selected
    ]
    ledger = it_dir.parent / _DFT_LABEL_CACHE / f"{it_dir.name}.jsonl"
    cached = _load_dft_labels(ledger)
    reused: dict[str, Any] = {}
    pending = []
    for spec in specs:
        result = cached.get(_dft_label_key(cfg, spec))
        if (
            result is None
            or result.backend != backend.name
            or not _cached_attempt_valid(it_dir.parent, i, spec, result, cfg)
        ):
            from ase.io.jsonio import encode

            attempt = create_attempt(
                it_dir.parent, iteration=i, kind="single_point", identity=spec.job_id
            )
            atomic_write_text(attempt / "structure.json", encode(spec.atoms))
            update_attempt(
                attempt,
                backend=backend.name,
                structure_identity=structure_identity(spec.atoms),
                method_signature=_dft_method_signature(cfg, set(spec.atoms.get_chemical_symbols())),
            )
            spec.work_dir = str(attempt / spec.job_id)
            pending.append(spec)
            continue
        reused[spec.job_id] = result
    if reused:
        log.info("Iter %d: reusing %d saved DFT labels from %s", i, len(reused), ledger)
    state.n_dft_reused = len(reused)

    by_job = dict(reused)

    def save(spec: DFTJobSpec, result: Any) -> None:
        by_job[spec.job_id] = result
        hashes = _artifact_hashes(Path(spec.work_dir), backend.name) if result.converged else {}
        update_attempt(
            Path(spec.work_dir).parent,
            status="converged" if result.converged else "failed",
            return_code=result.return_code,
            notes=result.notes,
            artifact_sha256=hashes,
        )
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


def _validate_candidate_ids(candidates: list) -> None:
    ids = [candidate.candidate_id for candidate in candidates]
    if any(
        not isinstance(value, str) or not value or value in {".", ".."} or Path(value).name != value
        for value in ids
    ):
        raise ValueError("candidate IDs must be nonempty single path components")
    if len(set(ids)) != len(ids):
        raise ValueError("candidate IDs must be unique within an iteration")


def _cached_attempt_valid(
    root: Path, iteration: int, spec: DFTJobSpec, result: Any, cfg: ALConfig
) -> bool:
    work_dir = Path(result.work_dir)
    attempt = work_dir.parent
    if work_dir.name != spec.job_id or attempt.parent.resolve() != (root / ATTEMPTS_DIR).resolve():
        return False
    if not work_dir.is_dir() or not (attempt / "attempt.json").is_file():
        return False
    record = json.loads((attempt / "attempt.json").read_text(encoding="utf-8"))
    hashes = record.get("artifact_sha256")
    if not isinstance(hashes, dict) or not hashes:
        return False
    try:
        if hashes != _artifact_hashes(work_dir, result.backend):
            return False
    except FileNotFoundError:
        return False
    return (
        record.get("attempt_id") == attempt.name
        and record.get("iteration") == iteration
        and record.get("kind") == "single_point"
        and record.get("identity") == spec.job_id
        and record.get("status") == "converged"
        and record.get("backend") == result.backend
        and record.get("structure_identity") == structure_identity(spec.atoms)
        and record.get("method_signature")
        == _dft_method_signature(cfg, set(spec.atoms.get_chemical_symbols()))
        and result.converged
    )


def _artifact_hashes(work_dir: Path, backend: str) -> dict[str, str]:
    if backend == "qe":
        names = ("pw.in", "pw.out")
    elif backend == "vasp":
        names = ("POSCAR", "INCAR", "KPOINTS", "POTCAR", "vasprun.xml")
    else:
        raise ValueError(f"unsupported DFT backend {backend!r} for attempt archive")
    return {name: sha256_file(work_dir / name) for name in names}


def _dft_label_key(cfg: ALConfig, spec: DFTJobSpec) -> str:
    elements = set(spec.atoms.get_chemical_symbols())
    return "|".join(
        (spec.job_id, structure_identity(spec.atoms), _dft_method_signature(cfg, elements))
    )


def _dft_label_log(ledger: Path) -> KeyedResultLog:
    from ase.io.jsonio import decode, encode

    return KeyedResultLog(ledger, encode=encode, decode=decode)


def _load_dft_labels(ledger: Path) -> dict[str, Any]:
    from matsim_agents.backends.dft.base import DFTResult

    return _dft_label_log(ledger).load(lambda result: DFTResult(**result))


def _save_dft_label(ledger: Path, key: str, result: Any) -> None:
    _dft_label_log(ledger).save(key, asdict(result))


def _record_provenance(cfg: ALConfig, frames: list, selected: list, i: int) -> None:
    """Record where each label came from, for grouped and chronological dataset splits.

    Method-independent identity and sampling provenance are shared. MD steps remain
    MD-only; NEB image indices are recorded as optimization coordinates, not time.
    """
    candidates = {candidate.candidate_id: candidate for candidate in selected}
    for frame in frames:
        work_dir = Path(frame.source_work_dir)
        candidate = candidates.get(work_dir.name)
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
        if work_dir.parent.parent.name == ATTEMPTS_DIR:
            info["dft_attempt_id"] = work_dir.parent.name
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
