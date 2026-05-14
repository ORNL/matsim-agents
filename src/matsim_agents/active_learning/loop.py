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

from matsim_agents.active_learning.calculator import build_ensemble, build_hydragnn_calculator
from matsim_agents.active_learning.candidates import sample_md_candidates
from matsim_agents.active_learning.config import ALConfig
from matsim_agents.active_learning.dft_backend import DFTJobSpec, make_backend
from matsim_agents.active_learning.dft_runner import run_dft_batch
from matsim_agents.active_learning.seeds import resolve_seed_structures
from matsim_agents.active_learning.trainer import (
    append_frames_to_extxyz,
    dft_results_to_frames,
    retrain_hydragnn,
)
from matsim_agents.active_learning.uncertainty import select_candidates

log = logging.getLogger(__name__)


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
    dataset_path: str | None = None
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


def _scan_resume(root: Path) -> tuple[int, Path | None]:
    """Return (start_iteration, current_logdir_or_None) based on existing state.

    The current_logdir is taken from the most recent completed iteration's
    ``state.json::new_logdir`` if present.
    """
    if not root.exists():
        return 0, None
    completed: list[tuple[int, Path]] = []
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
                completed.append((int(data["iteration"]), Path(data.get("new_logdir") or "")))
            else:
                log.warning("Removing failed/partial iteration dir %s", d)
                shutil.rmtree(d, ignore_errors=True)
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not parse %s (%s); removing", sf, exc)
            shutil.rmtree(d, ignore_errors=True)
    if not completed:
        return 0, None
    last_i, last_logdir = max(completed, key=lambda t: t[0])
    return last_i + 1, last_logdir if str(last_logdir) else None


# --------------------------------------------------------------------------- #
# Main driver                                                                 #
# --------------------------------------------------------------------------- #


def run_active_learning(cfg: ALConfig) -> None:
    """Run the full AL loop. Idempotent: safe to re-invoke after a job restart."""
    root = Path(cfg.loop.out_dir)
    root.mkdir(parents=True, exist_ok=True)

    dataset_path = root / (
        "dataset.extxyz" if cfg.loop.dataset_format == "extxyz" else "dataset.db"
    )

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
                cfg.hydragnn.logdir = resumed_logdir

    for i in range(start_iter, cfg.loop.n_iterations):
        it_dir = _iter_dir(root, i)
        it_dir.mkdir(parents=True, exist_ok=True)
        state = IterationState(iteration=i)
        t_iter0 = time.time()

        try:
            # --- 1. Build calculator(s) for this iteration --------------------
            t0 = time.time()
            primary_calc = build_hydragnn_calculator(cfg.hydragnn)
            ensemble_calcs: list = []
            if cfg.hydragnn.ensemble_paths:
                ensemble_calcs = build_ensemble(cfg.hydragnn)
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
            state.n_selected = len(selected)
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
            results = run_dft_batch(specs, backend)
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
                raise RuntimeError(
                    f"{state.n_dft_failed} {backend.name} jobs failed and fail_fast=True"
                )

            # --- 5. Append to dataset -----------------------------------------
            t0 = time.time()
            frames = dft_results_to_frames(results, iteration=i)
            n_appended = append_frames_to_extxyz(frames, dataset_path)
            state.dataset_path = str(dataset_path)
            state.timings_sec["append_dataset"] = time.time() - t0
            log.info("Iter %d: appended %d labelled frames to %s", i, n_appended, dataset_path)

            # --- 6. (Optional) retrain HydraGNN -------------------------------
            t0 = time.time()
            new_logdir = retrain_hydragnn(
                cfg.trainer,
                cfg.hydragnn,
                dataset_path=dataset_path,
                iteration=i,
                out_logdir=it_dir / "model",
            )
            state.new_logdir = str(new_logdir)
            state.timings_sec["retrain"] = time.time() - t0
            # Update logdir for next iteration (in-memory only — we re-resolve
            # from state.json on restart).
            cfg.hydragnn.logdir = new_logdir

            state.status = "complete"
        except Exception as exc:  # noqa: BLE001
            state.status = "failed"
            state.notes = repr(exc)
            log.exception("Iteration %d failed", i)
            _write_state(it_dir, state)
            raise
        finally:
            state.timings_sec["total"] = time.time() - t_iter0
            _write_state(it_dir, state)


def _write_state(it_dir: Path, state: IterationState) -> None:
    (it_dir / "state.json").write_text(json.dumps(state.to_dict(), indent=2))
