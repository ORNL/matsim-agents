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

import logging
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from matsim_agents.backends.mlip.config import MLIPConfig
from matsim_agents.backends.mlip.pool import (
    driver_on_mlip_device,
)
from matsim_agents.core.seeding import stream_seed
from matsim_agents.execution.placement import Placement
from matsim_agents.workflows.active_learning.campaign_init import (
    check_reference_coverage,
    elements_of,
)
from matsim_agents.workflows.active_learning.candidates import Candidate
from matsim_agents.workflows.active_learning.config import (
    ALConfig,
    MDSamplingStrategy,
    NEBSamplingStrategy,
)
from matsim_agents.workflows.active_learning.evaluate import (
    _apply_model_override,
)
from matsim_agents.workflows.active_learning.iteration_state import (
    IterationAssessor,
    IterationState,
    _iter_dir,
    _write_state,
)
from matsim_agents.workflows.active_learning.labelling import (
    _audit_force_errors,
    _completed_dft_count,
    _label_with_dft,
    _sampling_strategies,
    _select_for_labelling,
)
from matsim_agents.workflows.active_learning.md_sampler import sample_md_candidates
from matsim_agents.workflows.active_learning.neb_escalation import (
    _completed_dft_neb_count,
    _escalate_neb_pathways,
)
from matsim_agents.workflows.active_learning.promotion import (
    _compare_and_promote,
    _score_benchmarks,
    _validate_comparison_inputs,
)
from matsim_agents.workflows.active_learning.resume import _bind_root_seed, _scan_resume
from matsim_agents.workflows.active_learning.seed_sources import resolve_seed_structures
from matsim_agents.workflows.active_learning.surrogate import _SurrogateModels
from matsim_agents.workflows.active_learning.training import (
    _append_labelled_frames,
    _check_reference_method,
    _collect_training_deferrals,
    _retrain_candidate,
)

log = logging.getLogger(__name__)


CandidateSampler = Callable[[ALConfig, int, Path], list[Candidate]]


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
