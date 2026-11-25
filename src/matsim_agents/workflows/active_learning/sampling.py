"""Shared execution of configured MD and NEB sampling strategies."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

from matsim_agents.backends.mlip.config import MLIPConfig
from matsim_agents.core.seeding import stream_seed
from matsim_agents.workflows.active_learning.candidates import Candidate
from matsim_agents.workflows.active_learning.config import (
    AcquisitionConfig,
    MDSamplingStrategy,
    NebSamplingConfig,
    NEBSamplingStrategy,
    SamplingStrategy,
)
from matsim_agents.workflows.active_learning.md_sampler import sample_md_candidates
from matsim_agents.workflows.active_learning.neb_sampler import (
    NebSamplingOutcome,
    NebSamplingTask,
)


@dataclass(frozen=True)
class SampledNebPathway:
    strategy_index: int
    parameters: NebSamplingConfig
    directory: Path
    outcome: NebSamplingOutcome


@dataclass(frozen=True)
class SamplingBatch:
    candidates: list[Candidate]
    neb_pathways: list[SampledNebPathway]


def sample_strategies(
    strategies: Sequence[SamplingStrategy],
    primary_calculator,
    out_dir: str | Path,
    *,
    random_seed: int,
    iteration: int,
    seed_paths_by_strategy: Mapping[int, list] | None,
    mlip_config: MLIPConfig,
    acquisition: AcquisitionConfig | None,
    ensemble_calculators=(),
    md_pool_factory: Callable | None = None,
    md_sampler: Callable | None = None,
) -> SamplingBatch:
    """Run methods through their existing samplers and return the shared candidates."""
    root = Path(out_dir)
    md_strategy_count = sum(isinstance(strategy, MDSamplingStrategy) for strategy in strategies)
    neb_strategy_count = sum(isinstance(strategy, NEBSamplingStrategy) for strategy in strategies)
    candidates_by_strategy: dict[int, list[Candidate]] = {}
    pathways_by_strategy: dict[int, SampledNebPathway] = {}
    neb_requests: list[tuple[int, NebSamplingConfig, Path]] = []
    enable_dropout = acquisition is not None and acquisition.strategy in {
        "mc_dropout",
        "ensemble_then_dropout",
    }
    uncertainty_mode = None
    if acquisition is not None:
        if acquisition.strategy in {"ensemble", "ensemble_then_dropout"}:
            uncertainty_mode = "ensemble"
        elif acquisition.strategy == "mc_dropout":
            uncertainty_mode = "mc_dropout"

    pool_context = nullcontext(None) if md_pool_factory is None else md_pool_factory()
    with pool_context as pool:
        for strategy_index, strategy in enumerate(strategies):
            if isinstance(strategy, MDSamplingStrategy):
                md_dir = root / "md"
                if md_strategy_count > 1:
                    md_dir /= f"strategy-{strategy_index:03d}"
                md_candidates = (md_sampler or sample_md_candidates)(
                    strategy.parameters,
                    primary_calculator,
                    md_dir,
                    seed_paths=(
                        None
                        if seed_paths_by_strategy is None
                        else seed_paths_by_strategy.get(strategy_index)
                    ),
                    random_seed=stream_seed(random_seed, "md", iteration, strategy_index),
                    mlip_pool=pool,
                    mlip_cfg=mlip_config,
                    enable_mc_dropout=enable_dropout,
                )
                if md_strategy_count > 1:
                    prefix = f"strategy-{strategy_index:03d}"
                    for candidate in md_candidates:
                        candidate.candidate_id = f"{prefix}_{candidate.candidate_id}"
                        if candidate.sampling_run_id is not None:
                            candidate.sampling_run_id = f"{prefix}/{candidate.sampling_run_id}"
                        candidate.atoms.info.update(
                            candidate_id=candidate.candidate_id,
                            sampling_method=candidate.sampling_method,
                            sampling_strategy_index=strategy_index,
                        )
                        if candidate.sampling_run_id is not None:
                            candidate.atoms.info["sampling_run_id"] = candidate.sampling_run_id
                for candidate in md_candidates:
                    candidate.provenance["sampling_strategy_index"] = strategy_index
                candidates_by_strategy[strategy_index] = md_candidates
            elif isinstance(strategy, NEBSamplingStrategy):
                neb_dir = root / "neb"
                if neb_strategy_count > 1:
                    neb_dir /= f"strategy-{strategy_index:03d}"
                neb_requests.append((strategy_index, strategy.parameters, neb_dir))
            else:
                raise TypeError(f"unsupported sampling strategy {type(strategy).__name__}")

        if len(neb_requests) > 1 and pool is not None and getattr(pool, "uses_workers", False):
            from matsim_agents.workflows.active_learning.neb_sampler import (
                sample_neb_candidates_parallel,
            )

            tasks = [
                NebSamplingTask(
                    config_json=parameters.model_dump_json(),
                    mlip_config_json=mlip_config.model_dump_json(),
                    out_dir=str(directory),
                    transition_label=(
                        parameters.transition_id
                        or (
                            parameters.transition.transition_id
                            if parameters.transition is not None
                            else f"{parameters.initial_path.stem}-to-{parameters.final_path.stem}"
                        )
                    ),
                    iteration=iteration,
                    model_id=mlip_config.model_identifier(),
                    enable_mc_dropout=enable_dropout,
                    uncertainty_mode=uncertainty_mode,
                    dropout_passes=0 if acquisition is None else acquisition.mc_dropout_passes,
                    dropout_probability=0.1 if acquisition is None else acquisition.mc_dropout_p,
                    uncertainty_seed=stream_seed(
                        random_seed, "neb-pathway-uq", iteration, strategy_index
                    ),
                )
                for strategy_index, parameters, directory in neb_requests
            ]
            outcomes = sample_neb_candidates_parallel(tasks, pool)
            for (strategy_index, parameters, directory), outcome in zip(
                neb_requests, outcomes, strict=True
            ):
                pathways_by_strategy[strategy_index] = SampledNebPathway(
                    strategy_index=strategy_index,
                    parameters=parameters,
                    directory=directory,
                    outcome=outcome,
                )
                candidates_by_strategy[strategy_index] = outcome.candidates
        else:
            from matsim_agents.workflows.active_learning.neb_sampler import sample_neb_candidates

            for strategy_index, parameters, directory in neb_requests:
                outcome = sample_neb_candidates(
                    parameters,
                    primary_calculator,
                    directory,
                    iteration=iteration,
                    model_id=mlip_config.model_identifier(),
                    mlip_config=mlip_config,
                    uncertainty_calculators=ensemble_calculators,
                    uncertainty_mode=uncertainty_mode,
                    dropout_passes=0 if acquisition is None else acquisition.mc_dropout_passes,
                    dropout_probability=0.1 if acquisition is None else acquisition.mc_dropout_p,
                    uncertainty_seed=stream_seed(
                        random_seed, "neb-pathway-uq", iteration, strategy_index
                    ),
                )
                pathways_by_strategy[strategy_index] = SampledNebPathway(
                    strategy_index=strategy_index,
                    parameters=parameters,
                    directory=directory,
                    outcome=outcome,
                )
                candidates_by_strategy[strategy_index] = outcome.candidates

    candidates = [
        candidate
        for strategy_index in range(len(strategies))
        for candidate in candidates_by_strategy.get(strategy_index, [])
    ]
    pathways = [
        pathways_by_strategy[strategy_index] for strategy_index in sorted(pathways_by_strategy)
    ]
    return SamplingBatch(candidates=candidates, neb_pathways=pathways)


__all__ = ["SampledNebPathway", "SamplingBatch", "sample_strategies"]
