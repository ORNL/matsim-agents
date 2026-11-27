"""MLIP-only MD/NEB sampling through the standardized run interface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ase.io import write as ase_write

from matsim_agents.backends.mlip.pool import driver_on_mlip_device
from matsim_agents.core.seeding import stream_seed
from matsim_agents.workflows.active_learning.config import (
    MDSamplingStrategy,
    SamplingRunConfig,
)
from matsim_agents.workflows.active_learning.sampling import sample_strategies
from matsim_agents.workflows.active_learning.seed_sources import resolve_seed_structures


def _score_md_candidates(candidates, acquisition, primary, ensemble, random_seed: int) -> None:
    """Attach configuration uncertainty to MD candidates (NEB scores pathways itself)."""
    import numpy as np

    from matsim_agents.workflows.active_learning.uncertainty import (
        score_ensemble,
        score_mc_dropout,
    )

    md = [
        candidate
        for candidate in candidates
        if candidate.sampling_method == "md" and candidate.uncertainty is None
    ]
    if not md or acquisition.strategy == "random":
        return
    if acquisition.strategy == "mc_dropout":
        scores = score_mc_dropout(
            md,
            primary,
            acquisition.mc_dropout_passes,
            acquisition.mc_dropout_p,
            seed=stream_seed(random_seed, "md-uq"),
        )
    else:
        if not ensemble or len(ensemble) < 2:
            raise ValueError(f"{acquisition.strategy} requires >=2 ensemble calculators")
        scores = score_ensemble(md, ensemble)
        if acquisition.strategy == "ensemble_then_dropout":
            # Same screen as uncertainty.select_candidates; unrefined candidates get no value.
            keep_n = min(len(md), max(acquisition.n_select * 4, acquisition.n_select))
            top = np.argsort(scores)[-keep_n:]
            refined = score_mc_dropout(
                [md[i] for i in top],
                primary,
                acquisition.mc_dropout_passes,
                acquisition.mc_dropout_p,
                seed=stream_seed(random_seed, "md-uq"),
            )
            scores = np.full(len(md), np.nan, dtype=np.float64)
            scores[top] = refined
    for candidate, score in zip(md, scores, strict=True):
        candidate.uncertainty = float(score) if np.isfinite(score) else None


def run_sampling(config: SamplingRunConfig, directory: str | Path) -> dict[str, Any]:
    """Run configured samplers without DFT labelling, acquisition, or retraining."""
    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    placement = config.resources.place(root)
    from matsim_agents.workflows.active_learning.surrogate import _SurrogateModels

    models = _SurrogateModels(placement, keep=True)
    try:
        with driver_on_mlip_device(placement):
            acquisition = config.acquisition
            enable_dropout = acquisition is not None and acquisition.strategy in {
                "mc_dropout",
                "ensemble_then_dropout",
            }
            primary, ensemble = models.calculators(config.mlip, enable_dropout)
            md_strategies = [
                (index, strategy)
                for index, strategy in enumerate(config.sampling.strategies)
                if isinstance(strategy, MDSamplingStrategy)
            ]
            seed_paths_by_strategy = {}
            for strategy_index, strategy in md_strategies:
                seeds_dir = root / "seeds"
                if len(md_strategies) > 1:
                    seeds_dir /= f"strategy-{strategy_index:03d}"
                seed_stream = (
                    stream_seed(config.random_seed, "seed-source")
                    if len(md_strategies) == 1
                    else stream_seed(config.random_seed, "seed-source", strategy_index)
                )
                seed_paths_by_strategy[strategy_index] = resolve_seed_structures(
                    strategy.parameters.seed_source,
                    seeds_dir,
                    random_seed=seed_stream,
                )
            batch = sample_strategies(
                config.sampling.strategies,
                primary,
                root,
                random_seed=config.random_seed,
                iteration=0,
                seed_paths_by_strategy=seed_paths_by_strategy,
                mlip_config=config.mlip,
                acquisition=acquisition,
                ensemble_calculators=ensemble,
                md_pool_factory=models.md_pool,
            )
            if acquisition is not None:
                _score_md_candidates(
                    batch.candidates, acquisition, primary, ensemble, config.random_seed
                )
    finally:
        models.close()

    if batch.candidates:
        frames = []
        for candidate in batch.candidates:
            atoms = candidate.atoms.copy()
            atoms.info.update(
                candidate_id=candidate.candidate_id,
                sampling_method=candidate.sampling_method,
            )
            if candidate.model_id is not None:
                atoms.info["model_id"] = candidate.model_id
            if candidate.uncertainty is not None:
                atoms.info["uncertainty"] = candidate.uncertainty
            frames.append(atoms)
        ase_write(root / "candidates.extxyz", frames, format="extxyz")

    manifest = {
        "state": "complete",
        "n_candidates": len(batch.candidates),
        "candidate_file": (str(root / "candidates.extxyz") if batch.candidates else None),
        "candidates": [
            {
                "candidate_id": item.candidate_id,
                "sampling_method": item.sampling_method,
                "geometry_fingerprint": item.geometry_fingerprint,
                "model_id": item.model_id,
                "uncertainty": item.uncertainty,
                "provenance": item.provenance,
                "metadata": item.metadata,
            }
            for item in batch.candidates
        ],
        "neb_pathways": [
            {
                "directory": str(pathway.directory),
                "result": pathway.outcome.result.model_dump(mode="json"),
            }
            for pathway in batch.neb_pathways
        ],
    }
    (root / "sampling-results.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


__all__ = ["run_sampling"]
