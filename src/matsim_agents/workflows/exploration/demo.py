"""Synthetic scalability demonstrations.

Everything here uses a synthetic evaluator, so results measure software behaviour (throughput,
recovery, memory, sampling mechanics) and are never scientific evidence. Records produced by
these helpers carry ``label_source="synthetic"``.
"""

from __future__ import annotations

import time
import tracemalloc
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import bulk

from matsim_agents.core.seeding import numpy_rng
from matsim_agents.workflows.exploration.candidates import CandidateManifest, CandidateRecord
from matsim_agents.workflows.exploration.magnetic_surrogate import (
    LabeledExample,
    StratumMeanSurrogate,
    select_for_labeling,
)
from matsim_agents.workflows.exploration.sampling import sample_random

Evaluator = Callable[[CandidateRecord, Atoms], dict]


def synthetic_structure(n_cr: int, seed: int) -> Atoms:
    """Fe-Cr bcc 2x2x2 supercell (16 atoms) with ``n_cr`` randomly placed Cr atoms."""
    atoms = bulk("Fe", "bcc", a=2.87, cubic=True) * (2, 2, 2)
    sites = numpy_rng(seed).choice(len(atoms), size=n_cr, replace=False)
    symbols = atoms.get_chemical_symbols()
    for i in sites:
        symbols[int(i)] = "Cr"
    atoms.set_chemical_symbols(symbols)
    atoms.set_array("atom_id", np.arange(len(atoms), dtype=np.int64))
    return atoms


def make_candidate(index: int, seed: int) -> tuple[CandidateRecord, Atoms]:
    rng = numpy_rng(seed, index)
    n_cr = int(rng.integers(1, 9))
    atoms = synthetic_structure(n_cr, int(rng.integers(0, 2**31 - 1)))
    record = CandidateRecord.from_atoms(
        atoms,
        generation_method="synthetic_fecr",
        recipe={"index": index},
        seed=seed,
        stratum=f"Cr{n_cr}",
        metadata={"index": index},
    )
    return record, atoms


def synthetic_energy(record: CandidateRecord, atoms: Atoms) -> dict:
    """Deterministic fake 'energy' from composition and Cr clustering."""
    cr = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s == "Cr"]
    pair = 0.0
    if len(cr) > 1:
        d = atoms.get_distances(cr[0], cr[1:], mic=True)
        pair = float(np.sum(1.0 / d))
    return {"energy_eV": -0.05 * len(cr) + 0.02 * pair, "label_source": "synthetic"}


@dataclass(frozen=True)
class RunStats:
    n_total: int
    added: int
    duplicates: int
    counts: dict[str, int]
    seconds: float
    peak_mib: float
    evaluations: dict[str, int]


def ingest_synthetic_candidates(manifest: CandidateManifest, n: int, seed: int) -> tuple[int, int]:
    added = duplicates = 0
    for index in range(n):
        record, _ = make_candidate(index, seed)
        if manifest.add(record):
            added += 1
        else:
            duplicates += 1
    return added, duplicates


def evaluate_pending(
    manifest: CandidateManifest,
    seed: int,
    *,
    evaluator: Evaluator = synthetic_energy,
    crash_after: int | None = None,
    calls: dict[str, int] | None = None,
) -> int:
    """Evaluate pending candidates, regenerating each structure from its recipe. With
    ``crash_after`` the run stops abruptly after marking one more candidate ``running`` (a
    simulated kill). Returns the number completed in this call."""
    calls = calls if calls is not None else {}
    done = 0
    for record in manifest.pending():
        if crash_after is not None and done >= crash_after:
            manifest.set_status(record.candidate_id, "running")
            return done
        index = _index_of(record)
        regenerated, atoms = make_candidate(index, seed)
        if regenerated.candidate_id != record.candidate_id:
            manifest.set_status(record.candidate_id, "failed", {"error": "identity mismatch"})
            continue
        manifest.set_status(record.candidate_id, "running")
        calls[record.candidate_id] = calls.get(record.candidate_id, 0) + 1
        manifest.set_status(record.candidate_id, "done", {"result": evaluator(record, atoms)})
        done += 1
    return done


def _index_of(record: CandidateRecord) -> int:
    index = record.metadata.get("index")
    if index is None:
        raise ValueError("candidate lacks a generation index")
    return int(index)


def run_scale_demo(path: str | Path, n: int, seed: int, *, crash_fraction: float = 0.5) -> RunStats:
    """Ingest ``n`` candidates, kill the run partway, reopen, recover and finish. Measures
    wall time and the Python-heap peak (tracemalloc)."""
    manifest_path = Path(path)
    calls: dict[str, int] = {}
    tracemalloc.start()
    started = time.perf_counter()
    manifest = CandidateManifest(manifest_path)
    added, duplicates = ingest_synthetic_candidates(manifest, n, seed)
    evaluate_pending(manifest, seed, crash_after=int(crash_fraction * added), calls=calls)
    resumed = CandidateManifest(manifest_path)  # fresh process view of the on-disk log
    resumed.recover_interrupted()
    evaluate_pending(resumed, seed, calls=calls)
    seconds = time.perf_counter() - started
    peak = tracemalloc.get_traced_memory()[1] / 2**20
    tracemalloc.stop()
    return RunStats(n, added, duplicates, resumed.counts(), seconds, peak, calls)


@dataclass(frozen=True)
class StrategyResult:
    strategy: str
    seed: int
    budget: int
    mean_abs_error: float
    max_abs_error: float


def _landscape(n_strata: int, per_stratum: int, seed: int):
    """Strata with known true means and very different noise levels (heteroscedastic)."""
    rng = numpy_rng(seed)
    means = rng.normal(0.0, 1.0, n_strata)
    sigmas = np.geomspace(0.05, 2.0, n_strata)
    pool, truth = {}, {}
    for s in range(n_strata):
        for j in range(per_stratum):
            pool[f"s{s}-{j:03d}"] = f"S{s}"
        truth[f"S{s}"] = (float(means[s]), float(sigmas[s]))
    return pool, truth


def _stable_label(
    cid: str, stratum: str, truth: Mapping[str, tuple[float, float]], seed: int
) -> float:
    # hash() is salted per process; use a stable integer derived from the id instead
    digest = sum((i + 1) * ord(c) for i, c in enumerate(cid))
    mean, sigma = truth[stratum]
    rng = numpy_rng(seed, digest)
    return float(mean + sigma * rng.standard_normal())


def _error(model: StratumMeanSurrogate, truth) -> tuple[float, float]:
    errors = [abs(model.predict("x", s).mean - truth[s][0]) for s in truth]
    return float(np.mean(errors)), float(np.max(errors))


def compare_strategies(
    *,
    seed: int,
    budget: int = 80,
    initial: int = 16,
    batch: int = 8,
    n_strata: int = 8,
    per_stratum: int = 200,
) -> list[StrategyResult]:
    """Random vs uncertainty-driven selection on a synthetic landscape with known truth, scored
    by the error of the estimated stratum means."""
    pool, truth = _landscape(n_strata, per_stratum, seed)
    results = []
    for strategy in ("random", "adaptive"):
        labeled = {
            s.candidate_id for s in sample_random(list(pool), initial, seed=seed, round_index=0)
        }
        round_index = 1
        while len(labeled) < budget:
            examples = [
                LabeledExample(c, pool[c], _stable_label(c, pool[c], truth, seed), c)
                for c in sorted(labeled)
            ]
            remaining = {c: s for c, s in pool.items() if c not in labeled}
            k = min(batch, budget - len(labeled))
            if strategy == "random":
                picks = sample_random(list(remaining), k, seed=seed, round_index=round_index)
            else:
                model = StratumMeanSurrogate()
                model.fit(examples)
                picks = select_for_labeling(model, remaining, k, seed=seed, round_index=round_index)
            labeled |= {p.candidate_id for p in picks}
            round_index += 1
        final = StratumMeanSurrogate()
        final.fit(
            [
                LabeledExample(c, pool[c], _stable_label(c, pool[c], truth, seed), c)
                for c in sorted(labeled)
            ]
        )
        mean_err, max_err = _error(final, truth)
        results.append(StrategyResult(strategy, seed, len(labeled), mean_err, max_err))
    return results
