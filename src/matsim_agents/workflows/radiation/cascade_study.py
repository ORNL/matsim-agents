"""Planning, execution and summary of verification cascade runs (threshold scans, size, alloy).

Each task is one cascade on one cell with one imposed PKA; results are plain JSON files so the
tasks can be spread over processes and restarted. The PKA energy is imposed by hand, so the
events carry no nuclear data and every result is unqualified: a count from this module is a
measurement of the supplied potential under the stated numerics, nothing more.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.calculators.calculator import Calculator
from ase.optimize import LBFGS

from matsim_agents.core.seeding import numpy_rng, stream_seed
from matsim_agents.science.cascade.boundaries import assess_cascade_boundaries
from matsim_agents.science.cascade.events import CascadeEvent, structure_hash
from matsim_agents.science.cascade.protocol import (
    analyze_cascade,
    default_phases,
    run_protocol,
    thermalize,
)
from matsim_agents.science.cascade.recoil import (
    crystal_direction,
    cubic_family,
    isotropic_direction,
    sample_family_direction,
)
from matsim_agents.science.irradiation.contracts import RecoilSpecies

PHASE_DURATIONS_FS = {"threshold": (150.0, 500.0, 1000.0), "cascade": (200.0, 2000.0, 5000.0)}
SHELL_FRACTION = 0.05


def build_cell(element: str, a: float, n: int, counts: dict[str, int] | None, seed: int) -> Atoms:
    """Cubic ``n^3`` cell of a bcc/fcc element; a random substitutional alloy if ``counts``."""
    structure = {"Fe": "bcc", "W": "bcc", "Cr": "bcc", "Cu": "fcc", "Ni": "fcc", "Al": "fcc"}
    atoms = bulk(element, structure[element], a=a, cubic=True).repeat((n, n, n))
    if counts:
        if sum(counts.values()) != len(atoms):
            raise ValueError(f"counts sum to {sum(counts.values())}, cell has {len(atoms)} sites")
        symbols = [s for s, c in counts.items() for _ in range(c)]
        numpy_rng(seed, "alloy").shuffle(symbols)
        atoms.set_chemical_symbols(symbols)
    return atoms


def shell_mask(atoms: Atoms, fraction: float = SHELL_FRACTION) -> np.ndarray:
    """Atoms within ``fraction`` of the cell faces on any axis (the thermostatted layer)."""
    frac = atoms.get_scaled_positions(wrap=True)
    return np.any((frac < fraction) | (frac > 1.0 - fraction), axis=1)


def choose_pka(atoms: Atoms, symbol: str, rng: np.random.Generator) -> int:
    """Random atom of ``symbol`` in the central 40% of the cell on every axis."""
    frac = atoms.get_scaled_positions(wrap=True)
    central = np.all(np.abs(frac - 0.5) < 0.2, axis=1)
    pool = np.flatnonzero(central & (np.asarray(atoms.get_chemical_symbols()) == symbol))
    if not len(pool):
        raise ValueError(f"no {symbol} atom in the central region")
    return int(rng.choice(pool))


def direction_vector(atoms: Atoms, label: str, rng: np.random.Generator) -> np.ndarray:
    if label == "iso":
        return isotropic_direction(rng)
    family = cubic_family(tuple(int(c) for c in label))
    return sample_family_direction(rng, atoms, family)


def relax_positions(atoms: Atoms, fmax: float = 0.02, steps: int = 200) -> float:
    opt = LBFGS(atoms, logfile=None)
    opt.run(fmax=fmax, steps=steps)
    return float(np.max(np.linalg.norm(atoms.get_forces(), axis=1)))


def run_task(task: dict, calculators: dict[str, Calculator], out_dir: Path) -> dict:
    """Run one task, write ``task-<id>.json`` and return the record."""
    path = out_dir / f"task-{task['id']:05d}.json"
    if path.exists():
        return json.loads(path.read_text())
    t0 = time.time()
    seed = int(task["seed"])
    rng = numpy_rng(seed, "task")
    atoms = build_cell(task["element"], task["a"], task["n_cells"], task.get("counts"), seed)
    atoms.calc = calculators[task["short_range"]]
    residual = relax_positions(atoms)
    atoms.set_velocities(np.zeros((len(atoms), 3)))
    thermalize(atoms, task["temperature_K"], seed)
    reference = atoms.copy()
    reference.set_velocities(np.zeros((len(atoms), 3)))
    pka = choose_pka(atoms, task["pka_symbol"], rng)
    direction = direction_vector(atoms, task["direction"], rng)
    event = CascadeEvent(
        event_id=f"study-{task['id']}",
        parent_material_id=task["label"],
        parent_structure_hash=structure_hash(atoms),
        kind="displacement",
        recoil_species=RecoilSpecies(element=task["pka_symbol"]),
        parent_species=None,
        site_index=pka,
        energy_eV=float(task["energy_eV"]),
        scenario_id=task["study"],
        channel="elastic",
        seed=seed,
        nuclear_data_provenance="imposed PKA energy for a verification run; no nuclear data",
        rate_or_probability=1.0,
        rate_or_probability_basis="imposed, not a physical rate",
        direction_source="user",
        direction=tuple(float(x) for x in direction),
    )
    ball, spike, cool = PHASE_DURATIONS_FS[task["phase_set"]]
    result = run_protocol(
        atoms,
        event,
        direction=tuple(float(x) for x in direction),
        phases=default_phases(ball, spike, cool),
        boundary_mask=shell_mask(atoms),
        boundary_target_K=float(task["temperature_K"]),
        require_level=None,
        max_atom_speed=None,
    )
    cutoff = 1.05 * float(task["a"])
    analysis = analyze_cascade(
        result,
        reference,
        cluster_cutoff=cutoff,
        cluster_justification="second-neighbour distance of the cubic host, a modelling choice",
    )
    bounds = assess_cascade_boundaries(
        reference, result.final_atoms, pka_index=pka, boundary_mask=shell_mask(atoms)
    )
    s = analysis.summary
    record = {
        **task,
        "n_atoms": len(atoms),
        "pka_index": pka,
        "direction_vector": [float(x) for x in direction],
        "n_frenkel_pairs": s.n_frenkel_pairs,
        "n_vacancies": s.n_vacancies,
        "n_interstitials": s.n_interstitials,
        "n_antisites": s.n_antisites,
        "largest_vacancy_cluster": max((len(c) for c in s.vacancy_clusters), default=0),
        "largest_interstitial_cluster": max((len(c) for c in s.interstitial_clusters), default=0),
        "numerically_stable": bool(result.numerically_stable),
        "max_force_eV_per_A": result.max_force_eV_per_A,
        "max_abs_energy_drift_eV": max(abs(r.energy_drift_eV) for r in result.runs),
        "steps": int(sum(r.steps for r in result.runs)),
        "dt_min_used_fs": min(r.dt_min_used_fs for r in result.runs),
        "steps_at_dt_min": int(sum(r.steps_at_dt_min for r in result.runs)),
        "thermostat_loss_eV": result.energy_thermostat_eV,
        "initial_relax_max_force": residual,
        "boundary_ok": bounds.ok,
        "boundary_findings": list(bounds.findings),
        "damaged_extent_A": bounds.damaged_extent_A,
        "min_cell_length_A": bounds.min_cell_length_A,
        "caveats": list(analysis.caveats),
        "wall_s": time.time() - t0,
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record))
    tmp.replace(path)
    return record


def plan_threshold(
    element: str,
    a: float,
    energies_eV: list[float],
    directions: list[str],
    realizations: int,
    short_range: str,
    *,
    n_cells: int = 6,
    temperature_K: float = 50.0,
    base_seed: int,
    first_id: int = 0,
) -> list[dict]:
    tasks = []
    for d in directions:
        for e in energies_eV:
            for r in range(realizations):
                i = first_id + len(tasks)
                tasks.append(
                    {
                        "id": i,
                        "study": "threshold",
                        "label": f"{element}-bulk",
                        "element": element,
                        "a": a,
                        "n_cells": n_cells,
                        "counts": None,
                        "pka_symbol": element,
                        "energy_eV": e,
                        "direction": d,
                        "realization": r,
                        "short_range": short_range,
                        "temperature_K": temperature_K,
                        "phase_set": "threshold",
                        "seed": stream_seed(base_seed, "cascade-study", i),
                    }
                )
    return tasks


def plan_cascades(
    study: str,
    element: str,
    a: float,
    sizes: list[int],
    energy_eV: float,
    realizations: int,
    short_range: str,
    *,
    counts_for: Callable[[int], dict[str, int]] | None = None,
    pka_symbol: str | None = None,
    temperature_K: float = 300.0,
    base_seed: int,
    first_id: int = 0,
) -> list[dict]:
    tasks = []
    for n in sizes:
        for r in range(realizations):
            i = first_id + len(tasks)
            tasks.append(
                {
                    "id": i,
                    "study": study,
                    "label": f"{study}-{element}-{n}",
                    "element": element,
                    "a": a,
                    "n_cells": n,
                    "counts": counts_for(n) if counts_for else None,
                    "pka_symbol": pka_symbol or element,
                    "energy_eV": energy_eV,
                    "direction": "iso",
                    "realization": r,
                    "short_range": short_range,
                    "temperature_K": temperature_K,
                    "phase_set": "cascade",
                    "seed": stream_seed(base_seed, "cascade-study", i),
                }
            )
    return tasks


def displacement_probability(records: list[dict]) -> dict[float, tuple[float, int]]:
    """Fraction of runs per energy that left at least one Frenkel pair, and the run count."""
    by_e: dict[float, list[int]] = {}
    for r in records:
        by_e.setdefault(float(r["energy_eV"]), []).append(int(r["n_frenkel_pairs"] >= 1))
    return {e: (float(np.mean(v)), len(v)) for e, v in sorted(by_e.items())}


def threshold_estimates(prob: dict[float, tuple[float, int]]) -> dict:
    """Grid-resolved energy where P first reaches 0.5, and the mean of (1 - P) over the grid.

    The mean threshold ``integral (1 - P) dE`` equals the average threshold energy only if the
    grid starts below every threshold and ends above every one; both flags are returned.
    """
    energies = sorted(prob)
    p = [prob[e][0] for e in energies]
    first_half = next((e for e, v in zip(energies, p, strict=True) if v >= 0.5), None)
    mean = float(np.trapezoid([1.0 - v for v in p], energies)) + energies[0]
    return {
        "energy_at_p_half_eV": first_half,
        "mean_threshold_eV": mean,
        "grid_starts_below_all_thresholds": p[0] == 0.0,
        "grid_ends_above_all_thresholds": p[-1] == 1.0,
        "monotonic": all(b >= a - 1e-12 for a, b in zip(p, p[1:], strict=False)),
    }


def load_records(out_dir: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(out_dir.glob("task-*.json"))]


def mean_sem(values: list[float]) -> tuple[float, float]:
    v = np.asarray(values, dtype=float)
    sem = float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else float("nan")
    return float(v.mean()), sem


__all__ = [
    "build_cell",
    "choose_pka",
    "crystal_direction",
    "direction_vector",
    "displacement_probability",
    "load_records",
    "mean_sem",
    "plan_cascades",
    "plan_threshold",
    "relax_positions",
    "run_task",
    "shell_mask",
    "threshold_estimates",
]
