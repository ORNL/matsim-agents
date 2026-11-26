"""Turn cascade snapshots into active-learning ``Candidate`` objects (WP1.6).

Snapshots are selected from the cascade trajectory, tagged with provenance, the boundary
assumptions and a short-distance out-of-distribution diagnostic. The diagnostic compares the
smallest pair distance with the smallest distance in the training data that the caller supplies;
without it nothing is claimed in-distribution.
"""

from __future__ import annotations

from typing import Any

from ase import Atoms
from ase.neighborlist import neighbor_list

from matsim_agents.core.hashing import sha256_json
from matsim_agents.science.cascade.protocol import CascadeResult, Snapshot
from matsim_agents.science.defects.point_defects import geometry_fingerprint
from matsim_agents.workflows.active_learning.candidates import Candidate


def min_pair_distance(atoms: Atoms, cutoff: float = 6.0) -> float:
    d = neighbor_list("d", atoms, cutoff)
    return float(d.min()) if len(d) else float("inf")


def short_distance_diagnostic(
    atoms: Atoms, training_min_distance_A: float | None, margin: float = 0.95
) -> dict[str, Any]:
    d = min_pair_distance(atoms)
    if training_min_distance_A is None:
        return {
            "min_pair_distance_A": d,
            "ood_short_distance": None,
            "note": "no training minimum distance given: in-distribution status unknown",
        }
    return {
        "min_pair_distance_A": d,
        "training_min_distance_A": training_min_distance_A,
        "ood_short_distance": bool(d < margin * training_min_distance_A),
    }


def select_snapshots(
    snaps: tuple[Snapshot, ...], max_candidates: int, *, force_floor: float = 0.0
) -> list[Snapshot]:
    """Pick the highest-force snapshots (above ``force_floor``), returned in time order."""
    if max_candidates < 1:
        raise ValueError("max_candidates must be at least 1")
    pool = [s for s in snaps if s.max_force_eV_per_A >= force_floor]
    top = sorted(pool, key=lambda s: -s.max_force_eV_per_A)[:max_candidates]
    order = {id(s): i for i, s in enumerate(snaps)}
    return sorted(top, key=lambda s: order[id(s)])


def cascade_candidates(
    result: CascadeResult,
    *,
    max_candidates: int,
    training_min_distance_A: float | None,
    boundary_assumptions: dict[str, Any],
    model_id: str | None = None,
    sampling_run_id: str | None = None,
    force_floor: float = 0.0,
) -> list[Candidate]:
    if not result.snapshots:
        raise ValueError("the cascade result has no snapshots; run with snapshot_every_fs")
    if not boundary_assumptions:
        raise ValueError(
            "boundary_assumptions are required: labels depend on the boundary treatment"
        )
    out: list[Candidate] = []
    for snap in select_snapshots(result.snapshots, max_candidates, force_floor=force_floor):
        diag = short_distance_diagnostic(snap.atoms, training_min_distance_A)
        cid = (
            "cascade-"
            + sha256_json([result.event.event_id, snap.phase, round(snap.time_fs, 6)])[:16]
        )
        out.append(
            Candidate(
                candidate_id=cid,
                atoms=snap.atoms.copy(),
                predicted_energy_eV=snap.energy_eV,
                predicted_max_force_eV_per_A=snap.max_force_eV_per_A,
                sampling_method="cascade",
                parent_id=result.event.event_id,
                sampling_run_id=sampling_run_id,
                geometry_fingerprint=geometry_fingerprint(snap.atoms),
                model_id=model_id,
                provenance={
                    "event_id": result.event.event_id,
                    "seed": result.event.seed,
                    "protocol_hash": result.provenance.get("protocol_hash"),
                    "initial_structure_hash": result.provenance.get("initial_structure_hash"),
                    "stopping_source": result.provenance.get("stopping_source"),
                },
                metadata={
                    "phase": snap.phase,
                    "time_in_phase_fs": snap.time_fs,
                    "recoil_energy_eV": result.event.energy_eV,
                    "boundary_assumptions": dict(boundary_assumptions),
                    "numerically_stable": result.numerically_stable,
                    "qualification": result.qualification.level.name
                    if result.qualification
                    else "UNASSESSED",
                    **diag,
                },
            )
        )
    return out
