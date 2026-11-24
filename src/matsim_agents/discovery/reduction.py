"""Auditable redundancy reduction of a candidate pool (M13).

Order of operations: exact-hash duplicates, then translation and global-spin-reversal
equivalents, then optional per-stratum thinning. Candidates containing a rare local environment
are never removed by thinning. Every removal is recorded with its reason and representative, and
``kept + removed == input`` always holds.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list

from matsim_agents.discovery.candidates import CandidateRecord
from matsim_agents.discovery.magnetism import MagneticConfiguration

DECIMALS = 3


@dataclass(frozen=True)
class PoolItem:
    record: CandidateRecord
    atoms: Atoms
    magnetic: MagneticConfiguration | None = None


def _moments(item: PoolItem) -> np.ndarray:
    atoms, config = item.atoms, item.magnetic
    if config is None or not config.spin_polarized or config.initial_moments_muB is None:
        return np.zeros(len(atoms))
    by_id = dict(zip(config.atom_ids, config.initial_moments_muB, strict=True))
    ids = atoms.get_array("atom_id") if "atom_id" in atoms.arrays else range(len(atoms))
    return np.array([by_id.get(int(i), 0.0) for i in ids], dtype=float)


def symmetry_key(item: PoolItem, *, reverse_spin: bool) -> str:
    """Canonical hash minimised over lattice translations (and global spin reversal)."""
    atoms = item.atoms
    scaled = atoms.get_scaled_positions(wrap=True)
    symbols = atoms.get_chemical_symbols()
    moments = _moments(item)
    cell = np.round(atoms.cell.array, 6).tolist()
    signs = (1.0, -1.0) if reverse_spin else (1.0,)
    best: str | None = None
    for origin in scaled:
        shifted = np.round(scaled - origin, DECIMALS) % 1.0
        for sign in signs:
            rows = sorted(
                (s, *[f"{x:.{DECIMALS}f}" for x in row], f"{sign * m + 0.0:.{DECIMALS - 1}f}")
                for s, row, m in zip(symbols, shifted, moments, strict=True)
            )
            digest = hashlib.sha256(
                json.dumps({"c": cell, "r": rows}, separators=(",", ":")).encode()
            ).hexdigest()
            best = digest if best is None or digest < best else best
    assert best is not None
    return best


def local_environments(item: PoolItem, cutoff: float) -> frozenset[str]:
    """Per-atom environment strings, invariant to global spin reversal (relative alignment)."""
    atoms = item.atoms
    i_idx, j_idx = neighbor_list("ij", atoms, cutoff)
    symbols = atoms.get_chemical_symbols()
    moments = _moments(item)
    tags: list[list[str]] = [[] for _ in atoms]
    for i, j in zip(i_idx, j_idx, strict=True):
        product = moments[i] * moments[j]
        align = "=" if product > 1e-6 else "-" if product < -1e-6 else "0"
        tags[i].append(symbols[j] + align)
    return frozenset(f"{symbols[i]}|{','.join(sorted(tags[i]))}" for i in range(len(atoms)))


@dataclass(frozen=True)
class Removal:
    candidate_id: str
    reason: str
    representative: str


@dataclass(frozen=True)
class ReductionReport:
    n_input: int
    kept: tuple[str, ...]
    removed: tuple[Removal, ...]
    rare_retained: tuple[str, ...]

    @property
    def reasons(self) -> dict[str, int]:
        return dict(Counter(r.reason for r in self.removed))


def reduce_pool(
    items: Iterable[PoolItem],
    *,
    cutoff: float = 3.0,
    rare_max_count: int = 2,
    max_per_stratum: int | None = None,
) -> ReductionReport:
    """Deterministic: the result does not depend on input order."""
    pool = sorted(items, key=lambda it: it.record.candidate_id)
    removed: list[Removal] = []

    unique: dict[str, PoolItem] = {}
    for item in pool:
        cid = item.record.candidate_id
        if cid in unique:
            removed.append(Removal(cid, "exact_duplicate", cid))
        else:
            unique[cid] = item

    representative: dict[str, PoolItem] = {}
    survivors: list[PoolItem] = []
    for item in unique.values():
        key = symmetry_key(item, reverse_spin=True)
        rep = representative.get(key)
        if rep is None:
            representative[key] = item
            survivors.append(item)
            continue
        same_without_reversal = symmetry_key(item, reverse_spin=False) == symmetry_key(
            rep, reverse_spin=False
        )
        reason = "translation_equivalent" if same_without_reversal else "spin_reversal_equivalent"
        removed.append(Removal(item.record.candidate_id, reason, rep.record.candidate_id))

    environments = {it.record.candidate_id: local_environments(it, cutoff) for it in survivors}
    frequency = Counter(env for envs in environments.values() for env in envs)
    rare = {
        cid
        for cid, envs in environments.items()
        if any(frequency[env] <= rare_max_count for env in envs)
    }

    kept: list[PoolItem] = []
    if max_per_stratum is None:
        kept = survivors
    else:
        strata: dict[str, list[PoolItem]] = {}
        for item in survivors:
            strata.setdefault(item.record.stratum or item.record.composition, []).append(item)
        for _, members in sorted(strata.items()):
            protected = [m for m in members if m.record.candidate_id in rare]
            others = [m for m in members if m.record.candidate_id not in rare]
            room = max(max_per_stratum - len(protected), 0)
            chosen = others[:room]
            kept.extend(protected + chosen)
            for dropped in others[room:]:
                anchor = (chosen or protected)[0].record.candidate_id
                removed.append(Removal(dropped.record.candidate_id, "thinned", anchor))

    kept_ids = tuple(sorted(it.record.candidate_id for it in kept))
    return ReductionReport(
        n_input=len(pool),
        kept=kept_ids,
        removed=tuple(sorted(removed, key=lambda r: (r.reason, r.candidate_id))),
        rare_retained=tuple(sorted(rare & set(kept_ids))),
    )


def pool_from(
    atoms_list: Sequence[Atoms],
    records: Sequence[CandidateRecord],
    configs: Sequence[MagneticConfiguration | None] | None = None,
) -> list[PoolItem]:
    configs = configs or [None] * len(records)
    return [PoolItem(r, a, c) for r, a, c in zip(records, atoms_list, configs, strict=True)]
