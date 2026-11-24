"""Interaction coverage of a source dataset against a target alloy (descriptor level only).

Coverage here means "which local interactions were observed", never "transfer works"; that needs
the held-out experiments in ``campaign.transfer``.
"""

from __future__ import annotations

import itertools
from collections import Counter
from collections.abc import Iterable, Mapping

import numpy as np
from ase import Atoms
from pydantic import BaseModel, Field

from matsim_agents.campaign.registry import TARGET_ELEMENTS, CandidateRegistry, chemical_system
from matsim_agents.discovery.alloys import (
    first_neighbor_list,
    warren_cowley_sro,
)

PRIORITY_PAIRS: tuple[tuple[str, str], ...] = (
    ("Ta", "W"),
    ("Cr", "W"),
    ("V", "W"),
    ("Hf", "W"),
    ("Mo", "W"),
    ("Ta", "V"),
    ("Cr", "V"),
    ("Cr", "Mo"),
    ("Cr", "Ta"),
)
SRO_REGIME_EDGES = (0.05, 0.2)


def _pair_key(shell: int, a: str, b: str) -> str:
    return f"pair{shell}:{'-'.join(sorted((a, b)))}"


def cluster_key(elements: Iterable[str]) -> str:
    return "cluster:" + "-".join(sorted(set(elements)))


def key_order(key: str) -> int:
    kind, _, body = key.partition(":")
    return 2 if kind.startswith("pair") else len(body.split("-"))


class StructureDescriptor(BaseModel):
    composition: dict[str, float]
    pair_frequency: dict[str, dict[str, float]]
    warren_cowley: dict[str, dict[str, float]]
    local_composition: dict[str, dict[str, float]]
    motifs: dict[str, int]
    distortion: float
    sro_regime: str
    interactions: list[str]
    defect_interactions: list[str] = Field(default_factory=list)
    magnetic_environments: dict[str, int] = Field(default_factory=dict)


def sro_regime(alpha_by_shell: Mapping[str, Mapping[str, float]]) -> str:
    strength = max((abs(v) for shell in alpha_by_shell.values() for v in shell.values()), default=0)
    lo, hi = SRO_REGIME_EDGES
    return "random" if strength < lo else "weak" if strength < hi else "strong"


def _neighbours(n: int, i: np.ndarray, j: np.ndarray) -> list[list[int]]:
    groups: list[list[int]] = [[] for _ in range(n)]
    for a, b in zip(i.tolist(), j.tolist(), strict=True):
        groups[a].append(b)
    return groups


def structure_descriptor(
    atoms: Atoms,
    lattice_parameter: float,
    *,
    magnetic_moments: np.ndarray | None = None,
    defect_site_ids: Iterable[int] = (),
) -> StructureDescriptor:
    """Pair, SRO, local-composition, motif, distortion, defect and magnetic descriptors."""
    symbols = atoms.get_chemical_symbols()
    species = tuple(sorted(set(symbols)))
    n = len(atoms)
    sro = warren_cowley_sro(atoms, lattice_parameter, species)
    shells = {"1": sro.first_shell, "2": sro.second_shell}
    pair_frequency: dict[str, dict[str, float]] = {}
    alpha: dict[str, dict[str, float]] = {}
    interactions: set[str] = set()
    for shell, data in shells.items():
        counts = data.pair_counts
        total = counts.sum()
        pair_frequency[shell], alpha[shell] = {}, {}
        for (ia, a), (ib, b) in itertools.combinations_with_replacement(enumerate(species), 2):
            if counts[ia, ib] > 0:
                key = "-".join(sorted((a, b)))
                pair_frequency[shell][key] = float(counts[ia, ib] * (1 if a == b else 2) / total)
                interactions.add(_pair_key(int(shell), a, b))
            alpha[shell]["-".join(sorted((a, b)))] = float(data.alpha[ia, ib])
    local = {
        a: {
            b: float(p)
            for b, p in zip(species, sro.first_shell.conditional_probability[ia], strict=True)
        }
        for ia, a in enumerate(species)
    }

    i1, j1, v1 = first_neighbor_list(atoms, lattice_parameter)
    first = _neighbours(n, i1, j1)
    motifs: Counter[str] = Counter()
    site_cluster: list[set[str]] = []
    for site in range(n):
        around = Counter(symbols[k] for k in first[site])
        motifs[f"{symbols[site]}:" + "".join(f"{e}{c}" for e, c in sorted(around.items()))] += 1
        elems = {symbols[site], *around}
        site_cluster.append(elems)
        interactions.add(cluster_key(elems))
        for r in range(2, len(elems)):
            interactions.update(cluster_key(c) for c in itertools.combinations(sorted(elems), r))
    distances = np.linalg.norm(v1, axis=1)
    distortion = float(np.std(distances) / np.mean(distances)) if len(distances) else 0.0

    defect: set[str] = set()
    for site in defect_site_ids:
        defect.add(cluster_key(site_cluster[site]))
    magnetic: Counter[str] = Counter()
    if magnetic_moments is not None:
        moments = np.round(np.asarray(magnetic_moments, dtype=float), 1)
        for site in range(n):
            n_mag = sum(abs(moments[k]) > 0 for k in first[site])
            magnetic[f"{symbols[site]}@{moments[site]:+.1f}|nbr_mag={n_mag}"] += 1

    return StructureDescriptor(
        composition={s: symbols.count(s) / n for s in species},
        pair_frequency=pair_frequency,
        warren_cowley=alpha,
        local_composition=local,
        motifs=dict(motifs),
        distortion=distortion,
        sro_regime=sro_regime(alpha),
        interactions=sorted(interactions),
        defect_interactions=sorted(defect),
        magnetic_environments=dict(magnetic),
    )


def required_interactions(elements: Iterable[str] = TARGET_ELEMENTS) -> set[str]:
    """Every pair interaction (both shells) and every local-cluster element set of order >= 2."""
    ordered = sorted(set(elements))
    required = {
        _pair_key(shell, a, b)
        for shell in (1, 2)
        for a, b in itertools.combinations_with_replacement(ordered, 2)
    }
    for r in range(2, len(ordered) + 1):
        required.update(cluster_key(c) for c in itertools.combinations(ordered, r))
    return required


def _covered(required: set[str], sources: Iterable[StructureDescriptor]) -> set[str]:
    """Pairs need an exact observation; a cluster is covered by any observed superset cluster."""
    observed: set[str] = set()
    clusters: list[frozenset[str]] = []
    for src in sources:
        for key in src.interactions:
            observed.add(key)
            if key.startswith("cluster:"):
                clusters.append(frozenset(key[8:].split("-")))
    covered = {k for k in required if k.startswith("pair") and k in observed}
    for key in required:
        if key.startswith("cluster:"):
            need = frozenset(key[8:].split("-"))
            if any(need <= c for c in clusters):
                covered.add(key)
    return covered


class CoverageReport(BaseModel):
    n_required: int
    n_covered: int
    fraction: float
    covered: list[str]
    missing: list[str]
    fraction_by_order: dict[int, float]
    missing_higher_order: list[str]


def coverage_report(
    sources: Iterable[StructureDescriptor], required: set[str] | None = None
) -> CoverageReport:
    """Covered and explicitly missing interactions; higher-order gaps are listed, never implied."""
    required = required if required is not None else required_interactions()
    covered = _covered(required, list(sources))
    by_order: dict[int, list[bool]] = {}
    for key in required:
        by_order.setdefault(key_order(key), []).append(key in covered)
    missing = sorted(required - covered)
    return CoverageReport(
        n_required=len(required),
        n_covered=len(covered),
        fraction=len(covered) / len(required) if required else 1.0,
        covered=sorted(covered),
        missing=missing,
        fraction_by_order={k: sum(v) / len(v) for k, v in sorted(by_order.items())},
        missing_higher_order=[k for k in missing if key_order(k) >= 3],
    )


class CoverageGain(BaseModel):
    candidate_id: str
    newly_covered: list[str]
    gain: int


def rank_by_coverage_gain(
    candidates: Mapping[str, StructureDescriptor],
    sources: Iterable[StructureDescriptor] = (),
    required: set[str] | None = None,
) -> list[CoverageGain]:
    """Candidates ordered by newly covered required interactions; ties break on candidate id."""
    required = required if required is not None else required_interactions()
    sources = list(sources)
    base = _covered(required, sources)
    ranked = []
    for cid in sorted(candidates):
        new = sorted(_covered(required, [*sources, candidates[cid]]) - base)
        ranked.append(CoverageGain(candidate_id=cid, newly_covered=new, gain=len(new)))
    return sorted(ranked, key=lambda g: (-g.gain, g.candidate_id))


def pair_priority(registry: CandidateRegistry) -> list[tuple[str, int]]:
    """Priority binaries ordered by fewest existing records (least covered first)."""
    counts = registry.target_coverage()
    return sorted(
        ((chemical_system(p), counts.get(chemical_system(p), 0)) for p in PRIORITY_PAIRS),
        key=lambda item: (item[1], item[0]),
    )
