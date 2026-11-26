"""Defect clustering and surviving-defect counts from a Wigner-Seitz analysis.

A cluster is a connected component of defects whose periodic separation is at most ``cutoff``.
The cutoff is a modelling choice, not an observable: it is stored on every result together with
the caller's justification, and :func:`cluster_sensitivity` shows how the cluster distribution
moves with it. Vacancies and interstitials are clustered separately, and species-aware
clustering only links defects of the same species.

The survival fraction needs an explicit reference population (for example the Frenkel pairs at
the thermal-spike peak of the same cascade); without one it is ``None``. It is not a DPA
estimate. Everything here is counting on a given structure, not a physical validation.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from ase import Atoms
from ase.geometry import find_mic
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from matsim_agents.science.defects.wigner_seitz import WignerSeitzResult


def cluster_defects(
    positions: np.ndarray,
    cell: np.ndarray,
    cutoff: float,
    labels: Sequence[str] | None = None,
) -> tuple[tuple[int, ...], ...]:
    """Connected components of points closer than ``cutoff`` (periodic), largest first.

    With ``labels``, only points of equal label are linked. Components are sorted by size
    (descending) then by smallest member index; members are ascending.
    """
    positions = np.asarray(positions, dtype=float).reshape(-1, 3)
    if not np.isfinite(cutoff) or cutoff <= 0:
        raise ValueError(f"cutoff must be positive and finite, got {cutoff}")
    n = len(positions)
    if labels is not None and len(labels) != n:
        raise ValueError("labels must have one entry per point")
    if n == 0:
        return ()
    cell = np.asarray(cell, dtype=float)
    vectors = positions[None, :, :] - positions[:, None, :]
    _, lengths = find_mic(vectors.reshape(-1, 3), cell, pbc=True)
    close = lengths.reshape(n, n) <= cutoff
    if labels is not None:
        names = np.asarray(labels)
        close &= names[:, None] == names[None, :]
    rows, cols = np.nonzero(close)
    graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    count, component = connected_components(graph, directed=False)
    groups = [tuple(int(i) for i in np.flatnonzero(component == c)) for c in range(count)]
    return tuple(sorted(groups, key=lambda g: (-len(g), g[0])))


def size_distribution(clusters: Sequence[Sequence[int]]) -> dict[int, int]:
    """Number of clusters of each size."""
    return dict(sorted(Counter(len(c) for c in clusters).items()))


@dataclass(frozen=True)
class ClusterSensitivity:
    cutoff: float
    n_clusters: int
    largest: int
    size_distribution: dict[int, int]


def cluster_sensitivity(
    positions: np.ndarray,
    cell: np.ndarray,
    cutoffs: Sequence[float],
    labels: Sequence[str] | None = None,
) -> tuple[ClusterSensitivity, ...]:
    """Cluster statistics at each cutoff, to show how arbitrary the choice is."""
    if len(cutoffs) == 0:
        raise ValueError("at least one cutoff is required")
    out = []
    for cutoff in cutoffs:
        clusters = cluster_defects(positions, cell, float(cutoff), labels)
        out.append(
            ClusterSensitivity(
                cutoff=float(cutoff),
                n_clusters=len(clusters),
                largest=max((len(c) for c in clusters), default=0),
                size_distribution=size_distribution(clusters),
            )
        )
    return tuple(out)


@dataclass(frozen=True)
class SurvivingDefectSummary:
    """Counts and clusters of the defects found by one Wigner-Seitz analysis."""

    n_vacancies: int
    n_interstitials: int
    n_frenkel_pairs: int
    n_antisites: int
    n_multiply_occupied_sites: int
    vacancy_clusters: tuple[tuple[int, ...], ...]  # indices into ``vacancy_sites``
    interstitial_clusters: tuple[tuple[int, ...], ...]  # indices into ``interstitial_atoms``
    vacancy_sites: tuple[int, ...]
    interstitial_atoms: tuple[int, ...]
    vacancy_size_distribution: dict[int, int]
    interstitial_size_distribution: dict[int, int]
    frenkel_separations: tuple[float, ...]  # angstrom, minimum-total-distance pairing
    cluster_cutoff: float  # angstrom
    cluster_justification: str
    species_aware: bool
    reference_population: float | None
    reference_population_definition: str | None
    survival_fraction: float | None
    n_ambiguous: int
    scope: str = (
        "counts on one structure with the stated cutoff and reference population; "
        "not a DPA estimate and not a validated surviving-defect production"
    )


def summarize_surviving_defects(
    reference: Atoms,
    current: Atoms,
    result: WignerSeitzResult,
    *,
    cluster_cutoff: float,
    cluster_justification: str,
    species_aware: bool = False,
    reference_population: float | None = None,
    reference_population_definition: str | None = None,
) -> SurvivingDefectSummary:
    """Summarize ``result`` (from :func:`wigner_seitz_analysis`) with explicit choices.

    The survival fraction is ``n_frenkel_pairs / reference_population`` and requires both a
    positive ``reference_population`` and its ``reference_population_definition``.
    """
    if not cluster_justification.strip():
        raise ValueError("cluster_justification is required: the cutoff is a modelling choice")
    if (reference_population is None) != (reference_population_definition is None):
        raise ValueError(
            "reference_population and reference_population_definition must be given together"
        )
    if reference_population is not None and not (
        np.isfinite(reference_population) and reference_population > 0
    ):
        raise ValueError(f"reference_population must be positive, got {reference_population}")
    if result.n_sites != len(reference) or result.n_atoms != len(current):
        raise ValueError("result does not match the given reference and current structures")

    cell = reference.cell.array
    if result.cell_mode == "affine":
        scaled = current.get_scaled_positions(wrap=False) @ cell
    else:
        scaled = current.positions
    scaled = scaled - np.asarray(result.translation)
    vac_pos = reference.positions[list(result.vacant_sites)]
    int_pos = scaled[list(result.interstitial_atoms)]
    ref_symbols = reference.get_chemical_symbols()
    symbols = current.get_chemical_symbols()
    vac_labels = [ref_symbols[s] for s in result.vacant_sites] if species_aware else None
    int_labels = [symbols[a] for a in result.interstitial_atoms] if species_aware else None
    vac_clusters = cluster_defects(vac_pos, cell, cluster_cutoff, vac_labels)
    int_clusters = cluster_defects(int_pos, cell, cluster_cutoff, int_labels)

    n_frenkel = result.n_frenkel_pairs
    fraction = None if reference_population is None else n_frenkel / reference_population
    return SurvivingDefectSummary(
        n_vacancies=result.n_vacancies,
        n_interstitials=result.n_interstitials,
        n_frenkel_pairs=n_frenkel,
        n_antisites=len(result.antisites),
        n_multiply_occupied_sites=len(result.multiply_occupied_sites),
        vacancy_clusters=vac_clusters,
        interstitial_clusters=int_clusters,
        vacancy_sites=result.vacant_sites,
        interstitial_atoms=result.interstitial_atoms,
        vacancy_size_distribution=size_distribution(vac_clusters),
        interstitial_size_distribution=size_distribution(int_clusters),
        frenkel_separations=tuple(p.separation for p in result.frenkel_pairs),
        cluster_cutoff=float(cluster_cutoff),
        cluster_justification=cluster_justification,
        species_aware=species_aware,
        reference_population=None if reference_population is None else float(reference_population),
        reference_population_definition=reference_population_definition,
        survival_fraction=fraction,
        n_ambiguous=result.n_ambiguous,
    )


__all__ = [
    "ClusterSensitivity",
    "SurvivingDefectSummary",
    "cluster_defects",
    "cluster_sensitivity",
    "size_distribution",
    "summarize_surviving_defects",
]
