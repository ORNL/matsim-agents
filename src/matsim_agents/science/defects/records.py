"""Per-defect records built from a Wigner-Seitz analysis and its clustering.

One record per vacancy, interstitial atom or antisite, carrying type, identities, position,
species, the species within ``environment_cutoff``, cluster membership, detection method,
ambiguity and the reference provenance. Types are geometric labels from cluster size and
species: ``monovacancy``, ``divacancy``, ``vacancy_cluster``, ``self_interstitial``,
``foreign_interstitial``, ``interstitial_cluster`` and ``antisite``. Solute and gas complexes
are tags set only from caller-supplied species lists, since which species are solutes or gas
cannot be read from a structure. Transmutation-associated defects, extended defects and
Frenkel pairs as single entities are not assigned here (Frenkel pairing is in the analysis
result). A "self" interstitial means only that the species occurs in the reference. Nothing
here is validated against cascade data.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from ase import Atoms
from ase.geometry import find_mic

from matsim_agents.science.defects.survival import SurvivingDefectSummary, aligned_positions
from matsim_agents.science.defects.wigner_seitz import WignerSeitzResult

_METHOD = "wigner_seitz"


@dataclass(frozen=True)
class DefectRecord:
    defect_id: str
    defect_type: str
    kind: str  # "vacancy", "interstitial" or "antisite"
    site: int | None  # reference site index (vacancy, antisite)
    atom: int | None  # atom index (interstitial, antisite)
    position: tuple[float, float, float]  # angstrom, reference frame
    species: str  # reference species (vacancy) or occupant species
    environment: dict[str, int]  # species counts within environment_cutoff
    environment_cutoff: float
    cluster: int | None  # index into the summary's clusters of this kind
    cluster_size: int | None
    complexes: tuple[str, ...]  # subset of ("solute", "gas")
    detection_method: str
    ambiguity: dict[str, float | int]
    reference_provenance: str


def _environment(
    point: np.ndarray,
    positions: np.ndarray,
    symbols: Sequence[str],
    cell: np.ndarray,
    cutoff: float,
    exclude: int | None,
) -> dict[str, int]:
    _, lengths = find_mic(positions - point, cell, pbc=True)
    counts: Counter[str] = Counter()
    for i in np.flatnonzero(lengths <= cutoff):
        if int(i) != exclude:
            counts[symbols[int(i)]] += 1
    return dict(sorted(counts.items()))


def _tags(
    species: Sequence[str],
    environment: dict[str, int],
    solute_species: frozenset[str],
    gas_species: frozenset[str],
) -> tuple[str, ...]:
    present = set(species) | set(environment)
    tags = []
    if present & solute_species:
        tags.append("solute")
    if present & gas_species:
        tags.append("gas")
    return tuple(tags)


def build_defect_records(
    reference: Atoms,
    current: Atoms,
    result: WignerSeitzResult,
    summary: SurvivingDefectSummary,
    *,
    environment_cutoff: float,
    reference_provenance: str,
    solute_species: Sequence[str] = (),
    gas_species: Sequence[str] = (),
) -> tuple[DefectRecord, ...]:
    """Records for every vacancy, interstitial atom and antisite in ``result``.

    ``summary`` must come from the same ``result`` so cluster membership matches.
    """
    if not np.isfinite(environment_cutoff) or environment_cutoff <= 0:
        raise ValueError("environment_cutoff must be positive and finite")
    if not reference_provenance.strip():
        raise ValueError("reference_provenance is required")
    if result.n_sites != len(reference) or result.n_atoms != len(current):
        raise ValueError("result does not match the given reference and current structures")
    if (
        summary.vacancy_sites != result.vacant_sites
        or summary.interstitial_atoms != result.interstitial_atoms
    ):
        raise ValueError("summary does not match result")

    solutes, gases = frozenset(solute_species), frozenset(gas_species)
    cell = reference.cell.array
    positions = aligned_positions(reference, current, result)
    symbols = current.get_chemical_symbols()
    ref_symbols = reference.get_chemical_symbols()
    ref_species = set(ref_symbols)
    ambiguity = {
        "structure_n_ambiguous": result.n_ambiguous,
        "ambiguity_margin": result.ambiguity_margin,
    }

    def env(point: np.ndarray, exclude: int | None = None) -> dict[str, int]:
        return _environment(point, positions, symbols, cell, environment_cutoff, exclude)

    def membership(clusters: tuple[tuple[int, ...], ...], n: int) -> list[int]:
        out = [-1] * n
        for c, members in enumerate(clusters):
            for m in members:
                out[m] = c
        return out

    records: list[DefectRecord] = []

    vac_cluster = membership(summary.vacancy_clusters, len(result.vacant_sites))
    for k, site in enumerate(result.vacant_sites):
        c = vac_cluster[k]
        size = len(summary.vacancy_clusters[c])
        members = [ref_symbols[result.vacant_sites[m]] for m in summary.vacancy_clusters[c]]
        point = reference.positions[site]
        environment = env(point)
        defect_type = {1: "monovacancy", 2: "divacancy"}.get(size, "vacancy_cluster")
        records.append(
            DefectRecord(
                defect_id=f"vacancy-{site}",
                defect_type=defect_type,
                kind="vacancy",
                site=int(site),
                atom=None,
                position=tuple(float(x) for x in point),
                species=ref_symbols[site],
                environment=environment,
                environment_cutoff=float(environment_cutoff),
                cluster=c,
                cluster_size=size,
                complexes=_tags(members, environment, solutes, gases),
                detection_method=_METHOD,
                ambiguity=ambiguity,
                reference_provenance=reference_provenance,
            )
        )

    int_cluster = membership(summary.interstitial_clusters, len(result.interstitial_atoms))
    for k, atom in enumerate(result.interstitial_atoms):
        c = int_cluster[k]
        size = len(summary.interstitial_clusters[c])
        members = [symbols[result.interstitial_atoms[m]] for m in summary.interstitial_clusters[c]]
        point = positions[atom]
        environment = env(point, exclude=int(atom))
        if size > 1:
            defect_type = "interstitial_cluster"
        elif symbols[atom] in ref_species:
            defect_type = "self_interstitial"
        else:
            defect_type = "foreign_interstitial"
        records.append(
            DefectRecord(
                defect_id=f"interstitial-{atom}",
                defect_type=defect_type,
                kind="interstitial",
                site=None,
                atom=int(atom),
                position=tuple(float(x) for x in point),
                species=symbols[atom],
                environment=environment,
                environment_cutoff=float(environment_cutoff),
                cluster=c,
                cluster_size=size,
                complexes=_tags(members, environment, solutes, gases),
                detection_method=_METHOD,
                ambiguity=ambiguity,
                reference_provenance=reference_provenance,
            )
        )

    for anti in result.antisites:
        point = reference.positions[anti.site]
        environment = env(point, exclude=int(anti.atom))
        records.append(
            DefectRecord(
                defect_id=f"antisite-{anti.site}",
                defect_type="antisite",
                kind="antisite",
                site=int(anti.site),
                atom=int(anti.atom),
                position=tuple(float(x) for x in point),
                species=anti.occupant_species,
                environment=environment,
                environment_cutoff=float(environment_cutoff),
                cluster=None,
                cluster_size=None,
                complexes=_tags([anti.occupant_species], environment, solutes, gases),
                detection_method=_METHOD,
                ambiguity=ambiguity,
                reference_provenance=reference_provenance,
            )
        )
    return tuple(records)


__all__ = ["DefectRecord", "build_defect_records"]
