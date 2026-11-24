"""Reproducible ensembles of independent BCC alloy occupations at fixed composition.

Each configuration gets its own random stream derived from one ensemble seed with
``matsim_agents.seeding.stream_seed``. Independent seeds do not guarantee distinct or
statistically uncorrelated occupations: exact duplicates are detected and reported (never
retried), and symmetry-equivalent configurations are not detected here. Targeted-SRO members
are bounded-search configurations, not thermodynamic equilibrium structures.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from ase import Atoms
from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from matsim_agents.discovery.alloys import (
    AlloySpec,
    ShellSRO,
    SROResult,
    SROTarget,
    _shuffled_symbols,
    bcc_topology,
    build_bcc_supercell,
    generate_targeted_sro,
    warren_cowley_sro,
)
from matsim_agents.seeding import stream_seed

EnsembleMode = Literal["random", "targeted_sro"]


class AlloyEnsembleSpec(BaseModel):
    """Validated description of an ensemble: one parent alloy, a mode, a size and a seed."""

    model_config = ConfigDict(frozen=True)

    alloy: AlloySpec
    n_configurations: StrictInt = Field(gt=0)
    mode: EnsembleMode
    seed: StrictInt = Field(ge=0)
    sro_targets: tuple[SROTarget, ...] = ()
    max_swap_attempts: StrictInt = Field(default=1000, ge=0)
    tolerance: float = Field(default=1e-4, ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def _validate_targets(self) -> AlloyEnsembleSpec:
        if self.mode == "targeted_sro" and not self.sro_targets:
            raise ValueError("targeted_sro mode requires at least one SRO target")
        if self.mode == "random" and self.sro_targets:
            raise ValueError("random mode does not accept SRO targets")
        return self


@dataclass(frozen=True, eq=False)
class ConfigurationRecord:
    """Provenance and ordering descriptors of one configuration (no coordinates).

    ``seed`` seeds the random occupation; ``optimizer_seed`` seeds the targeted-SRO search and
    is None in random mode. ``converged``, ``objective`` and ``attempts`` are None in random mode.
    A non-converged targeted configuration is a valid outcome, not an error.
    """

    config_id: str
    parent: AlloySpec
    mode: EnsembleMode
    seed: int
    optimizer_seed: int | None
    counts: dict[str, int]
    fingerprint: str
    sro: SROResult
    converged: bool | None
    objective: float | None
    attempts: int | None

    def metadata(self) -> dict[str, Any]:
        """JSON-serializable record; SRO matrices are indexed by ``species``."""

        def shell(data: ShellSRO) -> dict[str, Any]:
            return {
                "pair_counts": data.pair_counts.tolist(),
                "alpha": data.alpha.tolist(),
            }

        return {
            "config_id": self.config_id,
            "parent": self.parent.model_dump(mode="json"),
            "mode": self.mode,
            "seed": self.seed,
            "optimizer_seed": self.optimizer_seed,
            "counts": dict(self.counts),
            "fingerprint": self.fingerprint,
            "species": list(self.sro.species),
            "first_shell": shell(self.sro.first_shell),
            "second_shell": shell(self.sro.second_shell),
            "converged": self.converged,
            "objective": self.objective,
            "attempts": self.attempts,
        }


@dataclass(frozen=True, eq=False)
class AlloyEnsemble:
    """All configurations of one ensemble, with the structures kept in memory.

    ``duplicate_groups`` lists the IDs of configurations sharing an identical occupation (same
    fingerprint); it says nothing about symmetry-equivalent occupations.
    """

    spec: AlloyEnsembleSpec
    records: tuple[ConfigurationRecord, ...]
    structures: tuple[Atoms, ...]
    duplicate_groups: tuple[tuple[str, ...], ...]

    @property
    def n_unique(self) -> int:
        """Number of distinct exact occupations."""
        return len({record.fingerprint for record in self.records})

    @property
    def expected_random_overlap(self) -> float:
        """E[site overlap] of two independent uniform shuffles of this composition: sum_i c_i^2."""
        return float(sum(c * c for c in self.spec.alloy.composition.values()))

    def overlap_matrix(self) -> np.ndarray:
        """Fraction of sites with the same species for every pair of configurations.

        A descriptive diagnostic only: values near ``expected_random_overlap`` are consistent
        with, but do not prove, statistical independence. Targeted-SRO members are not expected
        to match the random value.
        """
        numbers = np.stack([atoms.numbers for atoms in self.structures])
        return np.array([(numbers == row).mean(axis=1) for row in numbers])

    def overlap_summary(self) -> dict[str, float | None]:
        """Min, mean and max overlap over distinct pairs (None for a single configuration)."""
        matrix = self.overlap_matrix()
        pairs = matrix[np.triu_indices(len(matrix), k=1)]
        stats = (pairs.min(), pairs.mean(), pairs.max()) if len(pairs) else (None, None, None)
        keys = ("min", "mean", "max")
        summary = {k: None if v is None else float(v) for k, v in zip(keys, stats, strict=True)}
        return summary | {"expected_random": self.expected_random_overlap}

    def metadata(self) -> dict[str, Any]:
        return {
            "spec": self.spec.model_dump(mode="json"),
            "n_unique": self.n_unique,
            "duplicate_groups": [list(group) for group in self.duplicate_groups],
            "overlap": self.overlap_summary(),
            "configurations": [record.metadata() for record in self.records],
        }


def occupation_fingerprint(atoms: Atoms) -> str:
    """SHA-256 of the per-site species sequence; identical occupations of one lattice match."""
    symbols = atoms.get_chemical_symbols()
    species = sorted(set(symbols))
    position = {symbol: k for k, symbol in enumerate(species)}
    index = np.fromiter((position[s] for s in symbols), dtype="<u2", count=len(symbols))
    digest = hashlib.sha256()
    digest.update("|".join(species).encode())
    digest.update(b"\n")
    digest.update(index.tobytes())
    return digest.hexdigest()


def generate_alloy_ensemble(
    spec: AlloySpec,
    *,
    n_configurations: int,
    mode: str,
    seed: int,
    sro_targets: Sequence[SROTarget] | None = None,
    max_swap_attempts: int = 1000,
    tolerance: float = 1e-4,
) -> AlloyEnsemble:
    """Generate ``n_configurations`` independently seeded occupations of one BCC lattice.

    Per-configuration seeds come from ``matsim_agents.seeding.stream_seed``; the global
    NumPy state is untouched. The lattice and its neighbour shells are built once and shared.
    ``random`` mode shuffles the exact species counts. ``targeted_sro`` mode starts from such an
    occupation and runs ``generate_targeted_sro`` for at most ``max_swap_attempts`` swaps, then
    records the achieved SRO and convergence status whether or not the tolerance was met.
    Invalid sizes, modes or targets raise ``ValueError``.
    """
    ensemble = AlloyEnsembleSpec(
        alloy=spec,
        n_configurations=n_configurations,
        mode=mode,
        seed=seed,
        sro_targets=tuple(sro_targets or ()),
        max_swap_attempts=max_swap_attempts,
        tolerance=tolerance,
    )
    lattice = build_bcc_supercell(spec)
    topology = bcc_topology(lattice, spec.lattice_parameter)
    species = tuple(spec.counts)

    records: list[ConfigurationRecord] = []
    structures: list[Atoms] = []
    for k in range(n_configurations):
        occupation_seed = stream_seed(seed, "alloy-ensemble", "occupation", k)
        optimizer_seed = stream_seed(seed, "alloy-ensemble", "optimizer", k)
        atoms = lattice.copy()
        atoms.set_chemical_symbols(_shuffled_symbols(spec.counts, occupation_seed))
        converged = objective = attempts = None
        used_optimizer_seed = None
        if ensemble.mode == "targeted_sro":
            out = generate_targeted_sro(
                atoms,
                spec.lattice_parameter,
                ensemble.sro_targets,
                seed=optimizer_seed,
                tolerance=ensemble.tolerance,
                max_attempts=ensemble.max_swap_attempts,
                topology=topology,
            )
            atoms = out.atoms
            converged, objective, attempts = out.converged, out.objective, out.attempts
            used_optimizer_seed = optimizer_seed
        sro = warren_cowley_sro(atoms, spec.lattice_parameter, species, topology=topology)
        records.append(
            ConfigurationRecord(
                config_id=f"{ensemble.mode}-{k:04d}",
                parent=spec,
                mode=ensemble.mode,
                seed=occupation_seed,
                optimizer_seed=used_optimizer_seed,
                counts=dict(spec.counts),
                fingerprint=occupation_fingerprint(atoms),
                sro=sro,
                converged=converged,
                objective=objective,
                attempts=attempts,
            )
        )
        structures.append(atoms)

    by_fingerprint: dict[str, list[str]] = defaultdict(list)
    for record in records:
        by_fingerprint[record.fingerprint].append(record.config_id)
    duplicates = tuple(tuple(ids) for ids in by_fingerprint.values() if len(ids) > 1)
    return AlloyEnsemble(ensemble, tuple(records), tuple(structures), duplicates)
