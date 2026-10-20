"""Objective-aware train/validation/test splits of DFT-labelled datasets.

A split answers one scientific question, and the strategy decides which:

=====================  ========================================================
``random_frame``       Interpolation among similar samples of the training
                       distribution. Frames of one MD trajectory are correlated,
                       so this does not show transfer to new trajectories.
``temporal_block``     Extrapolation forward in time within trajectories:
                       contiguous blocks per trajectory, optionally separated by
                       a purge gap.
``trajectory_holdout`` Transfer to independent trajectories: whole trajectories
                       per partition.
``structure_holdout``  Transfer to new initial structures or structure families.
``composition_holdout`` Transfer to new compositions (exact formulas or bins).
``condition_holdout``  Transfer to new physical conditions (temperature,
                       pressure, strain, ...), with declared units.
``al_iteration_holdout`` Prediction of later, unobserved acquisition batches:
                       earlier AL iterations train, later ones test.
``generic_group``      Holdout of any declared metadata field.
=====================  ========================================================

Roles are separate from strategies: ``train`` updates weights (and any fitted
correction), ``validation`` serves early stopping, model comparison, and
promotion, and ``test`` is evaluated only after those choices are frozen.
Additional named test domains are external, immutable datasets with an explicit
membership rule. Every split writes its partitions and a content-addressed
``split-manifest.json``; every backend consumes the same files through
:func:`load_split`. Missing metadata, infeasible group counts, and leakage are
errors, never a silent fallback to a random split. A held-out group is held out
from fine-tuning only: foundation models may have seen it in pretraining.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections import defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
from ase import Atoms
from ase.io import read, write
from pydantic import BaseModel, Field, model_validator

from matsim_agents.datasets.governance import (
    DatasetManifest,
    DatasetValidationSummary,
    sha256_file,
    structure_identity,
    write_dataset_manifest,
)

SPLIT_MANIFEST = "split-manifest.json"
ROLES = ("train", "validation", "test")
# Dataset sidecar roles of the partition files.
_SIDECAR_ROLE = {"train": "training_pool", "validation": "validation", "test": "test"}
SplitStrategy = Literal[
    "random_frame",
    "temporal_block",
    "trajectory_holdout",
    "structure_holdout",
    "composition_holdout",
    "condition_holdout",
    "al_iteration_holdout",
    "generic_group",
]
_GROUPED = {
    "trajectory_holdout",
    "structure_holdout",
    "composition_holdout",
    "condition_holdout",
    "al_iteration_holdout",
    "generic_group",
}
_ORDERED = {"temporal_block", "al_iteration_holdout"}


class CompositionGrouping(BaseModel):
    """How compositions are compared.

    ``exact``: the reduced formula (``Nb2O5`` and ``Nb4O10`` are one group).
    ``bin``: atomic fractions in bins of ``bin_width`` (``Nb`` 0.30-0.35, ...), so
    nearby alloy compositions share a group.
    """

    mode: Literal["exact", "bin"] = "exact"
    bin_width: float = Field(0.05, gt=0, le=0.5)


class ConditionField(BaseModel):
    """A physical condition stored in each frame's ``info[name]``."""

    name: str = Field(min_length=1)
    unit: str = Field(min_length=1, description="Unit of the stored values, e.g. K or GPa.")
    bin_width: float | None = Field(
        None, gt=0, description="Group values in bins of this width (in ``unit``)."
    )


class SplitConfig(BaseModel):
    strategy: SplitStrategy
    train_fraction: float = Field(0.8, gt=0, le=1)
    validation_fraction: float = Field(0.0, ge=0, lt=1)
    test_fraction: float = Field(0.2, ge=0, lt=1)
    seed: int = 0
    group_by: str | None = Field(
        None,
        description=(
            "Metadata field defining the groups: required for generic_group; for "
            "structure_holdout initial_structure_id (default) or structure_family_id."
        ),
    )
    composition: CompositionGrouping = Field(default_factory=CompositionGrouping)
    conditions: list[ConditionField] = Field(default_factory=list)
    purge_gap_frames: int = Field(
        0, ge=0, description="temporal_block: frames dropped between consecutive blocks."
    )
    explicit_groups: dict[Literal["train", "validation", "test"], list[str]] | None = Field(
        None, description="Assign every group to a role by name instead of by seed."
    )
    min_groups_per_role: int = Field(
        2, ge=1, description="Fewer independent groups in a role is reported as a limitation."
    )
    near_duplicate_tolerance_A: float = Field(
        0.01,
        gt=0,
        description="RMS difference of neighbour-distance fingerprints below which frames "
        "in different roles are reported as near duplicates (diagnostic).",
    )

    @model_validator(mode="after")
    def _consistent(self) -> SplitConfig:
        total = self.train_fraction + self.validation_fraction + self.test_fraction
        if not math.isclose(total, 1.0, abs_tol=1e-9):
            raise ValueError(f"split fractions must sum to 1, got {total}")
        strategy = self.strategy
        if strategy == "generic_group" and not self.group_by:
            raise ValueError("generic_group requires group_by")
        if self.group_by and strategy not in {"generic_group", "structure_holdout"}:
            raise ValueError(
                f"group_by applies to generic_group and structure_holdout, not {strategy}"
            )
        if strategy == "structure_holdout" and self.group_by not in {
            None,
            "initial_structure_id",
            "structure_family_id",
        }:
            raise ValueError(
                "structure_holdout groups by initial_structure_id or structure_family_id"
            )
        if strategy == "condition_holdout" and not self.conditions:
            raise ValueError("condition_holdout requires at least one condition field")
        if self.conditions and strategy != "condition_holdout":
            raise ValueError("conditions apply to condition_holdout only")
        if self.purge_gap_frames and strategy != "temporal_block":
            raise ValueError("purge_gap_frames applies to temporal_block only")
        if self.explicit_groups is not None and strategy not in _GROUPED - {"al_iteration_holdout"}:
            raise ValueError(f"explicit_groups do not apply to {strategy}")
        return self

    def fraction(self, role: str) -> float:
        return {
            "train": self.train_fraction,
            "validation": self.validation_fraction,
            "test": self.test_fraction,
        }[role]


class TestDomainConfig(BaseModel):
    """An external, immutable evaluation dataset and the rule that makes it a holdout."""

    name: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    dataset: Path
    membership: SplitStrategy = Field(
        description="Groups (as the split strategy defines them) that must not appear in "
        "train or validation; random_frame requires only distinct configurations."
    )
    allow_overlap_with: list[str] = Field(
        default_factory=list, description="Other test domains this one may share frames with."
    )


class SplitCheck(BaseModel):
    name: str
    status: Literal["pass", "warn", "fail"]
    detail: str


class FrameEntry(BaseModel):
    index: int
    config_id: str
    role: str
    group_id: str
    composition: str
    trajectory_id: str | None = None
    time_index: int | None = None
    al_iteration: int | None = None
    initial_structure_id: str | None = None
    structure_family_id: str | None = None
    source_campaign: str | None = None
    conditions: dict[str, float] = Field(default_factory=dict)


class RoleMembership(BaseModel):
    path: str
    sha256: str
    n_frames: int
    n_groups: int
    config_ids: list[str] = Field(description="In the order of the partition file.")
    frame_group_ids: list[str] = Field(description="Group of each frame, as config_ids.")
    group_ids: list[str]


class DomainMembership(BaseModel):
    name: str
    dataset: str
    sha256: str
    membership: str
    n_frames: int
    n_groups: int
    config_ids: list[str]
    frame_group_ids: list[str]
    group_ids: list[str]
    allow_overlap_with: list[str]


class PurgedFrame(BaseModel):
    index: int
    config_id: str
    reason: str


class SplitManifest(BaseModel):
    schema_version: Literal[1] = 1
    objective: str
    strategy: str
    config: dict[str, Any]
    source: dict[str, Any]
    roles: dict[str, RoleMembership]
    frames: list[FrameEntry]
    purged: list[PurgedFrame] = Field(default_factory=list)
    test_domains: list[DomainMembership] = Field(default_factory=list)
    checks: list[SplitCheck]
    limitations: list[str]
    code_commit: str | None = None
    created_at_utc: str = ""
    content_sha256: str = ""

    def content_digest(self) -> str:
        payload = self.model_dump(
            mode="json", exclude={"created_at_utc", "content_sha256", "code_commit"}
        )
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    @property
    def failed_checks(self) -> list[SplitCheck]:
        return [check for check in self.checks if check.status == "fail"]


# --------------------------------------------------------------------------- #
# Frame metadata                                                              #
# --------------------------------------------------------------------------- #


def config_id(atoms: Atoms) -> str:
    """The frame's declared ``config_id``, or one derived from its geometry."""
    declared = atoms.info.get("config_id")
    return str(declared) if declared is not None else f"geom-{structure_identity(atoms)[:20]}"


def composition_label(atoms: Atoms, grouping: CompositionGrouping) -> str:
    symbols = atoms.get_chemical_symbols()
    if grouping.mode == "exact":
        return Atoms(symbols).get_chemical_formula(mode="hill", empirical=True)
    counts: dict[str, int] = defaultdict(int)
    for symbol in symbols:
        counts[symbol] += 1
    width = grouping.bin_width
    parts = []
    for element in sorted(counts):
        low = math.floor(counts[element] / len(symbols) / width + 1e-9) * width
        parts.append(f"{element}[{low:.4g},{low + width:.4g})")
    return "".join(parts)


def _optional_int(atoms: Atoms, key: str) -> int | None:
    value = atoms.info.get(key)
    return None if value is None else int(value)


def _optional_str(atoms: Atoms, key: str) -> str | None:
    value = atoms.info.get(key)
    return None if value is None else str(value)


def _entry(index: int, atoms: Atoms, cfg: SplitConfig) -> FrameEntry:
    conditions = {}
    for field in cfg.conditions:
        value = atoms.info.get(field.name)
        if value is not None:
            conditions[field.name] = float(value)
    return FrameEntry(
        index=index,
        config_id=config_id(atoms),
        role="",
        group_id="",
        composition=composition_label(atoms, cfg.composition),
        trajectory_id=_optional_str(atoms, "trajectory_id"),
        time_index=_optional_int(atoms, "time_index"),
        al_iteration=_optional_int(atoms, "al_iteration"),
        initial_structure_id=_optional_str(atoms, "initial_structure_id"),
        structure_family_id=_optional_str(atoms, "structure_family_id"),
        source_campaign=_optional_str(atoms, "source_campaign"),
        conditions=conditions,
    )


def _required(entries: list[FrameEntry], frames: Sequence[Atoms], keys: list[str], why: str):
    for key in keys:
        missing = [
            entry.index
            for entry, atoms in zip(entries, frames, strict=True)
            if atoms.info.get(key) is None
        ]
        if missing:
            shown = ", ".join(str(index) for index in missing[:10])
            raise ValueError(
                f"{why} needs info[{key!r}] on every frame; {len(missing)} frame(s) lack it "
                f"(indices {shown}{', ...' if len(missing) > 10 else ''}). Record it when "
                "the data are labelled, or choose a strategy whose metadata exist."
            )


def group_key(entry: FrameEntry, atoms: Atoms, cfg: SplitConfig, strategy: str) -> str:
    """The group of a frame under ``strategy`` (with ``cfg``'s grouping settings)."""
    if strategy == "random_frame":
        return entry.config_id
    if strategy in {"temporal_block", "trajectory_holdout"}:
        return str(entry.trajectory_id)
    if strategy == "structure_holdout":
        return str(atoms.info[cfg.group_by or "initial_structure_id"])
    if strategy == "composition_holdout":
        return entry.composition
    if strategy == "condition_holdout":
        parts = []
        for field in cfg.conditions:
            value = float(atoms.info[field.name])
            if field.bin_width is not None:
                low = math.floor(value / field.bin_width + 1e-9) * field.bin_width
                parts.append(f"{field.name}=[{low:g},{low + field.bin_width:g}){field.unit}")
            else:
                parts.append(f"{field.name}={value:g}{field.unit}")
        return ";".join(parts)
    if strategy == "al_iteration_holdout":
        return f"iteration-{entry.al_iteration}"
    return str(atoms.info[cfg.group_by])


def _required_keys(cfg: SplitConfig, strategy: str) -> list[str]:
    return {
        "temporal_block": ["trajectory_id", "time_index"],
        "trajectory_holdout": ["trajectory_id"],
        "structure_holdout": [cfg.group_by or "initial_structure_id"],
        "condition_holdout": [field.name for field in cfg.conditions],
        "al_iteration_holdout": ["al_iteration"],
        "generic_group": [cfg.group_by or ""],
    }.get(strategy, [])


# --------------------------------------------------------------------------- #
# Role assignment                                                             #
# --------------------------------------------------------------------------- #


def _active_roles(cfg: SplitConfig) -> list[str]:
    return [role for role in ROLES if cfg.fraction(role) > 0]


def _assign_random_frames(entries: list[FrameEntry], cfg: SplitConfig) -> None:
    # Test frames come first in the permutation, so a given seed and test
    # fraction select the same test frames whatever the validation fraction.
    order = np.arange(len(entries))
    np.random.default_rng(cfg.seed).shuffle(order)
    n = len(entries)
    n_test = max(1, round(n * cfg.test_fraction)) if cfg.test_fraction > 0 else 0
    n_val = max(1, round(n * cfg.validation_fraction)) if cfg.validation_fraction > 0 else 0
    if n_test + n_val >= n:
        raise ValueError(f"{n} frames cannot fill train, validation, and test")
    for rank, index in enumerate(order.tolist()):
        role = "test" if rank < n_test else "validation" if rank < n_test + n_val else "train"
        entries[index].role = role


def _assign_groups(entries: list[FrameEntry], cfg: SplitConfig) -> None:
    sizes: dict[str, int] = defaultdict(int)
    for entry in entries:
        sizes[entry.group_id] += 1
    groups = sorted(sizes)
    roles = _active_roles(cfg)
    if cfg.explicit_groups is not None:
        assignment = {
            group: role for role, members in cfg.explicit_groups.items() for group in members
        }
        unknown = sorted(set(assignment) - set(groups))
        unassigned = [group for group in groups if group not in assignment]
        if unknown or unassigned:
            raise ValueError(
                "explicit_groups must name every group exactly once: "
                f"unknown {unknown}, unassigned {unassigned}"
            )
        listed = [group for members in cfg.explicit_groups.values() for group in members]
        if len(listed) != len(set(listed)):
            raise ValueError("explicit_groups lists a group in more than one role")
    else:
        if len(groups) < len(roles):
            raise ValueError(
                f"{len(groups)} independent group(s) cannot fill the {len(roles)} roles "
                f"{roles}; collect data from more independent groups or drop a role"
            )
        order = [groups[i] for i in np.random.default_rng(cfg.seed).permutation(len(groups))]
        target = {role: cfg.fraction(role) * len(entries) for role in roles}
        filled = dict.fromkeys(roles, 0)
        assignment = {}
        # One group per role first, smallest roles first, then the largest deficit.
        for role, group in zip(sorted(roles, key=cfg.fraction), order, strict=False):
            assignment[group] = role
            filled[role] += sizes[group]
        for group in order[len(roles) :]:
            role = max(roles, key=lambda r: (target[r] - filled[r]) / max(target[r], 1e-9))
            assignment[group] = role
            filled[role] += sizes[group]
    for entry in entries:
        entry.role = assignment[entry.group_id]


def _assign_temporal(entries: list[FrameEntry], cfg: SplitConfig) -> list[PurgedFrame]:
    purged = []
    by_trajectory: dict[str, list[FrameEntry]] = defaultdict(list)
    for entry in entries:
        by_trajectory[entry.group_id].append(entry)
    roles = _active_roles(cfg)
    for trajectory, members in sorted(by_trajectory.items()):
        members.sort(key=lambda entry: (entry.time_index, entry.index))
        n = len(members)
        gaps = cfg.purge_gap_frames * (len(roles) - 1)
        usable = n - gaps
        if usable < len(roles):
            raise ValueError(
                f"trajectory {trajectory!r} has {n} frames, too few for {len(roles)} blocks "
                f"with a purge gap of {cfg.purge_gap_frames}"
            )
        counts = [max(1, round(usable * cfg.fraction(role))) for role in roles]
        counts[0] = usable - sum(counts[1:])
        position = 0
        for block, (role, count) in enumerate(zip(roles, counts, strict=True)):
            if block:
                for entry in members[position : position + cfg.purge_gap_frames]:
                    entry.role = "purged"
                    purged.append(
                        PurgedFrame(
                            index=entry.index,
                            config_id=entry.config_id,
                            reason=f"temporal purge gap before {role} in {trajectory}",
                        )
                    )
                position += cfg.purge_gap_frames
            for entry in members[position : position + count]:
                entry.role = role
            position += count
    return purged


def _assign_iterations(entries: list[FrameEntry], cfg: SplitConfig) -> None:
    sizes: dict[int, int] = defaultdict(int)
    for entry in entries:
        sizes[int(entry.al_iteration)] += 1
    iterations = sorted(sizes)
    roles = _active_roles(cfg)
    if len(iterations) < len(roles):
        raise ValueError(
            f"{len(iterations)} AL iteration(s) cannot fill the {len(roles)} roles {roles} "
            "in chronological order"
        )
    assignment: dict[int, str] = {}
    remaining = list(iterations)
    total = len(entries)
    for position, role in enumerate(roles):
        later = len(roles) - position - 1  # roles still to fill need one iteration each
        if not later:
            for iteration in remaining:
                assignment[iteration] = role
            break
        taken = 0
        while len(remaining) > later and (
            not taken or taken + sizes[remaining[0]] <= cfg.fraction(role) * total
        ):
            iteration = remaining.pop(0)
            assignment[iteration] = role
            taken += sizes[iteration]
    for entry in entries:
        entry.role = assignment[int(entry.al_iteration)]


# --------------------------------------------------------------------------- #
# Checks                                                                      #
# --------------------------------------------------------------------------- #


def _fingerprint(atoms: Atoms, neighbours: int = 8) -> np.ndarray | None:
    """Sorted per-atom nearest-neighbour distances: permutation and translation invariant."""
    if len(atoms) < 2 or len(atoms) > 1000:
        return None
    distances = atoms.get_all_distances(mic=bool(atoms.pbc.any()))
    distances.sort(axis=1)
    k = min(neighbours, len(atoms) - 1)
    rows = distances[:, 1 : k + 1]
    order = np.lexsort((*rows.T[::-1], atoms.numbers))
    return rows[order].ravel()


def _near_duplicates(
    entries: list[FrameEntry], frames: Sequence[Atoms], roles: Sequence[str], tol: float
) -> list[tuple[str, str, float]]:
    """Cross-role pairs whose neighbour-distance fingerprints differ by < ``tol`` RMS."""
    buckets: dict[tuple, list[int]] = defaultdict(list)
    for entry in entries:
        if entry.role in roles:
            atoms = frames[entry.index]
            volume = round(float(atoms.get_volume()), 0) if atoms.pbc.any() else 0.0
            buckets[(tuple(sorted(atoms.get_chemical_symbols())), volume)].append(entry.index)
    found = []
    for members in buckets.values():
        prints = {index: _fingerprint(frames[index]) for index in members}
        by_role: dict[str, list[int]] = defaultdict(list)
        for index in members:
            if prints[index] is not None:
                by_role[entries[index].role].append(index)
        present = sorted(by_role)
        for position, first_role in enumerate(present):
            for second_role in present[position + 1 :]:
                first, second = by_role[first_role], by_role[second_role]
                a = np.stack([prints[i] for i in first])
                b = np.stack([prints[i] for i in second])
                squared = (
                    (a * a).sum(axis=1)[:, None] + (b * b).sum(axis=1)[None, :] - 2.0 * a @ b.T
                )
                rms = np.sqrt(np.clip(squared, 0.0, None) / a.shape[1])
                for row, col in zip(*np.nonzero(rms < tol), strict=True):
                    found.append(
                        (
                            entries[first[row]].config_id,
                            entries[second[col]].config_id,
                            float(rms[row, col]),
                        )
                    )
    return found


def _role_checks(entries: list[FrameEntry], frames: Sequence[Atoms], cfg: SplitConfig):
    checks = []
    by_role: dict[str, list[FrameEntry]] = defaultdict(list)
    for entry in entries:
        if entry.role in ROLES:
            by_role[entry.role].append(entry)
    ids: dict[str, set[str]] = defaultdict(set)
    for entry in entries:
        if entry.role in ROLES:
            ids[entry.config_id].add(entry.role)
    shared = sorted(cid for cid, roles in ids.items() if len(roles) > 1)
    checks.append(
        SplitCheck(
            name="configuration_ids_in_one_role",
            status="fail" if shared else "pass",
            detail=f"configurations in several roles: {shared[:10]}"
            if shared
            else "every configuration ID is in exactly one role",
        )
    )
    group_roles: dict[str, set[str]] = defaultdict(set)
    for entry in entries:
        if entry.role in ROLES:
            group_roles[entry.group_id].add(entry.role)
    straddling = sorted(group for group, roles in group_roles.items() if len(roles) > 1)
    if cfg.strategy in _GROUPED:
        checks.append(
            SplitCheck(
                name="groups_in_one_role",
                status="fail" if straddling else "pass",
                detail=f"groups in several roles: {straddling[:10]}"
                if straddling
                else f"every {cfg.strategy} group is in exactly one role",
            )
        )
    elif cfg.strategy == "random_frame":
        trajectories: dict[str, set[str]] = defaultdict(set)
        for entry in entries:
            if entry.trajectory_id is not None:
                trajectories[entry.trajectory_id].add(entry.role)
        mixed = sorted(t for t, roles in trajectories.items() if len(roles) > 1)
        checks.append(
            SplitCheck(
                name="correlated_frames_across_roles",
                status="warn",
                detail=(
                    f"{len(mixed)} trajectory(ies) have frames in several roles; held-out "
                    "frames are correlated with training frames"
                    if mixed
                    else "random frames measure interpolation among similar samples"
                ),
            )
        )
    for role in _active_roles(cfg):
        groups = {entry.group_id for entry in by_role[role]}
        status = (
            "fail" if not groups else "warn" if len(groups) < cfg.min_groups_per_role else "pass"
        )
        checks.append(
            SplitCheck(
                name=f"independent_groups_{role}",
                status=status,
                detail=f"{len(by_role[role])} frame(s) in {len(groups)} independent group(s)"
                + (
                    f"; fewer than {cfg.min_groups_per_role}: results describe few groups"
                    if status == "warn"
                    else ""
                ),
            )
        )
    duplicates = _near_duplicates(entries, frames, ROLES, cfg.near_duplicate_tolerance_A)
    checks.append(
        SplitCheck(
            name="near_duplicates_across_roles",
            status="warn" if duplicates else "pass",
            detail=(
                f"{len(duplicates)} pair(s) closer than {cfg.near_duplicate_tolerance_A} A RMS "
                f"(sorted per-atom neighbour distances, same composition and volume), e.g. "
                f"{duplicates[:3]}"
                if duplicates
                else f"no cross-role pair closer than {cfg.near_duplicate_tolerance_A} A RMS "
                "in sorted per-atom neighbour distances (diagnostic, not a proof of novelty)"
            ),
        )
    )
    if cfg.strategy in _ORDERED:
        checks.append(_chronology_check(entries, cfg))
    return checks


def _chronology_check(entries: list[FrameEntry], cfg: SplitConfig) -> SplitCheck:
    roles = _active_roles(cfg)
    problems = []
    if cfg.strategy == "al_iteration_holdout":
        spans = {role: [e.al_iteration for e in entries if e.role == role] for role in roles}
        for earlier, later in zip(roles, roles[1:], strict=False):
            if max(spans[earlier]) >= min(spans[later]):
                problems.append(f"{earlier} reaches iteration {max(spans[earlier])}")
    else:
        by_trajectory: dict[str, list[FrameEntry]] = defaultdict(list)
        for entry in entries:
            by_trajectory[entry.group_id].append(entry)
        for trajectory, members in by_trajectory.items():
            times = {
                role: sorted(e.time_index for e in members if e.role == role) for role in roles
            }
            for earlier, later in zip(roles, roles[1:], strict=False):
                if times[earlier] and times[later]:
                    between = [
                        e for e in members if times[earlier][-1] < e.time_index < times[later][0]
                    ]
                    if times[earlier][-1] >= times[later][0]:
                        problems.append(f"{trajectory}: {earlier} overlaps {later} in time")
                    elif len(between) < cfg.purge_gap_frames:
                        problems.append(f"{trajectory}: gap before {later} is {len(between)}")
    return SplitCheck(
        name="chronological_order",
        status="fail" if problems else "pass",
        detail="; ".join(problems[:10])
        if problems
        else "every later role follows every earlier one"
        + (
            f" with a purge gap of {cfg.purge_gap_frames} frame(s)"
            if cfg.strategy == "temporal_block"
            else ""
        ),
    )


# --------------------------------------------------------------------------- #
# Test domains                                                                #
# --------------------------------------------------------------------------- #


def source_provenance(dataset: Path) -> DatasetManifest:
    """DFT provenance of a dataset from its sidecar (a full or minimal manifest)."""
    sidecar = dataset.with_suffix(dataset.suffix + ".manifest.json")
    if not sidecar.is_file():
        raise ValueError(f"{dataset} needs its dataset manifest (DFT provenance): {sidecar}")
    metadata = json.loads(sidecar.read_text())
    digest = sha256_file(dataset)
    if metadata.get("sha256") != digest:
        raise ValueError(f"{dataset} does not match the hash in {sidecar}")
    if not metadata.get("dft_backend"):
        raise ValueError(f"{sidecar} does not name the DFT backend")
    metadata.setdefault("dataset_id", digest[:16])
    metadata.setdefault("created_at_utc", "")
    metadata.setdefault("path", str(dataset))
    metadata.setdefault("energy_reference", f"{metadata['dft_backend']}:native_total_energy")
    metadata.setdefault("validation", {})
    return DatasetManifest.model_validate(metadata)


def _domain(
    domain: TestDomainConfig,
    cfg: SplitConfig,
    entries: list[FrameEntry],
    frames: Sequence[Atoms],
    source: DatasetManifest,
) -> tuple[DomainMembership, list[SplitCheck]]:
    """Membership and leakage checks of one external test domain."""
    path = Path(domain.dataset).resolve()
    domain_frames = list(read(path, index=":"))
    metadata = source_provenance(path)
    compatible = metadata.dft_backend == source.dft_backend and (
        metadata.method_signature == source.method_signature
        or (
            source.base_method_signature is not None
            and metadata.base_method_signature == source.base_method_signature
        )
    )
    checks = [
        SplitCheck(
            name=f"dft_method_{domain.name}",
            status="pass" if compatible else "fail",
            detail=f"{metadata.dft_backend} {metadata.method_signature} vs the split's "
            f"{source.dft_backend} {source.method_signature}",
        )
    ]
    keys = _required_keys(cfg, domain.membership)
    domain_entries = [_entry(index, atoms, cfg) for index, atoms in enumerate(domain_frames)]
    _required(domain_entries, domain_frames, keys, f"test domain {domain.name!r}")
    seen = [entry for entry in entries if entry.role in {"train", "validation"}]
    _required(
        seen,
        [frames[entry.index] for entry in seen],
        keys,
        f"checking test domain {domain.name!r} ({domain.membership}) against the split",
    )
    for entry, atoms in zip(domain_entries, domain_frames, strict=True):
        entry.role = f"domain:{domain.name}"
        entry.group_id = group_key(entry, atoms, cfg, domain.membership)
    seen_ids = {entry.config_id for entry in seen}
    seen_groups = {group_key(entry, frames[entry.index], cfg, domain.membership) for entry in seen}
    shared_ids = sorted({entry.config_id for entry in domain_entries} & seen_ids)
    checks.append(
        SplitCheck(
            name=f"configurations_unseen_{domain.name}",
            status="fail" if shared_ids else "pass",
            detail=f"configurations also in train/validation: {shared_ids[:10]}"
            if shared_ids
            else "no configuration of the domain is in train or validation",
        )
    )
    if domain.membership != "random_frame":
        shared_groups = sorted({entry.group_id for entry in domain_entries} & seen_groups)
        checks.append(
            SplitCheck(
                name=f"groups_unseen_{domain.name}",
                status="fail" if shared_groups else "pass",
                detail=f"{domain.membership} groups also in train/validation: {shared_groups[:10]}"
                if shared_groups
                else f"no {domain.membership} group of the domain is in train or validation",
            )
        )
    membership = DomainMembership(
        name=domain.name,
        dataset=str(path),
        sha256=sha256_file(path),
        membership=domain.membership,
        n_frames=len(domain_entries),
        n_groups=len({entry.group_id for entry in domain_entries}),
        config_ids=[entry.config_id for entry in domain_entries],
        frame_group_ids=[entry.group_id for entry in domain_entries],
        group_ids=sorted({entry.group_id for entry in domain_entries}),
        allow_overlap_with=list(domain.allow_overlap_with),
    )
    return membership, checks


def _domain_overlap_checks(domains: list[DomainMembership]) -> list[SplitCheck]:
    checks = []
    for position, first in enumerate(domains):
        for second in domains[position + 1 :]:
            allowed = (
                second.name in first.allow_overlap_with or first.name in second.allow_overlap_with
            )
            shared = sorted(set(first.config_ids) & set(second.config_ids))
            if shared and not allowed:
                status, detail = "fail", f"share configurations {shared[:10]} without permission"
            elif shared:
                status, detail = "pass", f"share {len(shared)} configuration(s), as declared"
            else:
                status, detail = "pass", "no shared configurations"
            checks.append(
                SplitCheck(name=f"domains_{first.name}_{second.name}", status=status, detail=detail)
            )
    return checks


# --------------------------------------------------------------------------- #
# Making and loading splits                                                   #
# --------------------------------------------------------------------------- #


def _code_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _limitations(cfg: SplitConfig, checks: list[SplitCheck]) -> list[str]:
    notes = [
        "Held-out groups are held out from fine-tuning only; foundation models may have "
        "seen similar structures in pretraining, so this is not a strict "
        "out-of-distribution test.",
    ]
    if cfg.strategy == "random_frame":
        notes.append(
            "random_frame measures interpolation among similar samples of the training "
            "distribution; it does not show transfer to new trajectories, structures, or "
            "compositions."
        )
    if cfg.strategy == "temporal_block":
        notes.append(
            "temporal_block tests extrapolation forward in time within the same trajectories, "
            "not transfer to independent trajectories."
        )
    if cfg.validation_fraction == 0:
        notes.append(
            "No validation partition: fine-tuners hold out part of the training partition "
            "for early stopping, and no model selection is made from evaluation data."
        )
    notes += [f"{check.name}: {check.detail}" for check in checks if check.status == "warn"]
    return notes


def make_split(
    dataset_path: str | Path,
    cfg: SplitConfig,
    output_dir: str | Path,
    *,
    objective: str,
    test_domains: Sequence[TestDomainConfig] = (),
) -> SplitManifest:
    """Partition ``dataset_path`` and write the partitions and their manifest.

    The output directory receives ``train.extxyz``, ``validation.extxyz``, and
    ``test.extxyz`` (the roles with a nonzero fraction), each with a dataset
    sidecar, and :data:`SPLIT_MANIFEST`. A directory that already holds a split
    is reused only if it is the same split; otherwise this raises. A split with
    a failed check is written for inspection, then raises.
    """

    if not objective.strip():
        raise ValueError("declare the scientific objective the split serves")
    source_path = Path(dataset_path).resolve()
    source = source_provenance(source_path)
    frames = list(read(source_path, index=":"))
    entries = [_entry(index, atoms, cfg) for index, atoms in enumerate(frames)]
    _required(entries, frames, _required_keys(cfg, cfg.strategy), cfg.strategy)
    for entry, atoms in zip(entries, frames, strict=True):
        entry.group_id = group_key(entry, atoms, cfg, cfg.strategy)

    purged: list[PurgedFrame] = []
    if cfg.strategy == "random_frame":
        _assign_random_frames(entries, cfg)
    elif cfg.strategy == "temporal_block":
        purged = _assign_temporal(entries, cfg)
    elif cfg.strategy == "al_iteration_holdout":
        _assign_iterations(entries, cfg)
    else:
        _assign_groups(entries, cfg)

    checks = _role_checks(entries, frames, cfg)
    checks.append(
        SplitCheck(
            name="fitted_corrections_use_training_only",
            status="pass",
            detail="fine-tuners receive the train partition to fit (and the validation "
            "partition for early stopping only); energy references come from elemental DFT "
            "calculations, not from these labels",
        )
    )
    domains = []
    for domain in test_domains:
        membership, domain_checks = _domain(domain, cfg, entries, frames, source)
        domains.append(membership)
        checks += domain_checks
    checks += _domain_overlap_checks(domains)
    if len({domain.name for domain in test_domains}) != len(test_domains):
        raise ValueError("test domain names must be unique")

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    roles = {}
    for role in _active_roles(cfg):
        members = [entry for entry in entries if entry.role == role]
        path = output / f"{role}.extxyz"
        staged = path.with_name(f".{path.name}.staged")
        write(staged, [frames[entry.index] for entry in members], format="extxyz")
        roles[role] = (path, staged, members)
    manifest = SplitManifest(
        objective=objective,
        strategy=cfg.strategy,
        config=cfg.model_dump(mode="json"),
        source={
            "path": str(source_path),
            "sha256": sha256_file(source_path),
            "n_frames": len(frames),
            "dataset_manifest": source.model_dump(mode="json"),
        },
        roles={
            role: RoleMembership(
                path=str(path),
                sha256=sha256_file(staged),
                n_frames=len(members),
                n_groups=len({entry.group_id for entry in members}),
                config_ids=[entry.config_id for entry in members],
                frame_group_ids=[entry.group_id for entry in members],
                group_ids=sorted({entry.group_id for entry in members}),
            )
            for role, (path, staged, members) in roles.items()
        },
        frames=entries,
        purged=purged,
        test_domains=domains,
        checks=checks,
        limitations=_limitations(cfg, checks),
        code_commit=_code_commit(),
    )
    manifest.content_sha256 = manifest.content_digest()
    manifest.created_at_utc = datetime.now(UTC).isoformat()

    target = output / SPLIT_MANIFEST
    if target.is_file():
        existing = SplitManifest.model_validate_json(target.read_text())
        for _path, staged, _members in roles.values():
            staged.unlink()
        if existing.content_sha256 != manifest.content_sha256:
            raise ValueError(
                f"{output} already holds a different split ({existing.content_sha256[:12]}); "
                "splits are immutable, write the new one elsewhere"
            )
        return existing
    for role, (path, staged, _members) in roles.items():
        staged.replace(path)
        write_dataset_manifest(
            path,
            dft_backend=source.dft_backend,
            energy_reference=source.energy_reference,
            validation=DatasetValidationSummary(accepted=manifest.roles[role].n_frames),
            parent_dataset_id=source.dataset_id,
            method_signature=source.method_signature,
            method_signature_elements=(
                set(source.method_signature_elements)
                if source.method_signature_elements is not None
                else None
            ),
            base_method_signature=source.base_method_signature,
            element_method_signatures=source.element_method_signatures,
            split_role=_SIDECAR_ROLE[role],
        )
    target.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    if manifest.failed_checks:
        failed = "; ".join(f"{c.name}: {c.detail}" for c in manifest.failed_checks)
        raise ValueError(f"split {target} failed its checks: {failed}")
    return manifest


def load_split(manifest_path: str | Path) -> SplitManifest:
    """Read a split and verify that its partitions and test domains are unchanged."""
    path = Path(manifest_path)
    if path.is_dir():
        path = path / SPLIT_MANIFEST
    manifest = SplitManifest.model_validate_json(path.read_text())
    if manifest.content_digest() != manifest.content_sha256:
        raise ValueError(f"{path} was modified after it was written")
    if manifest.failed_checks:
        raise ValueError(f"{path} records failed checks: {manifest.failed_checks}")
    for role, membership in manifest.roles.items():
        if sha256_file(membership.path) != membership.sha256:
            raise ValueError(f"the {role} partition {membership.path} changed after the split")
    for domain in manifest.test_domains:
        if sha256_file(domain.dataset) != domain.sha256:
            raise ValueError(f"test domain {domain.name} ({domain.dataset}) changed")
    return manifest


def main(argv: list[str] | None = None) -> int:
    """Make a split from an evaluation YAML (``objective``, ``split``, ``test_domains``)."""
    import argparse

    import yaml

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path, help="Evaluation YAML.")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    data = yaml.safe_load(args.config.read_text())
    data = data.get("evaluation", data)
    manifest = make_split(
        args.dataset,
        SplitConfig.model_validate(data["split"]),
        args.output_dir,
        objective=data["objective"],
        test_domains=[TestDomainConfig.model_validate(d) for d in data.get("test_domains", [])],
    )
    for role, membership in manifest.roles.items():
        print(f"{role}: {membership.n_frames} frames in {membership.n_groups} groups")
    for check in manifest.checks:
        print(f"[{check.status}] {check.name}: {check.detail}")
    print(args.output_dir / SPLIT_MANIFEST)
    return 0


__all__ = [
    "ROLES",
    "SPLIT_MANIFEST",
    "CompositionGrouping",
    "ConditionField",
    "SplitCheck",
    "SplitConfig",
    "SplitManifest",
    "TestDomainConfig",
    "composition_label",
    "config_id",
    "group_key",
    "load_split",
    "make_split",
    "source_provenance",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
