"""Carry collinear magnetic configurations through point defects and NEB images.

Vacancies keep every remaining atom's parent moment (matched by ``atom_id``). Atoms that have no
parent moment (interstitials, substitutions) get a moment only from an explicit value or a named
policy that is recorded as an assumption; nothing is invented silently.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from ase import Atoms
from pydantic import ValidationError

from matsim_agents.science.magnetism.config import MagneticConfiguration
from matsim_agents.science.magnetism.results import (
    ClassificationTolerances,
    MagneticResult,
    classify_moments,
)

InsertedPolicy = Literal["explicit", "zero_documented"]


@dataclass(frozen=True)
class InsertedMoment:
    """Moment for one atom without a parent value: explicit or a documented zero."""

    moment_muB: float | None = None
    policy: InsertedPolicy = "explicit"

    def resolve(self, atom_id: int) -> float:
        if self.policy == "zero_documented":
            return 0.0
        if self.moment_muB is None:
            raise ValueError(f"atom_id {atom_id} has no parent moment; give an explicit moment")
        return float(self.moment_muB)


def _atom_ids(atoms: Atoms) -> list[int]:
    if "atom_id" not in atoms.arrays:
        raise ValueError("structure has no atom_id array; stable identities are required")
    return [int(i) for i in atoms.get_array("atom_id")]


def derive_defect_configuration(
    parent: MagneticConfiguration,
    defect_atoms: Atoms,
    *,
    defect_structure_id: str,
    defect_kind: str,
    inserted: Mapping[int, InsertedMoment] | None = None,
    inserted_policy: InsertedMoment | None = None,
) -> MagneticConfiguration:
    """Magnetic configuration of ``defect_atoms`` derived from ``parent``.

    Removed atoms are the parent ``atom_id`` values absent from the defect structure. Atoms whose
    id is new, or whose species changed, need a moment from ``inserted`` (per id) or from
    ``inserted_policy`` (applied to all of them).
    """
    inserted = dict(inserted or {})
    ids = _atom_ids(defect_atoms)
    symbols = defect_atoms.get_chemical_symbols()
    parent_index = {a: i for i, a in enumerate(parent.atom_ids)}
    removed = sorted(set(parent.atom_ids) - set(ids))

    moments: list[float] = []
    polar: list[float | None] = []
    groups: list[str] = []
    assumptions = list(parent.assumptions)
    new_groups: dict[tuple[str, float], str] = {}
    needs_moments: list[int] = []
    for atom_id, symbol in zip(ids, symbols, strict=True):
        source = parent_index.get(atom_id)
        if source is not None and parent.species[source] == symbol:
            moments.append((parent.initial_moments_muB or parent.qe_starting_magnetization)[source])
            polar.append(
                None
                if parent.qe_starting_magnetization is None
                else parent.qe_starting_magnetization[source]
            )
            groups.append(parent.group_labels[source])
            continue
        spec = inserted.get(atom_id, inserted_policy)
        if spec is None:
            needs_moments.append(atom_id)
            continue
        value = spec.resolve(atom_id)
        moments.append(value)
        polar.append(None)
        key = (symbol, value)
        label = new_groups.setdefault(key, f"{symbol}_defect_{len(new_groups)}")
        groups.append(label)
        assumptions.append(
            f"atom_id {atom_id} ({symbol}) moment {value:+.3f} muB set by {spec.policy} policy"
        )
    if needs_moments:
        raise ValueError(
            f"atom_ids {needs_moments} have no parent moment; supply an explicit moment or a "
            "documented policy for them"
        )

    use_polar = all(p is not None for p in polar) and bool(polar)
    if not use_polar and parent.qe_starting_magnetization is not None:
        assumptions.append("QE starting polarizations must be regenerated for the new atoms")
    base = {
        "structure_id": defect_structure_id,
        "atom_ids": tuple(ids),
        "species": tuple(symbols),
        "group_labels": tuple(groups),
        "initial_moments_muB": tuple(moments) if parent.initial_moments_muB is not None else None,
        "qe_starting_magnetization": tuple(polar) if use_polar else None,
        "spin_polarized": parent.spin_polarized,
        "magnetism_model": parent.magnetism_model,
        "generation_method": f"defect_derived:{defect_kind}",
        "seed": parent.seed,
        "assumptions": tuple(assumptions),
        "provenance": {
            "parent_config_id": parent.config_id,
            "parent_fingerprint": parent.fingerprint(),
            "parent_structure_id": parent.structure_id,
            "parent_mode": parent.mode,
            "defect_kind": defect_kind,
            "removed_atom_ids": removed,
            "new_atom_ids": sorted(a for a in ids if a not in parent_index),
        },
    }
    for mode in (parent.mode, "custom_collinear"):
        try:
            cfg = MagneticConfiguration(
                config_id=f"{defect_structure_id}-{mode}-tmp", mode=mode, **base
            )
        except ValidationError:
            continue
        if mode != parent.mode:
            note = (
                f"parent mode {parent.mode} no longer holds after {defect_kind}; "
                "recorded as custom_collinear"
            )
            cfg = cfg.model_copy(update={"assumptions": (*cfg.assumptions, note)})
        return cfg.model_copy(
            update={"config_id": f"{defect_structure_id}-{mode}-{cfg.fingerprint()[:10]}"}
        )
    raise ValueError(f"cannot represent the magnetic configuration after {defect_kind}")


# --------------------------------------------------------------------------- #
# NEB                                                                         #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class NEBMagneticCheck:
    compatible: bool
    problems: tuple[str, ...]
    magnetic_type_map: Mapping[int, str]


def check_neb_magnetic_compatibility(
    images: Sequence[MagneticConfiguration],
) -> NEBMagneticCheck:
    """All NEB images must share atom_ids, species, groups and initial moments, so QE type
    labels (derived from them) are stable along the path."""
    problems: list[str] = []
    if len(images) < 2:
        return NEBMagneticCheck(False, ("need at least two images",), {})
    ref = images[0]
    ref_view = _per_atom(ref)
    for position, image in enumerate(images[1:], start=1):
        if set(image.atom_ids) != set(ref.atom_ids):
            problems.append(f"image {position}: atom_ids differ from the initial image")
            continue
        view = _per_atom(image)
        for atom_id, entry in ref_view.items():
            if view[atom_id] != entry:
                problems.append(f"image {position}: atom_id {atom_id} initialization differs")
        if image.spin_polarized != ref.spin_polarized:
            problems.append(f"image {position}: spin polarization differs")
    mapping = {a: group for a, (_, group, _, _) in ref_view.items()}
    return NEBMagneticCheck(not problems, tuple(problems), mapping)


def _per_atom(cfg: MagneticConfiguration) -> dict[int, tuple]:
    out = {}
    for i, atom_id in enumerate(cfg.atom_ids):
        moment = None if cfg.initial_moments_muB is None else round(cfg.initial_moments_muB[i], 9)
        out[atom_id] = (cfg.species[i], cfg.group_labels[i], moment, cfg.spin_polarized)
    return out


@dataclass(frozen=True)
class MagneticPathReport:
    qualified_for_barrier: bool
    flags: tuple[str, ...]
    classifications: tuple[str, ...]


def flag_magnetic_path(
    results: Sequence[MagneticResult],
    *,
    tol: ClassificationTolerances | None = None,
    jump_muB: float = 0.5,
) -> MagneticPathReport:
    """Flag collapse, classification change, sign switching or large moment jumps along a path."""
    tol = tol or ClassificationTolerances()
    flags: list[str] = []
    classes = tuple(classify_moments(r, tol) for r in results)
    for i, cls in enumerate(classes):
        if cls in ("not_converged", "insufficient_evidence"):
            flags.append(f"image {i}: {cls}")
    if "nonmagnetic" in classes and not all(c == "nonmagnetic" for c in classes):
        flags.append("magnetic collapse to nonmagnetic along the path")
    usable = [c for c in classes if c not in ("not_converged", "insufficient_evidence")]
    if len(set(usable)) > 1 and "magnetic collapse to nonmagnetic along the path" not in flags:
        flags.append(f"magnetic classification changes along the path: {sorted(set(usable))}")
    for i in range(1, len(results)):
        a, b = results[i - 1], results[i]
        if not a.local_moments_muB or set(a.atom_ids) != set(b.atom_ids):
            continue
        ma = dict(zip(a.atom_ids, a.local_moments_muB, strict=True))
        mb = dict(zip(b.atom_ids, b.local_moments_muB, strict=True))
        for atom_id in sorted(ma):
            x, y = ma[atom_id], mb[atom_id]
            if abs(x) >= tol.site_moment and abs(y) >= tol.site_moment and x * y < 0:
                flags.append(f"image {i}: atom_id {atom_id} switched moment sign")
        jumps = np.abs(np.array([mb[k] - ma[k] for k in sorted(ma)]))
        if jumps.size and float(jumps.max()) > jump_muB:
            flags.append(f"image {i}: moment jump of {float(jumps.max()):.2f} muB")
    return MagneticPathReport(not flags, tuple(flags), classes)
