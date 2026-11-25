"""Collinear magnetic configuration contract keyed by stable atom identity.

Assignments are stored per ``atom_id`` (never per array position). Three quantities are kept
apart and never converted silently: the initial moment in mu_B, the QE starting spin
polarization (a fraction of the valence charge in [-1, 1]), and the converged moment.
Noncollinear magnetism, spin-orbit coupling, spin spirals and disordered-local-moment models
are not representable here and are rejected rather than treated as collinear AFM.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Annotated, Any, Literal

import numpy as np
from ase import Atoms
from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

MagneticMode = Literal[
    "nonmagnetic", "ferromagnetic", "antiferromagnetic", "ferrimagnetic", "custom_collinear"
]
MAGNETIC_MODES: tuple[str, ...] = (
    "nonmagnetic",
    "ferromagnetic",
    "antiferromagnetic",
    "ferrimagnetic",
    "custom_collinear",
)
QualificationStatus = Literal["unqualified", "software_tested", "numerically_validated"]
UNSUPPORTED_MAGNETISM = (
    "noncollinear",
    "spin_orbit",
    "spin_spiral",
    "dlm",
    "disordered_local_moment",
)

Finite = Annotated[float, Field(allow_inf_nan=False)]
GROUP_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
ZERO_TOLERANCE = 1e-9


class MagneticConfiguration(BaseModel):
    """One collinear magnetic initialization of one structure."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    structure_id: str = Field(min_length=1)
    config_id: str = Field(min_length=1)
    mode: MagneticMode
    magnetism_model: Literal["collinear"] = "collinear"
    atom_ids: tuple[StrictInt, ...] = Field(min_length=1)
    species: tuple[str, ...] = Field(min_length=1)
    group_labels: tuple[str, ...] = Field(min_length=1)
    spin_polarized: bool
    initial_moments_muB: tuple[Finite, ...] | None = None
    qe_starting_magnetization: tuple[Finite, ...] | None = None
    converged_moments_muB: tuple[Finite, ...] | None = None
    moment_unit: Literal["muB"] = "muB"
    initialization_convention: str = "initial_moment_muB"
    generation_method: str = Field(min_length=1)
    seed: int | None = None
    assumptions: tuple[str, ...] = ()
    provenance: dict[str, Any] = Field(default_factory=dict)
    qualification: QualificationStatus = "unqualified"
    converged_classification: str | None = None

    @model_validator(mode="after")
    def _consistent(self) -> MagneticConfiguration:
        n = len(self.atom_ids)
        if len(self.species) != n or len(self.group_labels) != n:
            raise ValueError("atom_ids, species and group_labels must have the same length")
        if len(set(self.atom_ids)) != n:
            raise ValueError("atom_ids must be unique")
        if any(not GROUP_PATTERN.match(label) for label in self.group_labels):
            raise ValueError("group labels must be alphanumeric identifiers starting with a letter")
        for name in (
            "initial_moments_muB",
            "qe_starting_magnetization",
            "converged_moments_muB",
        ):
            values = getattr(self, name)
            if values is not None and len(values) != n:
                raise ValueError(f"{name} must have one value per atom")
        polar = self.qe_starting_magnetization
        if polar is not None and any(abs(v) > 1.0 for v in polar):
            raise ValueError("qe_starting_magnetization values must lie in [-1, 1]")
        reference = self.initial_moments_muB
        if reference is None and polar is None and self.mode != "nonmagnetic":
            raise ValueError("a magnetic mode needs initial moments or QE polarizations")
        self._check_groups()
        self._check_mode(reference if reference is not None else self.qe_starting_magnetization)
        return self

    def _check_groups(self) -> None:
        seen: dict[str, tuple[str, float | None, float | None]] = {}
        for i, label in enumerate(self.group_labels):
            moment = None if self.initial_moments_muB is None else self.initial_moments_muB[i]
            polar = (
                None
                if self.qe_starting_magnetization is None
                else self.qe_starting_magnetization[i]
            )
            entry = (self.species[i], moment, polar)
            if seen.setdefault(label, entry) != entry:
                raise ValueError(f"group {label!r} mixes species or different initial assignments")

    def _check_mode(self, values: tuple[float, ...] | None) -> None:
        if self.mode == "nonmagnetic":
            if values is not None and any(abs(v) > ZERO_TOLERANCE for v in values):
                raise ValueError("a nonmagnetic configuration cannot carry nonzero moments")
            return
        if not self.spin_polarized:
            raise ValueError("magnetic modes require a spin-polarized calculation")
        assert values is not None
        positive = [v for v in values if v > ZERO_TOLERANCE]
        negative = [v for v in values if v < -ZERO_TOLERANCE]
        if self.mode == "ferromagnetic":
            if positive and negative:
                raise ValueError("ferromagnetic moments must all share one sign")
            if not positive and not negative:
                raise ValueError("ferromagnetic order needs at least one nonzero moment")
        elif self.mode in ("antiferromagnetic", "ferrimagnetic"):
            if not positive or not negative:
                raise ValueError(f"{self.mode} order needs antiparallel moments")
            if self.mode == "ferrimagnetic" and math.isclose(
                sum(positive), -sum(negative), abs_tol=ZERO_TOLERANCE
            ):
                raise ValueError("ferrimagnetic order needs unequal antiparallel magnitudes")

    @property
    def n_atoms(self) -> int:
        return len(self.atom_ids)

    def canonical_payload(self) -> dict[str, Any]:
        """Order-independent content (sorted by atom_id), excluding provenance."""
        order = sorted(range(self.n_atoms), key=lambda i: self.atom_ids[i])

        def pick(values):
            return None if values is None else [values[i] for i in order]

        return {
            "structure_id": self.structure_id,
            "mode": self.mode,
            "magnetism_model": self.magnetism_model,
            "spin_polarized": self.spin_polarized,
            "atom_ids": [self.atom_ids[i] for i in order],
            "species": pick(self.species),
            "initial_moments_muB": pick(self.initial_moments_muB),
            "qe_starting_magnetization": pick(self.qe_starting_magnetization),
        }

    def fingerprint(self) -> str:
        blob = json.dumps(self.canonical_payload(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()

    def moments_for_atoms(self, atoms: Atoms) -> np.ndarray:
        """Initial moments aligned to the current order of ``atoms`` (matched by atom_id)."""
        return self._aligned(atoms, self.initial_moments_muB, "initial_moments_muB")

    def polarizations_for_atoms(self, atoms: Atoms) -> np.ndarray:
        return self._aligned(atoms, self.qe_starting_magnetization, "qe_starting_magnetization")

    def validate_against(self, atoms: Atoms) -> None:
        self._aligned(atoms, tuple(0.0 for _ in self.atom_ids), "check")

    def _aligned(self, atoms: Atoms, values: tuple[float, ...] | None, name: str) -> np.ndarray:
        if values is None:
            raise ValueError(f"configuration has no {name}")
        if "atom_id" not in atoms.arrays:
            raise ValueError("structure has no atom_id array; stable identities are required")
        ids = [int(i) for i in atoms.get_array("atom_id")]
        if sorted(ids) != sorted(self.atom_ids):
            raise ValueError("magnetic configuration atom_ids do not match the structure")
        position = {atom_id: i for i, atom_id in enumerate(self.atom_ids)}
        symbols = atoms.get_chemical_symbols()
        out = np.empty(len(ids))
        for index, atom_id in enumerate(ids):
            source = position[atom_id]
            if self.species[source] != symbols[index]:
                raise ValueError(f"species mismatch for atom_id {atom_id}")
            out[index] = values[source]
        return out

    def with_converged(
        self, moments_muB: tuple[float, ...], classification: str | None = None
    ) -> MagneticConfiguration:
        """Record converged moments; the requested ``mode`` is deliberately left unchanged."""
        return self.model_copy(
            update={
                "converged_moments_muB": tuple(float(v) for v in moments_muB),
                "converged_classification": classification,
            }
        )

    def to_json(self) -> str:
        return self.model_dump_json()

    @classmethod
    def from_json(cls, text: str) -> MagneticConfiguration:
        return cls.model_validate_json(text)


def qe_polarization_from_moment(
    moment_muB: float,
    *,
    valence_electrons: float,
    verified_valence: bool,
    qe_version: str,
) -> float:
    """Explicit mu_B -> QE starting_magnetization conversion (fraction of valence charge)."""
    if not verified_valence or not qe_version:
        raise ValueError(
            "conversion needs a verified pseudopotential valence count and a QE version"
        )
    if not math.isfinite(moment_muB) or valence_electrons <= 0:
        raise ValueError("moment must be finite and valence_electrons positive")
    value = moment_muB / valence_electrons
    if abs(value) > 1.0:
        raise ValueError("moment exceeds the valence charge; QE polarization must lie in [-1, 1]")
    return value


def reject_unsupported(kind: str) -> None:
    if kind.lower() in UNSUPPORTED_MAGNETISM:
        raise ValueError(
            f"{kind} magnetism is not supported; it is not represented as collinear order"
        )
