"""Typed contracts for NEB transitions: endpoints, correspondence, mechanism, images and results.

Records hold identities, status and provenance; coordinates stay in ASE ``Atoms`` files.
Atoms are matched by the stable integer ``atom_id`` array (see ``point_defects``), never by
array index. A *generated* endpoint is only a candidate; *relaxed* and *numerically_qualified*
need a method identity and a converged relaxation, and a qualified endpoint needs evidence.
These states, and the NEB result states, are deliberately distinct and are never promoted
implicitly. Nothing here establishes a mechanism or a barrier.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Annotated, Literal

import numpy as np
from ase import Atoms
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StringConstraints,
    field_serializer,
    field_validator,
    model_validator,
)

from matsim_agents.science.defects.point_defects import (
    DefectConfiguration,
    DefectType,
    ParameterValue,
    Vec3,
    geometry_fingerprint,
)

# Open identifier; known mechanisms are registered in matsim_agents.science.kinetics.transitions.
MechanismKind = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
EndpointRole = Literal["initial", "final"]
EndpointStatus = Literal["generated", "relaxed", "numerically_qualified"]
ResultState = Literal["failed", "incomplete", "unconverged", "converged_unqualified", "qualified"]

# QE neb.x needs more than three images (first and last included).
MIN_IMAGES = 4
# Endpoint method labels written by the active-learning NEB sampler: ``mlip:<model id>``.
MLIP_ENDPOINT_PREFIX = "mlip:"
CELL_TOLERANCE = 1e-6


def _freeze(value: Mapping) -> Mapping:
    return MappingProxyType(dict(sorted(value.items())))


class NebEndpoint(BaseModel):
    """Identity-level description of one endpoint structure (no coordinates)."""

    model_config = ConfigDict(frozen=True)

    structure_id: str = Field(min_length=1)
    role: EndpointRole
    parent_id: str = Field(min_length=1)
    parent_ordering_id: str | None = None
    defect_config_id: str | None = None
    defect_type: DefectType
    atom_ids: tuple[StrictInt, ...] = Field(min_length=1)
    species: tuple[str, ...]
    cell: tuple[Vec3, Vec3, Vec3]
    pbc: tuple[bool, bool, bool]
    geometry_fingerprint: str = Field(min_length=1)
    status: EndpointStatus = "generated"
    relaxation_converged: bool | None = None
    method: str | None = None  # numerical method identity for relaxed endpoints
    qualification_evidence: tuple[str, ...] = ()
    provenance: Mapping[str, str] = Field(default_factory=dict)

    @field_serializer("provenance")
    def _serialize_provenance(self, value: Mapping[str, str]) -> dict:
        return dict(value)

    @field_validator("provenance")
    @classmethod
    def _freeze_provenance(cls, value: Mapping[str, str]) -> Mapping[str, str]:
        return _freeze(value)

    @model_validator(mode="after")
    def _validate(self) -> NebEndpoint:
        if len(self.species) != len(self.atom_ids):
            raise ValueError("species and atom_ids must have the same length")
        if len(set(self.atom_ids)) != len(self.atom_ids):
            raise ValueError("atom_ids must be unique")
        if any(i < 0 for i in self.atom_ids):
            raise ValueError("atom_ids must be nonnegative")
        if not np.isfinite(np.array(self.cell)).all():
            raise ValueError("cell must be finite")
        if self.status != "generated":
            if not self.method:
                raise ValueError(f"a {self.status} endpoint needs a numerical method identity")
            if self.relaxation_converged is not True:
                raise ValueError(f"a {self.status} endpoint needs a converged relaxation")
        if self.status == "numerically_qualified" and not self.qualification_evidence:
            raise ValueError("a numerically_qualified endpoint needs qualification evidence")
        if self.status != "numerically_qualified" and self.qualification_evidence:
            raise ValueError("qualification evidence is only valid for a qualified endpoint")
        return self


def endpoint_from_atoms(
    atoms: Atoms,
    structure_id: str,
    role: EndpointRole,
    *,
    parent_id: str,
    defect_type: DefectType,
    parent_ordering_id: str | None = None,
    defect_config: DefectConfiguration | None = None,
    status: EndpointStatus = "generated",
    relaxation_converged: bool | None = None,
    method: str | None = None,
    qualification_evidence: tuple[str, ...] = (),
    provenance: Mapping[str, str] | None = None,
) -> NebEndpoint:
    """Describe ``atoms`` (which must carry an ``atom_id`` array and finite coordinates)."""
    if "atom_id" not in atoms.arrays:
        raise ValueError("structure has no atom_id array; stable identities are required")
    if not np.isfinite(atoms.positions).all():
        raise ValueError("structure has non-finite coordinates")
    return NebEndpoint(
        structure_id=structure_id,
        role=role,
        parent_id=parent_id,
        parent_ordering_id=parent_ordering_id,
        defect_config_id=None if defect_config is None else defect_config.config_id,
        defect_type=defect_type,
        atom_ids=tuple(int(i) for i in atoms.get_array("atom_id")),
        species=tuple(atoms.get_chemical_symbols()),
        cell=tuple(tuple(float(x) for x in row) for row in atoms.cell.array),
        pbc=tuple(bool(p) for p in atoms.pbc),
        geometry_fingerprint=geometry_fingerprint(atoms),
        status=status,
        relaxation_converged=relaxation_converged,
        method=method,
        qualification_evidence=qualification_evidence,
        provenance=provenance or {},
    )


class AtomCorrespondence(BaseModel):
    """One-to-one map from initial to final atoms by stable identity."""

    model_config = ConfigDict(frozen=True)

    atom_ids: tuple[StrictInt, ...]
    species: tuple[str, ...]
    initial_indices: tuple[StrictInt, ...]
    final_indices: tuple[StrictInt, ...]

    @model_validator(mode="after")
    def _validate(self) -> AtomCorrespondence:
        n = len(self.atom_ids)
        if not (len(self.species) == len(self.initial_indices) == len(self.final_indices) == n):
            raise ValueError("correspondence arrays must have equal length")
        for name, values in (
            ("atom_ids", self.atom_ids),
            ("initial_indices", self.initial_indices),
            ("final_indices", self.final_indices),
        ):
            if len(set(values)) != n:
                raise ValueError(f"{name} must be unique (one-to-one correspondence)")
        return self


def check_endpoint_compatibility(initial: NebEndpoint, final: NebEndpoint) -> None:
    """Raise ``ValueError`` unless the endpoints can be joined by a standard NEB path."""
    if (initial.role, final.role) != ("initial", "final"):
        raise ValueError("expected an initial endpoint and a final endpoint")
    if initial.parent_id != final.parent_id:
        raise ValueError("endpoints come from different parent structures")
    if initial.parent_ordering_id != final.parent_ordering_id:
        raise ValueError("endpoints come from different parent chemical orderings")
    if initial.defect_type != final.defect_type:
        raise ValueError("endpoints have different defect types")
    if len(initial.atom_ids) != len(final.atom_ids):
        raise ValueError(
            f"atom counts differ: {len(initial.atom_ids)} initial, {len(final.atom_ids)} final"
        )
    if set(initial.atom_ids) != set(final.atom_ids):
        raise ValueError("endpoints do not contain the same atomic identities")
    final_species = dict(zip(final.atom_ids, final.species, strict=True))
    for atom_id, symbol in zip(initial.atom_ids, initial.species, strict=True):
        if final_species[atom_id] != symbol:
            other = final_species[atom_id]
            raise ValueError(f"species of atom {atom_id} differs: {symbol} initial, {other} final")
    if initial.pbc != final.pbc or not all(initial.pbc):
        raise ValueError("endpoints must share fully periodic boundary conditions")
    if not np.allclose(initial.cell, final.cell, rtol=0.0, atol=CELL_TOLERANCE):
        raise ValueError("endpoint cells differ; only fixed-cell paths are supported")


def build_correspondence(initial: NebEndpoint, final: NebEndpoint) -> AtomCorrespondence:
    """Identity-based correspondence in initial atom order (checks compatibility first)."""
    check_endpoint_compatibility(initial, final)
    final_index = {atom_id: k for k, atom_id in enumerate(final.atom_ids)}
    return AtomCorrespondence(
        atom_ids=initial.atom_ids,
        species=initial.species,
        initial_indices=tuple(range(len(initial.atom_ids))),
        final_indices=tuple(final_index[i] for i in initial.atom_ids),
    )


class MigrationMechanism(BaseModel):
    """Proposed event. Always a hypothesis until geometry and energetics validate it."""

    model_config = ConfigDict(frozen=True)

    kind: MechanismKind
    moving_atom_ids: tuple[StrictInt, ...] = Field(min_length=1)
    parameters: Mapping[str, ParameterValue] = Field(default_factory=dict)
    notes: tuple[str, ...] = ()
    requires_validation: bool = True

    @field_serializer("parameters")
    def _serialize_parameters(self, value: Mapping[str, ParameterValue]) -> dict:
        return dict(value)

    @field_validator("parameters")
    @classmethod
    def _freeze_parameters(cls, value: Mapping[str, ParameterValue]) -> Mapping:
        return _freeze(value)

    @model_validator(mode="after")
    def _validate(self) -> MigrationMechanism:
        if len(set(self.moving_atom_ids)) != len(self.moving_atom_ids):
            raise ValueError("moving_atom_ids must be unique")
        return self


class NebImage(BaseModel):
    """Reference to one image structure (geometry lives in a structure file)."""

    model_config = ConfigDict(frozen=True)

    index: StrictInt = Field(ge=0)
    structure_ref: str = Field(min_length=1)
    geometry_fingerprint: str = Field(min_length=1)


class NebImageSequence(BaseModel):
    """Ordered images between fixed endpoints. A linear guess is not a minimum-energy path."""

    model_config = ConfigDict(frozen=True)

    images: tuple[NebImage, ...]
    method: str = Field(min_length=1)
    parameters: Mapping[str, ParameterValue] = Field(default_factory=dict)

    @field_serializer("parameters")
    def _serialize_parameters(self, value: Mapping[str, ParameterValue]) -> dict:
        return dict(value)

    @field_validator("parameters")
    @classmethod
    def _freeze_parameters(cls, value: Mapping[str, ParameterValue]) -> Mapping:
        return _freeze(value)

    @model_validator(mode="after")
    def _validate(self) -> NebImageSequence:
        if len(self.images) < MIN_IMAGES:
            raise ValueError(f"a NEB path needs at least {MIN_IMAGES} images")
        if [im.index for im in self.images] != list(range(len(self.images))):
            raise ValueError("image indices must be 0..n-1 in order")
        return self


class NebCalculationSpec(BaseModel):
    """Engine-independent NEB settings; engine-specific settings live with the backend."""

    model_config = ConfigDict(frozen=True)

    n_images: StrictInt
    engine: str = Field(min_length=1)
    method_id: str = Field(min_length=1)  # numerical method identity (functional, cutoffs, ...)
    ci_scheme: Literal["no-CI", "auto", "manual"] = "no-CI"
    parameters: Mapping[str, ParameterValue] = Field(default_factory=dict)

    @field_serializer("parameters")
    def _serialize_parameters(self, value: Mapping[str, ParameterValue]) -> dict:
        return dict(value)

    @field_validator("parameters")
    @classmethod
    def _freeze_parameters(cls, value: Mapping[str, ParameterValue]) -> Mapping:
        return _freeze(value)

    @field_validator("n_images")
    @classmethod
    def _validate_images(cls, value: int) -> int:
        if value < MIN_IMAGES:
            raise ValueError(f"n_images must be at least {MIN_IMAGES}, got {value}")
        return value


class NebResult(BaseModel):
    """Outcome of a NEB calculation.

    ``state`` separates a failed or incomplete run, a completed but unconverged one, a converged
    path whose evidence is not qualified, and a qualified result. Only a converged path with
    qualified endpoints may be ``qualified``; barriers are never invented for failed runs.
    """

    model_config = ConfigDict(frozen=True)

    state: ResultState
    method_id: str = Field(min_length=1)
    n_images: StrictInt = Field(ge=MIN_IMAGES)
    relative_energies_eV: tuple[float, ...] = ()
    forward_barrier_eV: float | None = Field(default=None, allow_inf_nan=False)
    reverse_barrier_eV: float | None = Field(default=None, allow_inf_nan=False)
    saddle_image_index: StrictInt | None = Field(default=None, ge=1)
    optimization_steps: StrictInt | None = Field(default=None, ge=0)
    max_force_eV_per_A: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    pathway_uncertainty_eV: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    forward_barrier_uncertainty_eV: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    reverse_barrier_uncertainty_eV: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    endpoints_qualified: bool = False
    climbing_image_used: bool = False
    warnings: tuple[str, ...] = ()
    provenance: Mapping[str, str] = Field(default_factory=dict)

    @field_serializer("provenance")
    def _serialize_provenance(self, value: Mapping[str, str]) -> dict:
        return dict(value)

    @field_validator("provenance")
    @classmethod
    def _freeze_provenance(cls, value: Mapping[str, str]) -> Mapping[str, str]:
        return _freeze(value)

    @model_validator(mode="after")
    def _validate(self) -> NebResult:
        if self.relative_energies_eV:
            if len(self.relative_energies_eV) != self.n_images:
                raise ValueError("relative_energies_eV must have one value per image")
            if not np.isfinite(np.array(self.relative_energies_eV)).all():
                raise ValueError("relative_energies_eV must be finite")
        has_barrier = self.forward_barrier_eV is not None or self.reverse_barrier_eV is not None
        if self.saddle_image_index is not None and self.saddle_image_index >= self.n_images - 1:
            raise ValueError("saddle_image_index must identify an interior image")
        if self.state in ("failed", "incomplete") and (has_barrier or self.relative_energies_eV):
            raise ValueError(f"a {self.state} calculation cannot carry energies or barriers")
        if self.state == "qualified" and not (
            self.endpoints_qualified
            and self.forward_barrier_eV is not None
            and self.reverse_barrier_eV is not None
        ):
            raise ValueError("a qualified result needs qualified endpoints and both barriers")
        if self.state != "qualified" and self.endpoints_qualified and not self.relative_energies_eV:
            raise ValueError("endpoints_qualified without energies is meaningless")
        return self


class NebTransition(BaseModel):
    """Endpoints, correspondence, mechanism and calculation settings for one migration event."""

    model_config = ConfigDict(frozen=True)

    transition_id: str = Field(min_length=1)
    initial: NebEndpoint
    final: NebEndpoint
    correspondence: AtomCorrespondence
    mechanism: MigrationMechanism
    calculation: NebCalculationSpec | None = None
    generation_parameters: Mapping[str, ParameterValue] = Field(default_factory=dict)
    provenance: Mapping[str, str] = Field(default_factory=dict)

    @field_serializer("generation_parameters", "provenance")
    def _serialize_mappings(self, value: Mapping) -> dict:
        return dict(value)

    @field_validator("generation_parameters", "provenance")
    @classmethod
    def _freeze_mappings(cls, value: Mapping) -> Mapping:
        return _freeze(value)

    @model_validator(mode="after")
    def _validate(self) -> NebTransition:
        if self.correspondence != build_correspondence(self.initial, self.final):
            raise ValueError("correspondence does not match the endpoints")
        known = set(self.initial.atom_ids)
        if not set(self.mechanism.moving_atom_ids) <= known:
            raise ValueError("mechanism names atoms that are not in the endpoints")
        return self


__all__ = [
    "AtomCorrespondence",
    "MIN_IMAGES",
    "MLIP_ENDPOINT_PREFIX",
    "MigrationMechanism",
    "NebCalculationSpec",
    "NebEndpoint",
    "NebImage",
    "NebImageSequence",
    "NebResult",
    "NebTransition",
    "build_correspondence",
    "check_endpoint_compatibility",
    "endpoint_from_atoms",
]
