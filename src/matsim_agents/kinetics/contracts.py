"""Typed records for defect transition networks.

Missing evidence is ``None``; a zero barrier is a measured value and is never a default.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from matsim_agents.discovery.neb_contracts import MechanismKind
from matsim_agents.discovery.point_defects import DefectType

Qualification = Literal["candidate", "unqualified", "qualified"]
AbsorbingKind = Literal["recombined", "annihilated", "sink"]
NebEvidenceState = Literal[
    "failed", "incomplete", "unconverged", "converged_unqualified", "qualified"
]


class DefectState(BaseModel):
    """One node: a defect configuration in a specific parent and chemical context."""

    model_config = ConfigDict(frozen=True)

    state_id: str = Field(min_length=1)
    config_id: str | None = None
    parent_id: str = Field(min_length=1)
    parent_ordering_id: str | None = None
    defect_type: DefectType | None = None
    n_atoms: StrictInt = Field(gt=0)
    atom_ids: tuple[StrictInt, ...] | None = None
    species_counts: dict[str, int] | None = None
    absorbing: bool = False
    absorbing_kind: AbsorbingKind | None = None
    provenance: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate(self) -> DefectState:
        if self.absorbing != (self.absorbing_kind is not None):
            raise ValueError("absorbing_kind is required exactly when the state is absorbing")
        if self.atom_ids is not None and len(set(self.atom_ids)) != len(self.atom_ids):
            raise ValueError("atom_ids must be unique")
        return self


class BarrierEvidence(BaseModel):
    """Numerical origin of a barrier; ``synthetic`` marks fixture values for software tests."""

    model_config = ConfigDict(frozen=True)

    method_id: str = Field(min_length=1)
    neb_state: NebEvidenceState
    forward_barrier_eV: float = Field(ge=0, allow_inf_nan=False)
    reverse_barrier_eV: float = Field(ge=0, allow_inf_nan=False)
    endpoints_qualified: bool = False
    climbing_image_used: bool = False
    n_images: StrictInt = Field(ge=4)
    synthetic: bool = False
    source_hash: str | None = None
    artifacts: dict[str, str] = Field(default_factory=dict)
    warnings: tuple[str, ...] = ()


class TransitionEvent(BaseModel):
    """One directed elementary event between two states."""

    model_config = ConfigDict(frozen=True)

    transition_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    target_id: str = Field(min_length=1)
    mechanism: MechanismKind
    moving_atom_ids: tuple[StrictInt, ...] = Field(min_length=1)
    multiplicity: StrictInt = Field(default=1, ge=1)
    attempt_frequency_hz: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    prefactor_note: str | None = None
    evidence: BarrierEvidence | None = None
    qualification: Qualification = "candidate"
    provenance: dict[str, str] = Field(default_factory=dict)

    @property
    def barrier_eV(self) -> float | None:
        """Barrier for this direction; ``None`` means no evidence, not zero."""
        return None if self.evidence is None else self.evidence.forward_barrier_eV

    @model_validator(mode="after")
    def _validate(self) -> TransitionEvent:
        if self.source_id == self.target_id:
            raise ValueError("a transition must connect two different states")
        if len(set(self.moving_atom_ids)) != len(self.moving_atom_ids):
            raise ValueError("moving_atom_ids must be unique")
        if self.qualification == "candidate" and self.evidence is not None:
            raise ValueError("a candidate has no evidence; use 'unqualified' or 'qualified'")
        if self.qualification != "candidate" and self.evidence is None:
            raise ValueError(f"a {self.qualification} transition needs barrier evidence")
        if self.qualification == "qualified":
            ev = self.evidence
            if ev is None or ev.neb_state != "qualified" or not ev.endpoints_qualified:
                raise ValueError("qualified needs qualified NEB evidence and endpoints")
            if self.attempt_frequency_hz is None:
                raise ValueError("qualified needs an explicit attempt frequency")
        return self


class TransitionRate(BaseModel):
    """Rate of one event at one temperature, labelled with the assumptions behind it."""

    model_config = ConfigDict(frozen=True)

    transition_id: str = Field(min_length=1)
    temperature_K: float = Field(gt=0, allow_inf_nan=False)
    rate_per_s: float = Field(ge=0, allow_inf_nan=False)
    qualification: Qualification
    assumptions: tuple[str, ...] = ()


class NetworkProvenance(BaseModel):
    model_config = ConfigDict(frozen=True)

    network_id: str = Field(min_length=1)
    parent_id: str = Field(min_length=1)
    parent_ordering_id: str | None = None
    synthetic: bool = False
    notes: tuple[str, ...] = ()
    sources: dict[str, str] = Field(default_factory=dict)


__all__ = [
    "AbsorbingKind",
    "BarrierEvidence",
    "DefectState",
    "NetworkProvenance",
    "Qualification",
    "TransitionEvent",
    "TransitionRate",
]
