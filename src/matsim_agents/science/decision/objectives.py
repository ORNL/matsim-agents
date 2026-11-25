"""Objective definitions: which observables count as evidence for an objective.

An :class:`ObjectiveDefinition` names the observables that may support one
scoring objective, what they mean physically, how they are obtained, and which
tempting proxies are *not* evidence for it. Applications define their
objectives in their policy file; :func:`check_objective_evidence` rejects
proxy evidence before any score is formed. Nothing here computes a score.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from matsim_agents.core.contracts import PROPERTY_UNITS
from matsim_agents.science.properties.calculators import (
    ELASTIC_PROPERTIES,
    PropertyEvidence,
    PropertyQualification,
)

Direction = Literal["maximize", "minimize"]


class ObservableSpec(BaseModel):
    """One observable usable for an objective, with its own direction."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    direction: Direction
    meaning: str = Field(min_length=1)

    @model_validator(mode="after")
    def _known(self) -> ObservableSpec:
        if self.name not in PROPERTY_UNITS:
            raise ValueError(f"observable {self.name!r} is not a registered property")
        return self

    @property
    def unit(self) -> str:
        return PROPERTY_UNITS[self.name]


class UnsupportedProxy(BaseModel):
    """A quantity that is not evidence for an objective, and why."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    reason: str = Field(min_length=1)


class ObjectiveDefinition(BaseModel):
    """What counts as evidence for one policy objective."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    physical_meaning: str
    required: tuple[ObservableSpec, ...] = Field(min_length=1)
    optional: tuple[ObservableSpec, ...] = ()
    compatible_conditions: tuple[str, ...] = Field(
        description="Condition fields that must match the evaluation scenario."
    )
    methods: tuple[str, ...]
    evidence_requirement: str
    minimum_qualification: PropertyQualification = PropertyQualification.NUMERICALLY_VALIDATED
    unsupported_proxies: tuple[UnsupportedProxy, ...] = ()
    qualification_note: str
    # Observables that may be used only together, never alone.
    insufficient_alone: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _consistent(self) -> ObjectiveDefinition:
        names = [o.name for o in (*self.required, *self.optional)]
        if len(names) != len(set(names)):
            raise ValueError(f"{self.name}: observable listed twice")
        proxies = {p.name for p in self.unsupported_proxies}
        if proxies & set(names):
            raise ValueError(f"{self.name}: an observable cannot also be an unsupported proxy")
        return self

    def observables(self) -> tuple[ObservableSpec, ...]:
        return (*self.required, *self.optional)

    def observable(self, name: str) -> ObservableSpec | None:
        return next((o for o in self.observables() if o.name == name), None)


class ObjectiveEvidenceError(ValueError):
    """Evidence was offered for an objective it cannot support."""


def check_objective_evidence(
    spec: ObjectiveDefinition, evidence: list[PropertyEvidence]
) -> list[str]:
    """Reject proxies and foreign observables; return problems with the evidence set.

    A non-empty return means the set cannot support a value for the objective.
    """
    name = spec.name
    proxies = {p.name: p.reason for p in spec.unsupported_proxies}
    problems: list[str] = []
    allowed = {o.name for o in spec.observables()}
    for ev in evidence:
        if ev.property in proxies:
            problems.append(f"{ev.property}: {proxies[ev.property]}")
        elif ev.property not in allowed:
            problems.append(f"{ev.property} is not an observable of {name}")
        for src in ev.derived_from:
            if src in proxies:
                problems.append(f"{ev.property} derived from {src}: {proxies[src]}")
    present = {ev.property for ev in evidence if ev.is_scoreable()}
    missing = [o.name for o in spec.required if o.name not in present]
    if missing:
        problems.append(f"missing required observable(s): {', '.join(missing)}")
    if present and present <= set(spec.insufficient_alone):
        problems.append(f"{', '.join(sorted(present))} alone is insufficient for {name}")
    return problems


def mechanical_screening(evidence: list[PropertyEvidence]) -> dict[str, float]:
    """Elastic values relabeled as a screening; never an integrity score."""
    return {
        f"mechanical_screening:{ev.property}": float(ev.value)  # type: ignore[arg-type]
        for ev in evidence
        if ev.property in ELASTIC_PROPERTIES
        and ev.is_scoreable()
        and isinstance(ev.value, int | float)
    }


__all__ = [
    "ObjectiveDefinition",
    "ObjectiveEvidenceError",
    "ObservableSpec",
    "UnsupportedProxy",
    "check_objective_evidence",
    "mechanical_screening",
]
