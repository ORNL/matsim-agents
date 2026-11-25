"""Evaluation scenarios and the comparability gate (F4).

A value is used for an objective only if every condition that objective
depends on matches the scenario. Conditions that ``PropertyConditions``
already carries are compared with :func:`conditions_match`; the
fusion-specific ones (neutron spectrum, cooling time, plasma ion and energy)
travel in :class:`EvidenceContext`. Several scenarios are reported side by
side with a worst case and a robustness ratio; they are averaged only when
the user supplies weights.
"""

from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict, Field, model_validator

from matsim_agents.campaign.property_recipes import conditions_match
from matsim_agents.execution.properties import PropertyConditions, PropertyEvidence
from matsim_agents.fusion.objectives import objective

_CORE_FIELDS = (
    "temperature_K",
    "irradiation",
    "magnetic_state",
    "defect_state",
    "transmutation_state",
    "loading",
)
_CONTEXT_FIELDS = (
    "neutron_spectrum",
    "cooling_time_s",
    "plasma_ion",
    "plasma_ion_energy_eV",
    "ion_incidence_angle_deg",
    "dose_geometry",
)
# Context fields that only constrain particular observables.
_PROPERTY_SCOPED = {"dose_geometry": frozenset({"shutdown_dose_rate"})}
_REL_TOL = 1e-9


class EvidenceContext(BaseModel):
    """Fusion conditions of one evidence record beyond ``PropertyConditions``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    neutron_spectrum: str | None = None
    cooling_time_s: float | None = Field(None, ge=0)
    plasma_ion: str | None = None
    plasma_ion_energy_eV: float | None = Field(None, gt=0)
    ion_incidence_angle_deg: float | None = Field(None, ge=0, lt=90, description="From normal.")
    dose_geometry: str | None = None


class FusionEvidence(BaseModel):
    """A property record and its fusion context."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    evidence: PropertyEvidence
    context: EvidenceContext = Field(default_factory=EvidenceContext)


class EvaluationScenario(BaseModel):
    """The operating state candidates are judged in."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scenario_id: str = Field(min_length=1)
    description: str = ""
    temperature_K: float = Field(gt=0)
    temperature_tolerance_K: float = Field(1.0, ge=0)
    irradiation: str | None = None
    neutron_spectrum: str | None = None
    cooling_time_s: float | None = Field(None, ge=0)
    plasma_ion: str | None = None
    plasma_ion_energy_eV: float | None = Field(None, gt=0)
    ion_incidence_angle_deg: float | None = Field(None, ge=0, lt=90, description="From normal.")
    dose_geometry: str | None = None
    magnetic_state: str | None = None
    defect_state: str | None = None
    transmutation_state: str | None = None
    loading: str | None = None

    def property_conditions(self, fields: tuple[str, ...] = _CORE_FIELDS) -> PropertyConditions:
        return PropertyConditions(**{f: getattr(self, f) for f in _CORE_FIELDS if f in fields})

    def mismatches(self, objective_name: str, item: FusionEvidence) -> list[str]:
        """Why ``item`` is not comparable for ``objective_name`` in this scenario."""
        return self.condition_mismatches(objective(objective_name).compatible_conditions, item)

    def condition_mismatches(self, fields: tuple[str, ...], item: FusionEvidence) -> list[str]:
        """Mismatches of ``item`` on the named condition fields only."""
        core = tuple(f for f in fields if f in _CORE_FIELDS)
        have = item.evidence.conditions
        have_sub = PropertyConditions(**{f: getattr(have, f) for f in core})
        problems = conditions_match(
            self.property_conditions(core),
            have_sub,
            temperature_tolerance_K=self.temperature_tolerance_K,
        )
        for f in fields:
            if f not in _CONTEXT_FIELDS:
                continue
            scope = _PROPERTY_SCOPED.get(f)
            if scope is not None and item.evidence.property not in scope:
                continue
            want, got = getattr(self, f), getattr(item.context, f)
            if want is None and got is None:
                continue
            if want is None or got is None or not _same(want, got):
                problems.append(f"{f} {got!r} != {want!r}")
        return [f"{item.evidence.property}: {p}" for p in problems]


def _same(a: object, b: object) -> bool:
    if isinstance(a, float | int) and isinstance(b, float | int):
        return math.isclose(float(a), float(b), rel_tol=_REL_TOL)
    return a == b


class ScenarioSet(BaseModel):
    """Several scenarios; optional user weights allow a weighted mean."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scenarios: tuple[EvaluationScenario, ...] = Field(min_length=1)
    weights: dict[str, float] | None = Field(
        None, description="User weights by scenario_id; required for any average."
    )

    @model_validator(mode="after")
    def _valid(self) -> ScenarioSet:
        ids = [s.scenario_id for s in self.scenarios]
        if len(ids) != len(set(ids)):
            raise ValueError("scenario ids must be unique")
        if self.weights is not None:
            if set(self.weights) != set(ids):
                raise ValueError("scenario weights must name every scenario exactly")
            if any(w < 0 or not math.isfinite(w) for w in self.weights.values()):
                raise ValueError("scenario weights must be finite and >= 0")
            if abs(math.fsum(self.weights.values()) - 1.0) > 1e-9:
                raise ValueError("scenario weights must sum to 1")
        return self

    def by_id(self, scenario_id: str) -> EvaluationScenario:
        for s in self.scenarios:
            if s.scenario_id == scenario_id:
                return s
        raise KeyError(scenario_id)


class ScenarioAggregate(BaseModel):
    """One candidate's scores across scenarios."""

    model_config = ConfigDict(frozen=True)

    per_scenario: dict[str, float | None]
    worst_case: float | None
    worst_scenario: str | None
    robustness: float | None = Field(
        None, description="worst / best score; 1 means scenario-insensitive."
    )
    weighted_mean: float | None = None
    notes: tuple[str, ...] = ()


def aggregate(scores: dict[str, float | None], scenarios: ScenarioSet) -> ScenarioAggregate:
    """Worst case and robustness; a mean only with user weights and no gaps."""
    declared = [s.scenario_id for s in scenarios.scenarios]
    unknown = sorted(set(scores) - set(declared))
    if unknown:
        raise ValueError(f"scores for undeclared scenario(s): {', '.join(unknown)}")
    scores = {k: scores.get(k) for k in declared}
    notes = []
    missing = sorted(k for k, v in scores.items() if v is None)
    if missing:
        notes.append(f"undetermined in scenario(s): {', '.join(missing)}")
        return ScenarioAggregate(
            per_scenario=scores, worst_case=None, worst_scenario=None, notes=tuple(notes)
        )
    worst_id = min(scores, key=lambda k: (scores[k], k))
    worst = scores[worst_id]
    best = max(scores.values())
    robustness = (worst / best) if best and best > 0 else None  # type: ignore[operator]
    mean = None
    if scenarios.weights is not None:
        mean = math.fsum(scenarios.weights[k] * v for k, v in scores.items())  # type: ignore[operator]
    elif len(scores) > 1:
        notes.append("no scenario weights supplied; no cross-scenario average")
    return ScenarioAggregate(
        per_scenario=scores,
        worst_case=worst,
        worst_scenario=worst_id,
        robustness=robustness,
        weighted_mean=mean,
        notes=tuple(notes),
    )


__all__ = [
    "EvaluationScenario",
    "EvidenceContext",
    "FusionEvidence",
    "ScenarioAggregate",
    "ScenarioSet",
    "aggregate",
]
