"""Versioned multi-objective scoring policies.

A policy states an application's priorities: which objectives count, how much,
in which direction, and what evidence each objective accepts (its
:class:`ObjectiveDefinition`). Applications supply policies as files; the
framework holds no application's objectives. A policy does not predict
properties, evaluate candidates, or select calculations.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from matsim_agents.science.decision.objectives import ObjectiveDefinition

WEIGHT_SUM_TOLERANCE = 1e-9


class PolicyError(ValueError):
    pass


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ObjectivePolicy(_Frozen):
    name: str
    weight: float = Field(ge=0.0)
    # Desirability is always maximized; an objective named low_* is already oriented.
    direction: Literal["maximize", "minimize"]

    @model_validator(mode="after")
    def _finite(self) -> ObjectivePolicy:
        if not math.isfinite(self.weight):
            raise ValueError(f"weight of {self.name} must be finite")
        if self.direction != "maximize":
            raise ValueError(
                f"{self.name}: objectives are desirabilities and are always maximized; "
                "orient the observable in the benchmark instead"
            )
        return self


class PolicyProvenance(_Frozen):
    source: str = Field(min_length=1)
    nature: str = Field(min_length=1)
    status: str = Field(min_length=1)


class RankingOptions(_Frozen):
    pareto_enabled: bool = True
    weighted_ranking_enabled: bool = True
    require_comparable_conditions: bool = True
    reject_missing_as_zero: bool = True

    @model_validator(mode="after")
    def _supported(self) -> RankingOptions:
        # The scorer always performs these; a False setting would misrecord the calculation.
        off = sorted(k for k, v in self.model_dump().items() if not v)
        if off:
            raise ValueError(f"ranking options {off} cannot be disabled in this version")
        return self


class ScoringPolicy(_Frozen):
    policy_version: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    application: str = Field(min_length=1)
    provenance: PolicyProvenance
    objectives: tuple[ObjectivePolicy, ...] = Field(min_length=1)
    definitions: dict[str, ObjectiveDefinition]
    ranking: RankingOptions = Field(default_factory=RankingOptions)

    @model_validator(mode="after")
    def _valid(self) -> ScoringPolicy:
        names = [o.name for o in self.objectives]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"duplicate objectives {duplicates}")
        for key, spec in self.definitions.items():
            if spec.name != key:
                raise ValueError(f"definition {key!r} is named {spec.name!r}")
        unknown = sorted(set(names) - set(self.definitions))
        if unknown:
            raise ValueError(f"unknown objectives {unknown}; defined {sorted(self.definitions)}")
        missing = [n for n in self.definitions if n not in names]
        if missing:
            raise ValueError(f"missing objectives {missing}; every defined objective is weighted")
        total = math.fsum(o.weight for o in self.objectives)
        if abs(total - 1.0) > WEIGHT_SUM_TOLERANCE:
            raise ValueError(f"weights sum to {total!r}, not 1")
        return self

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(o.name for o in self.objectives)

    def definition(self, name: str) -> ObjectiveDefinition:
        try:
            return self.definitions[name]
        except KeyError:
            raise PolicyError(f"unknown objective {name!r}") from None

    @property
    def weights(self) -> dict[str, float]:
        return {o.name: o.weight for o in self.objectives}

    @property
    def directions(self) -> dict[str, str]:
        return {o.name: o.direction for o in self.objectives}

    def objective(self, name: str) -> ObjectivePolicy:
        for o in self.objectives:
            if o.name == name:
                return o
        raise KeyError(name)

    def digest(self) -> str:
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def with_weights(self, weights: dict[str, float], *, policy_version: str) -> ScoringPolicy:
        """A new, separately versioned policy; weights are never edited in place."""
        if policy_version == self.policy_version:
            raise PolicyError("changed weights need a new policy_version")
        unknown = sorted(set(weights) - set(self.weights))
        if unknown:
            raise PolicyError(f"unknown objectives {unknown}")
        objectives = tuple(
            o.model_copy(update={"weight": weights.get(o.name, o.weight)}) for o in self.objectives
        )
        return ScoringPolicy.model_validate(
            {
                **self.model_dump(),
                "policy_version": policy_version,
                "objectives": [o.model_dump() for o in objectives],
            }
        )


def load_policy(path: str | Path) -> ScoringPolicy:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise PolicyError(f"{path}: a policy is a mapping")
    try:
        return ScoringPolicy.model_validate(data)
    except ValueError as exc:
        raise PolicyError(f"{path}: {exc}") from exc


class PolicyRegistry:
    """Policies by version; one version names exactly one set of weights."""

    def __init__(self) -> None:
        self._policies: dict[str, ScoringPolicy] = {}

    def register(self, policy: ScoringPolicy) -> ScoringPolicy:
        existing = self._policies.get(policy.policy_version)
        if existing is not None and existing.digest() != policy.digest():
            raise PolicyError(
                f"policy version {policy.policy_version!r} is already registered with "
                "different content"
            )
        self._policies[policy.policy_version] = policy
        return policy

    def get(self, version: str) -> ScoringPolicy:
        if version not in self._policies:
            raise PolicyError(f"policy version {version!r} is not registered")
        return self._policies[version]

    def versions(self) -> list[str]:
        return sorted(self._policies)


__all__ = [
    "ObjectivePolicy",
    "PolicyError",
    "PolicyProvenance",
    "PolicyRegistry",
    "RankingOptions",
    "ScoringPolicy",
    "load_policy",
]
