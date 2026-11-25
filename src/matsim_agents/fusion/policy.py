"""Versioned performance-scoring policies for fusion plasma-facing components.

A policy only states priorities: which objectives count, how much, and in which
direction. It does not predict properties, evaluate candidates, or select
calculations; those live in separate modules.
"""

from __future__ import annotations

import hashlib
import json
import math
from importlib import resources
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

FUSION_PFC_APPLICATION = "fusion_plasma_facing_component"
FUSION_PFC_OBJECTIVES = (
    "radiation_damage_recovery",
    "thermal_conductivity",
    "mechanical_integrity",
    "thermal_shock_fatigue",
    "plasma_erosion_resistance",
    "high_temperature_stability",
    "low_hydrogen_retention",
    "low_activation_shutdown_dose",
)
WEIGHT_SUM_TOLERANCE = 1e-9
DEFAULT_POLICY_VERSION = "fusion_pfc_initial_v1"


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


class FusionScoringPolicy(_Frozen):
    policy_version: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    application: Literal["fusion_plasma_facing_component"]
    provenance: PolicyProvenance
    objectives: tuple[ObjectivePolicy, ...]
    ranking: RankingOptions = Field(default_factory=RankingOptions)

    @model_validator(mode="after")
    def _valid(self) -> FusionScoringPolicy:
        names = [o.name for o in self.objectives]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"duplicate objectives {duplicates}")
        unknown = sorted(set(names) - set(FUSION_PFC_OBJECTIVES))
        if unknown:
            raise ValueError(f"unknown objectives {unknown}; allowed {list(FUSION_PFC_OBJECTIVES)}")
        missing = [n for n in FUSION_PFC_OBJECTIVES if n not in names]
        if missing:
            raise ValueError(f"missing objectives {missing}; a policy has all eight")
        total = math.fsum(o.weight for o in self.objectives)
        if abs(total - 1.0) > WEIGHT_SUM_TOLERANCE:
            raise ValueError(f"weights sum to {total!r}, not 1")
        return self

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

    def with_weights(
        self, weights: dict[str, float], *, policy_version: str
    ) -> FusionScoringPolicy:
        """A new, separately versioned policy; weights are never edited in place."""
        if policy_version == self.policy_version:
            raise PolicyError("changed weights need a new policy_version")
        unknown = sorted(set(weights) - set(self.weights))
        if unknown:
            raise PolicyError(f"unknown objectives {unknown}")
        objectives = tuple(
            o.model_copy(update={"weight": weights.get(o.name, o.weight)}) for o in self.objectives
        )
        return FusionScoringPolicy.model_validate(
            {
                **self.model_dump(),
                "policy_version": policy_version,
                "objectives": [o.model_dump() for o in objectives],
            }
        )


def load_policy(path: str | Path) -> FusionScoringPolicy:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise PolicyError(f"{path}: a policy is a mapping")
    try:
        return FusionScoringPolicy.model_validate(data)
    except ValueError as exc:
        raise PolicyError(f"{path}: {exc}") from exc


def builtin_policy(version: str = DEFAULT_POLICY_VERSION) -> FusionScoringPolicy:
    if not version.replace("_", "").replace("-", "").replace(".", "").isalnum():
        raise PolicyError(f"invalid policy version {version!r}")
    source = resources.files("matsim_agents.fusion") / "policies" / f"{version}.yaml"
    if not source.is_file():
        raise PolicyError(f"no built-in policy {version!r}")
    with resources.as_file(source) as path:
        policy = load_policy(path)
    if policy.policy_version != version:
        raise PolicyError(f"{source.name} declares version {policy.policy_version!r}")
    return policy


class PolicyRegistry:
    """Policies by version; one version names exactly one set of weights."""

    def __init__(self) -> None:
        self._policies: dict[str, FusionScoringPolicy] = {}

    def register(self, policy: FusionScoringPolicy) -> FusionScoringPolicy:
        existing = self._policies.get(policy.policy_version)
        if existing is not None and existing.digest() != policy.digest():
            raise PolicyError(
                f"policy version {policy.policy_version!r} is already registered with "
                "different content"
            )
        self._policies[policy.policy_version] = policy
        return policy

    def get(self, version: str) -> FusionScoringPolicy:
        if version not in self._policies:
            self.register(builtin_policy(version))
        return self._policies[version]

    def versions(self) -> list[str]:
        return sorted(self._policies)


__all__ = [
    "DEFAULT_POLICY_VERSION",
    "FUSION_PFC_APPLICATION",
    "FUSION_PFC_OBJECTIVES",
    "FusionScoringPolicy",
    "ObjectivePolicy",
    "PolicyError",
    "PolicyProvenance",
    "PolicyRegistry",
    "RankingOptions",
    "builtin_policy",
    "load_policy",
]
