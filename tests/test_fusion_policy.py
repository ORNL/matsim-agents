from __future__ import annotations

import math

import pytest
import yaml
from fusion_pfc_app import PFC_POLICY

from matsim_agents.science.decision.policy import (
    PolicyError,
    PolicyRegistry,
    ScoringPolicy,
    load_policy,
)

EXPECTED = {
    "radiation_damage_recovery": 0.25,
    "thermal_conductivity": 0.20,
    "mechanical_integrity": 0.18,
    "thermal_shock_fatigue": 0.12,
    "plasma_erosion_resistance": 0.10,
    "high_temperature_stability": 0.08,
    "low_hydrogen_retention": 0.04,
    "low_activation_shutdown_dose": 0.03,
}


def _data(**changes):
    data = load_policy(PFC_POLICY).model_dump(mode="json")
    data.update(changes)
    return data


def test_example_policy_matches_the_directive():
    policy = load_policy(PFC_POLICY)
    assert policy.policy_version == "fusion_pfc_initial_v1"
    assert policy.application == "fusion_plasma_facing_component"
    assert policy.weights == EXPECTED
    assert set(policy.directions.values()) == {"maximize"}
    assert math.isclose(sum(policy.weights.values()), 1.0)
    assert policy.ranking.pareto_enabled and policy.ranking.weighted_ranking_enabled
    assert policy.ranking.require_comparable_conditions and policy.ranking.reject_missing_as_zero
    assert "Márquez" in policy.provenance.source
    assert "preliminary" in policy.provenance.nature


def test_electrical_conductivity_is_not_an_objective():
    assert "electrical_conductivity" not in load_policy(PFC_POLICY).names
    data = _data()
    data["objectives"][0]["name"] = "electrical_conductivity"
    with pytest.raises(ValueError, match="unknown objectives"):
        ScoringPolicy.model_validate(data)


def test_minimize_direction_rejected():
    data = _data()
    data["objectives"][0]["direction"] = "minimize"
    with pytest.raises(ValueError, match="always maximized"):
        ScoringPolicy.model_validate(data)


@pytest.mark.parametrize("option", ["pareto_enabled", "weighted_ranking_enabled"])
def test_unimplemented_ranking_switches_rejected(option):
    data = _data()
    data["ranking"][option] = False
    with pytest.raises(ValueError, match="cannot be disabled"):
        ScoringPolicy.model_validate(data)


def test_weights_must_sum_to_one():
    data = _data()
    data["objectives"][0]["weight"] = 0.3
    with pytest.raises(ValueError, match="sum to"):
        ScoringPolicy.model_validate(data)


def test_negative_weight_rejected():
    data = _data()
    data["objectives"][0]["weight"] = -0.25
    data["objectives"][1]["weight"] = 0.70
    with pytest.raises(ValueError):
        ScoringPolicy.model_validate(data)


def test_duplicate_and_missing_objectives_rejected():
    data = _data()
    data["objectives"][1]["name"] = data["objectives"][0]["name"]
    with pytest.raises(ValueError, match="duplicate"):
        ScoringPolicy.model_validate(data)
    data = _data()
    data["objectives"] = data["objectives"][:7]
    with pytest.raises(ValueError, match="missing objectives"):
        ScoringPolicy.model_validate(data)


def test_direction_is_required():
    data = _data()
    del data["objectives"][0]["direction"]
    with pytest.raises(ValueError):
        ScoringPolicy.model_validate(data)


def test_changed_weights_need_a_new_version():
    policy = load_policy(PFC_POLICY)
    with pytest.raises(PolicyError, match="new policy_version"):
        policy.with_weights({}, policy_version=policy.policy_version)
    shifted = policy.with_weights(
        {"radiation_damage_recovery": 0.20, "thermal_conductivity": 0.25},
        policy_version="fusion_pfc_test_v2",
    )
    assert shifted.weights["thermal_conductivity"] == 0.25
    assert policy.weights["thermal_conductivity"] == 0.20


def test_registry_rejects_reused_version_with_other_content():
    registry = PolicyRegistry()
    policy = registry.register(load_policy(PFC_POLICY))
    assert registry.get("fusion_pfc_initial_v1") is policy
    clash = ScoringPolicy.model_validate(
        policy.with_weights(
            {"radiation_damage_recovery": 0.20, "thermal_conductivity": 0.25},
            policy_version="other",
        ).model_dump()
        | {"policy_version": "fusion_pfc_initial_v1"}
    )
    with pytest.raises(PolicyError, match="already registered"):
        registry.register(clash)


def test_load_policy_from_user_file(tmp_path):
    path = tmp_path / "p.yaml"
    path.write_text(yaml.safe_dump(_data(policy_version="user_v1")), encoding="utf-8")
    assert load_policy(path).policy_version == "user_v1"
    path.write_text("- not a mapping\n", encoding="utf-8")
    with pytest.raises(PolicyError):
        load_policy(path)


def test_unregistered_version_rejected():
    with pytest.raises(PolicyError, match="not registered"):
        PolicyRegistry().get("does_not_exist")
