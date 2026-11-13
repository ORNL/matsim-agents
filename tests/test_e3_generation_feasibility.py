from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.e3 import production, structure_generation
from benchmarks.e3.feasibility import (
    InvalidActionError,
    ScientificCapacityError,
    executable_catalog,
    validate_allocations,
)

FAILURES = json.loads((Path(__file__).parent / "fixtures/e3_generation_failures.json").read_text())


@pytest.mark.parametrize("material", list(FAILURES))
def test_original_failed_counts_are_rejected_before_gpu_dispatch(material):
    original = {**FAILURES[material], "material": material}
    action_space = production._action_space(material)
    available, unavailable = executable_catalog(action_space, [])
    if material == "nb_ta_o":
        assert any(
            row["action_id"] == original["action_id"]
            and "no installed AFLOW prototypes" in row["reason"]
            for row in unavailable
        )
    with pytest.raises((ScientificCapacityError, InvalidActionError)):
        validate_allocations(
            [{"action_id": original["action_id"], "count": original["count"]}],
            available,
            original["count"],
            {},
        )
    proposal = json.dumps(
        {
            "allocations": [{"action_id": original["action_id"], "count": original["count"]}],
            "cited_evidence_ids": ["campaign-design"],
            "rationale": "original failed allocation",
        }
    )
    with pytest.raises((ScientificCapacityError, InvalidActionError)):
        production._llm_allocation(
            SimpleNamespace(invoke=lambda _: SimpleNamespace(content=proposal)),
            material,
            action_space,
            [],
            original["count"],
            {},
        )


def test_lifepo4_aliases_share_joint_capacity_in_prompt_and_validator(tmp_path):
    actions = production._action_space("lifepo4")
    pristine = [
        action for action in actions if action["family"] == "olivine" and action["size_tier"] == 4
    ]
    proposal = {
        "allocations": [{"action_id": action["action_id"], "count": 1} for action in pristine],
        "cited_evidence_ids": ["campaign-design"],
        "rationale": "alias over-allocation",
    }

    def invoke(prompt):
        assert "sum counts across ALL action_ids" in prompt
        assert all(action["action_id"] in prompt for action in pristine)
        return SimpleNamespace(content=json.dumps(proposal))

    evidence_path = tmp_path / "proposal.json"
    with pytest.raises(ScientificCapacityError, match="shared group"):
        production._llm_allocation(
            SimpleNamespace(invoke=invoke),
            "lifepo4",
            actions,
            [],
            len(pristine),
            {},
            evidence_path=evidence_path,
            context={"round": 1, "seed": 20261113},
        )
    saved = json.loads(evidence_path.read_text())
    assert saved["raw_response"] == json.dumps(proposal)
    assert saved["parsed_allocations"] == proposal["allocations"]
    assert saved["capacity_validation"]["valid"] is False
    assert saved["accepted_final_allocation"] is None
    assert len(saved["rejected_actions"]) == len(pristine)
    assert saved["round"] == 1
    with pytest.raises(RuntimeError, match="refusing to repeat"):
        production._llm_allocation(
            SimpleNamespace(invoke=lambda _: pytest.fail("must not call again")),
            "lifepo4",
            actions,
            [],
            len(pristine),
            {},
            evidence_path=evidence_path,
        )


@pytest.mark.parametrize(
    "response",
    ["not JSON", '{"allocations":7}', '{"allocations":7,"rationale":"x","cited_evidence_ids":[]}'],
)
def test_unparseable_raw_llm_proposals_are_retained(tmp_path, response):
    path = tmp_path / "proposal.json"
    with pytest.raises((ValueError, TypeError)):
        production._llm_allocation(
            SimpleNamespace(invoke=lambda _: SimpleNamespace(content=response)),
            "lifepo4",
            production._action_space("lifepo4"),
            [],
            1,
            {},
            evidence_path=path,
        )
    saved = json.loads(path.read_text())
    assert saved["raw_response"] == response
    assert saved["status"] == "rejected"


@pytest.mark.parametrize("material", list(production.MATERIALS))
def test_both_policies_use_identical_full_round_catalog_and_capacities(material):
    actions = production._action_space(material)
    floors = dict.fromkeys(map(str, production.MATERIALS[material]["tiers"]), 1)
    deterministic = production._rule_allocation(actions, [], 64, floors)
    proposal = {
        "allocations": [
            {"action_id": row["action"]["action_id"], "count": row["count"]}
            for row in deterministic
        ],
        "cited_evidence_ids": ["campaign-design"],
        "rationale": "capacity parity",
    }
    llm, _, _ = production._llm_allocation(
        SimpleNamespace(invoke=lambda _: SimpleNamespace(content=json.dumps(proposal))),
        material,
        actions,
        [],
        64,
        floors,
    )
    assert llm == deterministic
    assert sum(row["count"] for row in llm) == 64


@pytest.mark.parametrize("material,expected", [("phosphorene", 2), ("zn_formate_mof", 1)])
def test_original_seeded_exhaustion_preserves_partial_structures_and_diagnostics(
    tmp_path,
    material,
    expected,
):
    action = {**FAILURES[material], "material": material}
    with pytest.raises(structure_generation.GenerationExhausted) as caught:
        structure_generation.generate_structures(material, action, tmp_path)
    error = caught.value
    assert len(error.records) == expected
    assert all(Path(row["structure_path"]).is_file() for row in error.records)
    assert error.diagnostics["requested"] == 3
    assert error.diagnostics["generation_attempts"] == 90
    assert error.diagnostics["accepted_unique"] == expected
    assert error.diagnostics["rejection_reasons"]["symmetry_equivalent"] == 90 - expected
    assert error.diagnostics["geometry_rejections"] == {}
    assert json.loads((tmp_path / "generation-diagnostics.json").read_text()) == error.diagnostics


def test_aflow_availability_retains_actual_low_and_high_symmetry_prototypes():
    available, unavailable = executable_catalog(production._action_space("nb_ta_o"), [])
    pentoxide = [
        row
        for row in available
        if row["formula"] == "Nb2O5"
        and row["size_tier"] == 2
        and row["generator"] == "aflow_prototype"
    ]
    groups = {
        prototype["space_group"]
        for row in pentoxide
        for prototype in row["feasibility"]["available_prototypes"]
    }
    assert groups == {12, 15, 166, 194}
    assert all(row["space_group_family"] != "orthorhombic-sg16-74" for row in pentoxide)
    assert unavailable


def test_preflight_refuses_missing_and_empty_gpu_workloads(tmp_path):
    with pytest.raises(RuntimeError, match="CPU generation preflight"):
        production._preflight_receipt(tmp_path, tmp_path / "manifest.json")


def test_legacy_rejected_proposal_does_not_consume_another_llm_call(tmp_path):
    manifest = production.build_campaign_manifest()
    spec = next(row for row in manifest["rounds"] if row["material"] == "lifepo4")
    stream = tmp_path / "lifepo4/llm_guided_adaptation/seed-20261113"
    stream.mkdir(parents=True)
    production._atomic_json(
        stream / "campaign-summary.json",
        {
            "stopped_round": 1,
            "termination_reason": "round_failed",
        },
    )
    with pytest.raises(ScientificCapacityError, match="one LLM call"):
        production._materialize_round(
            manifest,
            spec,
            tmp_path,
            [],
            SimpleNamespace(invoke=lambda _: pytest.fail("legacy call budget exhausted")),
        )
    saved = json.loads((stream / "round-01/acquisition/legacy-selection.json").read_text())
    assert saved["raw_response"] is None
    assert saved["llm_calls_consumed"] == 1


def test_insufficient_shared_capacity_fails_before_either_policy_can_acquire():
    actions = [
        row
        for row in production._action_space("lifepo4")
        if row["family"] == "olivine" and row["size_tier"] == 4
    ]
    with pytest.raises(ScientificCapacityError, match="cannot satisfy the round target"):
        production._rule_allocation(actions, [], 2, {})
    with pytest.raises(ScientificCapacityError, match="cannot satisfy the round target"):
        production._llm_allocation(
            SimpleNamespace(invoke=lambda _: pytest.fail("insufficient capacity before inference")),
            "lifepo4",
            actions,
            [],
            2,
            {},
        )


def test_generation_shortfall_drives_an_explicit_deterministic_decision(tmp_path, monkeypatch):
    manifest = production.build_campaign_manifest()
    spec = next(row for row in manifest["rounds"] if row["material"] == "lifepo4")
    spec = {
        **spec,
        "strategy": "rule_based_adaptation",
        "main_target": 1,
        "minimum_main_count_by_tier": {"4": 1},
    }
    original = structure_generation.generate_structures
    calls = []

    def exhaust_first(material, action, output):
        calls.append(action["action_id"])
        if len(calls) == 1:
            raise structure_generation.GenerationExhausted(
                "test seeded shortfall",
                [],
                {"generation_attempts": 40, "accepted_unique": 0, "rejection_reasons": {}},
            )
        return original(material, action, output)

    monkeypatch.setattr(structure_generation, "generate_structures", exhaust_first)
    monkeypatch.setattr(
        production,
        "_verified_checkpoint",
        lambda: (Path("unused"), production.E3_MLIP_CHECKPOINT_SHA256),
    )
    path, tasks_path, result = production._materialize_round(manifest, spec, tmp_path, [], None)
    assert len(production._read_tasks(tasks_path)) == 1
    assert result["status"] == "generation_complete"
    assert len(calls) == 2 and calls[0] != calls[1]
    archived = path.parent / "acquisition/attempt-001/round-manifest.json"
    assert archived.is_file()
    evidence = json.loads((path.parent / "acquisition/generation-history.json").read_text())
    assert evidence[0]["action_id"] == calls[0]
    assert evidence[0]["category"] == "generation_shortfall"
    assert evidence[0]["accepted_unique"] == 0
    assert not list(tmp_path.glob("**/tasks/**/result.json"))


def test_generation_infrastructure_failure_is_not_capacity_exhaustion(tmp_path, monkeypatch):
    manifest = production.build_campaign_manifest()
    spec = next(row for row in manifest["rounds"] if row["material"] == "lifepo4")
    spec = {
        **spec,
        "strategy": "rule_based_adaptation",
        "main_target": 1,
        "minimum_main_count_by_tier": {"4": 1},
    }

    def fail(*args):
        raise OSError("test unavailable filesystem")

    monkeypatch.setattr(structure_generation, "generate_structures", fail)
    with pytest.raises(OSError, match="unavailable filesystem"):
        production._materialize_round(manifest, spec, tmp_path, [], None)
    diagnostics = list(tmp_path.glob("**/acquisition/failure-*.json"))
    assert len(diagnostics) == 1
    assert json.loads(diagnostics[0].read_text())["category"] == "infrastructure_failure"
    assert not list(tmp_path.glob("**/acquisition/attempt-*"))


@pytest.mark.parametrize("tier,capacity", [(4, 2), (8, 4), (16, 8), (32, 16)])
def test_antisite_capacity_matches_parent_symmetry_orbits(tier, capacity):
    available, _ = executable_catalog(production._action_space("lifepo4"), [])
    antisites = [
        row["feasibility"]
        for row in available
        if row["family"] == "antisite_pair" and row["size_tier"] == tier
    ]
    assert {row["maximum_distinct_configurations"] for row in antisites} == {capacity}
    assert len({row["capacity_group"] for row in antisites}) == 1


def test_strained_olivine_nonaffine_variants_are_bounded_and_distinct(tmp_path):
    import numpy as np
    from ase.io import read

    action = next(
        row
        for row in production._action_space("lifepo4")
        if row["family"] == "olivine_strain" and row["size_tier"] == 4
    )
    records = structure_generation.generate_structures(
        "lifepo4",
        {**action, "random_seed": 20261113, "count": 3},
        tmp_path,
    )
    assert len(records) == 3
    base, _ = structure_generation._olivine(
        "LiFePO4",
        4,
        "olivine",
        None,
        None,
        np.random.default_rng(0),
    )
    for record in records:
        atoms = read(record["structure_path"])
        affine_parent = base.get_scaled_positions(wrap=False) @ atoms.cell.array
        displacements = np.linalg.norm(atoms.positions - affine_parent, axis=1)
        assert displacements.max() <= 0.040001
        assert atoms.get_chemical_symbols() == base.get_chemical_symbols()
        assert record["generation"]["operation"]["maximum_local_displacement_angstrom"] == 0.04
        assert record["generation"]["generator_version"] == "2"
        assert not structure_generation.structures_symmetry_equivalent(atoms, base)
