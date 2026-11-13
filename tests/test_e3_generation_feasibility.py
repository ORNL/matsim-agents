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


@pytest.mark.parametrize("material,expected", [("zn_formate_mof", 1)])
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


@pytest.mark.parametrize("material", ["lifepo4", "phosphorene", "nb_ta_o", "zn_formate_mof"])
def test_authorized_recovery_preserves_original_and_grants_only_one_call(tmp_path, material):
    manifest = production.build_campaign_manifest()
    manifest_path = tmp_path / "manifest.json"
    production._atomic_json(manifest_path, manifest)
    stream = tmp_path / material / "llm_guided_adaptation/seed-20261113"
    original = stream / "round-01"
    original.mkdir(parents=True)
    (original / "original-evidence.txt").write_text("original failure")
    production._atomic_json(
        stream / "campaign-summary.json",
        {
            "stopped_round": 1,
            "termination_reason": "round_failed",
        },
    )
    record = production.authorize_generation_recovery(
        manifest_path,
        tmp_path,
        material,
        "Approved one replacement decision",
    )
    archive = Path(record["original_attempt_archive"])
    assert (archive / "round-01/original-evidence.txt").read_text() == "original failure"
    assert (archive / "campaign-summary.json").is_file()
    assert record["additional_llm_calls"] == 1
    assert (
        production.authorize_generation_recovery(
            manifest_path,
            tmp_path,
            material,
            "Approved one replacement decision",
        )
        == record
    )
    spec = next(row for row in manifest["rounds"] if row["material"] == material)
    calls = []

    def invoke(prompt):
        calls.append(prompt)
        assert "authorized-generation-recovery" in prompt
        return SimpleNamespace(content="invalid proposal retained")

    with pytest.raises(ValueError, match="no JSON"):
        production._materialize_round(
            manifest,
            spec,
            tmp_path,
            [],
            SimpleNamespace(invoke=invoke),
        )
    with pytest.raises(ScientificCapacityError, match="one LLM call already consumed"):
        production._materialize_round(
            manifest,
            spec,
            tmp_path,
            [],
            SimpleNamespace(invoke=invoke),
        )
    assert len(calls) == 1
    proposal = json.loads((original / "acquisition/proposal-001.json").read_text())
    assert proposal["raw_response"] == "invalid proposal retained"
    assert manifest_path.read_text() == json.dumps(manifest, indent=2, sort_keys=True) + "\n"


@pytest.mark.parametrize(
    "artifact",
    ["generation-preflight.json", "round-summary.json", "tasks/a/attempt-01/result.json"],
)
def test_recovery_refuses_generated_or_optimized_work(tmp_path, artifact):
    manifest_path = tmp_path / "manifest.json"
    production._atomic_json(manifest_path, production.build_campaign_manifest())
    stream = tmp_path / "lifepo4/llm_guided_adaptation/seed-20261113"
    path = stream / "round-01" / artifact
    path.parent.mkdir(parents=True)
    path.write_text("{}")
    production._atomic_json(
        stream / "campaign-summary.json",
        {
            "stopped_round": 1,
            "termination_reason": "round_failed",
        },
    )
    with pytest.raises(ValueError, match="refusing to replace"):
        production.authorize_generation_recovery(
            manifest_path,
            tmp_path,
            "lifepo4",
            "Approved",
        )
    assert path.is_file()
    assert not (stream / "recovery-attempts").exists()


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

    def exhaust_first(material, action, output, **kwargs):
        calls.append(action["action_id"])
        if len(calls) == 1:
            raise structure_generation.GenerationExhausted(
                "test seeded shortfall",
                [],
                {"generation_attempts": 40, "accepted_unique": 0, "rejection_reasons": {}},
            )
        return original(material, action, output, **kwargs)

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

    def fail(*args, **kwargs):
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
        assert record["generation"]["generator_version"] == "3"
        assert not structure_generation.structures_symmetry_equivalent(atoms, base)


@pytest.mark.parametrize(
    "action_id,seed,count",
    [("25c1bb34540489d1", 3180278999, 3), ("a039846d4a2a0b50", 2606930876, 4)],
)
def test_seeded_collective_reconstructions_are_distinct_and_bounded(
    tmp_path,
    action_id,
    seed,
    count,
):
    import numpy as np
    from ase.io import read

    action = next(
        row for row in production._action_space("phosphorene") if row["action_id"] == action_id
    )
    base, _ = structure_generation._phosphorene(
        "P",
        action["size_tier"],
        action["family"],
        None,
        None,
        np.random.default_rng(0),
    )
    records = structure_generation.generate_structures(
        "phosphorene",
        {**action, "random_seed": seed, "count": count},
        tmp_path,
    )
    assert len(records) == count
    for record in records:
        atoms = read(record["structure_path"])
        displacement = atoms.positions - base.positions
        assert np.linalg.norm(displacement, axis=1).max() <= 0.080001
        np.testing.assert_allclose(displacement[:, 2], 0, atol=1e-8)
        np.testing.assert_allclose(displacement.mean(axis=0), 0, atol=1e-8)
        np.testing.assert_allclose(atoms.cell.array[2], base.cell.array[2])
        assert atoms.get_chemical_symbols() == base.get_chemical_symbols()


def test_recovery_proton_variants_keep_connectivity_bond_lengths_and_rotation_bound(tmp_path):
    import numpy as np
    from ase.io import read

    action = next(
        row
        for row in production._action_space("zn_formate_mof")
        if row["action_id"] == "3b4091f2627b3b49"
    )
    base, _ = structure_generation._mof(
        action["formula"],
        24,
        "cod_parent",
        None,
        None,
        np.random.default_rng(0),
    )
    records = structure_generation.generate_structures(
        "zn_formate_mof",
        {**action, "random_seed": 2096500330, "count": 4},
        tmp_path,
    )
    assert len(records) == 4
    for record in records:
        atoms = read(record["structure_path"])
        operation = record["generation"]["operation"]
        assert len(operation["hydrogen_sites"]) == 48
        for hydrogen, carbon in zip(
            operation["hydrogen_sites"],
            operation["carbon_sites"],
            strict=True,
        ):
            before = base.get_distance(carbon, hydrogen, mic=True, vector=True)
            after = atoms.get_distance(carbon, hydrogen, mic=True, vector=True)
            np.testing.assert_allclose(np.linalg.norm(after), np.linalg.norm(before), atol=1e-7)
            angle = np.degrees(
                np.arccos(
                    np.clip(
                        np.dot(before, after) / np.linalg.norm(before) / np.linalg.norm(after),
                        -1,
                        1,
                    )
                )
            )
            assert angle <= 15.00001
        assert structure_generation._mof_connectivity_signature(
            atoms
        ) == structure_generation._mof_connectivity_signature(base)


def test_actual_seeded_pyxtal_recovery_action_accepts_modern_rng(tmp_path):
    pytest.importorskip("pyxtal")
    action = next(
        row for row in production._action_space("nb_ta_o") if row["action_id"] == "3a5dea2fa58a17de"
    )
    records = structure_generation.generate_structures(
        "nb_ta_o",
        {**action, "random_seed": 2867809719, "count": 3},
        tmp_path,
    )
    assert len(records) == 3
    assert all(16 <= record["generation"]["operation"]["space_group"] <= 74 for record in records)


def test_seeded_generation_rejects_prior_round_duplicates_inside_attempt_budget(tmp_path):
    from ase.io import read

    action = next(
        row
        for row in production._action_space("lifepo4")
        if row["family"] == "olivine_strain" and row["size_tier"] == 4
    )
    action = {**action, "random_seed": 20261113, "count": 1}
    original = structure_generation.generate_structures("lifepo4", action, tmp_path / "first")
    prior = read(original[0]["structure_path"])
    replacement = structure_generation.generate_structures(
        "lifepo4",
        action,
        tmp_path / "next",
        excluded_structures=[prior],
    )
    assert len(replacement) == 1
    assert not structure_generation.structures_symmetry_equivalent(
        read(replacement[0]["structure_path"]),
        prior,
    )
    diagnostics = json.loads((tmp_path / "next/generation-diagnostics.json").read_text())
    assert diagnostics["rejection_reasons"]["prior_symmetry_equivalent"] >= 1
    assert diagnostics["generation_attempts"] <= 40


@pytest.mark.parametrize("variant", ["identical", "translated", "reordered", "basis", "distorted"])
def test_uncached_niggli_matching_preserves_frozen_equivalence(variant, monkeypatch):
    import numpy as np
    from pymatgen.io.ase import AseAtomsAdaptor

    parent, _ = structure_generation._mof(
        "ZnC2H2O4",
        24,
        "cod_parent",
        None,
        None,
        np.random.default_rng(0),
    )
    other = parent.copy()
    if variant == "translated":
        other.positions += other.cell.array[0] * 0.317
        other.wrap()
    elif variant == "reordered":
        other = other[list(reversed(range(len(other))))]
    elif variant == "basis":
        other.set_cell(other.cell.array[[1, 0, 2]], scale_atoms=False)
    elif variant == "distorted":
        other, _ = structure_generation._mof(
            "ZnC2H2O4",
            24,
            "bounded_perturbation",
            0.005,
            None,
            np.random.default_rng(20261113),
            operation_label="proton_orientation",
        )
    matcher = structure_generation.make_structure_matcher()
    expected = matcher.fit(
        AseAtomsAdaptor.get_structure(parent),
        AseAtomsAdaptor.get_structure(other),
    )

    def reject_cached_preprocessing(*args, **kwargs):
        pytest.fail("composition-keyed reduction cache must not be used")

    monkeypatch.setattr(matcher, "_get_reduced_structure", reject_cached_preprocessing)
    assert structure_generation.structures_symmetry_equivalent(parent, other, matcher) == expected


def test_source_revalidation_refuses_future_adaptive_rounds(tmp_path):
    with pytest.raises(ValueError, match="failed first rounds"):
        production.revalidate_preflight(
            tmp_path / "manifest.json",
            tmp_path,
            "lifepo4",
            2,
        )


@pytest.mark.parametrize("entrypoint", ["prepare", "replicate"])
def test_generation_exhaustion_records_llm_ingestible_outcome_without_dispatch(
    tmp_path,
    monkeypatch,
    entrypoint,
):
    manifest_path = tmp_path / "manifest.json"
    manifest = production.build_campaign_manifest()
    production._atomic_json(manifest_path, manifest)
    stream = tmp_path / "phosphorene/llm_guided_adaptation/seed-20261113"
    round_dir = stream / "round-03"
    for index in (1, 2):
        production._atomic_json(
            stream / f"round-{index:02d}/round-summary.json",
            {
                "status": "complete",
                "task_count": 64,
            },
        )
    action = next(
        row
        for row in production._action_space("phosphorene")
        if row["action_id"] == "2547ea956193a3e8"
    )
    available, _ = executable_catalog([action], [])
    action = available[0]
    partial = [{"action": action}, {"action": action}]
    diagnostic = {
        "action": action,
        "requested": 6,
        "accepted_unique": 2,
        "generation_attempts": 180,
        "rejection_reasons": {"prior_symmetry_equivalent": 143, "symmetry_equivalent": 35},
    }

    def exhaust(*args, **kwargs):
        production._atomic_json(round_dir / "round-manifest.json", {"main_target": 64})
        production._atomic_json(round_dir / "acquisition/catalog.json", {"available": available})
        production._atomic_json(round_dir / "acquisition/shortfall-action.json", diagnostic)
        production._write_jsonl(round_dir / "acquisition/partial-candidates.jsonl", partial)
        raise production.AcquisitionExhausted("seeded batch accepted 2/6")

    monkeypatch.setattr(production, "_materialize_round", exhaust)
    monkeypatch.setattr(production, "_chat_model", lambda: object())
    monkeypatch.setattr(production, "_verified_checkpoint_sha256", lambda: "verified")
    monkeypatch.setattr(
        production,
        "_execute_round_tasks",
        lambda *args, **kwargs: pytest.fail("no GPU dispatch on generation exhaustion"),
    )
    if entrypoint == "prepare":
        outcome = production.prepare_round(manifest_path, tmp_path, "phosphorene", 3)
    else:
        production._atomic_json(
            stream / "execution-policy.json",
            {
                "campaign_id": manifest["campaign_id"],
                "manifest_sha256": production._sha256_file(manifest_path),
                "hard_optimization_seconds": 300,
                "round_seconds": 1800,
                "workers": 64,
            },
        )
        outcome = production.run_replicate(
            manifest_path,
            tmp_path,
            "phosphorene",
            "llm_guided_adaptation",
            20261113,
            hard_optimization_seconds=300,
            round_index=3,
        )
    assert outcome["status"] == "stopped_generation_exhausted"
    assert outcome["round_complete"] is False
    assert outcome["new_optimization_evidence"] is False
    assert outcome["hypotheses_updated"] is False
    assert outcome["optimizations_attempted"] == 0
    assert outcome["requested"] == 64 and outcome["accepted_unique"] == 2
    assert outcome["shortfalls"] == [diagnostic]
    assert set(outcome["remaining_capacity_by_group"].values()) == {0}
    compact = production._compact_history([outcome])[0]
    assert compact["shortfalls"] == [diagnostic]
    assert compact["hypotheses_updated"] is False
    assert not (round_dir / "round-summary.json").exists()
    assert not (round_dir / "generation-preflight.json").exists()
    assert len(production._read_tasks(round_dir / "acquisition/partial-candidates.jsonl")) == 2


@pytest.mark.parametrize(
    "error", [OSError("filesystem failed"), InvalidActionError("invalid action")]
)
def test_preparation_does_not_disguise_infrastructure_or_invalid_requests(
    tmp_path,
    monkeypatch,
    error,
):
    manifest_path = tmp_path / "manifest.json"
    production._atomic_json(manifest_path, production.build_campaign_manifest())
    stream = tmp_path / "phosphorene/llm_guided_adaptation/seed-20261113"
    for index in (1, 2):
        production._atomic_json(
            stream / f"round-{index:02d}/round-summary.json",
            {
                "status": "complete",
                "task_count": 64,
            },
        )
    monkeypatch.setattr(production, "_chat_model", lambda: object())

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(production, "_materialize_round", fail)
    with pytest.raises(type(error), match=str(error)):
        production.prepare_round(manifest_path, tmp_path, "phosphorene", 3)
    assert not list(tmp_path.glob("**/outcome.json"))


def test_prepare_cli_exhaustion_blocks_submission_without_traceback(tmp_path, monkeypatch, capsys):
    outcome = {"status": "stopped_generation_exhausted", "round_complete": False}
    round_dir = tmp_path / "phosphorene/llm_guided_adaptation/seed-20261113/round-01"

    def supervise(*args, **kwargs):
        production._atomic_json(round_dir / "acquisition/outcome.json", outcome)

    monkeypatch.setattr(production, "_run_supervised_round", supervise)
    monkeypatch.setattr(
        production,
        "_preflight_receipt",
        lambda *args: pytest.fail("no successful receipt"),
    )
    assert (
        production.main(
            [
                "prepare",
                "--manifest",
                str(tmp_path / "manifest.json"),
                "--output-root",
                str(tmp_path),
                "--material",
                "phosphorene",
            ]
        )
        == 3
    )
    assert json.loads(capsys.readouterr().out) == outcome


def test_internal_prepare_reports_scientific_stop_to_supervisor(tmp_path, monkeypatch, capsys):
    outcome = {"status": "stopped_generation_exhausted", "round_complete": False}
    monkeypatch.setattr(production, "prepare_round", lambda *args: outcome)
    assert (
        production.main(
            [
                "prepare-round",
                "--manifest",
                str(tmp_path / "manifest.json"),
                "--output-root",
                str(tmp_path),
                "--material",
                "phosphorene",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out) == outcome


@pytest.mark.parametrize("changed", ["manifest", "tasks", "structure"])
def test_source_revalidation_rejects_changed_original_evidence(tmp_path, changed):
    import hashlib

    manifest_path = tmp_path / "manifest.json"
    production._atomic_json(manifest_path, production.build_campaign_manifest())
    round_dir = tmp_path / "lifepo4/llm_guided_adaptation/seed-20261113/round-01"
    round_dir.mkdir(parents=True)
    structure = round_dir / "original.extxyz"
    structure.write_text("changed original")
    task = {
        "candidate_id": "original",
        "structure_file_sha256": "original-file-digest",
        "structure_sha256": "original-geometry-digest",
        "structure_path": str(structure),
    }
    production._write_jsonl(round_dir / "tasks.jsonl", [task])
    production._atomic_json(
        round_dir / "generation-preflight.json",
        {
            "generation_seconds": 100,
            "manifest_sha256": (
                "changed"
                if changed == "manifest"
                else hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            ),
            "structures": {} if changed == "tasks" else {"original": "original-file-digest"},
        },
    )
    with pytest.raises(
        ValueError,
        match={
            "manifest": "cannot change",
            "tasks": "identities changed",
            "structure": "original structure changed",
        }[changed],
    ):
        production.revalidate_preflight(manifest_path, tmp_path, "lifepo4", 1)
    assert not list(round_dir.glob("acquisition/source-revalidation-*"))
