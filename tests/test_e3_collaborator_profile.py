"""Collaborator acquisition ceilings must not change scientific or GPU policies."""

import json
from pathlib import Path

import pytest

from benchmarks.e3 import production
from benchmarks.e3.feasibility import action_constraints
from benchmarks.e3.profiles import structures_per_round


def test_collaborator_profile_changes_only_acquisition_ceiling(monkeypatch):
    monkeypatch.delenv("MATSIM_E3_MLIP_STRUCTURES_PER_ROUND", raising=False)
    original = production.build_campaign_manifest()
    monkeypatch.setenv("MATSIM_E3_MLIP_STRUCTURES_PER_ROUND", "5000")
    large = production.build_campaign_manifest()
    production.validate_campaign_manifest(large)

    assert large["campaign_id"] == "e3-mlip-collaborator-5000"
    assert large["unique_configurations_per_material"] == 15000
    assert large["total_logical_optimizations"] == 90000
    assert len(large["rounds"]) == 18
    for key in ["optimizer", "checkpoint_sha256", "llm_guided_configuration", "action_spaces"]:
        assert large[key] == original[key]
    assert large["execution"] == original["execution"]
    for key in [
        "requested_gpu_hour_cap",
        "maximum_scheduled_gpu_hours",
        "walltime_seconds_per_array_element",
        "nodes_per_array_element",
    ]:
        assert large["resource_request"][key] == original["resource_request"][key]
    for small, big in zip(original["rounds"], large["rounds"], strict=True):
        assert big["main_target"] == big["logical_target"] == 5000
        assert big["replicate_unique_target"] == 15000
        for key in small.keys() - {"main_target", "logical_target", "replicate_unique_target"}:
            assert big[key] == small[key]
    for value in large["material_specs"].values():
        assert value["unique_target"] == 15000
    with pytest.raises(ValueError, match="campaign ID"):
        production.validate_campaign_manifest(original)

    monkeypatch.delenv("MATSIM_E3_MLIP_STRUCTURES_PER_ROUND")
    assert production.build_campaign_manifest() == original
    with pytest.raises(ValueError, match="campaign ID"):
        production.validate_campaign_manifest(large)


def test_collaborator_preflight_requires_5000_distinct_real_tasks(tmp_path, monkeypatch):
    monkeypatch.setenv("MATSIM_E3_MLIP_STRUCTURES_PER_ROUND", "5000")
    manifest = tmp_path / "manifest.json"
    production._atomic_json(manifest, production.build_campaign_manifest())
    tasks = [{"candidate_id": str(i), "structure_sha256": str(i)} for i in range(5000)]
    receipt = {
        "manifest_sha256": production._sha256_file(manifest),
        "generator_sha256": production._sha256_file(
            production.REPO / "benchmarks/e3/structure_generation.py"
        ),
        "catalog_sha256": production._sha256_file(production.REPO / "benchmarks/e3/feasibility.py"),
        "acquisition_sha256": production._sha256_file(Path(production.__file__)),
    }
    production._atomic_json(tmp_path / "generation-preflight.json", receipt)
    production._write_jsonl(tmp_path / "tasks.jsonl", tasks[:64])
    with pytest.raises(ValueError, match="5000 real tasks"):
        production._preflight_receipt(tmp_path, manifest)
    tasks[-1]["structure_sha256"] = tasks[0]["structure_sha256"]
    production._write_jsonl(tmp_path / "tasks.jsonl", tasks)
    with pytest.raises(ValueError, match="5000 distinct"):
        production._preflight_receipt(tmp_path, manifest)


def test_collaborator_ceiling_does_not_inflate_finite_parent_capacity(monkeypatch):
    strain = next(
        row for row in production._action_space("lifepo4") if row["family"] == "olivine_strain"
    )
    parent = next(
        row
        for row in production._action_space("lifepo4")
        if row["family"] == "olivine" and row["strain"] is None and row["defect"] is None
    )
    monkeypatch.delenv("MATSIM_E3_MLIP_STRUCTURES_PER_ROUND", raising=False)
    assert action_constraints(strain)["maximum_distinct_configurations"] == 192
    monkeypatch.setenv("MATSIM_E3_MLIP_STRUCTURES_PER_ROUND", "5000")
    assert action_constraints(strain)["maximum_distinct_configurations"] == 15000
    assert action_constraints(parent)["maximum_distinct_configurations"] == 1


@pytest.mark.parametrize("value", ["0", "65", "5001", "bad"])
def test_invalid_profile_is_not_silently_defaulted(monkeypatch, value):
    monkeypatch.setenv("MATSIM_E3_MLIP_STRUCTURES_PER_ROUND", value)
    with pytest.raises(ValueError, match="must be 64 or 5000"):
        structures_per_round()


def test_profile_manifest_cli_preserves_5000_cap(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MATSIM_E3_MLIP_STRUCTURES_PER_ROUND", "5000")
    manifest = tmp_path / "collaborator.json"
    assert production.main(["plan", "--output", str(manifest)]) == 0
    assert production.main(["validate", "--manifest", str(manifest)]) == 0
    assert json.loads(manifest.read_text())["total_logical_optimizations"] == 90000
    assert "90000" in capsys.readouterr().out


@pytest.mark.parametrize(("rank", "count"), [(0, 79), (63, 78)])
def test_5000_tasks_are_queued_across_64_workers(tmp_path, monkeypatch, rank, count):
    monkeypatch.setenv("MATSIM_E3_MLIP_STRUCTURES_PER_ROUND", "5000")
    monkeypatch.setattr(
        production,
        "_verified_checkpoint",
        lambda: (tmp_path / "synthetic-checkpoint", production.E3_MLIP_CHECKPOINT_SHA256),
    )
    tasks = [
        {"candidate_id": str(i), "optimizer": production._hard_optimizer_policy(300)}
        for i in range(5000)
    ]
    path = tmp_path / "tasks.jsonl"
    production._write_jsonl(path, tasks)
    called = []

    class Worker:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def run(self, request, *, seconds):
            candidate, _ = request
            assert seconds == 300
            assert all(
                candidate["optimizer"][key] is None
                for key in [
                    "relative_increase_threshold",
                    "max_cell_step_strain",
                    "max_cell_total_strain",
                    "stress_tolerance_eV_per_A3",
                ]
            )
            called.append(int(candidate["candidate_id"]))
            return {"status": "converged"}

    monkeypatch.setattr("matsim_agents.execution.deadline_worker.DeadlineWorker", Worker)
    result = production.run_worker(
        path, tmp_path / "results", rank=rank, world_size=64, hard_optimization_seconds=300
    )
    assert result == {"converged": count, "attempted": count}
    assert called == list(range(rank, 5000, 64))
