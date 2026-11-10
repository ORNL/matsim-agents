from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.e4.aurora_dft.planning import (  # noqa: E402
    ALLOCATIONS,
    build_candidate_inventory,
    build_run_matrix,
    sample_compositions,
    validate_run_matrix,
)
from benchmarks.e4.aurora_dft.registry import (  # noqa: E402
    CalculationRegistry,
    RegistryConflict,
    qualify_relaxation,
    qualify_static,
)


@pytest.fixture(scope="module")
def candidate_plan():
    candidates = build_candidate_inventory()
    return candidates, build_run_matrix(candidates)


def test_composition_sampling_is_reproducible_and_valid():
    first = sample_compositions(("Nb", "Ta", "V", "Hf"), 20, seed=101)
    second = sample_compositions(("Nb", "Ta", "V", "Hf"), 20, seed=101)
    assert first == second
    assert len(set(first)) == 20
    assert all(sum(counts) == 128 and min(counts) >= 4 for counts in first)
    assert max(first[0]) - min(first[0]) <= 1
    assert all(any(count / 128 >= 0.35 for count in composition) for composition in first[1:5])


def test_candidate_counts_compositions_and_family_allocations(candidate_plan):
    candidates, runs = candidate_plan
    assert len(candidates) == 3060
    assert len(runs) == 64
    expected = {"bcc_refractory": 2040, "fcc_hea": 510, "lifepo4": 510}
    for family, count in expected.items():
        rows = [item for item in candidates if item["material_family"] == family]
        assert len(rows) == count
        assert all(item["allocation"] == ALLOCATIONS[family] for item in rows)
    bcc = [item for item in candidates if item["material_family"] == "bcc_refractory"]
    assert len({tuple(item["elements"]) for item in bcc}) == 22
    assert all(sum(item["composition"].values()) == 128 for item in bcc)
    assert all(set(item["elements"]) == set(item["composition"]) for item in bcc)
    fcc = [item for item in candidates if item["material_family"] == "fcc_hea"]
    assert len({tuple(item["composition"].values()) for item in fcc}) == 102
    assert all(sum(item["composition"].values()) == 128 for item in fcc)
    lifepo4 = [item for item in candidates if item["material_family"] == "lifepo4"]
    categories = [item["configuration_category"] for item in lifepo4]
    assert categories.count("pristine") == 1
    assert categories.count("antisite") == 509
    assert all(item["atom_count"] == 112 for item in lifepo4)


def test_run_matrix_assigns_every_candidate_once_and_follows_requested_tiers(candidate_plan):
    candidates, runs = candidate_plan
    validate_run_matrix(candidates, runs)
    assert [run["nodes"] for run in runs if run["material_family"] == "bcc_refractory"] == [
        tier for _ in range(4) for tier in (1, 2, 4, 8, 16, 32, 64, 128)
    ]
    assert (
        sum(run["calculation_count"] for run in runs if run["material_family"] == "bcc_refractory")
        == 2040
    )
    assert all(run["authorization"] == "not_authorized" and run["queue"] is None for run in runs)


def test_run_matrix_rejects_duplicate_assignment(candidate_plan):
    candidates, runs = candidate_plan
    duplicated = [dict(run) for run in runs]
    duplicated[1]["calculation_ids"] = duplicated[0]["calculation_ids"]
    with pytest.raises(ValueError, match="multiple production runs"):
        validate_run_matrix(candidates, duplicated)


def test_registry_rejects_wrong_allocation_and_duplicate_method(tmp_path):
    registry = CalculationRegistry(tmp_path / "registry.sqlite")
    with pytest.raises(RegistryConflict, match="allocation"):
        registry.register("calc-1", "fcc_hea", "HydraGNN", "s1", "m1")
    registry.register("calc-1", "fcc_hea", ALLOCATIONS["fcc_hea"], "s1", "m1")
    with pytest.raises(RegistryConflict, match="already registered"):
        registry.register("calc-2", "fcc_hea", ALLOCATIONS["fcc_hea"], "s1", "m1")


def test_atomic_reservation_allows_only_one_active_run(tmp_path):
    registry_path = tmp_path / "registry.sqlite"
    CalculationRegistry(registry_path).register(
        "calc-1", "bcc_refractory", ALLOCATIONS["bcc_refractory"], "s1", "m1"
    )
    results: list[str] = []
    barrier = threading.Barrier(2)

    def attempt(run_id: str) -> None:
        local_registry = CalculationRegistry(registry_path)
        barrier.wait()
        try:
            local_registry.reserve("calc-1", run_id)
            results.append("reserved")
        except RegistryConflict:
            results.append("rejected")

    threads = [threading.Thread(target=attempt, args=(run_id,)) for run_id in ("run-a", "run-b")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(results) == ["rejected", "reserved"]


def test_restart_keeps_calculation_identity_and_accumulates_cost(tmp_path):
    registry = CalculationRegistry(tmp_path / "registry.sqlite")
    registry.register("calc-1", "lifepo4", ALLOCATIONS["lifepo4"], "s1", "m1")
    assert registry.reserve("calc-1", "run-1") == 1
    registry.transition("calc-1", "RUNNING", pbs_job_id="123.server")
    registry.finish_attempt("calc-1", "INTERRUPTED", allocated_node_hours=2.5)
    registry.request_restart("calc-1")
    assert registry.reserve("calc-1", "run-2") == 2
    registry.finish_attempt("calc-1", "COMPLETED", allocated_node_hours=1.5)
    result = registry.get("calc-1")
    assert result is not None
    assert result["calculation_id"] == "calc-1"
    assert result["total_node_hours"] == 4.0
    assert [attempt["run_id"] for attempt in result["attempts"]] == ["run-1", "run-2"]


def test_relaxation_qualification_requires_ionic_and_every_scf_step():
    assert (
        qualify_relaxation(
            ionic_terminated=True,
            scf_converged_each_step=[True, False],
            max_force_eV_A=0.01,
            force_threshold_eV_A=0.02,
        )["qualification_status"]
        == "unqualified"
    )
    assert (
        qualify_relaxation(
            ionic_terminated=True,
            scf_converged_each_step=[True, True],
            max_force_eV_A=0.01,
            force_threshold_eV_A=0.02,
        )["qualification_status"]
        == "qualified"
    )
    assert qualify_static(normal_termination=True, scf_converged=True) == {
        "scf_status": "converged",
        "relaxation_status": "not_applicable",
    }

    def test_pbs_templates_are_project_specific_and_cannot_launch_work():
        template_dir = ROOT / "deployments/aurora/e4-templates"
        forbidden_commands = ("qsub", "mpiexec", "vasp_std", "vasp_gam", "vasp_ncl")
        for family, allocation in ALLOCATIONS.items():
            text = (
                template_dir
                / f"{family.replace('_refractory', '-refractory').replace('_hea', '-hea')}.pbs.tmpl"
            ).read_text()
            assert f"#PBS -A {allocation}" in text
            assert "__QUEUE_REQUIRES_ALCF_VERIFICATION__" in text
            assert "__WALLTIME_REQUIRES_PILOT__" in text
            assert "exit 0" in text
            assert all(command not in text for command in forbidden_commands)
