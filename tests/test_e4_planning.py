from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.e4.consistency import compare_consistency_records  # noqa: E402
from benchmarks.e4.planning import (  # noqa: E402
    analyze_scaling,
    build_run_matrix,
    load_config,
    resolve_resource_profile,
)

CONFIG = load_config(ROOT / "benchmarks/e4/config/campaign.json")


def test_fixed_run_matrix_has_36_scaling_and_9_consistency_rows():
    rows = build_run_matrix(CONFIG)
    assert len(rows) == 45
    assert sum(r["experiment"] == "strong" for r in rows) == 18
    assert sum(r["experiment"] == "weak" for r in rows) == 18
    assert sum(r["experiment"] == "consistency" for r in rows) == 9
    assert len({r["run_id"] for r in rows}) == len(rows)
    assert {r["nodes"] for r in rows if r["experiment"] in {"strong", "weak"}} == {
        8,
        16,
        32,
        64,
        128,
        256,
    }
    assert all(r["authorization"] == "not_authorized" for r in rows)
    assert all(r["workload_tasks"] is None for r in rows if r["experiment"] == "strong")


def test_aurora_pool_placement_accounts_tiles_and_packages():
    small = resolve_resource_profile("aurora", 8, CONFIG)
    large = resolve_resource_profile("aurora", 256, CONFIG)
    assert small["counts"] == {"llm": 1, "mlip": 1, "dft": 6}
    assert large["counts"] == {"llm": 8, "mlip": 16, "dft": 232}
    assert small["logical_devices_total"] == 96
    assert small["physical_gpu_packages_total"] == 48
    assert large["logical_devices_total"] == 3072
    assert large["physical_gpu_packages_total"] == 1536
    assert small["idle_devices"] == large["idle_devices"] == 0


def test_consistency_profiles_report_physical_and_logical_devices_separately():
    expected = {
        "aurora": (12, 6, "xpu-tile"),
        "frontier": (8, 4, "rocm-gcd"),
        "perlmutter": (4, 4, "cuda-gpu"),
    }
    for facility, (logical, physical, unit) in expected.items():
        profile = resolve_resource_profile(facility, 3, CONFIG, consistency=True)
        assert profile["logical_devices_per_node"] == logical
        assert profile["physical_gpu_packages_per_node"] == physical
        assert profile["device_unit"] == unit


def _measurement(experiment, nodes, repetition, elapsed, workload_hash="fixed", tasks=8, valid=8):
    return {
        "run_id": f"{experiment}-{nodes}-{repetition}",
        "experiment": experiment,
        "facility": "aurora",
        "nodes": nodes,
        "repetition": repetition,
        "workload_hash": workload_hash,
        "attempted_tasks": tasks,
        "valid_tasks": valid,
        "failed_tasks": tasks - valid,
        "application_makespan_seconds": elapsed,
        "allocation_seconds": elapsed + 30,
        "source_commit": "abc",
        "science_config_sha256": "s",
        "resource_config_sha256": "r",
    }


def test_strong_scaling_requires_complete_fixed_workload():
    records = [_measurement("strong", 8, 1, 100), _measurement("strong", 16, 1, 60)]
    rows = analyze_scaling(records)
    assert rows[0]["speedup"] == 1.0
    assert rows[1]["speedup"] == 100 / 60
    bad = [_measurement("strong", 8, 1, 100), _measurement("strong", 16, 1, 50, valid=7)]
    assert analyze_scaling(bad)[1]["speedup"] is None
    with pytest.raises(ValueError, match="task counts differ"):
        analyze_scaling(
            [_measurement("strong", 8, 1, 100), _measurement("strong", 16, 1, 50, tasks=9)]
        )


def test_scaling_uses_smallest_node_count_as_baseline():
    strong = analyze_scaling(
        [
            _measurement("strong", 1, 1, 100),
            _measurement("strong", 2, 1, 60),
            _measurement("strong", 4, 1, 35),
            _measurement("strong", 8, 1, 20),
        ]
    )
    assert [row["speedup"] for row in strong] == [1, 100 / 60, 100 / 35, 5]
    assert [row["efficiency"] for row in strong] == [1, (100 / 60) / 2, (100 / 35) / 4, 5 / 8]


def test_weak_scaling_uses_valid_task_throughput_and_node_hours():
    rows = analyze_scaling(
        [
            _measurement("weak", 8, 1, 80),
            _measurement("weak", 16, 1, 160, tasks=16, valid=16),
        ]
    )
    assert rows[0]["weak_efficiency"] == 1.0
    assert rows[1]["weak_efficiency"] == 0.5
    assert rows[1]["allocated_node_hours"] == 16 * 190 / 3600
    with pytest.raises(ValueError, match="tasks per node"):
        analyze_scaling([_measurement("weak", 8, 1, 80), _measurement("weak", 16, 1, 160, tasks=8)])


def test_consistency_requires_matching_method_for_energy_comparison():
    records = []
    for facility in ("aurora", "frontier", "perlmutter"):
        for repetition in range(1, 4):
            records.append(
                {
                    "facility": facility,
                    "repetition": repetition,
                    "contract_sha256": "contract",
                    "structure_sha256": "structure",
                    "model_checkpoint_sha256": "model",
                    "dft_method_sha256": "qe-method",
                    "scientific_conclusion": "supported",
                    "mlip_converged": True,
                    "dft_converged": True,
                    "evidence_admissible": True,
                    "mlip_energy_eV_per_atom": -5.0,
                    "dft_energy_eV_per_atom": -5.1,
                    "max_force_eV_per_A": 0.01,
                    "geometry_rmsd_to_canonical_A": 0.01,
                }
            )
    assert compare_consistency_records(records)["status"] == "passed"
    records[-1]["dft_method_sha256"] = "different-method"
    result = compare_consistency_records(records)
    assert result["status"] == "failed"
    assert result["dft_energy_range_eV_per_atom"] is None
    assert result["same_scientific_conclusion"] is True
