import csv
import json
from pathlib import Path

from matsim_agents.execution.dft_inventory import inventory_dft_runs


def _write_run(root: Path, *, backend: str, output_text: str, stages: int = 1) -> Path:
    run_dir = root / "run-1"
    work_dir = run_dir / "calculations" / backend
    work_dir.mkdir(parents=True)
    output_name = "pw.out" if backend == "qe" else "OUTCAR"
    (work_dir / output_name).write_text(output_text)
    result = {
        "run_id": "run-1",
        "run_directory": str(run_dir),
        "status": "complete",
        "stages": [
            {
                "backend": backend,
                "converged": True,
                "steps": 1,
                "artifacts": {"work_dir": str(work_dir)},
            }
            for _ in range(stages)
        ],
    }
    (run_dir / "results.json").write_text(json.dumps(result))
    return run_dir


def test_inventory_reads_qe_native_cpu_and_wall_timers(tmp_path):
    _write_run(
        tmp_path,
        backend="qe",
        output_text="     PWSCF        :  1m 6.88s CPU   12.86s WALL\n",
    )

    report = inventory_dft_runs(tmp_path)

    assert report["run_count"] == 1
    stage = report["runs"][0]["dft_stages"][0]
    assert stage["application_cpu_seconds"] == 66.88
    assert stage["application_wall_seconds"] == 12.86
    assert report["runs"][0]["allocation"] is None


def test_inventory_reads_vasp_timer_and_counts_allocation_once_per_run(tmp_path):
    run_dir = _write_run(
        tmp_path,
        backend="vasp",
        output_text="Elapsed time (sec):       15.572\n",
        stages=2,
    )
    accounting = tmp_path / "accounting.csv"
    with accounting.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["run_directory", "elapsed_seconds", "nodes", "gpus"]
        )
        writer.writeheader()
        writer.writerow({"run_directory": run_dir, "elapsed_seconds": 600, "nodes": 2, "gpus": 8})

    report = inventory_dft_runs(tmp_path, accounting)

    run = report["runs"][0]
    assert len(run["dft_stages"]) == 2
    assert all(stage["application_wall_seconds"] == 15.572 for stage in run["dft_stages"])
    assert run["allocation"]["node_hours"] == 1 / 3
    assert run["allocation"]["gpu_hours"] == 4 / 3
    assert report["allocation_record_count"] == 1


def test_inventory_leaves_allocation_unknown_without_explicit_record(tmp_path):
    _write_run(tmp_path, backend="vasp", output_text="no timer\n")

    run = inventory_dft_runs(tmp_path)["runs"][0]

    assert run["allocation"] is None
    assert run["dft_stages"][0]["application_wall_seconds"] is None


def test_accounting_csv_requires_explicit_join_schema(tmp_path):
    accounting = tmp_path / "accounting.csv"
    accounting.write_text("job_id,elapsed\n1,60\n")

    try:
        inventory_dft_runs(tmp_path, accounting)
    except ValueError as exc:
        assert "run_directory, elapsed_seconds, nodes, gpus" in str(exc)
    else:
        raise AssertionError("invalid accounting schema should be rejected")
