from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from matsim_agents.workflows import runner
from matsim_agents.workflows.run_records import find_run, list_runs, load_record

REPO = Path(__file__).resolve().parents[1]


def test_alloy_spec_runs_and_records(tmp_path) -> None:
    spec = tmp_path / "alloy.yaml"
    spec.write_text(
        yaml.safe_dump(
            {
                "crystal_structure": "bcc",
                "repeats": [2, 2, 2],
                "lattice_parameter": 3.16,
                "ordering": "random",
                "seed": 1,
                "counts": {"W": 12, "Ta": 4},
            }
        )
    )
    directory, record = runner.run(spec, runs_dir=tmp_path / "runs")
    assert record.state == "complete"
    assert record.summary["atoms"] == 16
    assert (directory / "structure.extxyz").is_file()
    assert load_record(directory).run_id == record.run_id
    assert find_run(record.run_id, [tmp_path / "runs"]) == directory.resolve()
    assert [r.run_id for _, r in list_runs([tmp_path / "runs"])] == [record.run_id]
    assert runner.exit_code(record) == 0


def test_inconsistent_alloy_is_recorded_as_failed(tmp_path) -> None:
    spec = tmp_path / "alloy.yaml"
    spec.write_text(
        yaml.safe_dump(
            {
                "crystal_structure": "bcc",
                "repeats": [2, 2, 2],
                "lattice_parameter": 3.16,
                "counts": {"W": 12},
            }
        )
    )
    with pytest.raises(ValueError):
        runner.validate(spec)


def test_relaxation_with_missing_structure_is_blocked(tmp_path) -> None:
    config = tmp_path / "relax.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "mode": "dft",
                "structure_path": str(tmp_path / "absent.cif"),
                "dft": {"backend": "qe", "pseudo_dir": str(tmp_path)},
            }
        )
    )
    _, record = runner.run(config, runs_dir=tmp_path / "runs")
    assert record.state == "blocked"
    assert any("absent.cif" in item for item in record.blockers)
    assert any("approval" in item for item in record.blockers)
    assert runner.exit_code(record) == 3


def test_workflow_wrapper_resolves_to_native_kind() -> None:
    kind, _ = runner._resolve(REPO / "examples/workflows/debate-workflow.yaml")
    assert kind == "debate"


@pytest.mark.parametrize(
    "path",
    [
        "examples/mlip/mace-mp-medium.yaml",
        "examples/fusion_pfc/policy.yaml",
        "examples/mcp/mcp.example.yaml",
    ],
)
def test_auxiliary_inputs_are_refused_with_guidance(path) -> None:
    with pytest.raises(runner.UnrunnableInput, match="not run directly"):
        runner.validate(REPO / path)


def test_unknown_file_is_refused(tmp_path) -> None:
    stray = tmp_path / "stray.yaml"
    stray.write_text("alpha: 1\n")
    with pytest.raises(runner.UnrunnableInput, match="not a recognized"):
        runner.validate(stray)
