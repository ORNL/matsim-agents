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


def test_incomplete_example_config_is_recorded_not_raised(tmp_path) -> None:
    source = REPO / "examples/paper_cases/al_hea_fcc_cantor.yaml"
    _, record = runner.run(source, runs_dir=tmp_path)
    assert record.state == "failed"
    assert "invalid or incomplete configuration" in (record.failure_reason or "")
    assert runner.exit_code(record) == 1


def test_neb_example_runs_as_dry_run(tmp_path) -> None:
    source = REPO / "examples/radiation_damage/neb_vacancy_hop.example.yaml"
    directory, record = runner.run(source, runs_dir=tmp_path)
    assert record.state == "complete"
    assert record.summary["dry_run"] is True
    assert (directory / "neb").is_dir()


def test_external_campaign_dry_run_does_not_execute(tmp_path) -> None:
    source = REPO / "examples/external/spectra-pka/campaign.yaml"
    _, record = runner.run(source, runs_dir=tmp_path)
    assert record.state in {"complete", "failed"}
    assert record.summary.get("numerical_validation", "NOT_VALIDATED") == "NOT_VALIDATED"


def test_external_execution_is_blocked_when_the_tool_is_missing(tmp_path) -> None:
    source = REPO / "examples/external/spectra-pka/campaign.yaml"
    options = runner.RunOptions(dispatch=True)
    _, record = runner.run(source, runs_dir=tmp_path, options=options)
    assert record.state == "blocked"
    assert any("SPECTRA-PKA" in item for item in record.blockers)


def _application(tmp_path, source, **overrides):
    import yaml

    data = yaml.safe_load((REPO / source).read_text())
    for key in ("policy", "workflow_config"):
        if key in data:
            data[key] = str((REPO / source).parent / data[key])
    data.update(overrides)
    target = tmp_path / "application.yaml"
    target.write_text(yaml.safe_dump(data))
    return target


def test_cantor_application_builds_the_exact_fcc_structure(tmp_path) -> None:
    from ase.io import read

    config = _application(tmp_path, "examples/cantor_fcc_256/application.yaml", mode="generate")
    directory, record = runner.run(config, runs_dir=tmp_path / "runs")
    assert record.state == "complete"
    atoms = read(directory / "structures" / "Co52Cr51Fe51Mn51Ni51.extxyz")
    symbols = atoms.get_chemical_symbols()
    assert len(atoms) == 256
    assert {el: symbols.count(el) for el in set(symbols)} == {
        "Co": 52,
        "Cr": 51,
        "Fe": 51,
        "Mn": 51,
        "Ni": 51,
    }


def test_perovskite_application_generates_deduplicated_candidates(tmp_path) -> None:
    config = _application(tmp_path, "examples/perovskite_oxides/application.yaml", mode="generate")
    _, record = runner.run(config, runs_dir=tmp_path / "runs")
    assert record.state == "complete"
    labels = [item["candidate"] for item in record.summary["structures"]]
    assert len(labels) == len(set(labels)) > 0
    assert {label.split("/")[0] for label in labels} == {
        "SrTiO3",
        "BaTiO3",
        "CaZrO3",
        "LaAlO3",
    }


def test_relax_mode_without_relaxation_config_is_rejected(tmp_path) -> None:
    config = _application(
        tmp_path, "examples/perovskite_oxides/application.yaml", mode="mlip_relax"
    )
    with pytest.raises(ValueError, match="needs a relaxation config"):
        runner.validate(config)


def test_mlip_relax_mode_reports_a_missing_backend_as_blocked(tmp_path) -> None:
    import yaml

    relaxation = tmp_path / "relax.yaml"
    relaxation.write_text(
        yaml.safe_dump({"mode": "mlip", "structure_path": "unused", "mlip": {"backend": "mace"}})
    )
    config = _application(
        tmp_path,
        "examples/perovskite_oxides/application.yaml",
        mode="mlip_relax",
        relaxation=str(relaxation),
    )
    _, record = runner.run(config, runs_dir=tmp_path / "runs")
    assert record.state in {"blocked", "failed"}
    assert record.native_run_directory is None or record.state == "failed"


@pytest.mark.parametrize("command", ["validate", "plan", "run"])
def test_malformed_yaml_is_a_config_error_not_a_traceback(tmp_path, command):
    from typer.testing import CliRunner

    from matsim_agents.cli import app

    bad = tmp_path / "bad.yaml"
    bad.write_text("name: [unclosed\n  bad: : :\n")
    args = [command, str(bad)] + (
        ["--runs-dir", str(tmp_path / "runs")] if command == "run" else []
    )
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 2, result.output
    assert not isinstance(result.exception, yaml.YAMLError)
    assert "malformed YAML" in result.output


EXAMPLES = sorted((REPO / "examples").rglob("*.y*ml"))


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: str(p.relative_to(REPO)))
def test_every_example_validates_or_names_what_is_missing(path, monkeypatch) -> None:
    for name in ("MODEL_ROOT", "PROJ_ROOT", "MACE_CHECKPOINT"):
        monkeypatch.delenv(name, raising=False)
    try:
        kind, preflight = runner.validate(path)
    except ValueError as exc:  # unrunnable inputs and unset user inputs; never a traceback
        assert str(exc)
        return
    assert kind in runner.HANDLERS
    assert isinstance(preflight.blockers, list)


def test_investigation_examples_are_blocked_without_a_model(monkeypatch) -> None:
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    for name in ("single_llm.yaml", "multi_llm_debate.yaml"):
        kind, preflight = runner.validate(REPO / "examples" / "investigation" / name)
        assert kind == "investigation"
        assert preflight.blockers


@pytest.mark.parametrize(
    "relative",
    ["evaluation/composition-holdout.yaml", "characterization/fusion_alloy_selection.yaml"],
)
def test_auxiliary_inputs_are_refused_with_their_role(relative) -> None:
    with pytest.raises(runner.UnrunnableInput, match="not run directly"):
        runner.validate(REPO / "examples" / relative)
