"""Radiation-damage CLI and report. Energies and results are synthetic fixtures."""

import json

import yaml
from typer.testing import CliRunner

from matsim_agents.cli import app
from matsim_agents.workflows.radiation_damage import run_radiation_damage
from matsim_agents.workflows.radiation_report import (
    REPORT_FILE,
    STANDING_CAVEATS,
    build_report,
    read_report,
    write_report,
)
from tests.test_neb_workflow import _fake
from tests.test_radiation_damage_workflow import _config

runner = CliRunner()


def _write_yaml(tmp_path, **over):
    cfg = _config(tmp_path, **over).model_dump(mode="json")
    path = tmp_path / "rd.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_run_is_a_dry_run_by_default_and_writes_a_report(tmp_path):
    result = runner.invoke(
        app, ["radiation-damage", "run", str(_write_yaml(tmp_path)), "--out", str(tmp_path / "run")]
    )
    assert result.exit_code == 0, result.output
    assert "dry run" in result.output and "blocker" in result.output
    data = read_report(tmp_path / "run")
    assert data["mode"] == "dry_run" and data["kinetics"] is None
    assert data["qualification"] == "not_qualified"


def test_status_and_report_commands_read_the_run(tmp_path):
    out = tmp_path / "run"
    runner.invoke(app, ["radiation-damage", "run", str(_write_yaml(tmp_path)), "--out", str(out)])
    status = runner.invoke(app, ["radiation-damage", "status", str(out)])
    assert status.exit_code == 0 and "neb_run: skipped" in status.output
    report = runner.invoke(app, ["radiation-damage", "report", str(out)])
    assert report.exit_code == 0 and "kinetics: not run" in report.output
    assert "caveat:" in report.output
    as_json = runner.invoke(app, ["radiation-damage", "report", str(out), "--json"])
    assert json.loads(as_json.output)["caveats"] == list(STANDING_CAVEATS)


def test_second_run_without_resume_fails_and_resume_works(tmp_path):
    out = tmp_path / "run"
    args = ["radiation-damage", "run", str(_write_yaml(tmp_path)), "--out", str(out)]
    assert runner.invoke(app, args).exit_code == 0
    again = runner.invoke(app, args)
    assert again.exit_code == 1 and "resume" in again.output
    assert runner.invoke(app, [*args, "--resume"]).exit_code == 0


def test_unknown_config_key_is_rejected(tmp_path):
    path = _write_yaml(tmp_path)
    data = yaml.safe_load(path.read_text())
    data["surprise"] = 1
    path.write_text(yaml.safe_dump(data))
    result = runner.invoke(
        app, ["radiation-damage", "run", str(path), "--out", str(tmp_path / "r")]
    )
    assert result.exit_code == 1 and "surprise" in result.output
    assert not (tmp_path / "r").exists()


def test_report_on_an_empty_directory_fails_cleanly(tmp_path):
    result = runner.invoke(app, ["radiation-damage", "report", str(tmp_path)])
    assert result.exit_code == 1 and "run the workflow first" in result.output


def test_capabilities_command_lists_every_capability_as_not_qualified():
    result = runner.invoke(app, ["radiation-damage", "capabilities", "--json"])
    items = json.loads(result.output)
    assert result.exit_code == 0 and len(items) >= 8
    assert {i["qualification"] for i in items} == {"not_qualified"}


def test_report_of_a_dispatched_run_marks_kinetics_exploratory(tmp_path):
    config = _config(tmp_path)
    run = run_radiation_damage(
        config,
        tmp_path / "run",
        dispatch=True,
        allow_unrelaxed_endpoints=True,
        launcher=_fake(),
    )
    report = build_report(run)
    assert report["mode"] == "dispatched"
    assert report["kinetics"]["status"] == "exploratory"
    assert report["qualification"] == "not_qualified"
    assert report["caveats"] == list(STANDING_CAVEATS)
    path = write_report(run)
    assert path.name == REPORT_FILE and read_report(run.directory) == report
    probabilities = [o["probability"] for o in report["kinetics"]["outcomes"].values()]
    assert all(0.0 <= p <= 1.0 for p in probabilities)


def test_report_with_wrong_schema_is_refused(tmp_path):
    (tmp_path / REPORT_FILE).write_text('{"schema": "x"}')
    try:
        read_report(tmp_path)
    except ValueError as exc:
        assert "not a" in str(exc)
    else:
        raise AssertionError("expected ValueError")
