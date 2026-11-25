"""CLI tests for ``external spectra-pka``. Mock executable; not nuclear-physics verification."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from typer.testing import CliRunner

from matsim_agents.cli import app
from tests.spectra_pka_fixtures import make_campaign, write_mock_executable

runner = CliRunner()


def _config(tmp_path: Path, *, with_exe: bool = True) -> Path:
    campaign = make_campaign(tmp_path)
    exe = write_mock_executable(tmp_path / "spectra-pka") if with_exe else tmp_path / "absent"
    data = {
        "external_tools": {
            "spectra_pka": {
                "enabled": True,
                "executable": str(exe),
                "nuclear_data_root": str(tmp_path / "data"),
                "workdir": str(tmp_path / "work"),
            }
        },
        "campaign": json.loads(campaign.model_dump_json()),
    }
    path = tmp_path / "campaign.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def test_help_lists_group_and_commands():
    top = runner.invoke(app, ["--help"])
    assert "external" in top.output
    sub = runner.invoke(app, ["external", "spectra-pka", "--help"])
    assert sub.exit_code == 0
    for name in ("check", "prepare", "run", "status", "import"):
        assert name in sub.output


def test_check_not_installed(tmp_path):
    result = runner.invoke(
        app, ["external", "spectra-pka", "check", "-c", str(_config(tmp_path, with_exe=False))]
    )
    assert result.exit_code == 0
    assert '"NOT_INSTALLED"' in result.output


def test_run_is_dry_by_default_then_executes(tmp_path):
    cfg = str(_config(tmp_path))
    dry = runner.invoke(app, ["external", "spectra-pka", "run", "-c", cfg])
    assert dry.exit_code == 0 and "DRY_RUN" in dry.output
    assert not list((tmp_path / "work").rglob("spka.out"))
    real = runner.invoke(app, ["external", "spectra-pka", "run", "-c", cfg, "--execute"])
    assert real.exit_code == 0 and "STRUCTURALLY_VALID" in real.output
    imported = runner.invoke(app, ["external", "spectra-pka", "import", "-c", cfg])
    assert imported.exit_code == 0 and "W182" in imported.output
    assert "NOT_VALIDATED" in imported.output


def test_run_without_install_fails_cleanly(tmp_path):
    cfg = str(_config(tmp_path, with_exe=False))
    result = runner.invoke(app, ["external", "spectra-pka", "run", "-c", cfg])
    assert result.exit_code == 1 and "not installed" in result.output


def test_shipped_example_campaign_parses(monkeypatch):
    from matsim_agents.integrations.external.spectra_pka.config import load_config_file

    for var in ("SPECTRA_PKA_EXECUTABLE_PATH", "PKA_DATA_DIRECTORY", "NEUTRON_SPECTRUM_FILE"):
        monkeypatch.setenv(var, "/tmp/placeholder")
    load_config_file(Path("examples/external/spectra-pka/campaign.yaml"))
