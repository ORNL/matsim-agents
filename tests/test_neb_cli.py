"""NEB CLI registration, dry run, and error paths (no QE needed)."""

import copy
import re
from pathlib import Path

import yaml
from typer.testing import CliRunner

from matsim_agents.cli import app

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples/radiation_damage/neb_vacancy_hop.example.yaml"
runner = CliRunner()


def _config(tmp_path, mutate=None):
    data = yaml.safe_load(EXAMPLE.read_text())
    pseudo = tmp_path / "pseudo"
    pseudo.mkdir(exist_ok=True)
    data["qe"]["pseudo_dir"] = str(pseudo)
    if mutate:
        mutate(data)
    path = tmp_path / "neb.yaml"
    path.write_text(yaml.safe_dump(copy.deepcopy(data)))
    return path


def test_neb_group_and_commands_are_registered():
    result = runner.invoke(app, ["neb", "--help"])
    assert result.exit_code == 0
    for command in ("prepare", "validate", "inspect", "resume"):
        assert command in result.output
    prepare_help = runner.invoke(app, ["neb", "prepare", "--help"]).output
    assert "--dispatch" in re.sub(r"\x1b\[[0-9;]*m", "", prepare_help)


def test_prepare_is_a_dry_run_and_writes_expected_artifacts(tmp_path):
    out = tmp_path / "w"
    result = runner.invoke(app, ["neb", "prepare", str(_config(tmp_path)), "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert "dry run" in result.output
    assert (out / "pathway" / "pathway.json").is_file()
    assert (out / "qe" / "neb.stage1.in").is_file()
    assert (out / "neb-workflow.json").is_file()
    assert not (out / "qe" / "neb.stage1.out").exists()
    checked = runner.invoke(app, ["neb", "validate", str(out)])
    assert checked.exit_code == 0 and "endpoint_relaxation_pending" in checked.output


def test_inspect_before_any_run_fails_cleanly(tmp_path):
    out = tmp_path / "w"
    runner.invoke(app, ["neb", "prepare", str(_config(tmp_path)), "--out", str(out)])
    result = runner.invoke(app, ["neb", "inspect", str(out)])
    assert result.exit_code == 1 and "no stage has been run" in result.output


def test_invalid_configs_are_rejected(tmp_path):
    def bad_key(data):
        data["surprise"] = 1

    def bad_mechanism(data):
        data["defect"]["mechanism"] = "teleport"

    def too_few_images(data):
        data["qe"]["n_images"] = 3

    for mutate, text in (
        (bad_key, "unknown keys"),
        (bad_mechanism, "unsupported mechanism"),
        (too_few_images, "n_images"),
    ):
        result = runner.invoke(
            app, ["neb", "prepare", str(_config(tmp_path, mutate)), "--out", str(tmp_path / "x")]
        )
        assert result.exit_code == 1 and text in result.output, result.output


def test_rejected_pathway_exits_nonzero_without_input(tmp_path):
    def tiny_step(data):
        data["images"]["max_step"] = 0.01

    out = tmp_path / "w"
    result = runner.invoke(
        app, ["neb", "prepare", str(_config(tmp_path, tiny_step)), "--out", str(out)]
    )
    assert result.exit_code == 1 and "rejected" in result.output
    assert not list(out.glob("qe/neb.stage*.in"))


def test_resume_without_a_run_stage_is_refused(tmp_path):
    out = tmp_path / "w"
    runner.invoke(app, ["neb", "prepare", str(_config(tmp_path)), "--out", str(out)])
    result = runner.invoke(app, ["neb", "resume", str(out), "--nstep-path", "200"])
    assert result.exit_code == 1 and "not been run" in result.output


def test_example_config_is_valid_yaml_with_known_keys():
    data = yaml.safe_load(EXAMPLE.read_text())
    assert set(data) <= {"parent", "defect", "hop_distance", "qe", "images", "method_id"}
