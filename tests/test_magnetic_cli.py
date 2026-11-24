"""``matsim-agents magnetic`` commands (M11). No DFT is run."""

from __future__ import annotations

import json

import numpy as np
import pytest
from ase.build import bulk
from ase.io import write
from typer.testing import CliRunner

from matsim_agents.cli import app

runner = CliRunner()


@pytest.fixture
def fe_cr_file(tmp_path):
    atoms = bulk("Fe", "bcc", a=2.87, cubic=True) * (2, 2, 2)
    atoms.symbols[0] = "Cr"
    path = tmp_path / "fecr.extxyz"
    write(str(path), atoms)
    return path


def test_policies_lists_every_policy_as_unqualified():
    result = runner.invoke(app, ["magnetic", "policies", "--json"])
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert "fe_cr_rafm" in report
    assert all(not v["scientifically_qualified"] for v in report.values())


def test_generate_writes_configurations_and_manifest(tmp_path, fe_cr_file):
    out = tmp_path / "states"
    result = runner.invoke(app, ["magnetic", "generate", str(fe_cr_file), "-o", str(out)])
    assert result.exit_code == 0, result.output
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["policy"] == "fe_cr_rafm"
    assert [s["mode"] for s in manifest["states"]] == [
        "ferromagnetic",
        "custom_collinear",
        "nonmagnetic",
    ]
    assert len({s["fingerprint"] for s in manifest["states"]}) == 3
    first = out / f"{manifest['states'][0]['config_id']}.json"
    shown = runner.invoke(app, ["magnetic", "show", str(first)])
    assert shown.exit_code == 0 and "Mode: ferromagnetic" in shown.output
    assert "qualification: unqualified" in shown.output


def test_generate_is_deterministic(tmp_path, fe_cr_file):
    for name in ("a", "b"):
        runner.invoke(app, ["magnetic", "generate", str(fe_cr_file), "-o", str(tmp_path / name)])
    ma = json.loads((tmp_path / "a" / "manifest.json").read_text())
    mb = json.loads((tmp_path / "b" / "manifest.json").read_text())
    assert ma == mb


def test_generate_rejects_unknown_policy_and_unmatched_composition(tmp_path, fe_cr_file):
    bad = runner.invoke(
        app, ["magnetic", "generate", str(fe_cr_file), "-o", str(tmp_path / "x"), "--policy", "no"]
    )
    assert bad.exit_code == 2
    gold = bulk("Au", "fcc", a=4.08)
    gold.set_array("atom_id", np.arange(len(gold), dtype=np.int64))
    path = tmp_path / "au.extxyz"
    write(str(path), gold)
    none = runner.invoke(app, ["magnetic", "generate", str(path), "-o", str(tmp_path / "y")])
    assert none.exit_code == 1 and "No policy matches" in none.output
