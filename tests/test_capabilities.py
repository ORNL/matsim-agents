"""Tests for the backend capability registry and ``matsim-agents capabilities``."""

from __future__ import annotations

import json
import sys
import typing
from pathlib import Path

import pytest
from typer.testing import CliRunner

from matsim_agents.active_learning.config import LLMSeedConfig
from matsim_agents.backends.dft.config import DFTConfig
from matsim_agents.backends.dft.vasp_relax import _CALC_MODE_MAP
from matsim_agents.backends.llm.provider import DEFAULT_MODELS
from matsim_agents.backends.mlip.config import MLIPConfig
from matsim_agents.cli import app
from matsim_agents.execution import capabilities as caps
from matsim_agents.execution.contracts import DFT_EVIDENCE

ROOT = Path(__file__).resolve().parents[1]


def _names(kind: str) -> set[str]:
    return {c.name for c in caps.CAPABILITIES if c.kind == kind}


def _literal(model, field: str) -> set[str]:
    return set(typing.get_args(model.model_fields[field].annotation))


def test_registry_matches_configured_backends():
    assert _names("dft") == _literal(DFTConfig, "backend")
    assert _names("mlip") == _literal(MLIPConfig, "backend")
    assert _names("llm") == set(DEFAULT_MODELS) == _literal(LLMSeedConfig, "provider")


def test_vasp_tasks_match_its_calculation_modes():
    assert set(caps.get_capability("vasp").tasks) == set(_CALC_MODE_MAP)


def test_only_dft_backends_claim_dft_evidence():
    for c in caps.CAPABILITIES:
        assert bool(DFT_EVIDENCE & set(c.evidence_levels)) == (c.kind == "dft"), c.name


def test_get_capability_is_case_insensitive_and_rejects_unknown_names():
    assert caps.get_capability("QE").name == "qe"
    with pytest.raises(KeyError, match="unknown backend"):
        caps.get_capability("castep")


def test_check_reports_missing_launcher_modules_and_keys(monkeypatch):
    monkeypatch.setattr(caps, "_module_found", lambda module: False)
    qe = caps.check(caps.get_capability("qe"), environ={})
    assert not qe.available and qe.missing == ("env MATSIM_QE_LAUNCHER",)
    absent = caps.check(caps.get_capability("qe"), environ={"MATSIM_QE_LAUNCHER": "no-such-pw.x"})
    assert absent.missing == ("executable no-such-pw.x",)
    openai = caps.check(caps.get_capability("openai"), environ={})
    assert openai.missing == ("python module langchain_openai", "env OPENAI_API_KEY")


def test_check_passes_with_launcher_on_path(monkeypatch):
    monkeypatch.setattr(caps.shutil, "which", lambda exe, path=None: f"/usr/bin/{exe}")
    env = {"MATSIM_VASP_LAUNCHER": "srun -n 4 vasp_std"}
    assert caps.check(caps.get_capability("vasp"), environ=env).available


def test_check_resolves_launcher_on_the_target_path(monkeypatch, tmp_path):
    target, caller = tmp_path / "target", tmp_path / "caller"
    for d in (target, caller):
        d.mkdir()
    exe = target / "pw-wrapper"
    exe.write_text("#!/bin/sh\n", encoding="utf-8")
    exe.chmod(0o755)
    monkeypatch.setenv("PATH", str(caller))
    qe = caps.get_capability("qe")
    env = {"MATSIM_QE_LAUNCHER": "pw-wrapper"}
    assert caps.check(qe, environ={**env, "PATH": str(target)}).available
    monkeypatch.setenv("PATH", str(target))
    assert caps.check(qe, environ={**env, "PATH": str(caller)}).missing == (
        "executable pw-wrapper",
    )
    assert not caps.check(qe, environ={**env, "PATH": ""}).available


def _fake_venv(tmp_path: Path, exit_code: int) -> Path:
    python = tmp_path / "venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text(f"#!/bin/sh\nexit {exit_code}\n", encoding="utf-8")
    python.chmod(0o755)
    return python.parents[1]


def test_check_finds_mlip_modules_in_companion_venv(monkeypatch, tmp_path):
    monkeypatch.setattr(caps, "_module_found", lambda module: False)
    venv = _fake_venv(tmp_path, 0)
    uma = caps.check(caps.get_capability("uma"), environ={"MATSIM_FAIRCHEM_VENV": str(venv)})
    assert uma.available and uma.python == str(venv / "bin" / "python")


def test_check_reports_companion_venv_without_the_modules(monkeypatch, tmp_path):
    monkeypatch.setattr(caps, "_module_found", lambda module: False)
    venv = _fake_venv(tmp_path, 1)
    mace = caps.check(caps.get_capability("mace"), environ={"MATSIM_MACE_VENV": str(venv)})
    assert mace.missing == (
        "python module mace.calculators",
        "venv $MATSIM_MACE_VENV or .venv-mace",
    )


def _boobytrapped_package(tmp_path: Path) -> str:
    pkg = tmp_path / "trapped_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("raise RuntimeError('parent ran')\n", encoding="utf-8")
    (pkg / "calculators.py").write_text("", encoding="utf-8")
    return str(tmp_path)


def test_dotted_module_lookup_does_not_run_parent_package(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(_boobytrapped_package(tmp_path))
    assert caps._module_found("trapped_pkg.calculators")
    assert not caps._module_found("trapped_pkg.missing")
    assert "trapped_pkg" not in sys.modules


def test_venv_probe_does_not_run_parent_package(monkeypatch, tmp_path):
    monkeypatch.setenv("PYTHONPATH", _boobytrapped_package(tmp_path))
    python = Path(sys.executable)
    assert caps._venv_has_modules(python, ["trapped_pkg.calculators"])
    assert not caps._venv_has_modules(python, ["trapped_pkg.missing"])


def test_check_all_filters_by_kind():
    assert {r.kind for r in caps.check_all(["mlip"], environ={})} == {"mlip"}
    assert len(caps.check_all(environ={})) == len(caps.CAPABILITIES)


def test_capabilities_doc_is_generated_from_the_registry():
    assert (ROOT / "docs" / "capabilities.md").read_text(encoding="utf-8") == caps.render_markdown()


def test_cli_check_json_and_require(monkeypatch):
    monkeypatch.setattr(caps, "_module_found", lambda module: False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    runner = CliRunner()
    result = runner.invoke(app, ["capabilities", "check", "--kind", "llm", "--json"])
    assert result.exit_code == 0, result.output
    assert {r["name"] for r in json.loads(result.output)} == _names("llm")
    assert runner.invoke(app, ["capabilities", "check", "--require", "openai"]).exit_code == 1
    assert runner.invoke(app, ["capabilities", "check", "--require", "castep"]).exit_code == 2


def test_cli_check_json_stays_parseable_when_a_required_backend_is_missing(monkeypatch):
    monkeypatch.setattr(caps, "_module_found", lambda module: False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    args = ["capabilities", "check", "--kind", "llm", "--json", "--require", "openai"]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 1
    assert {r["name"] for r in json.loads(result.stdout)} == _names("llm")
    assert "openai" in result.stderr


def test_cli_matrix_writes_the_markdown(tmp_path):
    out = tmp_path / "capabilities.md"
    result = CliRunner().invoke(app, ["capabilities", "matrix", "-o", str(out)])
    assert result.exit_code == 0, result.output
    assert out.read_text(encoding="utf-8") == caps.render_markdown()
