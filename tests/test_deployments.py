from __future__ import annotations

import importlib.util
from pathlib import Path


def test_deployment_assets_are_portable() -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "diagnostics" / "validate_deployments.py"
    spec = importlib.util.spec_from_file_location("validate_deployments", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.validate() == []


def test_perlmutter_campaign_imports_current_checkout() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (
        root
        / "deployments/perlmutter/jobs/job-campaign-formula-discovery-all-models-perlmutter.sh"
    )
    content = script.read_text(encoding="utf-8")

    source_path_export = 'export PYTHONPATH="${REPO}/src${PYTHONPATH:+:${PYTHONPATH}}"'
    assert source_path_export in content
    assert content.index(source_path_export) < content.index(".venv-uma/bin/activate")


def test_perlmutter_campaign_exposes_adaptive_acquisition_controls() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (
        root
        / "deployments/perlmutter/jobs/job-campaign-formula-discovery-all-models-perlmutter.sh"
    )
    content = script.read_text(encoding="utf-8")

    assert '--acquisition-mode \\"\\${MATSIM_CAMPAIGN_ACQUISITION_MODE:-legacy}\\"' in content
    assert '--lambda-initial \\"\\${MATSIM_CAMPAIGN_LAMBDA_INITIAL:-0.5}\\"' in content
    assert '--maximum-per-relaxed-family \\"\\${MATSIM_CAMPAIGN_MAX_PER_FAMILY:-1}\\"' in content
    assert (
        '--reserved-dft-per-formula \\"\\${MATSIM_CAMPAIGN_RESERVED_DFT_PER_FORMULA:-0}\\"'
        in content
    )
    assert "\\${MATSIM_CAMPAIGN_STOPPING_ARGS:-}" in content
    assert "\\${MATSIM_CAMPAIGN_FINAL_REVIEW_ARGS:-}" in content
