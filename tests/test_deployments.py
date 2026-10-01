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


def test_perlmutter_multinode_server_extends_raylet_startup_wait() -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "deployments/perlmutter/jobs/job-serve-multinode-perlmutter.sh"
    content = script.read_text(encoding="utf-8")

    assert "RAY_raylet_start_wait_time_s=${RAY_raylet_start_wait_time_s:-300}" in content


def test_perlmutter_campaign_exposes_adaptive_acquisition_controls() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (
        root
        / "deployments/perlmutter/jobs/job-campaign-formula-discovery-all-models-perlmutter.sh"
    )
    content = script.read_text(encoding="utf-8")

    assert (
        '--acquisition-mode \\"\\${MATSIM_CAMPAIGN_ACQUISITION_MODE:-insertion-order}\\"'
        in content
    )
    assert '--lambda-initial \\"\\${MATSIM_CAMPAIGN_LAMBDA_INITIAL:-0.5}\\"' in content
    assert '--maximum-per-relaxed-family \\"\\${MATSIM_CAMPAIGN_MAX_PER_FAMILY:-1}\\"' in content
    assert (
        '--reserved-dft-per-formula \\"\\${MATSIM_CAMPAIGN_RESERVED_DFT_PER_FORMULA:-0}\\"'
        in content
    )
    assert "\\${MATSIM_CAMPAIGN_STOPPING_ARGS:-}" in content
    assert "\\${MATSIM_CAMPAIGN_FINAL_REVIEW_ARGS:-}" in content
    assert (
        '--degeneracy-tolerance-ev-per-atom '
        '\\"\\${MATSIM_CAMPAIGN_DEGENERACY_TOLERANCE_EV_PER_ATOM:-0.01}\\"'
        in content
    )


def test_perlmutter_campaign_supports_debate_only_and_uma_only_modes() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (
        root
        / "deployments/perlmutter/jobs/job-campaign-formula-discovery-all-models-perlmutter.sh"
    )
    content = script.read_text(encoding="utf-8")

    discovery_call = content.index("campaign_formula_discovery.py")
    debate_exit = content.index(
        'if [[ "$CAMPAIGN_MODE" == "debate-only" || "$CAMPAIGN_MODE" == '
        '"single-llm-once" ]]',
        discovery_call,
    )
    execution_call = content.index("campaign_execute.py")
    assert discovery_call < debate_exit < execution_call
    assert '[[ "$CAMPAIGN_MODE" == "single-llm-once" ]] && EXPECTED_NODES=1' in content
    assert "NAMES=(glm-4.7-flash)" in content
    assert "DISCOVERY_ARGS+=(--single-call)" in content
    assert (
        '[[ "$CAMPAIGN_MODE" == "debate-only" || "$CAMPAIGN_MODE" == '
        '"single-llm-once" ]]' in content
    )
    assert 'EXECUTION_ARGS=(--execution-mode "$CAMPAIGN_MODE")' in content
    assert '[[ "$CAMPAIGN_MODE" == "dft" ]] && EXECUTION_ARGS+=(--approve-dft)' in content
    assert '[[ "$CAMPAIGN_MODE" == "dft" && "${MATSIM_CAMPAIGN_DFT_REFINE:-1}"' in content
    assert '[[ "$CAMPAIGN_MODE" == "dft" && "${MATSIM_CAMPAIGN_RETRAIN:-0}"' in content

    executor_path = root / "deployments/perlmutter/jobs/campaign_execute.py"
    spec = importlib.util.spec_from_file_location("campaign_execute_deployment", executor_path)
    assert spec is not None and spec.loader is not None
    executor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(executor)
    assert executor.WorkflowStatus.FAILED == "failed"

    def test_perlmutter_campaign_requires_held_out_validation_for_promotion() -> None:
        root = Path(__file__).resolve().parents[1]
        job = (
            root
            / "deployments/perlmutter/jobs/job-campaign-formula-discovery-all-models-perlmutter.sh"
        ).read_text(encoding="utf-8")
        executor = (root / "deployments/perlmutter/jobs/campaign_execute.py").read_text(
            encoding="utf-8"
        )

        assert "MATSIM_CAMPAIGN_PROMOTION_VALIDATION_SET:?" in job
        assert "--promotion-validation-set" in job
        assert "--promotion-max-energy-mae" in job
        assert "--promotion-max-force-mae" in job
        assert 'parser.error("--promote-model requires --promotion-validation-set")' in executor
