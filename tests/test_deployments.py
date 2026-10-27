from __future__ import annotations

import importlib.util
import os
import subprocess
from pathlib import Path

import pytest
import yaml
from ase.build import bulk
from ase.io import write


def test_deployment_assets_are_portable() -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "diagnostics" / "validate_deployments.py"
    spec = importlib.util.spec_from_file_location("validate_deployments", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.validate() == []


@pytest.mark.parametrize(
    "case",
    [
        "al_zn_formate.yaml",
        "al_phosphorene.yaml",
        "al_lifepo4.yaml",
        "al_hea_fcc_cantor.yaml",
        "al_hea_bcc.yaml",
        "al_cu_bht_2d.yaml",
    ],
)
def test_perlmutter_hydragnn_examples_pass_declared_checkout(case) -> None:
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "examples/paper_cases" / case).read_text())
    assert config["mlip"]["hydragnn"]["hydragnn_root"] == "${HYDRAGNN_DIR}"


@pytest.mark.parametrize(
    "case",
    [
        "examples/active_learning/al_config.example.yaml",
        "examples/active_learning/al_config.prompt.example.yaml",
        "examples/paper_cases/al_hea_bcc_frontier.yaml",
        "examples/paper_cases/al_lifepo4_frontier.yaml",
        "examples/paper_cases/al_phosphorene_frontier.yaml",
    ],
)
def test_frontier_newhead_examples_pass_finetuning_checkout(case) -> None:
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / case).read_text())
    hydragnn = config.get("mlip", {}).get("hydragnn", config.get("hydragnn"))
    assert hydragnn["ft_repo"] == "${HYDRAGNN_FT_REPO}"
    assert "/lustre/orion/" in config["vars"]["HYDRAGNN_FT_REPO"]


def test_perlmutter_campaign_imports_current_checkout() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (
        root / "deployments/perlmutter/jobs/job-campaign-formula-discovery-all-models-perlmutter.sh"
    )
    content = script.read_text(encoding="utf-8")

    source_path_export = 'export PYTHONPATH="${REPO}/src${PYTHONPATH:+:${PYTHONPATH}}"'
    assert source_path_export in content
    assert content.index(source_path_export) < content.index(".venv-uma/bin/activate")

    standalone = root / "deployments/perlmutter/jobs/campaign_formula_discovery.py"
    standalone_content = standalone.read_text(encoding="utf-8")
    assert 'SOURCE_ROOT = ROOT / "src"' in standalone_content
    assert "sys.path.insert(0, str(SOURCE_ROOT))" in standalone_content


def test_perlmutter_multinode_server_extends_raylet_startup_wait() -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "deployments/perlmutter/jobs/job-serve-multinode-perlmutter.sh"
    content = script.read_text(encoding="utf-8")

    assert "RAY_raylet_start_wait_time_s=${RAY_raylet_start_wait_time_s:-300}" in content


def test_vasp_campaign_preserves_explicit_method_signature() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (
        root
        / "deployments/perlmutter/jobs/job-campaign-formula-discovery-all-models-vasp-perlmutter.sh"
    )
    content = script.read_text(encoding="utf-8")

    assert (
        'MATSIM_CAMPAIGN_DFT_METHOD_SIGNATURE="${MATSIM_CAMPAIGN_DFT_METHOD_SIGNATURE:-'
        "vasp-6.6.1-pbe64-encut520-kspacing0.25-o2-triplet-v1}" in content
    )
    assert '"$MATSIM_VASP_BIN" != "$DEFAULT_VASP_BIN"' in content
    assert '"$MATSIM_VASP_POTCAR_DIR" != "$DEFAULT_VASP_POTCAR_DIR"' in content
    assert "custom VASP binary or POTCAR directory requires" in content


def test_vasp_campaign_requires_signature_for_custom_inputs(tmp_path) -> None:
    root = Path(__file__).resolve().parents[1]
    script = (
        root
        / "deployments/perlmutter/jobs/job-campaign-formula-discovery-all-models-vasp-perlmutter.sh"
    )
    environment = os.environ.copy()
    environment.update(
        PROJECT_ROOT=str(tmp_path),
        MATSIM_VASP_BIN=str(tmp_path / "custom-vasp"),
        MATSIM_VASP_POTCAR_DIR=str(tmp_path / "custom-potcars"),
    )
    environment.pop("MATSIM_CAMPAIGN_DFT_METHOD_SIGNATURE", None)

    result = subprocess.run(
        ["bash", str(script)],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "custom VASP binary or POTCAR directory requires" in result.stderr


def test_perlmutter_campaign_exposes_adaptive_acquisition_controls() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (
        root / "deployments/perlmutter/jobs/job-campaign-formula-discovery-all-models-perlmutter.sh"
    )
    content = script.read_text(encoding="utf-8")

    assert (
        '--acquisition-mode \\"\\${MATSIM_CAMPAIGN_ACQUISITION_MODE:-insertion-order}\\"' in content
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
        "--degeneracy-tolerance-ev-per-atom "
        '\\"\\${MATSIM_CAMPAIGN_DEGENERACY_TOLERANCE_EV_PER_ATOM:-0.01}\\"' in content
    )
    assert "--expand-unary-polymorphs" in content
    assert '--unary-random "${MATSIM_CAMPAIGN_UNARY_RANDOM:-50}"' in content
    assert '--surrogate-minimum-unique-unary "${MATSIM_CAMPAIGN_MIN_UNIQUE_UNARY:-2}"' in content


def test_perlmutter_campaign_supports_debate_only_and_uma_only_modes() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (
        root / "deployments/perlmutter/jobs/job-campaign-formula-discovery-all-models-perlmutter.sh"
    )
    content = script.read_text(encoding="utf-8")

    discovery_call = content.index("campaign_formula_discovery.py")
    debate_exit = content.index(
        'if [[ "$CAMPAIGN_MODE" == "debate-only" || "$CAMPAIGN_MODE" == "single-llm-once" ]]',
        discovery_call,
    )
    execution_call = content.index("campaign_execute.py")
    assert discovery_call < debate_exit < execution_call
    assert '[[ "$CAMPAIGN_MODE" == "single-llm-once" ]] && EXPECTED_NODES=1' in content
    assert "single-llm-once) LAYOUT=campaign-panel-single.yaml ;;" in content
    assert "debate-only) LAYOUT=campaign-panel-debate.yaml ;;" in content
    assert 'resources serve "$RESOURCES"' in content
    assert content.index('resources serve "$RESOURCES"') < discovery_call
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


def test_reference_preparation_deduplicates_unary_polymorphs(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "deployments/perlmutter/jobs/prepare_nb_ta_o_references.py"
    spec = importlib.util.spec_from_file_location("prepare_nb_ta_o_references", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    bcc = tmp_path / "bcc.extxyz"
    bcc_copy = tmp_path / "bcc-copy.extxyz"
    fcc = tmp_path / "fcc.extxyz"
    write(bcc, bulk("Nb", "bcc", a=3.3))
    write(bcc_copy, bulk("Nb", "bcc", a=3.3))
    write(fcc, bulk("Nb", "fcc", a=3.3))
    phases = {
        "Nb-bcc": {"formula": "Nb", "path": str(bcc)},
        "Nb-bcc-copy": {"formula": "Nb", "path": str(bcc_copy)},
        "Nb-fcc": {"formula": "Nb", "path": str(fcc)},
    }

    retained, duplicates = module._deduplicate_unary_phases(phases)

    assert list(retained) == ["Nb-bcc", "Nb-fcc"]
    assert duplicates == ["Nb-bcc-copy"]


def test_reference_preparation_merges_curated_before_generated_unary_dedup(tmp_path, monkeypatch):
    import json
    import sys
    from types import SimpleNamespace

    root = Path(__file__).resolve().parents[1]
    script = root / "deployments/perlmutter/jobs/prepare_nb_ta_o_references.py"
    spec = importlib.util.spec_from_file_location("prepare_curated_references", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    curated_structure = tmp_path / "curated-nb.extxyz"
    write(curated_structure, bulk("Nb", "bcc", a=3.3))
    curated_manifest = tmp_path / "curated.json"
    curated_manifest.write_text(
        json.dumps(
            {
                "phases": {
                    "curated-Nb": {
                        "formula": "Nb",
                        "path": curated_structure.name,
                        "settings": {"custom": "retained"},
                    }
                }
            }
        )
    )
    monkeypatch.setattr(
        module,
        "generate_seeds",
        lambda composition, *_args, **_kwargs: (
            [
                SimpleNamespace(
                    source="prototype",
                    structure_path=str(curated_structure),
                    candidate_id="copy",
                    prototype_id="bcc",
                    space_group=229,
                    random_seed=None,
                )
            ]
            if composition.formula == "Nb"
            else []
        ),
    )
    output = tmp_path / "output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(script),
            "--output-dir",
            str(output),
            "--competing-formulas",
            "--expand-unary-polymorphs",
            "--curated-manifest",
            str(curated_manifest),
        ],
    )
    assert module.main() == 0
    result = json.loads((output / "reference_structures.json").read_text())
    assert list(result["phases"])[0] == "curated-Nb"
    assert result["phases"]["curated-Nb"]["settings"] == {"custom": "retained"}
    assert set(result["generation"]["duplicate_unary_phases_removed"]) == {"Nb", "Nb-aflow-0000"}


def test_perlmutter_campaign_requires_held_out_validation_for_promotion() -> None:
    root = Path(__file__).resolve().parents[1]
    job = (
        root / "deployments/perlmutter/jobs/job-campaign-formula-discovery-all-models-perlmutter.sh"
    ).read_text(encoding="utf-8")
    executor = (root / "deployments/perlmutter/jobs/campaign_execute.py").read_text(
        encoding="utf-8"
    )

    assert "MATSIM_CAMPAIGN_PROMOTION_VALIDATION_FRACTION" in job
    assert "--promotion-validation-set" in job
    assert "--promotion-max-energy-mae" in job
    assert "--promotion-max-force-mae" in job
    assert "--promotion-max-relative-regression" not in job
    assert "MATSIM_CAMPAIGN_PROMOTION_MIN_RELATIVE_IMPROVEMENT:-0.05" in job
    assert "--promotion-min-relative-improvement" in job
    assert "promotion_min_relative_improvement=args.promotion_min_relative_improvement" in executor
    assert "--promotion-validation-fraction" in executor
    assert "args.promotion_validation_fraction <= 0" in executor
    phase_policy = executor.split("phase_policy = PhaseExplorationPolicy(", 1)[1].split(
        "formula_runner =", 1
    )[0]
    assert "promote_model=args.promote_model" in phase_policy
    assert "promotion_approved=args.approve_model_promotion" in phase_policy


@pytest.mark.parametrize(
    "manifest",
    [
        {"phases": {"Nb": "/tmp/Nb.extxyz"}},
        {"schema_version": 2, "phases": {"Nb": "/tmp/Nb.extxyz"}},
        {
            "schema_version": 2,
            "phases": {"Nb": {"formula": "Nb", "path": "/tmp/Nb.extxyz"}},
        },
        {
            "schema_version": 2,
            "phases": {
                "Nb": {
                    "phase_id": "Nb",
                    "formula": "Nb",
                    "path": "/tmp/Nb.extxyz",
                    "legacy": True,
                }
            },
        },
    ],
)
def test_campaign_executor_rejects_legacy_or_invalid_reference_manifests(
    tmp_path, monkeypatch, manifest
) -> None:
    import json
    from types import SimpleNamespace

    from matsim_agents.active_learning.config import ALConfig
    from matsim_agents.campaign.state import CampaignState

    root = Path(__file__).resolve().parents[1]
    executor_path = root / "deployments/perlmutter/jobs/campaign_execute.py"
    spec = importlib.util.spec_from_file_location("campaign_execute_manifest_test", executor_path)
    assert spec and spec.loader
    executor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(executor)

    campaign_state = tmp_path / "campaign.json"
    campaign_state.touch()
    al_config = tmp_path / "al.yaml"
    al_config.touch()
    reference_manifest = tmp_path / "references.json"
    reference_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(
        CampaignState,
        "load",
        lambda _path: SimpleNamespace(budget=SimpleNamespace(), reference_energies=None),
    )
    monkeypatch.setattr(
        ALConfig,
        "from_yaml",
        lambda _path: SimpleNamespace(
            dft=SimpleNamespace(backend="qe"),
            trainer=SimpleNamespace(
                validation_reference_set=None,
                training_references=None,
            ),
        ),
    )

    with pytest.raises(SystemExit) as error:
        executor.main(
            [
                "--campaign-state",
                str(campaign_state),
                "--al-config",
                str(al_config),
                "--model",
                "reviewer",
                "openai",
                "model",
                "",
                "--approve-dft",
                "--dft-reference-structures",
                str(reference_manifest),
                "--dft-method-signature",
                "qe-test-v1",
            ]
        )
    assert error.value.code == 2


def test_bounded_campaign_settings_cover_all_scientific_stages(monkeypatch) -> None:
    from matsim_agents.active_learning.config import ALConfig

    root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("PROJECT_ROOT", str(root))
    config = ALConfig.from_yaml(
        root / "deployments/perlmutter/jobs/config/campaign-nb-ta-o-uma-qe-bounded.yaml"
    )
    assert config.acquisition.strategy == "mc_dropout"
    assert config.acquisition.n_select == 10
    assert config.trainer.validation_fraction == 0.2
    assert config.dft.qe.ecutwfc_ry == 80
    assert config.dft.qe.ecutrho_ry == 640
    script = (root / "deployments/perlmutter/jobs/submit-nb-ta-o-combined-bounded.sh").read_text()
    for setting in (
        "MATSIM_CAMPAIGN_MODE=dft",
        "MATSIM_CAMPAIGN_MAX_DFT=64",
        "MATSIM_CAMPAIGN_MAX_CANDIDATES=3",
        "MATSIM_CAMPAIGN_RETRAIN=1",
        "MATSIM_CAMPAIGN_PROMOTE_MODEL=1",
        "MATSIM_CAMPAIGN_PROMOTION_VALIDATION_FRACTION=0.2",
        "MATSIM_CAMPAIGN_PROMOTION_MIN_EVALUATED_FRAMES=2",
        "MATSIM_CAMPAIGN_CONTINUE_ON_PROMOTION_REJECTION=1",
        "MATSIM_CAMPAIGN_DFT_REFINE=1",
        "MATSIM_CAMPAIGN_DFT_REFINE_CANDIDATES=2",
        "MATSIM_CAMPAIGN_SURROGATE_HULL=1",
        "configure_uma_model_artifacts",
        "-N 16 -t 06:00:00",
    ):
        assert setting in script


@pytest.mark.parametrize(
    ("mode", "backend", "refine", "workflow"),
    [
        ("dft", "qe", "1", "7llm-uma-al-qe-hull"),
        ("dft", "vasp", "1", "7llm-uma-al-vasp-hull"),
        ("dft", "qe", "0", "7llm-uma-al-qe"),
        ("uma-only", "qe", "0", "7llm-uma-screen"),
        ("debate-only", "qe", "0", "7llm-debate"),
        ("single-llm-once", "qe", "0", "1llm-debate"),
    ],
)
def test_campaign_descriptive_run_naming(tmp_path, mode, backend, refine, workflow):
    root = Path(__file__).resolve().parents[1]
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "--allow-empty",
            "-m",
            "test",
        ],
        check=True,
        capture_output=True,
    )
    sha = subprocess.check_output(
        ["git", "-C", str(tmp_path), "rev-parse", "--short=7", "HEAD"], text=True
    ).strip()
    env = dict(os.environ)
    for key in list(env):
        if key.startswith("MATSIM_CAMPAIGN_") or key.startswith("GIT_"):
            del env[key]
    env["MATSIM_CAMPAIGN_DFT_REFINE"] = refine
    env["MATSIM_CAMPAIGN_RUN_TAG"] = "obsolete-custom-tag"
    command = (
        'source "$1"; configure_campaign_run_name "$2" "$3" "$4"; '
        'printf "%s--j12345|%s" "$MATSIM_CAMPAIGN_RUN_TAG" '
        '"$MATSIM_CAMPAIGN_SOURCE_DIRTY"'
    )
    args = [
        "bash",
        "-eu",
        "-c",
        command,
        "test",
        str(root / "deployments/perlmutter/setup/campaign-naming.sh"),
        str(tmp_path),
        mode,
        backend,
    ]
    assert subprocess.check_output(args, env=env, text=True) == (
        f"nb-ta-o--{workflow}--{sha}--j12345|0"
    )
    (tmp_path / "untracked.txt").touch()
    assert subprocess.check_output(args, env=env, text=True).endswith("--j12345|1")

    env["MATSIM_CAMPAIGN_SOURCE_REVISION"] = subprocess.check_output(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"], text=True
    ).strip()
    env["MATSIM_CAMPAIGN_SOURCE_DIRTY"] = "0"
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "--allow-empty",
            "-m",
            "later",
        ],
        check=True,
        capture_output=True,
    )
    assert subprocess.check_output(args, env=env, text=True) == (
        f"nb-ta-o--{workflow}--{sha}--j12345|0"
    )


def test_campaign_launcher_uses_new_names_without_legacy_branch():
    root = Path(__file__).resolve().parents[1]
    job = (
        root / "deployments/perlmutter/jobs/job-campaign-formula-discovery-all-models-perlmutter.sh"
    ).read_text()
    assert 'RUN_NAME="${CAMPAIGN_RUN_TAG}--j${SLURM_JOB_ID}"' in job
    assert "MATSIM_CAMPAIGN_RUN_NAMING" not in job
    assert 'source "$REPO/deployments/perlmutter/setup/campaign-naming.sh"' in job
    assert '>"$OUTPUT/run_identity.txt"' in job.replace(" > ", ">")


def _module_source(root: Path, module: str) -> Path | None:
    base = root / "src" / Path(*module.split("."))
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def _top_level_names(path: Path) -> set[str]:
    import ast

    names: set[str] = set()
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update((alias.asname or alias.name).split(".")[0] for alias in node.names)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def test_shell_embedded_matsim_imports_resolve() -> None:
    """Python embedded in launchers/scripts must use existing matsim_agents paths."""
    import re

    root = Path(__file__).resolve().parents[1]
    tracked = subprocess.run(
        ["git", "ls-files", "deployments", "scripts"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    from_re = re.compile(r"from\s+(matsim_agents(?:\.\w+)*)\s+import\s+(\([^)]*\)|[^\n#]+)")
    module_re = re.compile(r"(?:-m|^\s*import)\s+(matsim_agents(?:\.\w+)+)", re.M)
    problems = []
    for name in tracked:
        if not name.endswith(".sh"):
            continue
        text = (root / name).read_text(errors="ignore")
        for match in module_re.finditer(text):
            if _module_source(root, match.group(1)) is None:
                problems.append(f"{name}: no module {match.group(1)}")
        for match in from_re.finditer(text):
            module = match.group(1)
            source = _module_source(root, module)
            if source is None:
                problems.append(f"{name}: no module {module}")
                continue
            defined = _top_level_names(source)
            for item in match.group(2).strip("()").replace("\n", ",").split(","):
                symbol = item.split(" as ")[0].strip()
                if not symbol or symbol == "*":
                    continue
                if symbol not in defined and _module_source(root, f"{module}.{symbol}") is None:
                    problems.append(f"{name}: {module} has no {symbol}")
    assert problems == []
