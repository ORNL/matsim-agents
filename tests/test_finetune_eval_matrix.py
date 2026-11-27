"""The fine-tune + evaluation matrix expands into validated per-run inputs."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
LAUNCHERS = ROOT / "deployments" / "perlmutter" / "launchers"
EVALUATION = ROOT / "examples" / "evaluation" / "random-frame-interpolation.yaml"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def matrix_module(monkeypatch):
    monkeypatch.setenv("PROJECT_ROOT", str(ROOT))
    # Normally exported by deployments/common/hydragnn-env.sh.
    monkeypatch.setenv("HYDRAGNN_ROOT", "/pinned/HydraGNN")
    monkeypatch.setenv("HYDRAGNN_BEST6_LOGDIR", "/weights/BEST6")
    return _load("finetune_eval_matrix", LAUNCHERS / "finetune_eval_matrix.py")


def _evaluation() -> dict:
    return yaml.safe_load(EVALUATION.read_text())["evaluation"]


def test_matrix_covers_every_harvested_variant(matrix_module):
    harvest = _load("harvest_finetune_eval", ROOT / "scripts/utils/harvest_finetune_eval.py")
    matrix = matrix_module.load_matrix(LAUNCHERS / "config/finetune-eval-matrix.yaml")
    assert {variant for _label, variant, _endpoint in harvest.ROWS} <= set(matrix["variants"])
    assert set(harvest.CASES) == set(matrix["cases"])


def test_every_run_is_written_with_its_model_and_recipe(matrix_module, tmp_path):
    matrix = matrix_module.load_matrix(LAUNCHERS / "config/finetune-eval-matrix.yaml")
    runs = matrix_module.expand(matrix, _evaluation(), tmp_path)
    assert len(runs) == len(matrix["variants"]) * len(matrix["cases"])
    assert {backend for _variant, _case, backend, _out in runs} == {"hydragnn", "uma", "mace"}
    for variant, case, backend, out_dir in runs:
        assert out_dir == tmp_path / variant / case
        mlip = yaml.safe_load((out_dir / "mlip.yaml").read_text())["mlip"]
        evaluation = yaml.safe_load((out_dir / "evaluation.yaml").read_text())["evaluation"]
        assert mlip["backend"] == backend
        assert evaluation.get("fine_tune", {"backend": backend})["backend"] == backend
    zn = yaml.safe_load((tmp_path / "uma/zn-formate-mof-uma-al-001/mlip.yaml").read_text())
    assert zn["mlip"]["task_name"] == "omol"
    lifepo4 = yaml.safe_load((tmp_path / "uma/lifepo4-al-001/mlip.yaml").read_text())
    assert lifepo4["mlip"]["task_name"] == "omat"
    frozen = yaml.safe_load((tmp_path / "hydragnn-frozen/lifepo4-al-001/mlip.yaml").read_text())
    assert frozen["mlip"]["inference_head"] == "OMat24"
    assert "fine_tuner" not in frozen["mlip"]


def test_selection_filters_and_rejects_unknown_names(matrix_module, tmp_path):
    matrix = matrix_module.load_matrix(LAUNCHERS / "config/finetune-eval-matrix.yaml")
    runs = matrix_module.expand(
        matrix, _evaluation(), tmp_path, variants="uma-lora", cases="lifepo4-al-001"
    )
    assert [(variant, case) for variant, case, _b, _o in runs] == [("uma-lora", "lifepo4-al-001")]
    with pytest.raises(ValueError, match="unknown variants: orb"):
        matrix_module.expand(matrix, _evaluation(), tmp_path, variants="orb")


def test_a_recipe_for_another_backend_is_rejected_before_writing(matrix_module, tmp_path):
    matrix = {
        "cases": {"case": {}},
        "variants": {
            "bad": {"mlip": {"backend": "uma"}, "fine_tune": {"backend": "mace", "lora": True}}
        },
    }
    with pytest.raises(ValueError, match="selected MLIP is 'uma'"):
        matrix_module.expand(matrix, _evaluation(), tmp_path)
    assert not (tmp_path / "bad").exists()
