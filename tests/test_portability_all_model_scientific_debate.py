from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.portability.all_model_scientific_debate import (
    CATALOG,
    THERMOELECTRIC_HYPOTHESIS,
    execute_all_model_scientific_debate,
    load_catalog,
)


class _FakeModel:
    def __init__(self, model: str):
        self.model = model
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        assert THERMOELECTRIC_HYPOTHESIS in messages[-1].content
        return SimpleNamespace(content=f"{self.model} scientific argument {self.calls}")


def _catalog(tmp_path, count=3):
    path = tmp_path / "catalog.json"
    path.write_text(
        json.dumps(
            [
                {
                    "name": f"model-{index}",
                    "model": f"org/model-{index}",
                    "base_url_env": f"MODEL_{index}_URL",
                }
                for index in range(count)
            ]
        )
    )
    return path


def _models_root(tmp_path, *indices):
    root = tmp_path / "models"
    for index in indices:
        (root / f"model-{index}").mkdir(parents=True)
    return root


def test_every_catalog_model_completes_two_scientific_interaction_rounds(tmp_path):
    catalog = _catalog(tmp_path)
    models = {}

    def factory(*, provider, model, base_url):
        assert provider == "vllm"
        assert base_url.endswith("/v1")
        models[model] = _FakeModel(model)
        return models[model]

    result = execute_all_model_scientific_debate(
        output=tmp_path / "output",
        rounds=2,
        catalog=catalog,
        models_root=_models_root(tmp_path, 0, 1, 2),
        environment={f"MODEL_{index}_URL": f"http://node:{8000 + index}/v1" for index in range(3)},
        model_factory=factory,
    )
    assert result["status"] == "passed"
    assert result["required_models"] == [f"org/model-{index}" for index in range(3)]
    assert result["turn_count"] == 6
    assert all(model.calls == 3 for model in models.values())
    dialogue = json.loads(Path(result["dialogue_path"]).read_text())
    contributions = [entry for entry in dialogue if entry["kind"] == "model_argument"]
    assert len(contributions) == 6
    assert len({entry["contribution_id"] for entry in contributions}) == 6


def test_actual_first_class_catalog_defines_the_enabled_enclave(tmp_path):
    entries = load_catalog(CATALOG)
    enabled = [entry for entry in entries if entry.get("debate_enabled", True)]
    models_root = tmp_path / "models"
    for entry in entries:
        (models_root / entry["model"].rsplit("/", 1)[-1]).mkdir(parents=True)
    environment = {
        entry["base_url_env"]: f"http://model-{index}:8000/v1"
        for index, entry in enumerate(entries)
    }

    def factory(*, provider, model, base_url):
        return _FakeModel(model)

    result = execute_all_model_scientific_debate(
        output=tmp_path / "catalog-enclave",
        rounds=2,
        catalog=CATALOG,
        models_root=models_root,
        environment=environment,
        model_factory=factory,
    )
    assert result["status"] == "passed"
    assert result["required_models"] == [entry["model"] for entry in enabled]
    assert result["turn_count"] == 2 * len(enabled)
    assert "Pixtral" in result["disabled_models"]["mistral-large-3"]


def test_enclave_fails_closed_for_missing_model_endpoint(tmp_path):
    with pytest.raises(ValueError, match="MODEL_1_URL"):
        execute_all_model_scientific_debate(
            output=tmp_path / "output",
            rounds=2,
            catalog=_catalog(tmp_path, count=2),
            models_root=_models_root(tmp_path, 0, 1),
            environment={"MODEL_0_URL": "http://node:8000/v1"},
        )


def test_enclave_excludes_catalog_models_not_stored_locally(tmp_path):
    def factory(*, provider, model, base_url):
        return _FakeModel(model)

    result = execute_all_model_scientific_debate(
        output=tmp_path / "output",
        rounds=2,
        catalog=_catalog(tmp_path),
        models_root=_models_root(tmp_path, 0, 2),
        environment={
            "MODEL_0_URL": "http://node:8000/v1",
            "MODEL_2_URL": "http://node:8002/v1",
        },
        model_factory=factory,
    )

    assert result["status"] == "passed"
    assert result["required_models"] == ["org/model-0", "org/model-2"]
    assert result["skipped_nonlocal_models"] == ["model-1"]


def test_enclave_excludes_models_disabled_by_catalog_policy(tmp_path):
    catalog = _catalog(tmp_path)
    entries = json.loads(catalog.read_text())
    entries[1]["debate_enabled"] = False
    entries[1]["debate_skip_reason"] = "known runtime incompatibility"
    catalog.write_text(json.dumps(entries))

    result = execute_all_model_scientific_debate(
        output=tmp_path / "output",
        rounds=2,
        catalog=catalog,
        models_root=_models_root(tmp_path, 0, 1, 2),
        environment={
            "MODEL_0_URL": "http://node:8000/v1",
            "MODEL_2_URL": "http://node:8002/v1",
        },
        model_factory=lambda **kwargs: _FakeModel(kwargs["model"]),
    )

    assert result["status"] == "passed"
    assert result["required_models"] == ["org/model-0", "org/model-2"]
    assert result["disabled_models"] == {"model-1": "known runtime incompatibility"}


@pytest.mark.parametrize("exclusion", ["model-1", "org/model-1"])
def test_enclave_excludes_model_for_facility_runtime(tmp_path, exclusion):
    result = execute_all_model_scientific_debate(
        output=tmp_path / "output",
        rounds=2,
        catalog=_catalog(tmp_path),
        models_root=_models_root(tmp_path, 0, 1, 2),
        excluded_models={exclusion},
        environment={
            "MODEL_0_URL": "http://node:8000/v1",
            "MODEL_2_URL": "http://node:8002/v1",
        },
        model_factory=lambda **kwargs: _FakeModel(kwargs["model"]),
    )

    assert result["status"] == "passed"
    assert result["required_models"] == ["org/model-0", "org/model-2"]
    assert result["runtime_excluded_models"] == ["model-1"]


def test_enclave_rejects_fewer_than_two_rounds(tmp_path):
    with pytest.raises(ValueError, match="at least two"):
        execute_all_model_scientific_debate(
            output=tmp_path / "output",
            rounds=1,
            catalog=_catalog(tmp_path, count=2),
        )
