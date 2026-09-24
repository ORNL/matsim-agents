from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from ase import Atoms
from ase.io import write

from benchmarks.portability.active_learning_scientific_debate import (
    execute_active_learning_scientific_debate,
)

_AL_CONFIG_YAML = """
mlip:
  backend: uma
  uma:
    model_name: uma-s-1p1
md:
  seed_source:
    kind: paths
    paths:
      - /tmp/seed.vasp
acquisition:
  strategy: random
  n_select: 2
dft:
  backend: qe
  qe:
    pw_bin: /tmp/pw.x
    pw_wrapper: /tmp/wrapper.sh
    pseudo_dir: /tmp/pseudo
trainer:
  enabled: false
loop:
  n_iterations: 1
  out_dir: ${MATSIM_AL_OUT_DIR}
"""


class _FakeModel:
    def __init__(self, model: str):
        self.model = model
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        return SimpleNamespace(content=f"{self.model} scientific argument {self.calls}")


def _catalog(tmp_path, count=2):
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


def _fake_al_runner(cfg) -> None:
    """Stand in for the real MD+DFT loop: write the same artifacts it would."""
    out_dir = Path(cfg.loop.out_dir)
    iteration_dir = out_dir / "iteration_0000"
    iteration_dir.mkdir(parents=True)
    state = {
        "iteration": 0,
        "status": "complete",
        "n_candidates": 4,
        "n_selected": 2,
        "n_dft_converged": 2,
        "n_dft_failed": 0,
        "dft_backend": "qe",
        "score_min": 0.01,
        "score_max": 0.05,
        "score_mean": 0.03,
    }
    (iteration_dir / "state.json").write_text(json.dumps(state))

    frames = []
    for index in range(2):
        atoms = Atoms("Si2", positions=[[0, 0, 0], [1.36, 1.36, 1.36]], cell=[5.43] * 3, pbc=True)
        atoms.info["energy"] = -10.5 - index
        atoms.info["dft_backend"] = "qe"
        atoms.info["source_work_dir"] = f"candidate-{index}"
        atoms.arrays["forces"] = np.array([[0.01, 0.0, 0.0], [-0.01, 0.0, 0.0]])
        frames.append(atoms)
    write(out_dir / "dataset.extxyz", frames)


def test_active_learning_evidence_reaches_every_debate_participant(tmp_path):
    catalog = _catalog(tmp_path)
    models = {}

    def factory(*, provider, model, base_url):
        assert provider == "vllm"
        models[model] = _FakeModel(model)
        return models[model]

    al_config = tmp_path / "al.yaml"
    al_config.write_text(_AL_CONFIG_YAML)

    result = execute_active_learning_scientific_debate(
        output=tmp_path / "output",
        al_config=al_config,
        rounds=2,
        catalog=catalog,
        models_root=_models_root(tmp_path, 0, 1),
        environment={f"MODEL_{index}_URL": f"http://node:{8000 + index}/v1" for index in range(2)},
        model_factory=factory,
        al_runner=_fake_al_runner,
    )

    assert result["status"] == "passed"
    assert result["active_learning"]["state"]["n_dft_converged"] == 2
    assert "candidate-0" in result["active_learning"]["evidence"]
    assert "candidate-1" in result["active_learning"]["evidence"]
    assert result["debate"]["status"] == "passed"
    assert result["debate"]["turn_count"] == 4
    assert all(model.calls == 3 for model in models.values())

    dialogue_path = Path(result["debate"]["dialogue_path"])
    assert dialogue_path.is_file()

    result_path = tmp_path / "output" / "active_learning_scientific_debate_result.json"
    assert json.loads(result_path.read_text())["status"] == "passed"
