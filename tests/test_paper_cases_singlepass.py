from __future__ import annotations

import importlib.util
from argparse import Namespace
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "examples" / "paper_cases" / "singlepass.py"
SPEC = importlib.util.spec_from_file_location("paper_cases_singlepass", SCRIPT)
assert SPEC and SPEC.loader
singlepass = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(singlepass)


def test_stage_seeds_copies_case_files_only(tmp_path, monkeypatch) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "lifepo4_olivine.extxyz").write_text("seed\n")
    (source / "hea_bcc_random.extxyz").write_text("other\n")
    monkeypatch.setattr(singlepass, "SEED_ROOT", source)

    staged = singlepass.stage_seeds("lifepo4", tmp_path / "run" / "seeds")

    assert len(staged) == 1
    assert staged[0].name == "lifepo4_olivine.extxyz"
    assert staged[0].read_text() == "seed\n"


def test_run_case_invokes_current_graph_with_staged_paths(tmp_path, monkeypatch) -> None:
    seed = tmp_path / "seed"
    seed.mkdir()
    (seed / "lifepo4_olivine.extxyz").write_text("seed\n")
    monkeypatch.setattr(singlepass, "SEED_ROOT", seed)
    captured = {}

    class FakeGraph:
        def invoke(self, state, config):
            captured["state"] = state
            captured["config"] = config
            return {"analysis": "ranked", "results": [object()]}

    monkeypatch.setattr(singlepass, "build_graph", FakeGraph)
    args = Namespace(
        mlip_backend="uma",
        logdir=None,
        hydragnn_branch_mlp_checkpoint=None,
        checkpoint=None,
        uma_model_name="uma-s-1p1",
        uma_task="omat",
        mlp_device="cpu",
    )

    singlepass.run_case("lifepo4", tmp_path / "run", args)

    expected_seed = tmp_path / "run/lifepo4/seeds/lifepo4_olivine.extxyz"
    assert str(expected_seed.resolve()) in captured["state"].objective
    assert captured["config"]["configurable"]["mlip_backend"] == "uma"
    assert (tmp_path / "run/lifepo4/analysis.txt").read_text() == "ranked\n"