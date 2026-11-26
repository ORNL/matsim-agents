"""Cascade-study driver and summary scripts: resume safety and per-setting grouping."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "cascade"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _plan(monkeypatch, out, *extra):
    run = _load("run_cascade_study")
    argv = ["x", "--study", "size", "--out", str(out), "--a", "2.87", "--sizes", "3,4"]
    monkeypatch.setattr(sys, "argv", [*argv, "--energy", "500", "--plan-only", *extra])
    return run.main()


def test_resume_with_changed_settings_is_refused(tmp_path, monkeypatch):
    assert _plan(monkeypatch, tmp_path) == 0
    assert _plan(monkeypatch, tmp_path, "--shards", "4", "--shard", "1") == 0
    with pytest.raises(SystemExit, match="different settings.*seed"):
        _plan(monkeypatch, tmp_path, "--seed", "2")


def test_concurrent_shards_never_read_a_partial_plan(tmp_path):
    import os
    import subprocess

    argv = [sys.executable, str(SCRIPTS / "run_cascade_study.py"), "--study", "size"]
    argv += ["--out", str(tmp_path), "--a", "2.87", "--sizes", "3,4", "--energy", "500"]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    procs = [
        subprocess.Popen(
            [*argv, "--plan-only", "--shards", "16", "--shard", str(i)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for i in range(16)
    ]
    errors = [p.communicate()[1] for p in procs if p.wait() != 0]
    assert not errors, errors[0]
    assert json.loads((tmp_path / "plan.json").read_text())["n_tasks"] > 0
    assert not list(tmp_path.glob("plan.json.*.tmp"))


def _rec(sr, n, fp):
    return {
        "study": "size",
        "short_range": sr,
        "energy_eV": 500.0,
        "element": "Fe",
        "pka_symbol": "Fe",
        "temperature_K": 300.0,
        "counts": None,
        "n_atoms": n,
        "n_frenkel_pairs": fp,
        "boundary_ok": True,
        "numerically_stable": True,
        "max_abs_energy_drift_eV": 0.0,
    }


def test_short_range_models_are_summarized_separately():
    summ = _load("summarize_cascade_study")
    # Opposite size trends that pooled would look converged.
    recs = [_rec("none", 54, v) for v in (4, 5, 6)] + [_rec("none", 128, v) for v in (9, 10, 11)]
    recs += [_rec("zbl", 54, v) for v in (9, 10, 11)] + [_rec("zbl", 128, v) for v in (4, 5, 6)]
    out = summ.summarize_cascades(recs, tolerance=0.15)
    groups = out["groups"]
    assert len(groups) == 2 and all(k.startswith(("sr=none", "sr=zbl")) for k in groups)
    assert not any(g["size_convergence"]["converged"] for g in groups.values())
    json.dumps(out)
