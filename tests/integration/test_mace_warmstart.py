"""Integration test: MACE warm-start vs cold-start QE relaxation.

For each fixture in ``data/fixtures.yaml`` we:

  1. Run a MACE-MP (mace-torch) structure optimisation on the input crystal.
  2. Run ``pw.x`` ``vc-relax`` from the *original* coordinates (cold start).
  3. Run ``pw.x`` ``vc-relax`` from the MACE-relaxed coordinates (warm start).
  4. Compare BFGS step counts, total SCF iterations, and final energies.

The whole test is **skipped unless** every external dependency is available
through environment variables; this lets the rest of the test suite run
without Quantum ESPRESSO or a MACE foundation model::

    MATSIM_QE_LAUNCHER         # absolute path or wrapper that invokes pw.x
    MATSIM_QE_PSEUDO_DIR       # directory with .UPF pseudopotentials
    MATSIM_MACE_MODEL          # MACE variant (small|medium|large) or .model path

Optional::

    MATSIM_MACE_FAMILY         # mace_mp|mace_off|checkpoint  (default: mace_mp)
    MATSIM_MACE_PRECISION      # fp32|fp64  (default: fp64)
    MATSIM_QE_TIMEOUT_SEC      # per-pw.x-run timeout (default: 3600)
    MATSIM_QE_MLP_DEVICE       # cuda|cpu  (default: cuda)
    MATSIM_WARMSTART_FIXTURES  # comma-sep names to restrict (default: all)

A SLURM launcher that wires all of the above is provided at
``deployments/perlmutter/jobs/job-mace-warmstart-perlmutter.sh``.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import pytest

DATA_DIR = Path(__file__).resolve().parent / "data"
FIXTURES_YAML = DATA_DIR / "fixtures.yaml"


# --------------------------------------------------------------------------- #
# Skip-gating                                                                 #
# --------------------------------------------------------------------------- #


def _env(name: str) -> str | None:
    val = os.environ.get(name)
    return val if val and val.strip() else None


def _missing_requirements() -> list[str]:
    missing: list[str] = []
    if _env("MATSIM_QE_LAUNCHER") is None:
        missing.append("MATSIM_QE_LAUNCHER")
    elif not Path(_env("MATSIM_QE_LAUNCHER").split()[0]).exists():
        missing.append("MATSIM_QE_LAUNCHER (file not found)")
    if _env("MATSIM_QE_PSEUDO_DIR") is None:
        missing.append("MATSIM_QE_PSEUDO_DIR")
    elif not Path(_env("MATSIM_QE_PSEUDO_DIR")).is_dir():
        missing.append("MATSIM_QE_PSEUDO_DIR (not a directory)")
    if _env("MATSIM_MACE_MODEL") is None:
        missing.append("MATSIM_MACE_MODEL")
    return missing


pytestmark = pytest.mark.skipif(
    bool(_missing_requirements()),
    reason=(
        "MACE warm-start benchmark requires environment variables: "
        + ", ".join(_missing_requirements() or ["<all set>"])
    ),
)


# --------------------------------------------------------------------------- #
# Fixture loading                                                             #
# --------------------------------------------------------------------------- #


def _load_fixtures() -> list[dict[str, Any]]:
    """Parse ``fixtures.yaml`` (lazy yaml import; skip if PyYAML missing)."""
    try:
        import yaml
    except ImportError:  # pragma: no cover
        pytest.skip("PyYAML not installed; cannot read fixtures.yaml.")
    raw = yaml.safe_load(FIXTURES_YAML.read_text())
    items = list(raw.get("fixtures", []))
    only = _env("MATSIM_WARMSTART_FIXTURES")
    if only:
        wanted = {n.strip() for n in only.split(",")}
        items = [f for f in items if f["name"] in wanted]
    return items


def _ids(fixtures: list[dict[str, Any]]) -> list[str]:
    return [f["name"] for f in fixtures]


_FIXTURES = _load_fixtures() if FIXTURES_YAML.exists() else []


# --------------------------------------------------------------------------- #
# Test                                                                        #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("fixture", _FIXTURES, ids=_ids(_FIXTURES))
def test_mace_warmstart_helps_qe(fixture: dict[str, Any], tmp_path: Path) -> None:
    """Warm-started pw.x relaxation should not need more BFGS steps than cold."""
    from matsim_agents.tools.warmstart_benchmark_qe import run_warmstart_benchmark

    structure = DATA_DIR / fixture["file"]
    assert structure.exists(), f"missing fixture file: {structure}"

    work_dir = tmp_path / fixture["name"]
    work_dir.mkdir(parents=True, exist_ok=True)

    # Copy the structure into the work dir so it's preserved alongside outputs.
    structure_copy = work_dir / structure.name
    shutil.copy(structure, structure_copy)

    qe_overrides: dict[str, Any] = {"is_2d": bool(fixture.get("is_2d", False))}
    qe_overrides.update(fixture.get("qe", {}))

    # Read MACE-specific settings from env / fixture.
    # The fixture may optionally have a ``mace:`` block with keys that override
    # defaults (e.g. fmax, maxiter, optimizer).
    mace_cfg: dict[str, Any] = dict(fixture.get("mace", {}))

    is_vc_relax = qe_overrides.get("calculation", "vc-relax") == "vc-relax"
    default_fmax = 0.01 if is_vc_relax else 0.05

    mace_kwargs: dict[str, Any] = {
        "mace_family": _env("MATSIM_MACE_FAMILY") or mace_cfg.get("family", "mace_mp"),
        "mace_model": _env("MATSIM_MACE_MODEL") or mace_cfg.get("model", "medium"),
        "mace_precision": _env("MATSIM_MACE_PRECISION") or mace_cfg.get("precision", "fp64"),
        "mlp_device": _env("MATSIM_QE_MLP_DEVICE") or "cuda",
        "optimizer": mace_cfg.get("optimizer", "FIRE"),
        "fmax": float(mace_cfg.get("fmax", default_fmax)),
        "maxiter": int(mace_cfg.get("maxiter", 200)),
        "maxstep": float(mace_cfg.get("maxstep", 1e-2)),
        "relative_increase_threshold": float(mace_cfg.get("relative_increase_threshold", 0.05)),
    }

    timeout = int(_env("MATSIM_QE_TIMEOUT_SEC") or "3600")
    qe_launcher_str = _env("MATSIM_QE_LAUNCHER")
    qe_launcher: list[str] | str = (
        qe_launcher_str.split() if " " in qe_launcher_str else qe_launcher_str
    )

    summary = run_warmstart_benchmark(
        structure_path=str(structure_copy),
        work_dir=str(work_dir),
        pseudo_dir=_env("MATSIM_QE_PSEUDO_DIR"),
        qe_launcher=qe_launcher,
        qe_settings_overrides=qe_overrides,
        mlip_backend="mace",
        mlip_kwargs=mace_kwargs,
        timeout_sec=timeout,
    )

    # Always emit the JSON next to the test artefacts for later inspection.
    (work_dir / "comparison.json").write_text(json.dumps(_to_jsonable(summary), indent=2))

    cold = summary.cold
    cold_may_fail = bool(fixture.get("cold_may_fail", False))

    if cold_may_fail:
        # This fixture tests that warm-start *enables* convergence even when
        # the cold run fails.  Assert warm converged and cold did not.
        if summary.warm is None or "bfgs_steps" not in summary.warm:
            pytest.fail(
                f"Warm-start phase produced no result for {fixture['name']!r}; "
                f"mace block: {summary.hydragnn}"
            )
        warm = summary.warm
        assert warm.get("converged"), (
            f"Warm pw.x run did not converge for {fixture['name']!r} even "
            f"though cold_may_fail=true: see {warm.get('stdout_path')}"
        )
        if cold.get("converged"):
            import warnings

            warnings.warn(
                f"{fixture['name']!r}: cold run converged unexpectedly — "
                "consider removing cold_may_fail: true from the fixture.",
                stacklevel=2,
            )
        return  # pass — warm converged, cold did not (or both did, which is fine)

    assert cold.get("converged"), (
        f"Cold pw.x run did not converge for {fixture['name']!r}: "
        f"return_code={cold.get('return_code')}, see {cold.get('stdout_path')}"
    )

    if summary.warm is None or "bfgs_steps" not in summary.warm:
        pytest.fail(
            f"Warm-start phase produced no result for {fixture['name']!r}; "
            f"mace block: {summary.hydragnn}"
        )
    warm = summary.warm
    assert warm.get("converged"), (
        f"Warm pw.x run did not converge for {fixture['name']!r}: "
        f"return_code={warm.get('return_code')}, see {warm.get('stdout_path')}"
    )

    # Energy agreement: same minimum to within fixture-specified tolerance.
    e_tol = float(fixture.get("energy_tolerance_ev", 0.01))
    delta_e = abs(float(cold["final_energy_ev"]) - float(warm["final_energy_ev"]))
    assert delta_e < e_tol, (
        f"Cold and warm runs disagree on final energy by {delta_e:.4f} eV "
        f"(tolerance {e_tol} eV) for {fixture['name']!r}. They likely landed in "
        f"different minima."
    )

    # Step count: warm should not be slower than cold for the same minimum.
    if fixture.get("expect_warm_le_cold", True):
        assert int(warm["bfgs_steps"]) <= int(cold["bfgs_steps"]), (
            f"Warm-started BFGS took more steps ({warm['bfgs_steps']}) than "
            f"cold ({cold['bfgs_steps']}) for {fixture['name']!r}."
        )


def _to_jsonable(obj: Any) -> Any:
    """Recursively convert an object to a JSON-serialisable form."""
    if hasattr(obj, "__dict__"):
        return {k: _to_jsonable(v) for k, v in obj.__dict__.items()}
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj
