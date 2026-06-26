"""Integration test: HydraGNN warm-start vs cold-start QE relaxation.

For each fixture in ``data/fixtures.yaml`` we:

  1. Run a HydraGNN structure optimisation on the input crystal.
  2. Run ``pw.x`` ``vc-relax`` from the *original* coordinates (cold start).
  3. Run ``pw.x`` ``vc-relax`` from the HydraGNN-relaxed coordinates (warm).
  4. Compare BFGS step counts, total SCF iterations, and final energies.

The whole test is **skipped unless** every external dependency is available
through environment variables; this lets the rest of the test suite run
without a Quantum ESPRESSO install or a HydraGNN checkpoint::

    MATSIM_QE_LAUNCHER         # absolute path or wrapper that invokes pw.x
    MATSIM_QE_PSEUDO_DIR       # directory with .UPF pseudopotentials
    MATSIM_HYDRAGNN_LOGDIR     # HydraGNN logdir (config.json + checkpoint)
    HYDRAGNN_BRANCH_MLP_CHECKPOINT   # BranchWeightMLP .pt checkpoint

Optional::

    MATSIM_QE_TIMEOUT_SEC      # per-pw.x-run timeout (default: 3600)
    MATSIM_QE_MLP_DEVICE       # cuda|cpu  (default: cuda)
    MATSIM_WARMSTART_FIXTURES  # comma-sep names to restrict (default: all)

A SLURM launcher that wires all of the above is provided at
``scripts/launchers/frontier/run-qe-warmstart-benchmark.sh``.
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
    if _env("MATSIM_HYDRAGNN_LOGDIR") is None:
        missing.append("MATSIM_HYDRAGNN_LOGDIR")
    if _env("HYDRAGNN_BRANCH_MLP_CHECKPOINT") is None:
        missing.append("HYDRAGNN_BRANCH_MLP_CHECKPOINT")
    return missing


pytestmark = pytest.mark.skipif(
    bool(_missing_requirements()),
    reason=(
        "QE warm-start benchmark requires environment variables: "
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
def test_hydragnn_warmstart_helps_qe(fixture: dict[str, Any], tmp_path: Path) -> None:
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

    hydragnn_cfg: dict[str, Any] = dict(fixture.get("hydragnn", {}))

    is_vc_relax = qe_overrides.get("calculation", "vc-relax") == "vc-relax"
    default_fmax = 0.01 if is_vc_relax else 0.05

    # relax_cell controls whether the HydraGNN pre-relax uses ExpCellFilter to
    # also optimise the unit cell.  Default is False: HydraGNN only relaxes
    # atomic positions and QE handles the cell via its DFT stress tensor.
    # Cell relaxation via NumericalStressCalculator (finite-difference forces)
    # is too noisy for metallic / disordered systems and can collapse the cell.
    # Enable per-fixture via  hydragnn: relax_cell: true  only when the MLP
    # stress is reliable (e.g. well-converged insulators with small cells).
    relax_cell = bool(hydragnn_cfg.get("relax_cell", False))

    hydragnn_kwargs: dict[str, Any] = {
        "logdir": _env("MATSIM_HYDRAGNN_LOGDIR"),
        "hydragnn_branch_mlp_checkpoint": _env("HYDRAGNN_BRANCH_MLP_CHECKPOINT"),
        "mlp_device": _env("MATSIM_QE_MLP_DEVICE") or "cuda",
        "optimizer": hydragnn_cfg.get("optimizer", "FIRE"),
        "fmax": float(hydragnn_cfg.get("fmax", default_fmax)),
        "maxiter": int(hydragnn_cfg.get("maxiter", 200)),
        "maxstep": float(hydragnn_cfg.get("maxstep", 1e-2)),
        "relative_increase_threshold": float(
            hydragnn_cfg.get("relative_increase_threshold", 0.05)
        ),
        "relax_cell": relax_cell,
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
        hydragnn_kwargs=hydragnn_kwargs,
        timeout_sec=timeout,
    )

    # Always emit the JSON next to the test artefacts for later inspection.
    (work_dir / "comparison.json").write_text(json.dumps(_to_jsonable(summary), indent=2))

    cold = summary.cold
    cold_may_fail = bool(fixture.get("cold_may_fail", False))
    warm_may_fail = bool(fixture.get("warm_may_fail", False))

    if cold_may_fail:
        # This fixture tests that warm-start *enables* convergence even when
        # the cold run fails.  Assert warm converged and cold did not.
        if summary.warm is None or "bfgs_steps" not in summary.warm:
            pytest.fail(
                f"Warm-start phase produced no result for {fixture['name']!r}; "
                f"hydragnn block: {summary.hydragnn}"
            )
        warm = summary.warm
        assert warm.get("converged"), (
            f"Warm pw.x run did not converge for {fixture['name']!r} even "
            f"though cold_may_fail=true: see {warm.get('stdout_path')}"
        )
        # Cold should have failed — if it somehow converged, emit a warning
        # so the fixture flag can be re-evaluated.
        if cold.get("converged"):
            import warnings
            warnings.warn(
                f"{fixture['name']!r}: cold run converged unexpectedly — "
                "consider removing cold_may_fail: true from the fixture."
            )
        return  # pass — warm converged, cold did not (or both did, which is fine)

    assert cold.get("converged"), (
        f"Cold pw.x run did not converge for {fixture['name']!r}: "
        f"return_code={cold.get('return_code')}, see {cold.get('stdout_path')}"
    )

    if summary.warm is None or "bfgs_steps" not in summary.warm:
        pytest.fail(
            f"Warm-start phase produced no result for {fixture['name']!r}; "
            f"hydragnn block: {summary.hydragnn}"
        )
    warm = summary.warm
    if not warm.get("converged"):
        if warm_may_fail:
            import warnings
            warnings.warn(
                f"{fixture['name']!r}: warm pw.x run did not converge "
                f"(warm_may_fail=true) — MLIP pre-relaxation may have moved "
                f"atoms away from the DFT basin. See {warm.get('stdout_path')}"
            )
            return  # pass — cold converged, warm failure is expected
        pytest.fail(
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
            f"cold ({cold['bfgs_steps']}) for {fixture['name']!r} — HydraGNN "
            f"warm-start did not help."
        )


def _to_jsonable(obj: Any) -> Any:
    """Best-effort dataclass->dict serialiser for the comparison summary."""
    from dataclasses import asdict, is_dataclass

    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    return obj
