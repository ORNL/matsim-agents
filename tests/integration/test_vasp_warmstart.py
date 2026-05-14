"""Integration test: HydraGNN warm-start vs cold-start VASP relaxation.

Mirror of :mod:`tests.integration.test_qe_warmstart` for the VASP backend.

For each fixture in ``data/fixtures.yaml`` we:

  1. Run a HydraGNN structure optimisation on the input crystal.
  2. Run ``vasp_std`` ``vc-relax`` from the *original* coordinates (cold start).
  3. Run ``vasp_std`` ``vc-relax`` from the HydraGNN-relaxed coordinates (warm).
  4. Compare ionic step counts, total electronic-SCF iterations, and final
     energies.

The whole test is **skipped unless** every external dependency is available
through environment variables; this keeps the rest of the test suite runnable
without a VASP install or a HydraGNN checkpoint::

    MATSIM_VASP_LAUNCHER       # absolute path or wrapper that invokes vasp_std
    MATSIM_VASP_POTCAR_DIR     # directory with per-element POTCARs
                               # (e.g. <dir>/Si/POTCAR, <dir>/Mg/POTCAR ...)
    MATSIM_HYDRAGNN_LOGDIR     # HydraGNN logdir (config.json + checkpoint)
    MATSIM_HYDRAGNN_MLP_CKPT   # BranchWeightMLP .pt checkpoint

Optional::

    MATSIM_VASP_TIMEOUT_SEC    # per-vasp-run timeout (default: 3600)
    MATSIM_VASP_MLP_DEVICE     # cuda|cpu  (default: cuda)
    MATSIM_WARMSTART_FIXTURES  # comma-sep names to restrict (default: all)
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
    if _env("MATSIM_VASP_LAUNCHER") is None:
        missing.append("MATSIM_VASP_LAUNCHER")
    elif not Path(_env("MATSIM_VASP_LAUNCHER").split()[0]).exists():
        missing.append("MATSIM_VASP_LAUNCHER (file not found)")
    if _env("MATSIM_VASP_POTCAR_DIR") is None:
        missing.append("MATSIM_VASP_POTCAR_DIR")
    elif not Path(_env("MATSIM_VASP_POTCAR_DIR")).is_dir():
        missing.append("MATSIM_VASP_POTCAR_DIR (not a directory)")
    if _env("MATSIM_HYDRAGNN_LOGDIR") is None:
        missing.append("MATSIM_HYDRAGNN_LOGDIR")
    if _env("MATSIM_HYDRAGNN_MLP_CKPT") is None:
        missing.append("MATSIM_HYDRAGNN_MLP_CKPT")
    return missing


pytestmark = pytest.mark.skipif(
    bool(_missing_requirements()),
    reason=(
        "VASP warm-start benchmark requires environment variables: "
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
def test_hydragnn_warmstart_helps_vasp(fixture: dict[str, Any], tmp_path: Path) -> None:
    """Warm-started VASP relaxation should not need more ionic steps than cold."""
    from matsim_agents.tools.warmstart_benchmark_vasp import run_warmstart_benchmark

    structure = DATA_DIR / fixture["file"]
    assert structure.exists(), f"missing fixture file: {structure}"

    work_dir = tmp_path / fixture["name"]
    work_dir.mkdir(parents=True, exist_ok=True)

    # Copy the structure into the work dir so it's preserved alongside outputs.
    structure_copy = work_dir / structure.name
    shutil.copy(structure, structure_copy)

    # VASP-specific overrides live under fixture['vasp']; fall back to the
    # composition-aware defaults from recommend_settings if absent.
    vasp_overrides: dict[str, Any] = dict(fixture.get("vasp") or {})

    hydragnn_kwargs: dict[str, Any] = {
        "logdir": _env("MATSIM_HYDRAGNN_LOGDIR"),
        "mlp_checkpoint": _env("MATSIM_HYDRAGNN_MLP_CKPT"),
        "mlp_device": _env("MATSIM_VASP_MLP_DEVICE") or "cuda",
        "fmax": 0.05,
        "maxiter": 200,
    }

    timeout = int(_env("MATSIM_VASP_TIMEOUT_SEC") or "3600")
    vasp_launcher_str = _env("MATSIM_VASP_LAUNCHER")
    vasp_launcher: list[str] | str = (
        vasp_launcher_str.split() if " " in vasp_launcher_str else vasp_launcher_str
    )

    summary = run_warmstart_benchmark(
        structure_path=str(structure_copy),
        work_dir=str(work_dir),
        potcar_dir=_env("MATSIM_VASP_POTCAR_DIR"),
        vasp_launcher=vasp_launcher,
        vasp_settings_overrides=vasp_overrides,
        hydragnn_kwargs=hydragnn_kwargs,
        timeout_sec=timeout,
    )

    # Always emit the JSON next to the test artefacts for later inspection.
    (work_dir / "comparison.json").write_text(json.dumps(_to_jsonable(summary), indent=2))

    cold = summary.cold
    assert cold.get("converged"), (
        f"Cold VASP run did not converge for {fixture['name']!r}: "
        f"return_code={cold.get('return_code')}, see {cold.get('work_dir')}/vasp.out"
    )

    if summary.warm is None or "n_ionic_steps" not in summary.warm:
        pytest.fail(
            f"Warm-start phase produced no result for {fixture['name']!r}; "
            f"hydragnn block: {summary.hydragnn}"
        )
    warm = summary.warm
    assert warm.get("converged"), (
        f"Warm VASP run did not converge for {fixture['name']!r}: "
        f"return_code={warm.get('return_code')}, see {warm.get('work_dir')}/vasp.out"
    )

    # Energy agreement: same minimum to within fixture-specified tolerance.
    e_tol = float(fixture.get("energy_tolerance_ev", 0.01))
    e_cold = cold.get("final_energy_eV")
    e_warm = warm.get("final_energy_eV")
    assert e_cold is not None and e_warm is not None, (
        f"Missing final energies for {fixture['name']!r}: cold={e_cold}, warm={e_warm}"
    )
    delta_e = abs(float(e_cold) - float(e_warm))
    assert delta_e < e_tol, (
        f"Cold and warm runs disagree on final energy by {delta_e:.4f} eV "
        f"(tolerance {e_tol} eV) for {fixture['name']!r}. They likely landed in "
        f"different minima."
    )

    # Step count: warm should not be slower than cold for the same minimum.
    if fixture.get("expect_warm_le_cold", True):
        assert int(warm["n_ionic_steps"]) <= int(cold["n_ionic_steps"]), (
            f"Warm-started VASP took more ionic steps ({warm['n_ionic_steps']}) "
            f"than cold ({cold['n_ionic_steps']}) for {fixture['name']!r} \u2014 "
            f"HydraGNN warm-start did not help."
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
