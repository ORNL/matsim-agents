from __future__ import annotations

from pathlib import Path

import pytest

from matsim_agents.backends.mlip.uma_artifacts import resolve_uma_artifact_bundle


def test_resolve_uma_artifact_bundle(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    bundle_dir = tmp_path / "uma-s-1p1"
    bundle_dir.mkdir()
    for name in ("checkpoint.pt", "atom_refs.yaml", "form_elem_refs.yaml"):
        (bundle_dir / name).write_text("data", encoding="utf-8")
    monkeypatch.setenv("MATSIM_UMA_ARTIFACT_DIR", str(tmp_path))

    bundle = resolve_uma_artifact_bundle("uma-s-1p1")

    assert bundle is not None
    assert bundle.checkpoint == bundle_dir / "checkpoint.pt"
    assert bundle.atom_refs == bundle_dir / "atom_refs.yaml"
    assert bundle.form_elem_refs == bundle_dir / "form_elem_refs.yaml"


def test_incomplete_configured_uma_bundle_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("MATSIM_UMA_ARTIFACT_DIR", str(tmp_path))

    with pytest.raises(FileNotFoundError, match="Incomplete durable UMA bundle"):
        resolve_uma_artifact_bundle("uma-s-1p1")


def test_deployments_do_not_store_required_assets_in_cache_or_scratch() -> None:
    repo = Path(__file__).resolve().parents[1]
    forbidden = (
        "VLLM_SRC=$PROJ/cache",
        "TRITON_SRC=$PROJ/cache",
        'FAIRCHEM_CACHE_DIR="${FAIRCHEM_CACHE_DIR:-${SCRATCH',
        'HF_HOME="${HF_HOME:-${PROJ}/models/hf_cache',
        'STAGE_BASE="${SCRATCH',
        'STAGE_ROOT="${SCRATCH',
    )
    violations: list[str] = []
    for path in (repo / "deployments").rglob("*.sh"):
        text = path.read_text(encoding="utf-8")
        for pattern in forbidden:
            if pattern in text:
                violations.append(f"{path.relative_to(repo)}: {pattern}")

    assert not violations, "required artifacts must use durable project storage:\n" + "\n".join(
        violations
    )
