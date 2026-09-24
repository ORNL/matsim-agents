"""Resolve UMA checkpoints from durable local artifact bundles."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class UMAArtifactBundle:
    checkpoint: Path
    atom_refs: Path
    form_elem_refs: Path | None


def resolve_uma_artifact_bundle(model_name: str) -> UMAArtifactBundle | None:
    root_value = os.environ.get("MATSIM_UMA_ARTIFACT_DIR")
    if not root_value:
        return None
    bundle_dir = Path(root_value).expanduser() / model_name
    checkpoint = bundle_dir / "checkpoint.pt"
    atom_refs = bundle_dir / "atom_refs.yaml"
    form_elem_refs = bundle_dir / "form_elem_refs.yaml"
    if not checkpoint.is_file() or not atom_refs.is_file():
        raise FileNotFoundError(
            f"Incomplete durable UMA bundle for {model_name!r}: {bundle_dir}. "
            "Run deployments/perlmutter/download/download-uma-perlmutter.sh."
        )
    return UMAArtifactBundle(
        checkpoint=checkpoint,
        atom_refs=atom_refs,
        form_elem_refs=form_elem_refs if form_elem_refs.is_file() else None,
    )


def load_uma_predict_unit_from_bundle(
    bundle: UMAArtifactBundle,
    *,
    device: str,
    inference_settings: str = "default",
) -> Any:
    from fairchem.core.units.mlip_unit import load_predict_unit
    from omegaconf import OmegaConf

    atom_refs = OmegaConf.load(bundle.atom_refs)
    form_elem_refs = None
    if bundle.form_elem_refs is not None:
        form_elem_refs = OmegaConf.load(bundle.form_elem_refs)["refs"]
    return load_predict_unit(
        bundle.checkpoint,
        device=device,
        inference_settings=inference_settings,
        atom_refs=atom_refs,
        form_elem_refs=form_elem_refs,
    )