"""Download a UMA bundle (checkpoint + refs) into a durable directory and verify it loads."""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

from fairchem.core.calculate import pretrained_mlip
from fairchem.core.units.mlip_unit import load_predict_unit
from huggingface_hub import hf_hub_url
from omegaconf import OmegaConf

model_name, artifact_root = sys.argv[1:3]
metadata = pretrained_mlip._MODEL_CKPTS.checkpoints[model_name]
bundle = Path(artifact_root) / model_name
bundle.mkdir(parents=True, exist_ok=True)
token = os.environ.get("HF_TOKEN")
manifest = {"model": model_name, "repo_id": metadata.repo_id, "revision": metadata.revision}
manifest_path = bundle / "manifest.json"


def _manifest_matches() -> bool:
    try:
        return json.loads(manifest_path.read_text()) == manifest
    except (OSError, ValueError):
        return False


# A manifest is written only after a complete download, so a match means the files are current.
reuse = _manifest_matches()
if not reuse:
    manifest_path.unlink(missing_ok=True)


def download(filename: str, subfolder: str | None, destination: Path) -> None:
    if reuse and destination.is_file() and destination.stat().st_size:
        return
    url = hf_hub_url(metadata.repo_id, filename, subfolder=subfolder, revision=metadata.revision)
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    tmp = destination.with_suffix(destination.suffix + f".part.{os.getpid()}")
    try:
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request) as response, tmp.open("wb") as out:
            while chunk := response.read(8 * 1024 * 1024):
                out.write(chunk)
        tmp.replace(destination)
    finally:
        tmp.unlink(missing_ok=True)


download(metadata.filename, metadata.subfolder, bundle / "checkpoint.pt")
download(metadata.atom_refs["filename"], metadata.atom_refs["subfolder"], bundle / "atom_refs.yaml")
if metadata.form_elem_refs is not None:
    download(
        metadata.form_elem_refs["filename"],
        metadata.form_elem_refs["subfolder"],
        bundle / "form_elem_refs.yaml",
    )
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

atom_refs = OmegaConf.load(bundle / "atom_refs.yaml")
form_path = bundle / "form_elem_refs.yaml"
form_elem_refs = OmegaConf.load(form_path)["refs"] if form_path.is_file() else None
predictor = load_predict_unit(
    bundle / "checkpoint.pt", device="cpu", atom_refs=atom_refs, form_elem_refs=form_elem_refs
)
print(f"OK: {model_name} loaded from {bundle} -> {type(predictor).__name__}")
