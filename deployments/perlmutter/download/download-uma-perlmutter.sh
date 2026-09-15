#!/bin/bash
#SBATCH -J dl-uma
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
# ---------------------------------------------------------------------------
# Pre-fetch the UMA (fairchem-core) universal MLIP checkpoint(s) into the shared
# Hugging Face cache, so downstream AL / warm-start jobs do not spend GPU
# wall-time downloading on first use.
#
# UMA weights are NOT fetched by the `download-models-*.sh` scripts (those cover
# the LLM chat models). fairchem normally downloads UMA lazily on the first call
# to `pretrained_mlip.get_predict_unit(...)`; this script does that once, on the
# CPU partition, and verifies the model loads.
#
# Submit (default model uma-s-1p1):
#   sbatch deployments/perlmutter/download/download-uma-perlmutter.sh
#
# Multiple / alternate models:
#   UMA_MODELS="uma-s-1p1 uma-m-1p1" \
#   sbatch deployments/perlmutter/download/download-uma-perlmutter.sh
#
# Alternate cache location:
#   MATSIM_UMA_ARTIFACT_DIR=/path/to/project/models/artifacts/uma \
#   sbatch deployments/perlmutter/download/download-uma-perlmutter.sh
#
# Notes:
# - facebook/UMA is a GATED repo: you must accept the license on Hugging Face
#   and provide a token. This script reads ~/.cache/huggingface/token if HF_TOKEN
#   is not already set (run `hf auth login` once beforehand).
# - Downloads are resumable; rerunning skips files already in the cache.
# - Runs from the matsim-owned .venv-uma built with INSTALL_UMA=1.
# ---------------------------------------------------------------------------

set -euo pipefail

# ── paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
REPO_DEFAULT="$(cd "${SCRIPT_DIR}/../../.." 2>/dev/null && pwd)"
REPO="${PROJECT_ROOT:-${REPO_DEFAULT}}"
[[ ! -f "${REPO}/pyproject.toml" ]] && \
  REPO=${PROJECT_ROOT:?export PROJECT_ROOT}
PROJ="$(dirname "${REPO}")"

# Isolated FairChem/UMA compatibility environment.
VENV_ROOT="$REPO/.hpc-build/perlmutter"
VENV="${MATSIM_FAIRCHEM_VENV:-${REPO}/.venv-uma}"
RUN_DIR="$PROJ/runs/download-uma-${SLURM_JOB_ID:-manual}"
mkdir -p "$RUN_DIR"

if [[ ! -d "$VENV" ]]; then
  echo "ERROR: matsim .venv-uma not found: $VENV" >&2
  echo "Build it first with:" >&2
  echo "  INSTALL_UMA=1 bash deployments/perlmutter/setup/install.sh" >&2
  exit 1
fi

# ── modules and matsim-owned environment ─────────────────────────────────────
source "$REPO/deployments/perlmutter/setup/perlmutter-module-stack.sh"
load_perlmutter_modules
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

# Durable shared model artifacts on CFS. Downloads stream directly into this
# tree and atomically rename on completion, bypassing Hugging Face cache locks.
ARTIFACT_ROOT="${MATSIM_MODEL_ARTIFACTS_ROOT:-${PROJ}/models/artifacts}"
UMA_ARTIFACT_DIR="${MATSIM_UMA_ARTIFACT_DIR:-${ARTIFACT_ROOT}/uma}"
mkdir -p "${UMA_ARTIFACT_DIR}"
if [[ -z "${HF_TOKEN:-}" && -f "${HOME}/.cache/huggingface/token" ]]; then
  export HF_TOKEN="$(< "${HOME}/.cache/huggingface/token")"
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "WARNING: HF_TOKEN not set and ~/.cache/huggingface/token not found." >&2
  echo "         The download will fail for the gated facebook/UMA repo." >&2
  echo "         Run 'hf auth login' once, or export HF_TOKEN=hf_..." >&2
fi

# ── model list ───────────────────────────────────────────────────────────────
UMA_MODELS="${UMA_MODELS:-uma-s-1p1}"

echo "[$(date)] Durable UMA artifact directory: $UMA_ARTIFACT_DIR"
echo "[$(date)] UMA models to pre-fetch: $UMA_MODELS"
echo "[$(date)] Using venv: $VENV"

rc=0
for model_name in $UMA_MODELS; do
  log="$RUN_DIR/${model_name}.download.log"
  echo
  echo "[$(date)] Fetching + validating UMA model: $model_name"
  if python - "$model_name" "$UMA_ARTIFACT_DIR" >"$log" 2>&1 <<'PY'
import json
import os
from pathlib import Path
import sys
import urllib.request

from fairchem.core.calculate import pretrained_mlip
from fairchem.core.units.mlip_unit import load_predict_unit
from huggingface_hub import hf_hub_url
from omegaconf import OmegaConf

model_name, artifact_root = sys.argv[1:]
metadata = pretrained_mlip._MODEL_CKPTS.checkpoints[model_name]
bundle = Path(artifact_root) / model_name
bundle.mkdir(parents=True, exist_ok=True)
token = os.environ.get("HF_TOKEN")


def download(filename, subfolder, destination):
    if destination.is_file() and destination.stat().st_size:
        return
    url = hf_hub_url(
        metadata.repo_id,
        filename,
        subfolder=subfolder,
        revision=metadata.revision,
    )
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    request = urllib.request.Request(url, headers=headers)
    temporary = destination.with_suffix(destination.suffix + f".part.{os.getpid()}")
    try:
        with urllib.request.urlopen(request) as response, temporary.open("wb") as output:
            while chunk := response.read(8 * 1024 * 1024):
                output.write(chunk)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


download(metadata.filename, metadata.subfolder, bundle / "checkpoint.pt")
download(metadata.atom_refs["filename"], metadata.atom_refs["subfolder"], bundle / "atom_refs.yaml")
if metadata.form_elem_refs is not None:
    download(
        metadata.form_elem_refs["filename"],
        metadata.form_elem_refs["subfolder"],
        bundle / "form_elem_refs.yaml",
    )
(bundle / "manifest.json").write_text(
    json.dumps({"model": model_name, "repo_id": metadata.repo_id, "revision": metadata.revision}, indent=2) + "\n"
)
atom_refs = OmegaConf.load(bundle / "atom_refs.yaml")
form_path = bundle / "form_elem_refs.yaml"
form_elem_refs = OmegaConf.load(form_path)["refs"] if form_path.is_file() else None
predictor = load_predict_unit(
    bundle / "checkpoint.pt",
    device="cpu",
    atom_refs=atom_refs,
    form_elem_refs=form_elem_refs,
)
print(f"OK: {model_name} loaded from durable bundle -> {type(predictor).__name__}")
PY
  then
    echo "[$(date)] DONE: $model_name"
  else
    echo "[$(date)] FAILED: $model_name (see $log)"
    rc=1
  fi
done

echo
echo "[$(date)] Completed UMA download job. Logs in $RUN_DIR"
exit "$rc"
