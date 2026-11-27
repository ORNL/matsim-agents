#!/bin/bash
# Fetch UMA bundles into the shared artifacts tree; needs HF_TOKEN and the facebook/UMA license.
# Usage: bash deployments/frontier/download/download-uma-frontier.sh   (UMA_MODELS="uma-s-1p1 ...")
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PROJ="$(dirname "${REPO}")"
VENV="${MATSIM_FAIRCHEM_VENV:-${REPO}/.venv-uma}"
UMA_ARTIFACT_DIR="${MATSIM_UMA_ARTIFACT_DIR:-${PROJ}/models/artifacts/uma}"

if [[ -z "${HF_TOKEN:-}" && -f "${HOME}/.cache/huggingface/token" ]]; then
  HF_TOKEN="$(< "${HOME}/.cache/huggingface/token")"
  export HF_TOKEN
fi
[[ -n "${HF_TOKEN:-}" ]] || { echo "ERROR: HF_TOKEN not set" >&2; exit 1; }

export PYTHONNOUSERSITE=1 XALT_EXECUTABLE_TRACKING=no
mkdir -p "${UMA_ARTIFACT_DIR}"
for model in ${UMA_MODELS:-uma-s-1p1}; do
  echo "[$(date)] fetching ${model} -> ${UMA_ARTIFACT_DIR}"
  "${VENV}/bin/python" "${SCRIPT_DIR}/fetch_uma_bundle.py" "${model}" "${UMA_ARTIFACT_DIR}"
done
