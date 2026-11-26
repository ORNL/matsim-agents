#!/bin/bash
#SBATCH -J lammps-mliap-finetune
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -N 1
#SBATCH -p batch
# ---------------------------------------------------------------------------
# HydraGNN, MACE and UMA fine-tunes on a small EMT-labelled Cu set (CPU), each followed by the
# LAMMPS re-export of the resulting checkpoint: parity plus NVE stability, written as
# <backend>/finetune-export.json. Checks the pipeline, not model accuracy. MATSIM_MLIAP_FT_LORA=1
# makes the MACE and UMA fine-tunes low-rank.
#
# Submit: MLIAP_OUT=<dir> sbatch -A <ACCOUNT> -t 00:30:00 -q normal \
#   deployments/frontier/jobs/lammps/job-lammps-mliap-finetune-frontier.sh
# ---------------------------------------------------------------------------

set -euo pipefail

REPO="${PROJECT_ROOT:-$(d="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." 2>/dev/null && pwd)"; [[ -f "$d/pyproject.toml" ]] && echo "$d" || echo "${SLURM_SUBMIT_DIR:?}")}"
OUT="${MLIAP_OUT:?set MLIAP_OUT}"
PROJ="$(dirname "$REPO")"
export HYDRAGNN_DIR="${HYDRAGNN_DIR:-$PROJ/HydraGNN-e4b-frozen-81d5b883}"
export HYDRAGNN_WEIGHTS="${HYDRAGNN_WEIGHTS:-$PROJ/HydraGNN}"

# shellcheck disable=SC1091
source "${LAMMPS_ROOT:-$REPO/external/lammps}/activate-mliap.sh"
unset PYTHONHOME
export PYTHONPATH="$HYDRAGNN_DIR:$REPO/src:$REPO:${PYTHONPATH:-}" XALT_EXECUTABLE_TRACKING=no PYTHONNOUSERSITE=1
mkdir -p "$OUT"

status=0
ft() { # backend, python interpreter
    local backend="$1" py="$2"
    "$py" "$REPO/scripts/lammps/mliap_finetune_export.py" --backend "$backend" \
        --out "$OUT/$backend" --epochs "${MATSIM_MLIAP_FT_EPOCHS:-1}" \
        ${MATSIM_MLIAP_FT_LORA:+--lora} || status=1
}

ft hydragnn "$REPO/.venv/bin/python"
ft mace "$REPO/.venv-mace/bin/python"

export MATSIM_UMA_ARTIFACT_DIR="${MATSIM_UMA_ARTIFACT_DIR:-$PROJ/models/artifacts/uma}"
ft uma "$REPO/.venv-uma/bin/python"
exit "$status"
