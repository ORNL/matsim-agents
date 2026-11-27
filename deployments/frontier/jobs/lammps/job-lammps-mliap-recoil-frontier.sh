#!/bin/bash
#SBATCH -J lammps-mliap-recoil
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -N 1
#SBATCH -p batch
# ---------------------------------------------------------------------------
# PKA recoil (MATSIM_MLIAP_PKA_EV, default 200 eV along <110>) in LAMMPS ML-IAP with MACE,
# HydraGNN and UMA, each with additive ZBL, on MATSIM_MLIAP_MPI_RANKS ranks (default 4) of one
# node, CPU. Each stage runs parity first, then NVE (MATSIM_MLIAP_RECOIL_STEPS, default 200 steps
# of MATSIM_MLIAP_TIMESTEP_FS, default 0.2 fs) from the recoiled cell, and records energy drift and the position difference to ASE.
# A software check of the bridge under a recoil, not a cascade result and not a model validation.
#
# Submit: MLIAP_OUT=<dir> MATSIM_MLIAP_SEED=<root seed> sbatch -A <ACCOUNT> -t 01:30:00 -q normal \
#   deployments/frontier/jobs/lammps/job-lammps-mliap-recoil-frontier.sh
# ---------------------------------------------------------------------------

set -euo pipefail

REPO="${PROJECT_ROOT:-$(d="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." 2>/dev/null && pwd)"; [[ -f "$d/pyproject.toml" ]] && echo "$d" || echo "${SLURM_SUBMIT_DIR:?}")}"
OUT="${MLIAP_OUT:?set MLIAP_OUT}"
SEED="${MATSIM_MLIAP_SEED:?set MATSIM_MLIAP_SEED (root seed of the run)}"
RANKS="${MATSIM_MLIAP_MPI_RANKS:-4}"
PKA_EV="${MATSIM_MLIAP_PKA_EV:-200}"
STEPS="${MATSIM_MLIAP_RECOIL_STEPS:-200}"
DT_FS="${MATSIM_MLIAP_TIMESTEP_FS:-0.2}"

# shellcheck disable=SC1091
source "${LAMMPS_ROOT:-$REPO/external/lammps}/activate-mliap.sh"
unset PYTHONHOME
export PYTHONPATH="$REPO/src:$REPO:${PYTHONPATH:-}" XALT_EXECUTABLE_TRACKING=no PYTHONNOUSERSITE=1
mkdir -p "$OUT"

status=0
run() { # name python args...
    local name="$1" py="$2"
    shift 2
    srun -N 1 -n "$RANKS" -c 1 --gpus-per-node=0 "$py" "$REPO/scripts/lammps/mliap_parity.py" \
        "$@" --zbl --pka-eV "$PKA_EV" --md-steps "$STEPS" --md-timestep-fs "$DT_FS" \
        --seed "$SEED" --out "$OUT/$name.json" || status=1
}
run mace-recoil "$REPO/.venv-mace/bin/python" --calc mace --model small

PROJ="$(dirname "$REPO")"
HG_SRC="${HYDRAGNN_DIR:-$PROJ/HydraGNN-e4b-frozen-81d5b883}"
HG_WEIGHTS="${HYDRAGNN_WEIGHTS:-$PROJ/HydraGNN}"
if [[ -d "$HG_SRC/hydragnn" && -d "$HG_WEIGHTS/examples/multidataset_hpo_sc26" ]]; then
    export HYDRAGNN_DIR="$HG_SRC" HYDRAGNN_WEIGHTS="$HG_WEIGHTS"
    PYTHONPATH="$HG_SRC:$PYTHONPATH" run hydragnn-recoil "$REPO/.venv/bin/python" --calc hydragnn
else
    echo "hydragnn: skipped (frozen source or weights missing)" | tee "$OUT/hydragnn-recoil.skipped"
fi

export MATSIM_UMA_ARTIFACT_DIR="${MATSIM_UMA_ARTIFACT_DIR:-$PROJ/models/artifacts/uma}"
UMA_MODEL="${UMA_MODEL:-uma-s-1p1}"
if [[ -f "$MATSIM_UMA_ARTIFACT_DIR/$UMA_MODEL/checkpoint.pt" && -x "$REPO/.venv-uma/bin/python" ]]; then
    export UMA_CHECKPOINT="$MATSIM_UMA_ARTIFACT_DIR/$UMA_MODEL/checkpoint.pt"
    run uma-recoil "$REPO/.venv-uma/bin/python" --calc uma --model "$UMA_MODEL"
else
    echo "uma: skipped (no bundle in $MATSIM_UMA_ARTIFACT_DIR)" | tee "$OUT/uma-recoil.skipped"
fi
exit "$status"
