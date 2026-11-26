#!/bin/bash
#SBATCH -J lammps-mliap-parity
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -N 1
#SBATCH -p batch
# ---------------------------------------------------------------------------
# ASE vs LAMMPS ML-IAP energy/force parity on a compute node (single rank, CPU).
# Needs external/lammps/activate-mliap.sh from
# deployments/frontier/setup/lammps/build-lammps-mliap-frontier.sh. MACE uses the
# cached MACE-MP-0 small weights. HydraGNN needs the frozen source (HYDRAGNN_DIR) and the
# BEST6 weights tree (HYDRAGNN_WEIGHTS); UMA needs the bundle from
# deployments/frontier/download/download-uma-frontier.sh and is skipped without it.
#
# Submit: MLIAP_OUT=<dir> sbatch -A <ACCOUNT> -t 00:30:00 -q debug \
#   deployments/frontier/jobs/lammps/job-lammps-mliap-parity-frontier.sh
# ---------------------------------------------------------------------------

set -euo pipefail

REPO="${PROJECT_ROOT:-$(d="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." 2>/dev/null && pwd)"; [[ -f "$d/pyproject.toml" ]] && echo "$d" || echo "${SLURM_SUBMIT_DIR:?}")}"
OUT="${MLIAP_OUT:?set MLIAP_OUT}"

# shellcheck disable=SC1091
source "${LAMMPS_ROOT:-$REPO/external/lammps}/activate-mliap.sh"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 XALT_EXECUTABLE_TRACKING=no OMP_NUM_THREADS=1
export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
cd "$OUT"

status=0
run() { # name python args...
    local name="$1" py="$2"
    shift 2
    "$py" "$REPO/scripts/lammps/mliap_parity.py" "$@" --out "$OUT/$name.json" || status=1
}

run emt "$REPO/.venv/bin/python" --calc emt
run emt-zbl "$REPO/.venv/bin/python" --calc emt --a 2.3 --rattle 0.02 --zbl
run mace "$REPO/.venv-mace/bin/python" --calc mace --model small
run mace-zbl "$REPO/.venv-mace/bin/python" --calc mace --model small --a 2.3 --rattle 0.02 --zbl
PROJ="$(dirname "$REPO")"
HG_SRC="${HYDRAGNN_DIR:-$PROJ/HydraGNN-e4b-frozen-81d5b883}"
HG_WEIGHTS="${HYDRAGNN_WEIGHTS:-$PROJ/HydraGNN}"
if [[ -d "$HG_SRC/hydragnn" && -d "$HG_WEIGHTS/examples/multidataset_hpo_sc26" ]]; then
    export HYDRAGNN_DIR="$HG_SRC" HYDRAGNN_WEIGHTS="$HG_WEIGHTS"
    HG_PP="$HG_SRC:$PYTHONPATH"
    PYTHONPATH="$HG_PP" run hydragnn "$REPO/.venv/bin/python" --calc hydragnn
    PYTHONPATH="$HG_PP" run hydragnn-zbl "$REPO/.venv/bin/python" --calc hydragnn \
        --a 2.3 --rattle 0.02 --zbl
else
    echo "hydragnn: skipped (frozen source or weights missing)" | tee "$OUT/hydragnn.skipped"
fi

export MATSIM_UMA_ARTIFACT_DIR="${MATSIM_UMA_ARTIFACT_DIR:-$PROJ/models/artifacts/uma}"
UMA_MODEL="${UMA_MODEL:-uma-s-1p1}"
if [[ -f "$MATSIM_UMA_ARTIFACT_DIR/$UMA_MODEL/checkpoint.pt" && -x "$REPO/.venv-uma/bin/python" ]]; then
    export UMA_CHECKPOINT="$MATSIM_UMA_ARTIFACT_DIR/$UMA_MODEL/checkpoint.pt"
    run uma "$REPO/.venv-uma/bin/python" --calc uma --model "$UMA_MODEL"
    run uma-zbl "$REPO/.venv-uma/bin/python" --calc uma --model "$UMA_MODEL" \
        --a 2.3 --rattle 0.02 --zbl
else
    echo "uma: skipped (no bundle in $MATSIM_UMA_ARTIFACT_DIR)" | tee "$OUT/uma.skipped"
fi
exit "$status"
