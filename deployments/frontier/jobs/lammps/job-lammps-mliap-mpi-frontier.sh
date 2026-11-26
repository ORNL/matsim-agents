#!/bin/bash
#SBATCH -J lammps-mliap-mpi
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -N 1
#SBATCH -p batch
# ---------------------------------------------------------------------------
# Multi-rank ML-IAP check: EMT, MACE, HydraGNN and UMA parity plus NVE runs on
# MATSIM_MLIAP_MPI_RANKS ranks (default 4) of one node, CPU. The script asserts that the MPI world
# size equals SLURM_NTASKS and records it as "mpi_ranks" in the result, so the output proves the
# rank count. Every rank evaluates the full system (all-gather), so this checks correctness only,
# not scaling.
#
# Submit: MLIAP_OUT=<dir> sbatch -A <ACCOUNT> -t 00:20:00 -q debug \
#   deployments/frontier/jobs/lammps/job-lammps-mliap-mpi-frontier.sh
# ---------------------------------------------------------------------------

set -euo pipefail

REPO="${PROJECT_ROOT:-$(d="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." 2>/dev/null && pwd)"; [[ -f "$d/pyproject.toml" ]] && echo "$d" || echo "${SLURM_SUBMIT_DIR:?}")}"
OUT="${MLIAP_OUT:?set MLIAP_OUT}"
RANKS="${MATSIM_MLIAP_MPI_RANKS:-4}"

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
        "$@" --out "$OUT/$name.json" || status=1
}
run emt-mpi "$REPO/.venv/bin/python" --calc emt --md-steps 200 --md-timestep-fs 0.2 --pka-eV 200
run mace-mpi "$REPO/.venv-mace/bin/python" --calc mace --model small --md-steps 20

PROJ="$(dirname "$REPO")"
HG_SRC="${HYDRAGNN_DIR:-$PROJ/HydraGNN-e4b-frozen-81d5b883}"
HG_WEIGHTS="${HYDRAGNN_WEIGHTS:-$PROJ/HydraGNN}"
if [[ -d "$HG_SRC/hydragnn" && -d "$HG_WEIGHTS/examples/multidataset_hpo_sc26" ]]; then
    export HYDRAGNN_DIR="$HG_SRC" HYDRAGNN_WEIGHTS="$HG_WEIGHTS"
    PYTHONPATH="$HG_SRC:$PYTHONPATH" run hydragnn-mpi "$REPO/.venv/bin/python" \
        --calc hydragnn --md-steps 20
else
    echo "hydragnn: skipped (frozen source or weights missing)" | tee "$OUT/hydragnn-mpi.skipped"
fi

export MATSIM_UMA_ARTIFACT_DIR="${MATSIM_UMA_ARTIFACT_DIR:-$PROJ/models/artifacts/uma}"
UMA_MODEL="${UMA_MODEL:-uma-s-1p1}"
if [[ -f "$MATSIM_UMA_ARTIFACT_DIR/$UMA_MODEL/checkpoint.pt" && -x "$REPO/.venv-uma/bin/python" ]]; then
    export UMA_CHECKPOINT="$MATSIM_UMA_ARTIFACT_DIR/$UMA_MODEL/checkpoint.pt"
    run uma-mpi "$REPO/.venv-uma/bin/python" --calc uma --model "$UMA_MODEL" --md-steps 20
else
    echo "uma: skipped (no bundle in $MATSIM_UMA_ARTIFACT_DIR)" | tee "$OUT/uma-mpi.skipped"
fi
exit "$status"
