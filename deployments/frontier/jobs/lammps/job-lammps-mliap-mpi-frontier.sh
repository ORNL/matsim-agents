#!/bin/bash
#SBATCH -J lammps-mliap-mpi
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -N 1
#SBATCH -p batch
# ---------------------------------------------------------------------------
# Multi-rank ML-IAP check: EMT and MACE parity plus a 200-step NVE run on
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
exit "$status"
