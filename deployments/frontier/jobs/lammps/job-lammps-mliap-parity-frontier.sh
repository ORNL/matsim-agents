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
# cached MACE-MP-0 small weights; UMA is skipped (no weights on Frontier).
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
echo "uma: skipped (no UMA weights on Frontier)" | tee "$OUT/uma.skipped"
exit "$status"
