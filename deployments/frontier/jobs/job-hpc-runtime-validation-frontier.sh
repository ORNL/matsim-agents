#!/bin/bash
#SBATCH -J matsim-hpc-runtime
#SBATCH -N 1
#SBATCH -p batch
#SBATCH -q debug
#SBATCH --gpus-per-node=8
#SBATCH -t 00:30:00
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
# Scheduler and launcher validation on a real allocation (see deployments/common/validate_hpc_runtime.py).
#   sbatch -A <allocation> deployments/frontier/jobs/job-hpc-runtime-validation-frontier.sh

set -uo pipefail
REPO="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
export XALT_EXECUTABLE_TRACKING=no PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO/src:$REPO"
export MATSIM_QE_LAUNCHER="${MATSIM_QE_LAUNCHER:-bash $REPO/deployments/frontier/launchers/run-pw-gpu-frontier.sh}"
export MATSIM_QE_PSEUDO_DIR="${MATSIM_QE_PSEUDO_DIR:-$REPO/external/quantum-espresso/src/pseudo}"
OUT="${VALIDATION_OUTPUT_DIR:-$REPO/.hpc-build/frontier/runtime-validation/${SLURM_JOB_ID:-local}}"
PY="${MATSIM_FRONTIER_VENV:-$REPO/.venv}/bin/python"

status=0
for phase in allocation qe; do
  "$PY" "$REPO/deployments/common/validate_hpc_runtime.py" "$phase" --facility frontier --output "$OUT" || status=1
done
echo "runtime validation exit status: $status"
exit "$status"
