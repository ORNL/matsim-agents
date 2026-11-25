#!/bin/bash
#SBATCH -J matsim-hpc-runtime
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH --gpus-per-node=4
#SBATCH -c 32
#SBATCH -t 00:30:00
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
# Scheduler and launcher validation (see deployments/common/validate_hpc_runtime.py).
#   sbatch -A <allocation> deployments/perlmutter/jobs/job-hpc-runtime-validation-perlmutter.sh

set -uo pipefail
REPO="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
source "$REPO/deployments/perlmutter/setup/perlmutter-module-stack.sh"
load_perlmutter_modules_gpu
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${MATSIM_PERLMUTTER_VENV:-$REPO/.venv}"

export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO/src:$REPO"
export MATSIM_QE_LAUNCHER="${MATSIM_QE_LAUNCHER:-bash $REPO/deployments/perlmutter/launchers/run-pw-gpu-perlmutter.sh}"
export MATSIM_QE_PSEUDO_DIR="${MATSIM_QE_PSEUDO_DIR:-$REPO/external/quantum-espresso/src/pseudo}"
OUT="${VALIDATION_OUTPUT_DIR:-$REPO/.hpc-build/perlmutter/runtime-validation/${SLURM_JOB_ID:-local}}"

status=0
for phase in allocation qe; do
  python "$REPO/deployments/common/validate_hpc_runtime.py" "$phase" \
    --facility perlmutter-gpu --output "$OUT" || status=1
done
echo "runtime validation exit status: $status"
exit "$status"
