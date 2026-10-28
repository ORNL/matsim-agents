#!/bin/bash
#SBATCH -J matsim-torch-pyg
#SBATCH -N 1
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -t 00:10:00
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err

set -euo pipefail
REPO="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
source /sw/frontier/miniforge3/23.11.0-0/etc/profile.d/conda.sh
source "$REPO/deployments/frontier/setup/frontier-module-stack.sh"
load_frontier_rocm72_modules
source activate "${MATSIM_FRONTIER_VENV:-$REPO/.venv}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export PYTORCH_ROCM_ARCH=gfx90a ROCM_ARCH=gfx90a
export MIOPEN_DISABLE_CACHE=1 MIOPEN_USER_DB_PATH="/tmp/miopen-${SLURM_JOB_ID:-$$}"
mkdir -p "$MIOPEN_USER_DB_PATH"
python "$REPO/deployments/common/qualify_torch_pyg_device.py" \
  --device cuda --output "${QUALIFICATION_OUTPUT:-${REPO}/.hpc-build/frontier/qualification/${SLURM_JOB_ID:-local}/torch-pyg.json}"