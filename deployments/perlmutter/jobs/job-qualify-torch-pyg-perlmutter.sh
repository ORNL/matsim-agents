#!/bin/bash
#SBATCH -J matsim-torch-pyg
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH --gpus-per-node=1
#SBATCH -c 8
#SBATCH -q debug
#SBATCH -t 00:10:00
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err

set -euo pipefail
REPO="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
source "$REPO/deployments/perlmutter/setup/perlmutter-module-stack.sh"
load_perlmutter_modules_gpu
PYTHON="${MATSIM_PERLMUTTER_VENV:-$REPO/.venv}/bin/python"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
"$PYTHON" "$REPO/deployments/common/qualify_torch_pyg_device.py" \
  --device cuda --output "${QUALIFICATION_OUTPUT:-${REPO}/.hpc-build/perlmutter/qualification/${SLURM_JOB_ID:-local}/torch-pyg.json}"