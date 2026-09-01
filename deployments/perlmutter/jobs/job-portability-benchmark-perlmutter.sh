#!/bin/bash
#SBATCH -J matsim-portability
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 00:20:00
#SBATCH --gpus-per-node=4
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err

set -euo pipefail

REPO="${PROJECT_ROOT:?export PROJECT_ROOT to the matsim-agents checkout}"
RUNS_ROOT="${RUNS_ROOT:-${REPO}/runs}"
RUN_DIR="${RUNS_ROOT}/portability/perlmutter-${SLURM_JOB_ID:-$$}"

python "${REPO}/benchmarks/portability/run.py" \
  --facility perlmutter --suite smoke --backend qe --execute --output "${RUN_DIR}"
python "${REPO}/benchmarks/portability/validate.py" "${RUN_DIR}"

