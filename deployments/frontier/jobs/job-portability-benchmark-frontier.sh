#!/bin/bash
#SBATCH -J matsim-portability
#SBATCH -N 1
#SBATCH -t 00:20:00
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err

set -euo pipefail

REPO="${PROJECT_ROOT:?export PROJECT_ROOT to the matsim-agents checkout}"
RUNS_ROOT="${RUNS_ROOT:-${REPO}/runs}"
RUN_DIR="${RUNS_ROOT}/portability/frontier-${SLURM_JOB_ID:-$$}"

# The portability job is deliberately thin: the activated environment and
# science configuration are identical in spirit on every machine; only this
# scheduler envelope and the facility overlay differ.
python "${REPO}/benchmarks/portability/run.py" \
  --facility frontier --suite all --backend qe --execute --output "${RUN_DIR}"
python "${REPO}/benchmarks/portability/validate.py" "${RUN_DIR}"
