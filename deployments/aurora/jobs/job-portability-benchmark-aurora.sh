#!/bin/bash
#PBS -N matsim-portability
#PBS -l select=1
#PBS -l place=scatter
#PBS -l walltime=00:20:00
#PBS -l filesystems=home:flare
#PBS -q debug
#PBS -k doe
#PBS -j oe

set -eo pipefail

REPO="${PROJECT_ROOT:?export PROJECT_ROOT to the matsim-agents checkout}"
RUNS_ROOT="${RUNS_ROOT:-${REPO}/runs}"
RUN_DIR="${RUNS_ROOT}/portability/aurora-${PBS_JOBID:-$$}"

python "${REPO}/benchmarks/portability/run.py" \
  --facility aurora --suite smoke --backend qe --execute --output "${RUN_DIR}"
python "${REPO}/benchmarks/portability/validate.py" "${RUN_DIR}"

