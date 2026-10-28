#!/usr/bin/env bash
#PBS -N matsim-torch-pyg-xpu
#PBS -l select=1
#PBS -l place=scatter
#PBS -l walltime=00:10:00
#PBS -l filesystems=home:flare
#PBS -q debug
#PBS -k doe
#PBS -j oe
set -eo pipefail

REPO="${PROJECT_ROOT:-${PBS_O_WORKDIR:?Set PROJECT_ROOT to the Matsim checkout}}"
[[ -f "${REPO}/pyproject.toml" ]] || exit 2
source "${REPO}/deployments/aurora/setup/aurora-module-stack.sh"
load_aurora_frameworks_modules
load_aurora_xpu_modules
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export ZE_FLAT_DEVICE_HIERARCHY=FLAT ZE_AFFINITY_MASK=0
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cd "${REPO}"
timeout --kill-after=10 540 mpiexec -n 1 --ppn 1 \
    "${REPO}/.venv/bin/python" \
    "${REPO}/deployments/aurora/jobs/qualify_torch_pyg_xpu.py" \
    --output "${QUALIFICATION_OUTPUT:-${REPO}/.hpc-build/aurora/qualification/${PBS_JOBID:-local}/torch-pyg.json}"