#!/bin/bash
# Submit (authorized): sbatch --parsable -q debug -A LRN110 --job-name=qe-1024-validation \
#   --output=slurm-%j.out --error=slurm-%j.err qe_1024_job.sbatch
# debug QoS: MaxWall 02:00:00, 1 job per user. CALC_DIR must hold a gate-validated pw.in.
#SBATCH -p batch
#SBATCH -N 8
#SBATCH --gpus-per-node=8
#SBATCH -t 02:00:00

set -euo pipefail

REPO="${PROJECT_ROOT:-/lustre/orion/lrn110/proj-shared/matsim-agents}"
PW_BIN="${PW_BIN:-${REPO}/external/quantum-espresso/install-gpu/bin/pw.x}"
CALC_DIR="${CALC_DIR:?set CALC_DIR to a prepared calculation directory}"
NPOOL="${NPOOL:-1}"

[[ -x "${PW_BIN}" ]] || { echo "pw.x not found: ${PW_BIN}" >&2; exit 2; }
[[ -f "${CALC_DIR}/pw.in" ]] || { echo "no pw.in in ${CALC_DIR}" >&2; exit 2; }
grep -Eq "nat *= *10[0-9]{2}" "${CALC_DIR}/pw.in" || { echo "pw.in is not a full-size cell" >&2; exit 3; }

module reset
module load Core/25.03 PrgEnv-cray cce craype-accel-amd-gfx90a rocm/6.2.4 cray-fftw

export OMP_NUM_THREADS=7
export OMP_TARGET_OFFLOAD=MANDATORY
export MPICH_GPU_SUPPORT_ENABLED=1

cd "${CALC_DIR}"
sha256sum pw.in > pw.in.sha256
NRANKS=$(( SLURM_JOB_NUM_NODES * 8 ))
srun -N"${SLURM_JOB_NUM_NODES}" -n"${NRANKS}" -c7 --threads-per-core=1 --cpu-bind=cores \
  --gpus-per-node=8 --gpu-bind=closest \
  "${PW_BIN}" -npool "${NPOOL}" -in pw.in > pw.out

grep -q 'convergence has been achieved' pw.out
grep '^!' pw.out
