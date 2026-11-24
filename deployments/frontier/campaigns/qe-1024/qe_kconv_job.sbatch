#!/bin/bash
# Submit: sbatch --parsable -q debug -A LRN110 --job-name=qe-kconv --output=slurm-%j.out \
#   --error=slurm-%j.err qe_kconv_job.sbatch   (KCONV_DIR set via --export=ALL,KCONV_DIR=...)
# Runs the k-point meshes in order (coarse to dense) so partial results survive a timeout.
#SBATCH -p batch
#SBATCH -N 4
#SBATCH --gpus-per-node=8
#SBATCH -t 02:00:00

set -uo pipefail

REPO="${PROJECT_ROOT:-/lustre/orion/lrn110/proj-shared/matsim-agents}"
PW_BIN="${PW_BIN:-${REPO}/external/quantum-espresso/install-gpu/bin/pw.x}"
KCONV_DIR="${KCONV_DIR:?set KCONV_DIR to the prepared test directory}"

[[ -x "${PW_BIN}" ]] || { echo "pw.x not found: ${PW_BIN}" >&2; exit 2; }

module reset
module load Core/25.03 PrgEnv-cray cce craype-accel-amd-gfx90a rocm/6.2.4 cray-fftw

export OMP_NUM_THREADS=7
export OMP_TARGET_OFFLOAD=MANDATORY
export MPICH_GPU_SUPPORT_ENABLED=1

NRANKS=$(( SLURM_JOB_NUM_NODES * 8 ))
NPOOL="${NPOOL:-16}"
for mesh in 4 6 8; do
  cd "${KCONV_DIR}/k${mesh}" || exit 2
  echo "[kconv] mesh ${mesh} start $(date)"
  srun -N"${SLURM_JOB_NUM_NODES}" -n"${NRANKS}" -c7 --threads-per-core=1 --cpu-bind=cores \
    --gpus-per-node=8 --gpu-bind=closest \
    "${PW_BIN}" -npool "${NPOOL}" -in pw.in > pw.out 2> pw.err
  echo "[kconv] mesh ${mesh} exit $? $(date)"
done
