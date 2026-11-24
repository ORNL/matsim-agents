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
# pw.x was built under cpe/24.11 (cray-mpich 8.1.31, rocm 6.2.4); the default cpe pulls rocm 7 libs.
module load cpe/24.11
module load Core/25.03 PrgEnv-cray cce/18.0.1 craype-accel-amd-gfx90a rocm/6.2.4 cray-fftw
MPICH=/opt/cray/pe/mpich/8.1.31
export LD_LIBRARY_PATH="${MPICH}/gtl/lib:${MPICH}/ofi/cray/17.0/lib:${LD_LIBRARY_PATH:-}"

export OMP_NUM_THREADS=7
export OMP_TARGET_OFFLOAD=MANDATORY
export MPICH_GPU_SUPPORT_ENABLED=1

NRANKS=$(( SLURM_JOB_NUM_NODES * 8 ))
NPOOL="${NPOOL:-16}"
# Slurm reports COMPLETED whatever pw.x did, so a mesh counts as successful only if srun
# exited 0 and pw.out shows a converged SCF and JOB DONE. Any failure makes the job fail.
failed=0
for mesh in 4 6 8; do
  cd "${KCONV_DIR}/k${mesh}" || exit 2
  echo "[kconv] mesh ${mesh} start $(date)"
  srun -N"${SLURM_JOB_NUM_NODES}" -n"${NRANKS}" -c7 --threads-per-core=1 --cpu-bind=cores \
    --gpus-per-node=8 --gpu-bind=closest \
    "${PW_BIN}" -npool "${NPOOL}" -in pw.in > pw.out 2> pw.err
  rc=$?
  if (( rc == 0 )) && grep -q 'convergence has been achieved' pw.out \
      && grep -q 'JOB DONE' pw.out; then
    echo "[kconv] mesh ${mesh} OK exit ${rc} $(date)"
  else
    echo "[kconv] mesh ${mesh} FAILED exit ${rc} (no converged SCF or no JOB DONE) $(date)" >&2
    failed=1
  fi
done
exit "${failed}"
