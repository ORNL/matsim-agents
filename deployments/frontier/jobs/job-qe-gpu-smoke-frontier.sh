#!/bin/bash
#SBATCH -J qe-gpu-smoke
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 1
#SBATCH --gpus-per-node=8
#SBATCH -t 00:10:00
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err

set -euo pipefail

REPO="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
QE_PREFIX="${QE_PREFIX:-${REPO}/external/quantum-espresso}"
PW_BIN="${PW_BIN:-${QE_PREFIX}/install-gpu/bin/pw.x}"
ROCM_MODULE="${ROCM_MODULE:-rocm/6.2.4}"
WORK_DIR="${QE_SMOKE_DIR:-${REPO}/.hpc-build/frontier/qe-smoke/${SLURM_JOB_ID:-local}}"

[[ -x "${PW_BIN}" ]] || { echo "pw.x not found: ${PW_BIN}" >&2; exit 2; }
mkdir -p "${WORK_DIR}/tmp"
cd "${WORK_DIR}"
cat > pw.in <<EOF
&CONTROL
  calculation = 'scf'
  prefix = 'si'
  pseudo_dir = '${QE_PREFIX}/src/pseudo'
  outdir = '${WORK_DIR}/tmp'
  tprnfor = .true.
  tstress = .true.
/
&SYSTEM
  ibrav = 0
  nat = 2
  ntyp = 1
  ecutwfc = 20
  ecutrho = 160
  occupations = 'fixed'
/
&ELECTRONS
  conv_thr = 1.0d-8
/
ATOMIC_SPECIES
  Si 28.0850 Si_r.upf

CELL_PARAMETERS angstrom
  0.0000000000  2.7500000000  2.7500000000
  2.7500000000  0.0000000000  2.7500000000
  2.7500000000  2.7500000000  0.0000000000

ATOMIC_POSITIONS angstrom
  Si 0.0000000000 0.0000000000 0.0000000000
  Si 1.4250000000 1.3450000000 1.3950000000

K_POINTS automatic
  2 2 2  0 0 0
EOF

module reset
module load Core/25.03
module load PrgEnv-cray
module load cce
module load craype-accel-amd-gfx90a
module load "${ROCM_MODULE}"
module load cray-fftw

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-7}"
export OMP_TARGET_OFFLOAD=MANDATORY
export MPICH_GPU_SUPPORT_ENABLED=1

echo "QE GPU smoke test on $(hostname), job ${SLURM_JOB_ID:-local}"
echo "pw.x: ${PW_BIN}"
echo "GPU offload: mandatory; MPI ranks: 8; OpenMP threads/rank: ${OMP_NUM_THREADS}"
srun -N1 -n8 -c"${OMP_NUM_THREADS}" --threads-per-core=1 --cpu-bind=cores \
  --gpus-per-node=8 --gpu-bind=closest \
  "${PW_BIN}" -in pw.in > pw.out

grep -q 'convergence has been achieved' pw.out
grep -q '^!' pw.out
echo "QE SCF converged; result: ${WORK_DIR}/pw.out"