#!/bin/bash
# =============================================================================
# _qe-step-frontier.sh
#
# Inner-step launcher for ONE Quantum ESPRESSO pw.x single-point on Frontier
# (AMD MI250X / gfx90a). Called by the active-learning Python driver via:
#
#   bash _qe-step-frontier.sh <work_dir> <pw_bin> <input_file> \
#                             <nodes> <ranks_per_node> <threads_per_rank>
#
# This script does the full module reset/swap so the driver shell (which runs
# under PrgEnv-gnu + the HydraGNN venv + rocm/7.2.0) never has to. It is safe
# to invoke concurrently from multiple Python threads inside one SLURM
# allocation: each call issues its own `srun` step.
#
# Module pin matches deployments/frontier/setup/build-qe-gpu-frontier.sh
# (rocm/6.2.4 — the older ROCm is required by QE's GPU-offload patches).
# =============================================================================

set -euo pipefail

if [[ $# -lt 6 ]]; then
  echo "Usage: $0 <work_dir> <pw_bin> <input_file> <nodes> <ranks_per_node> <threads_per_rank>" >&2
  exit 2
fi

WORK_DIR="$1"
PW_BIN="$2"
INPUT="$3"
NNODES="$4"
RANKS_PER_NODE="$5"
THREADS_PER_RANK="$6"
TOTAL_RANKS=$(( NNODES * RANKS_PER_NODE ))
# Devices per node come from the placement; several ranks may share one device.
GPUS_PER_NODE="${MATSIM_DEVICES_PER_NODE:-$RANKS_PER_NODE}"

[[ -d "${WORK_DIR}" ]] || { echo "work_dir not a directory: ${WORK_DIR}" >&2; exit 2; }
[[ -x "${PW_BIN}"   ]] || { echo "pw.x not executable: ${PW_BIN}" >&2; exit 2; }
[[ -f "${INPUT}"    ]] || { echo "input file missing: ${INPUT}" >&2; exit 2; }

# ── Module reset to the QE build's toolchain ───────────────────────────────
# IMPORTANT: do NOT inherit modules from the parent shell — QE needs
# PrgEnv-cray + cce + craype-accel-amd-gfx90a + rocm/6.2.4 + cray-fftw,
# while the parent driver runs under PrgEnv-gnu + rocm/7.2.0.
# `module reset` is mandatory; otherwise Lmod will refuse to swap PrgEnv.
if command -v module >/dev/null 2>&1; then
  :
elif [[ -f /usr/share/lmod/lmod/init/bash ]]; then
  # shellcheck disable=SC1091
  source /usr/share/lmod/lmod/init/bash
fi

ROCM_MODULE="${ROCM_MODULE:-rocm/6.2.4}"

module reset
module load PrgEnv-cray
module load cce
module load craype-accel-amd-gfx90a
module load "${ROCM_MODULE}"
module load cray-fftw
module unload darshan-runtime 2>/dev/null || true

# Strip env vars that would perturb the QE launch (set by the Python venv)
unset LD_PRELOAD
unset VLLM_CUDART_SO_PATH
unset PYTHONPATH

# ── QE runtime tuning ───────────────────────────────────────────────────────
export OMP_NUM_THREADS="${THREADS_PER_RANK}"
export OMP_PLACES="cores"
export OMP_PROC_BIND="close"
export MPICH_GPU_SUPPORT_ENABLED=1
export OMP_TARGET_OFFLOAD="${OMP_TARGET_OFFLOAD:-DEFAULT}"
export OMP_STACKSIZE="${OMP_STACKSIZE:-512M}"

cd "${WORK_DIR}"

echo "[qe-step] $(date) host=$(hostname) work_dir=${WORK_DIR}"
NODE_ARGS=()
if [[ -n "${MATSIM_ASSIGNED_NODES:-}" ]]; then
  NODE_ARGS=(--nodelist "${MATSIM_ASSIGNED_NODES}")
fi
echo "[qe-step] assigned_nodes=${MATSIM_ASSIGNED_NODES:-scheduler-selected}"
echo "[qe-step] srun -N ${NNODES} -n ${TOTAL_RANKS} -c ${THREADS_PER_RANK} \\"
echo "             --gpus-per-node=${GPUS_PER_NODE} --gpu-bind=closest ${PW_BIN} -in ${INPUT}"

# Slurm picks which GPUs a step receives, so a GPU count alone cannot honour the
# placement's device indices. When the placement uses only some of the node's
# GPUs, take them all for the step and expose only the placed ones inside it.
STEP_GPUS="${GPUS_PER_NODE}"
GPU_ARGS=(--gpus-per-node="${STEP_GPUS}" --gpu-bind=closest)
STEP_ENV=()
STEP_SHARE=(--exclusive)
if [[ -n "${MATSIM_DEVICE_MASK:-}" && -n "${MATSIM_NODE_DEVICES:-}" \
      && "${STEP_GPUS}" != "${MATSIM_NODE_DEVICES}" ]]; then
  GPU_ARGS=(--gpus-per-node="${MATSIM_NODE_DEVICES}" --gpu-bind=none)
  # Other pools may hold the node's remaining GPUs (e.g. a vLLM step); the
  # placement keeps devices disjoint, so overlap instead of waiting for them.
  STEP_SHARE=(--overlap)
  STEP_ENV=(env -u HIP_VISIBLE_DEVICES "ROCR_VISIBLE_DEVICES=${MATSIM_DEVICE_MASK}")
fi
echo "[step] gpu_args=${GPU_ARGS[*]} placed_devices=${MATSIM_DEVICE_MASK:-all}"
exec srun \
  "${STEP_SHARE[@]}" \
  "${NODE_ARGS[@]}" \
  -N "${NNODES}" \
  -n "${TOTAL_RANKS}" \
  -c "${THREADS_PER_RANK}" \
  "${GPU_ARGS[@]}" \
  "${STEP_ENV[@]}" \
  "${PW_BIN}" -in "${INPUT}"
