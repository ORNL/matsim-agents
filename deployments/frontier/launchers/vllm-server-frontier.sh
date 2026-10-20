#!/bin/bash
# Run `vllm serve` (ROCm build in the client environment) on a Frontier node.
#
# matsim-agents starts this for every managed-vllm server whose `launcher` names
# it (see deployments/frontier/resources/*.yaml) and appends the model and the
# server arguments derived from the placement; ROCR_VISIBLE_DEVICES holds the
# server's GCDs. As in the qualified recipes
# (smoke-tests/smoke-vllm-singlenode-frontier.sh,
# jobs/job-discovery-chat-vllm-frontier.sh), vLLM runs inside an `srun` step:
# the step takes every GCD of the node (Slurm would otherwise choose which
# GCDs it gets) and HIP_VISIBLE_DEVICES exposes only the placed ones.
#
#   bash vllm-server-frontier.sh <model> --served-model-name ... --port ...
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
VENV="${MATSIM_VLLM_VENV:-$REPO/.venv}"
PLACED="${ROCR_VISIBLE_DEVICES:?matsim-agents sets ROCR_VISIBLE_DEVICES from the placement}"

source /sw/frontier/miniforge3/23.11.0-0/etc/profile.d/conda.sh
source "${REPO}/deployments/frontier/setup/frontier-module-stack.sh"
load_frontier_rocm72_modules
source activate "${VENV}"
[[ -x "${VENV}/bin/vllm" ]] || { echo "vllm not found in ${VENV}" >&2; exit 2; }

export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ftp_proxy FTP_PROXY all_proxy ALL_PROXY
export no_proxy='*' NO_PROXY='*'
# ROCm settings from the qualified Frontier vLLM jobs.
export VLLM_CUDART_SO_PATH=/opt/rocm-7.2.0/lib/libamdhip64.so
export VLLM_NCCL_SO_PATH=/opt/rocm-7.2.0/lib/librccl.so.1
export LD_LIBRARY_PATH="${VENV}/lib/python3.11/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export HSA_NO_SCRATCH_RECLAIM=1 PYTORCH_ROCM_ARCH=gfx90a ROCM_ARCH=gfx90a
export RCCL_UNROLL_FACTOR=0 RCCL_P2P_BATCH_ENABLE=0 RCCL_P2P_BATCH_THRESHOLD=0
source "${REPO}/deployments/frontier/setup/tvm-ffi-artifact.sh"
require_tvm_ffi_artifact "${REPO}" "${VENV}"
rm -f ~/.cache/tvm-ffi/*.lock 2>/dev/null || true
JIT_TMP="/tmp/vllm-jit.${USER}.${SLURM_JOB_ID:-0}.$$"
mkdir -p "${JIT_TMP}/tmp" "${JIT_TMP}/miopen"
export TRITON_CACHE_DIR="${JIT_TMP}/triton" TORCHINDUCTOR_CACHE_DIR="${JIT_TMP}/inductor"
export VLLM_CACHE_ROOT="${JIT_TMP}/vllm" XDG_CACHE_HOME="${JIT_TMP}/xdg-cache"
export TORCH_HOME="${JIT_TMP}/torch-home" TMPDIR="${JIT_TMP}/tmp"
export MIOPEN_DISABLE_CACHE=1 MIOPEN_USER_DB_PATH="${JIT_TMP}/miopen"

echo "[vllm-server] $(date) host=$(hostname) placed GCDs=${PLACED}"
# Ray (used by vLLM on AMD GPUs) requires HIP_VISIBLE_DEVICES, not ROCR_VISIBLE_DEVICES.
SERVE=(env -u ROCR_VISIBLE_DEVICES "HIP_VISIBLE_DEVICES=${PLACED}" "${VENV}/bin/vllm" serve "$@")
if [[ "${MATSIM_SERVER_IN_STEP:-0}" == "1" ]]; then
  # matsim-agents already started this inside an srun step on the server's node.
  exec "${SERVE[@]}"
fi
NODE_GCDS="${SLURM_GPUS_ON_NODE:-8}"
NODE_GCDS="${NODE_GCDS##*:}"
CPUS=$(( 7 * $(tr ',' '\n' <<<"${PLACED}" | grep -c .) ))
exec srun -N1 -n1 --overlap -w "$(hostname)" -c "${CPUS}" \
  --gpus-per-node="${NODE_GCDS}" --gpu-bind=none \
  "${SERVE[@]}"
