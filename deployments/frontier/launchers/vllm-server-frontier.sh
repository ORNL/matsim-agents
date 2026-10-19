#!/bin/bash
# Run `vllm serve` (ROCm build in the client environment) on a Frontier node.
#
# matsim-agents starts this for every managed-vllm server whose `launcher` names
# it (see deployments/frontier/resources/*.yaml) and appends the model and the
# server arguments derived from the placement; ROCR_VISIBLE_DEVICES is already
# set to the server's GCDs. Mirrors the qualified recipe of
# deployments/frontier/jobs/job-llm-check-frontier.sh.
#
#   bash vllm-server-frontier.sh <model> --served-model-name ... --port ...
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
VENV="${MATSIM_VLLM_VENV:-$REPO/.venv}"
MASK="${ROCR_VISIBLE_DEVICES:-}"

source /sw/frontier/miniforge3/23.11.0-0/etc/profile.d/conda.sh
source "${REPO}/deployments/frontier/setup/frontier-module-stack.sh"
load_frontier_rocm72_modules
source activate "${VENV}"
[[ -x "${VENV}/bin/vllm" ]] || { echo "vllm not found in ${VENV}" >&2; exit 2; }
# The module stack may reset device variables; keep the placement's GCD mask.
unset HIP_VISIBLE_DEVICES
[[ -n "${MASK}" ]] && export ROCR_VISIBLE_DEVICES="${MASK}"

export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ftp_proxy FTP_PROXY all_proxy ALL_PROXY
export no_proxy='*' NO_PROXY='*'
source "${REPO}/deployments/frontier/setup/tvm-ffi-artifact.sh"
require_tvm_ffi_artifact "${REPO}" "${VENV}"
rm -f ~/.cache/tvm-ffi/*.lock 2>/dev/null || true
JIT_TMP="/tmp/vllm-jit.${USER}.${SLURM_JOB_ID:-0}.$$"
mkdir -p "${JIT_TMP}/tmp" "${JIT_TMP}/miopen"
export TRITON_CACHE_DIR="${JIT_TMP}/triton" TORCHINDUCTOR_CACHE_DIR="${JIT_TMP}/inductor"
export VLLM_CACHE_ROOT="${JIT_TMP}/vllm" XDG_CACHE_HOME="${JIT_TMP}/xdg-cache"
export TORCH_HOME="${JIT_TMP}/torch-home" TMPDIR="${JIT_TMP}/tmp"
export MIOPEN_DISABLE_CACHE=1 MIOPEN_USER_DB_PATH="${JIT_TMP}/miopen"

echo "[vllm-server] $(date) host=$(hostname) ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-all}"
exec "${VENV}/bin/vllm" serve "$@"
