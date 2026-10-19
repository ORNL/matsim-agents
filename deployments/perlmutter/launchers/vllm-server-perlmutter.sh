#!/bin/bash
# Run `vllm serve` from the isolated vLLM environment on a Perlmutter GPU node.
#
# matsim-agents starts this for every managed-vllm server whose `launcher` names
# it (see deployments/perlmutter/resources/*.yaml) and appends the model and the
# server arguments derived from the placement; the device mask is already set.
#
#   bash vllm-server-perlmutter.sh <model> --served-model-name ... --port ...
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
VLLM_VENV="${MATSIM_VLLM_VENV:-$REPO/venv_vllm}"
[[ -x "$VLLM_VENV/bin/vllm" ]] || { echo "vllm not found in $VLLM_VENV" >&2; exit 2; }

# The driver runs from the client environment; keep its Python settings out of vLLM.
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV CONDA_PREFIX
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_DO_NOT_TRACK=1 VLLM_NO_USAGE_STATS=1
export PYTHONNOUSERSITE=1 VLLM_USE_FLASHINFER_SAMPLER=0
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ftp_proxy FTP_PROXY all_proxy ALL_PROXY
export no_proxy='*' NO_PROXY='*'
# JIT caches on node-local storage: CFS does not support the locking they need.
JIT_TMP="${TMPDIR:-/tmp}/vllm-jit.${USER}.${SLURM_JOB_ID:-0}.$$"
mkdir -p "$JIT_TMP"
export FLASHINFER_WORKSPACE_BASE="$JIT_TMP"
export TRITON_CACHE_DIR="$JIT_TMP/triton"
export TORCHINDUCTOR_CACHE_DIR="$JIT_TMP/inductor"
export VLLM_CACHE_ROOT="$JIT_TMP/vllm"
export XDG_CACHE_HOME="$JIT_TMP/xdg-cache" HF_HOME="$JIT_TMP/hf-home" TORCH_HOME="$JIT_TMP/torch-home"
export TMPDIR="$JIT_TMP/tmp"
mkdir -p "$TMPDIR"
# FlashInfer JIT builds need the Python headers of the client environment.
PY_HDR="$REPO/.venv/include/python3.11"
export CPATH="${PY_HDR}:${CPATH:-}"
export C_INCLUDE_PATH="${PY_HDR}:${C_INCLUDE_PATH:-}"

echo "[vllm-server] $(date) host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all}"
exec "$VLLM_VENV/bin/vllm" serve "$@"
