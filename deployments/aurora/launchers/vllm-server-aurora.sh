#!/bin/bash
# Run vLLM (XPU, frameworks module) on the Aurora node of the calling process.
#
# matsim-agents starts this for every managed-vllm server whose `launcher` names
# it (see deployments/aurora/resources/*.yaml) and appends the model and the
# server arguments derived from the placement; ZE_AFFINITY_MASK is already set to
# the server's tiles. Mirrors aurora-vllm.sh: the server runs through `mpiexec`
# (bare Python on compute nodes cannot see the PVC tiles) with PMI/PALS rank
# variables cleared, and aurora_vllm_entrypoint.py keeps the model-registry
# subprocess off Level Zero. When matsim-agents already started this through
# mpiexec on another node (MATSIM_SERVER_IN_STEP=1), it runs the server directly.
#
#   bash vllm-server-aurora.sh <model> --served-model-name ... --port ...
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRYPOINT="$(cd "${SCRIPT_DIR}/../smoke-tests" && pwd)/aurora_vllm_entrypoint.py"
SMOKE_TEST_DIR="$(cd "${SCRIPT_DIR}/../smoke-tests" && pwd)"
[[ $# -ge 1 ]] || { echo "usage: $0 <model> [vllm serve arguments]" >&2; exit 2; }
MODEL="$1"
shift
: "${ZE_AFFINITY_MASK:?matsim-agents sets ZE_AFFINITY_MASK from the placement}"

export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 PYTHONFAULTHANDLER=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export RAY_USAGE_STATS_ENABLED=0 RAY_DISABLE_IMPORT_WARNING=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ftp_proxy FTP_PROXY all_proxy ALL_PROXY
export no_proxy='*' NO_PROXY='*'
# vLLM's multiproc TP workers are not MPI ranks: keep oneCCL on its default KVS.
export FI_MR_CACHE_MONITOR=userfaultfd FI_CXI_RX_MATCH_MODE=hybrid
export TORCH_DISTRIBUTED_USE_TORCHCOMMS=1
export PYTHONPATH="${SMOKE_TEST_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
# PBS TMPDIR paths exceed the Unix-socket length limit used by vLLM's ZMQ IPC.
export TMPDIR=/tmp
CACHE="/tmp/vllm-cache.${USER}.${PBS_JOBID:-0}.$$"
export VLLM_CACHE_ROOT="${CACHE}/vllm" TRITON_CACHE_DIR="${CACHE}/triton"
mkdir -p "${VLLM_CACHE_ROOT}" "${TRITON_CACHE_DIR}"
ulimit -c 0

echo "[vllm-server] $(date) host=$(hostname) ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK}"
# The API server must not see PMI/PALS rank variables, or vLLM treats it as an MPI rank.
SERVE=(
    env -u PMI_RANK -u PMI_SIZE -u PMI_FD
        -u PALS_APID -u PALS_RANKID -u PALS_NODEID -u PALS_SPOOL_DIR
        -u MPI_LOCALRANKID -u MPI_LOCALNRANKS
        -u OMPI_COMM_WORLD_RANK -u OMPI_COMM_WORLD_SIZE
    python "${ENTRYPOINT}"
        --model "${MODEL}"
        --dtype bfloat16
        --distributed-executor-backend mp
        "$@"
)
if [[ "${MATSIM_SERVER_IN_STEP:-0}" == "1" ]]; then
    # matsim-agents already started this through mpiexec on the server's node.
    exec "${SERVE[@]}"
fi
# Pin the rank to this node: in a multi-node job mpiexec would otherwise use the
# first host of the PBS nodefile, which need not be the node the plan chose.
exec mpiexec --hosts "$(hostname)" -n 1 --ppn 1 "${SERVE[@]}"
