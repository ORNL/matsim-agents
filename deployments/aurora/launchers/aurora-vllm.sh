#!/bin/bash
# Shared vLLM server lifecycle for Aurora jobs (source this file).
#
# Mirrors the qualified recipe in smoke-tests/smoke-vllm-singlenode-aurora.sh:
# the frameworks module provides vLLM (XPU), the server is launched through
# `mpiexec` (bare Python on compute nodes cannot see the PVC tiles) with the
# PMI/PALS rank variables cleared, and aurora_vllm_entrypoint.py keeps the
# model-registry subprocess off Level Zero.
#
#   aurora_vllm_env <run_dir> <tensor_parallel_size>
#   aurora_vllm_start <model_path> <served_name> <tp> <port> <max_model_len> <log> [extra args...]
#   aurora_vllm_wait <port> <served_name> <timeout_seconds>
#   aurora_vllm_stop
#
# aurora_vllm_start sets AURORA_VLLM_PID; aurora_vllm_wait fails if the server
# exits early or the exact served model never appears in /v1/models.

AURORA_VLLM_PID=""
AURORA_VLLM_ENTRYPOINT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../smoke-tests" && pwd)/aurora_vllm_entrypoint.py"

aurora_vllm_env() {
    local run_dir="$1" tp="$2" mask="" i
    export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 PYTHONFAULTHANDLER=1
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
    export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
    export RAY_USAGE_STATS_ENABLED=0 RAY_DISABLE_IMPORT_WARNING=1
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ftp_proxy FTP_PROXY all_proxy ALL_PROXY
    export no_proxy='*' NO_PROXY='*'
    # vLLM's multiproc TP workers are not MPI ranks: keep oneCCL on its
    # default internal KVS (no CCL_KVS_MODE=mpi / CCL_PROCESS_LAUNCHER=pmix).
    export FI_MR_CACHE_MONITOR=userfaultfd FI_CXI_RX_MATCH_MODE=hybrid
    export TORCH_DISTRIBUTED_USE_TORCHCOMMS=1
    # FLAT: each tile is a root device, so torch.xpu.device_count() == #tiles.
    export ZE_FLAT_DEVICE_HIERARCHY=FLAT
    for ((i = 0; i < tp; i++)); do
        mask+="${mask:+,}${i}"
    done
    export ZE_AFFINITY_MASK="${mask}"
    # Keep ONEAPI_DEVICE_SELECTOR as set by the frameworks module.
    export VLLM_CACHE_ROOT="${run_dir}/vllm-cache" TRITON_CACHE_DIR="${run_dir}/triton-cache"
    mkdir -p "${VLLM_CACHE_ROOT}" "${TRITON_CACHE_DIR}"
    # PBS TMPDIR paths exceed the Unix-socket length limit used by vLLM's ZMQ IPC.
    export TMPDIR=/tmp
    ulimit -c 0
}

aurora_vllm_start() {
    local model="$1" name="$2" tp="$3" port="$4" max_len="$5" log="$6"
    shift 6
    mpiexec -n 1 --ppn 1 \
        env -u PMI_RANK -u PMI_SIZE -u PMI_FD \
            -u PALS_APID -u PALS_RANKID -u PALS_NODEID -u PALS_SPOOL_DIR \
            -u MPI_LOCALRANKID -u MPI_LOCALNRANKS \
            -u OMPI_COMM_WORLD_RANK -u OMPI_COMM_WORLD_SIZE \
        python "${AURORA_VLLM_ENTRYPOINT}" \
            --model "${model}" \
            --served-model-name "${name}" \
            --tensor-parallel-size "${tp}" \
            --dtype bfloat16 \
            --max-model-len "${max_len}" \
            --port "${port}" \
            --host 127.0.0.1 \
            --trust-remote-code \
            --enforce-eager \
            --distributed-executor-backend mp \
            "$@" >"${log}" 2>&1 &
    AURORA_VLLM_PID=$!
}

aurora_vllm_wait() {
    local port="$1" name="$2" timeout="$3" elapsed=0 models
    while ((elapsed < timeout)); do
        if ! kill -0 "${AURORA_VLLM_PID}" 2>/dev/null; then
            echo "[vllm] server exited before readiness" >&2
            return 1
        fi
        if models=$(curl -fsS "http://127.0.0.1:${port}/v1/models" 2>/dev/null) &&
            MODELS="${models}" EXPECTED="${name}" python -c \
                'import json, os, sys; ids = [m.get("id") for m in json.loads(os.environ["MODELS"]).get("data", [])]; sys.exit(0 if os.environ["EXPECTED"] in ids else 1)'; then
            echo "[vllm] ${name} ready after ${elapsed}s"
            return 0
        fi
        sleep 10
        elapsed=$((elapsed + 10))
    done
    echo "[vllm] ${name} not ready within ${timeout}s" >&2
    return 1
}

aurora_vllm_stop() {
    if [[ -n "${AURORA_VLLM_PID}" ]]; then
        kill "${AURORA_VLLM_PID}" 2>/dev/null || true
        wait "${AURORA_VLLM_PID}" 2>/dev/null || true
        AURORA_VLLM_PID=""
    fi
}
