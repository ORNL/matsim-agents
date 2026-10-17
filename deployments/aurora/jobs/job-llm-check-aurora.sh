#!/bin/bash
#PBS -N matsim-llm-check
#PBS -l select=1
#PBS -l place=scatter
#PBS -l walltime=01:30:00
#PBS -l filesystems=home:flare
#PBS -q debug
#PBS -k doe
#PBS -j oe
# ---------------------------------------------------------------------------
# Qualify one locally stored model through a live vLLM server on Aurora.
#
# The job owns the complete server lifecycle: launch on XPU tiles, exact-model
# readiness, matsim-agents' six-stage LLM qualification, optional live
# scientific portability run, and cleanup. Mirrors the Frontier and
# Perlmutter LLM-check jobs.
#
# Optional:
#   MATSIM_MODEL_DIR           local checkpoint directory
#   MATSIM_MODEL_NAME          served identifier (default: checkpoint basename)
#   MATSIM_VLLM_TP             tensor-parallel size in PVC tiles (default: 4)
#   MATSIM_VLLM_PORT           server port (default: 8000)
#   MATSIM_VLLM_MAXLEN         context length (default: 8192)
#   MATSIM_LLM_CONCURRENCY     simultaneous qualification requests (default: 2)
#   MATSIM_VLLM_READY_TIMEOUT  server startup timeout in seconds (default: 2400)
#   MATSIM_RUN_PORTABILITY=1   run live scientific portability after qualification
#
# Submit with the site allocation supplied to qsub, not embedded here:
#   qsub -A <project> -v PROJECT_ROOT=$PWD deployments/aurora/jobs/job-llm-check-aurora.sh
# ---------------------------------------------------------------------------

set -eo pipefail  # NOTE: no -u; lmod's bash init breaks under nounset

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-${PBS_O_WORKDIR:-$PWD}/$0}")" 2>/dev/null && pwd)"
RUNTIME_ENV="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." 2>/dev/null && pwd)}/deployments/common/runtime-env.sh"
[[ -f "${RUNTIME_ENV}" ]] || { echo "ERROR: export PROJECT_ROOT before submission" >&2; exit 2; }
source "${RUNTIME_ENV}"
REPO="$(resolve_repo_root "${SCRIPT_DIR}")"
PROJ="$(dirname "${REPO}")"
VENV="${MATSIM_AURORA_VENV:-${REPO}/.venv}"
MODEL_DIR="${MATSIM_MODEL_DIR:-${PROJ}/models/Qwen2.5-14B-Instruct}"
MODEL_NAME="${MATSIM_MODEL_NAME:-$(basename "${MODEL_DIR}")}"
init_run_dirs "${PROJ}" "llm-check-aurora" "${PBS_JOBID:-$$}"

VLLM_PORT="${MATSIM_VLLM_PORT:-8000}"
VLLM_TP="${MATSIM_VLLM_TP:-4}"
VLLM_MAXLEN="${MATSIM_VLLM_MAXLEN:-8192}"
READY_TIMEOUT="${MATSIM_VLLM_READY_TIMEOUT:-2400}"
CONCURRENCY="${MATSIM_LLM_CONCURRENCY:-2}"
BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
VLLM_LOG="${RUN_DIR}/vllm-server.log"
CHECK_ROOT="${RUN_DIR}/llm-check"
CHECK_CONFIG="${RUN_DIR}/llm-check-config.json"

[[ -d "${MODEL_DIR}" ]] || { echo "ERROR: model directory not found: ${MODEL_DIR}" >&2; exit 2; }
mkdir -p "${CHECK_ROOT}"

# ── environment: frameworks provides vLLM; .venv adds matsim-agents ─────────
source "${REPO}/deployments/aurora/setup/aurora-module-stack.sh"
load_aurora_frameworks_modules
[[ -f "${VENV}/bin/activate" ]] || { echo "ERROR: client environment not found: ${VENV}" >&2; exit 2; }
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
source "${REPO}/deployments/aurora/launchers/aurora-vllm.sh"
aurora_vllm_env "${RUN_DIR}" "${VLLM_TP}"
trap aurora_vllm_stop EXIT

echo "[$(date)] Starting ${MODEL_NAME} from ${MODEL_DIR} on ${VLLM_TP} tile(s)"
aurora_vllm_start "${MODEL_DIR}" "${MODEL_NAME}" "${VLLM_TP}" "${VLLM_PORT}" \
    "${VLLM_MAXLEN}" "${VLLM_LOG}"
if ! aurora_vllm_wait "${VLLM_PORT}" "${MODEL_NAME}" "${READY_TIMEOUT}"; then
    tail -n 80 "${VLLM_LOG}" >&2 || true
    exit 1
fi
curl -fsS "${BASE_URL}/models" >"${RUN_DIR}/models.json"

cd "${REPO}"
export MATSIM_LLM_PROVIDER=vllm
export MATSIM_LLM_MODEL="${MODEL_NAME}"
export MATSIM_VLLM_BASE_URL="${BASE_URL}"
export MATSIM_VLLM_API_KEY=EMPTY

MODEL_NAME="${MODEL_NAME}" BASE_URL="${BASE_URL}" CHECK_ROOT="${CHECK_ROOT}" \
VLLM_TP="${VLLM_TP}" VLLM_MAXLEN="${VLLM_MAXLEN}" CONCURRENCY="${CONCURRENCY}" \
CHECK_CONFIG="${CHECK_CONFIG}" \
  python -c \
  'import json, os, pathlib; cfg={"provider":"vllm","model":os.environ["MODEL_NAME"],"base_url":os.environ["BASE_URL"],"api_key":"EMPTY","output_root":os.environ["CHECK_ROOT"],"temperature":0.0,"timeout_sec":120,"expected_accelerators":int(os.environ["VLLM_TP"]),"tensor_parallel_size":int(os.environ["VLLM_TP"]),"context_length":int(os.environ["VLLM_MAXLEN"]),"concurrent_requests":int(os.environ["CONCURRENCY"])}; pathlib.Path(os.environ["CHECK_CONFIG"]).write_text(json.dumps(cfg, indent=2)+"\n", encoding="utf-8")'

echo "[$(date)] Running six-stage LLM qualification"
matsim-agents llm-check "${CHECK_CONFIG}" | tee "${RUN_DIR}/llm-check-console.log"

shopt -s nullglob
CHECK_RUNS=("${CHECK_ROOT}"/*/)
shopt -u nullglob
[[ "${#CHECK_RUNS[@]}" -eq 1 ]] || {
    echo "ERROR: expected exactly one qualification run in ${CHECK_ROOT}, found ${#CHECK_RUNS[@]}" >&2
    exit 1
}
CHECK_RUN="${CHECK_RUNS[0]}"
CHECK_RUN="${CHECK_RUN}" python -c \
  'import json, os, pathlib, sys; result=json.loads((pathlib.Path(os.environ["CHECK_RUN"])/"result.json").read_text()); required={"readiness","load","generation","structured","discussion","distributed"}; ok=result.get("status")=="complete" and required.issubset({k for k,v in result.get("stages",{}).items() if v}); sys.exit(0 if ok else 1)'
ln -s "${CHECK_RUN}" "${RUN_DIR}/successful-llm-check"
echo "[$(date)] Qualification passed: ${CHECK_RUN}"

if [[ "${MATSIM_RUN_PORTABILITY:-0}" == "1" ]]; then
    echo "[$(date)] Running live scientific portability suite"
    python benchmarks/portability/run.py \
        --facility aurora \
        --suite all \
        --backend qe \
        --execute \
        --live-llm \
        --llm-check-run "${CHECK_RUN}" \
        --output "${RUN_DIR}/portability-live"
fi

echo "[$(date)] LLM deployment qualification complete"
echo "  Run root:      ${RUN_DIR}"
echo "  Qualification: ${CHECK_RUN}"
echo "  Server log:    ${VLLM_LOG}"
