#!/bin/bash
#PBS -A CM2US
#PBS -N matsim-serve-multinode
#PBS -l select=2
#PBS -l place=scatter
#PBS -l walltime=02:00:00
#PBS -l filesystems=home:flare
#PBS -q debug-scaling
#PBS -k doe
#PBS -j oe
# ---------------------------------------------------------------------------
# Multi-node HuggingFace tensor-parallel serve on ALCF Aurora (Intel PVC).
#
# Aurora's `vLLM` story on PVC is not production-ready; this script is the
# Aurora analog of `scripts/advanced/frontier/job-serve-multinode-frontier.sh`,
# but uses HuggingFace transformers' built-in TP planner (`tp_plan="auto"`)
# layered on `torch.distributed` with the **oneCCL** backend.
#
# Layout (default):
#   • 1 rank per PVC tile (6 GPUs × 2 tiles = 12 ranks/node).
#   • `mpiexec --ppn 12 --cpu-bind=list:...` for CPU pinning.
#   • A small per-rank launcher script pins each rank to its tile via
#     `ZE_AFFINITY_MASK=$PALS_LOCAL_RANKID` and runs `_mpi_xpu_loader.py`.
#
# Required env:
#   MATSIM_MODEL_DIR   — absolute path to the local model directory
#                        (default: $PROJ/models/Qwen2.5-72B-Instruct)
#
# Optional env:
#   MATSIM_MN_PROMPT          — prompt string (default: 2+2 sanity check)
#   MATSIM_MN_MAX_NEW_TOKENS  — generation length (default: 128)
#   PPN                       — ranks per node (default: 12 = one per tile)
#
# Submit (2 nodes, default model):
#   qsub scripts/advanced/aurora/job-serve-multinode-aurora.sh
#
# Submit (4 nodes, override model):
#   qsub -l select=4 \
#        -v MATSIM_MODEL_DIR=$PROJ/models/Mixtral-8x22B-Instruct-v0.1 \
#        scripts/advanced/aurora/job-serve-multinode-aurora.sh
# ---------------------------------------------------------------------------

set -eo pipefail  # NOTE: no -u; lmod's bash init breaks under nounset

# ── repo / paths ────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-${PBS_O_WORKDIR:-$PWD}/$0}")" 2>/dev/null && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../../.." 2>/dev/null && pwd)"
[[ ! -f "${REPO}/pyproject.toml" ]] && \
  REPO=/lus/flare/projects/CM2US/mlupopa/matsim-agents
PROJ="$(dirname "${REPO}")"

VENV="${MATSIM_AURORA_VENV:-${PROJ}/HydraGNN/installation_DOE_supercomputers/HydraGNN-Installation-Aurora/hydragnn_venv}"
LOADER="${SCRIPT_DIR}/_mpi_xpu_loader.py"

MODEL_DIR="${MATSIM_MODEL_DIR:-${PROJ}/models/Mistral-Small-24B-Instruct-2501}"
MODEL_NAME="${MATSIM_MODEL_NAME:-$(basename "${MODEL_DIR}")}"

JOBID="${PBS_JOBID:-$$}"
RUN_DIR="${PROJ}/runs/serve-multinode-aurora-${JOBID}"
mkdir -p "${RUN_DIR}"

# ── input validation ────────────────────────────────────────────────────────
if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "ERROR: MATSIM_MODEL_DIR does not exist: ${MODEL_DIR}" >&2
  echo "       (Set MATSIM_MODEL_DIR via -v at qsub time.)" >&2
  exit 2
fi
if [[ ! -f "${LOADER}" ]]; then
  echo "ERROR: TP loader missing: ${LOADER}" >&2
  exit 2
fi

# ── modules + venv ──────────────────────────────────────────────────────────
if command -v module >/dev/null 2>&1; then
  module reset
  module load frameworks
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

# ── job geometry ────────────────────────────────────────────────────────────
NNODES="$(wc -l < "${PBS_NODEFILE}")"
PPN="${PPN:-12}"               # ranks per node = PVC tiles per node
NTOTRANKS=$(( NNODES * PPN ))

# 12-rank CPU binding used by HydraGNN on Aurora — one tile per rank.
CPU_BIND="--cpu-bind=list:1-8:9-16:17-24:25-32:33-40:41-48:53-60:61-68:69-76:77-84:85-92:93-100"

# ── runtime env (HF + torch-distributed/CCL on XPU) ─────────────────────────
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

# Compute nodes have no outbound internet → keep HF stack fully offline.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# matsim-agents config (rank 0 only consults these; harmless on workers)
export MATSIM_LLM_PROVIDER=huggingface
export MATSIM_HF_MODEL_PATH="${MODEL_DIR}"

# Aurora oneCCL / fabric tunings copied from HydraGNN multi-node training.
export CCL_KVS_MODE=mpi
export CCL_KVS_CONNECTION_TIMEOUT=900
export FI_MR_CACHE_MONITOR=userfaultfd
export FI_CXI_RX_MATCH_MODE=hybrid
export TORCH_DISTRIBUTED_USE_TORCHCOMMS=1

# torch.distributed env-init rendezvous (used by ``init_method="env://"``).
HEAD=$(head -n 1 "${PBS_NODEFILE}")
export MASTER_ADDR="${HEAD}"
export MASTER_PORT="${MASTER_PORT:-29500}"

# ── per-rank launcher: pin each rank to its PVC tile ────────────────────────
# Aurora exposes 6 PVC GPUs × 2 tiles via ZE_AFFINITY_MASK in the form
# "<gpu>.<tile>".  PALS_LOCAL_RANKID runs 0..11; map directly.
TILE_LAUNCHER="${RUN_DIR}/launch_one_rank.sh"
cat > "${TILE_LAUNCHER}" <<'EOF'
#!/bin/bash
set -euo pipefail
TILE_TABLE=(0.0 0.1 1.0 1.1 2.0 2.1 3.0 3.1 4.0 4.1 5.0 5.1)
LR="${PALS_LOCAL_RANKID:-${PMI_LOCAL_RANK:-0}}"
export ZE_AFFINITY_MASK="${TILE_TABLE[$LR]}"
export ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:gpu}"
exec python -u "$@"
EOF
chmod +x "${TILE_LAUNCHER}"

# ── diagnostics ─────────────────────────────────────────────────────────────
echo "=========================================="
echo "[Aurora multi-node HF/TP serve]"
echo "Date:           $(date)"
echo "Job ID:         ${JOBID}"
echo "Nodes:          ${NNODES}  (head=${HEAD})"
echo "Ranks/node:     ${PPN}"
echo "World size:     ${NTOTRANKS}"
echo "Master:         ${MASTER_ADDR}:${MASTER_PORT}"
echo "Model:          ${MODEL_NAME}"
echo "Model dir:      ${MODEL_DIR}"
echo "Loader:         ${LOADER}"
echo "Tile launcher:  ${TILE_LAUNCHER}"
echo "Run dir:        ${RUN_DIR}"
echo "=========================================="

cat "${PBS_NODEFILE}" | sed 's/^/  node: /'

# ── launch ──────────────────────────────────────────────────────────────────
cd "${REPO}"

set -x
mpiexec -n "${NTOTRANKS}" --ppn "${PPN}" \
        ${CPU_BIND} \
        --hostfile "${PBS_NODEFILE}" \
        "${TILE_LAUNCHER}" "${LOADER}" \
        2>&1 | tee "${RUN_DIR}/serve-multinode.log"
status=${PIPESTATUS[0]}
set +x

echo "[$(date)] mpiexec exit status: ${status}"
exit "${status}"
