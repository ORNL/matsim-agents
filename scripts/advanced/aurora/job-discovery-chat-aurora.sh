#!/bin/bash
#PBS -A CM2US
#PBS -N discovery-chat
#PBS -l select=1
#PBS -l place=scatter
#PBS -l walltime=02:00:00
#PBS -l filesystems=home:flare
#PBS -q debug
#PBS -k doe
#PBS -j oe
# ---------------------------------------------------------------------------
# matsim-agents: end-to-end discovery-chat run on ALCF Aurora.
#
# Mirrors scripts/advanced/{frontier,perlmutter}/job-discovery-chat-*.sh:
#   • Local HuggingFace LLM (Qwen2.5-72B-Instruct via transformers, no server)
#   • HydraGNN MLFF relaxation (multidataset BEST6 fp64 ensemble)
#   • Same RHEA-style query, same flags (--n-orderings 2, --min-atoms 64,
#     --auto-confirm, --maxiter 500, --fmax 0.02)
#
# Submit:
#   qsub scripts/advanced/aurora/job-discovery-chat-aurora.sh
#
# Override at submit time:
#   qsub -v MATSIM_MODEL_DIR=/path/to/Qwen3-32B \
#        scripts/advanced/aurora/job-discovery-chat-aurora.sh
# ---------------------------------------------------------------------------

set -eo pipefail  # NOTE: no -u; lmod's bash init breaks under nounset

# ── paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-${PBS_O_WORKDIR:-$PWD}/$0}")" 2>/dev/null && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../../.." 2>/dev/null && pwd)"
[[ ! -f "${REPO}/pyproject.toml" ]] && \
  REPO=/lus/flare/projects/CM2US/mlupopa/matsim-agents
PROJ="$(dirname "${REPO}")"

VENV="${MATSIM_AURORA_VENV:-${PROJ}/HydraGNN/installation_DOE_supercomputers/HydraGNN-Installation-Aurora/hydragnn_venv}"
HYDRAGNN_EXAMPLE=$PROJ/HydraGNN/examples/multidataset_hpo_sc26
LOGDIR=${MATSIM_HYDRAGNN_LOGDIR:-$HYDRAGNN_EXAMPLE/multidataset_hpo-BEST6-fp64}
MLP_CHECKPOINT=${MATSIM_HYDRAGNN_MLP_CKPT:-$HYDRAGNN_EXAMPLE/mlp_branch_weights.pt}

MODEL_DIR=${MATSIM_MODEL_DIR:-$PROJ/models/Mistral-Small-24B-Instruct-2501}
MODEL_NAME=${MATSIM_MODEL_NAME:-$(basename "$MODEL_DIR")}

JOBID="${PBS_JOBID:-$$}"
RUN_DIR=$PROJ/runs/discovery-chat-aurora-${JOBID}
OUTPUT_DIR=$RUN_DIR/outputs
mkdir -p "$RUN_DIR" "$OUTPUT_DIR"

# ── modules & venv ───────────────────────────────────────────────────────────
if command -v module >/dev/null 2>&1; then
  module reset
  module load frameworks
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

export PYTHONPATH=$HYDRAGNN_EXAMPLE:$PROJ/HydraGNN:${PYTHONPATH:-}
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

# Compute nodes have no outbound internet → force fully offline HF stack.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Point matsim-agents at the local model directory
export MATSIM_LLM_PROVIDER=huggingface
export MATSIM_HF_MODEL_PATH=$MODEL_DIR

# Intel PVC runtime defaults
export ZE_AFFINITY_MASK="${ZE_AFFINITY_MASK:-0}"
export ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:gpu}"

# ── diagnostics ──────────────────────────────────────────────────────────────
echo "=========================================="
echo "[Aurora discovery-chat]"
echo "Date:          $(date)"
echo "Job ID:        ${JOBID}"
echo "Host:          $(hostname)"
echo "Run dir:       $RUN_DIR"
echo "Repo:          $REPO"
echo "Venv:          $VENV"
echo "LLM model:     $MODEL_NAME ($MODEL_DIR)"
echo "HydraGNN log:  $LOGDIR"
echo "MLP ckpt:      $MLP_CHECKPOINT"
echo "=========================================="

echo "[$(date)] Python: $(which python) ($(python --version 2>&1))"
python - <<'PY'
import torch
print(f"[torch] {torch.__version__}  xpu={getattr(torch, 'xpu', None) is not None}")
try:
    import hydragnn  # noqa
    import matsim_agents  # noqa
    print("[imports] hydragnn + matsim_agents OK")
except Exception as e:
    print(f"[imports] FAILED: {e}")
    raise
PY

# ── RHEA query (identical to Frontier / Perlmutter) ─────────────────────────
QUERY="Propose 4 to 5 refractory high-entropy alloy compositions using elements \
from Mo, Nb, Ta, W, V, Cr, Hf, Zr, Ti that are known for combined \
high-temperature resistance and mechanical strength. For each composition \
specify the relevant crystal phases (e.g. BCC, B2, HCP) and explain the \
physical justification. Then relax each proposed structure using the MLFF \
and report the final energies and which phases are most stable."

echo "[$(date)] Submitting RHEA query to matsim-agents (HuggingFace provider) ..."
echo "$QUERY" | matsim-agents chat \
    --logdir          "$LOGDIR" \
    --mlp-checkpoint  "$MLP_CHECKPOINT" \
    --output-dir      "$OUTPUT_DIR" \
    --llm-provider    huggingface \
    --llm-model       "$MODEL_DIR" \
    --ase-structure-optimizer FIRE \
    --maxiter         500 \
    --fmax            0.02 \
    --min-atoms       64 \
    --n-orderings     2 \
    --auto-confirm \
    2>&1 | tee "$RUN_DIR/matsim-agents.log"

echo "[$(date)] Discovery-chat finished. Artifacts in $OUTPUT_DIR"
