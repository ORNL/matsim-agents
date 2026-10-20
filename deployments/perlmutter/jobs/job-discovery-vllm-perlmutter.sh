#!/bin/bash
#SBATCH -J discovery-vllm
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 01:30:00
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --gpus-per-node=4
#SBATCH -c 32
# ---------------------------------------------------------------------------
# matsim-agents: discovery run on NERSC Perlmutter with a vLLM server that
# matsim-agents starts, waits for, and stops itself.
#
# Placement comes from a resource layout (default
# deployments/perlmutter/resources/discovery-vllm.yaml):
#   • vLLM server (Qwen2.5-14B)  : GPU 0    [venv_vllm, via vllm-server-perlmutter.sh]
#   • MLIP relaxations           : GPUs 1-3 [.venv, one worker process per GPU]
#   • LLM proposer + critic both use the plan's "proposer" server.
#
# Usage:
#   sbatch deployments/perlmutter/jobs/job-discovery-vllm-perlmutter.sh
#
# Another model or layout at submission:
#   MATSIM_RESOURCES=/path/to/layout.yaml MODEL_ROOT=$PROJ/models \
#     sbatch deployments/perlmutter/jobs/job-discovery-vllm-perlmutter.sh
#   matsim-agents resources plan <layout.yaml>   # preview the placement first
# ---------------------------------------------------------------------------

set -euo pipefail

# ── paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
# Slurm copies the submitted script into a spool dir before executing it, so
# BASH_SOURCE[0] does NOT point at this file's real location under sbatch --
# fall back to the known repo-absolute path when the relative lookup misses.
RUNTIME_ENV="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." 2>/dev/null && pwd)}/deployments/common/runtime-env.sh"
[[ -f "${RUNTIME_ENV}" ]] || { echo "ERROR: export PROJECT_ROOT before submission" >&2; exit 2; }
source "${RUNTIME_ENV}"
REPO="$(resolve_repo_root "${SCRIPT_DIR}")"
PROJ="$(dirname "${REPO}")"
VENV=$REPO/.venv          # discovery client + HydraGNN MLFF
HYDRAGNN_EXAMPLE=$PROJ/HydraGNN/examples/multidataset_hpo_sc26
LOGDIR=${MATSIM_HYDRAGNN_LOGDIR:-$HYDRAGNN_EXAMPLE/multidataset_hpo-BEST6-fp64}
HYDRAGNN_BRANCH_MLP_CHECKPOINT=${HYDRAGNN_BRANCH_MLP_CHECKPOINT:-$HYDRAGNN_EXAMPLE/mlp_branch_weights.pt}
init_run_dirs "$PROJ" "discovery-vllm" "${SLURM_JOB_ID:-$$}"

# ── placement: vLLM server and MLIP workers come from the resource layout ────
export MODEL_ROOT="${MODEL_ROOT:-$PROJ/models}"
export PROJECT_ROOT="$REPO"
RESOURCES="${MATSIM_RESOURCES:-$REPO/deployments/perlmutter/resources/discovery-vllm.yaml}"

source "$REPO/deployments/perlmutter/setup/perlmutter-module-stack.sh"
load_perlmutter_modules_gpu
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$VENV"

export PYTHONPATH=$HYDRAGNN_EXAMPLE:$PROJ/HydraGNN:${PYTHONPATH:-}
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ── diagnostics ──────────────────────────────────────────────────────────────
echo "[$(date)] Client python: $(which python) ($(python --version 2>&1))"
echo "[$(date)] Resources:     $RESOURCES"
matsim-agents resources plan "$RESOURCES"
python - <<'PY'
import torch
print(f"[torch] {torch.__version__}  cuda={torch.cuda.is_available()}  devices={torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
PY

# ── 4) discovery query ───────────────────────────────────────────────────────
QUERY="${MATSIM_DISCOVERY_QUERY:-}"
if [[ -z "$QUERY" ]]; then
  QUERY="$(cat <<'EOF'
Propose 3 to 5 candidate inorganic materials for high-temperature structural applications.
For each candidate, provide the formula, likely crystal family, and a brief physics-based
justification. Then relax the proposed structures with the MLFF and summarize relative
stability from final energies and residual forces.
EOF
)"
fi

# ── active-learning handoff mode ─────────────────────────────────────────────
# MATSIM_AL_MODE = dry-run (default; plan/report only) | run (execute AL loop).
# When MATSIM_AL_MODE=run, supply MATSIM_AL_CONFIG = path to a base AL YAML
# (e.g. examples/paper_cases/al_hea_bcc.yaml). Discovery overrides its
# seed_source to the detected formula on handoff.
AL_MODE=${MATSIM_AL_MODE:-dry-run}
AL_ARGS=()
if [[ "$AL_MODE" == "run" ]]; then
  AL_ARGS+=( --al-run )
  if [[ -n "${MATSIM_AL_CONFIG:-}" ]]; then
    AL_CFG="${MATSIM_AL_CONFIG}"
    [[ "$AL_CFG" = /* ]] || AL_CFG="$REPO/$AL_CFG"
    [[ -f "$AL_CFG" ]] || { echo "[ERROR] MATSIM_AL_CONFIG not found: $AL_CFG" >&2; exit 1; }
    AL_ARGS+=( --al-config "$AL_CFG" )
  fi
else
  AL_ARGS+=( --al-dry-run )
fi

echo "[$(date)] Submitting discovery query to matsim-agents (managed vLLM, AL mode=$AL_MODE) ..."
echo "$QUERY" | matsim-agents chat \
    --logdir          "$LOGDIR" \
    --hydragnn-branch-mlp-checkpoint "$HYDRAGNN_BRANCH_MLP_CHECKPOINT" \
    --output-dir      "$OUTPUT_DIR" \
    --resources       "$RESOURCES" \
    --llm-peer-review \
    --peer-review-rounds "${MATSIM_PEER_REVIEW_ROUNDS:-2}" \
    --ase-structure-optimizer FIRE \
    --maxiter         500 \
    --fmax            0.02 \
    "${AL_ARGS[@]}" \
    --auto-confirm \
    2>&1 | tee "$RUN_DIR/matsim-agents.log"

echo "[$(date)] matsim-agents finished. Artifacts in $OUTPUT_DIR"
