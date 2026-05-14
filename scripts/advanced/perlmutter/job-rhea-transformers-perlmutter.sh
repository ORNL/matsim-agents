#!/bin/bash
#SBATCH -A amsc001
#SBATCH -J rhea-hf-pm
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -t 02:00:00
#SBATCH --gpus-per-node=4
#SBATCH -c 32
#SBATCH -o /global/cfs/projectdirs/amsc001/cm2us/mlupopa/runs/rhea-hf-pm-%j/job-%j.out
#SBATCH -e /global/cfs/projectdirs/amsc001/cm2us/mlupopa/runs/rhea-hf-pm-%j/job-%j.out
# ---------------------------------------------------------------------------
# matsim-agents end-to-end discovery validation on NERSC Perlmutter.
#
# Validates the integrated stack:
#   • Local HuggingFace LLM (Qwen2.5-72B-Instruct via transformers, no server)
#   • HydraGNN MLFF relaxation (multidataset BEST6 fp64 ensemble)
#   • Quantum ESPRESSO pw.x GPU build (cold-vs-warm-start benchmark)
#
# Module stack is the *unified HydraGNN-aligned* one
# (cpe/24.07, PrgEnv-gnu/8.5.0, cray-mpich/8.1.30, cudatoolkit/12.9,
# gcc-native/13.2). The QE launcher swaps to PrgEnv-nvidia + NVHPC 25.5
# (which bundles CUDA 12.9 — same major.minor as the PyTorch wheel) only
# inside its own subshell; see scripts/launchers/perlmutter/run-pw-gpu-perlmutter.sh.
#
# Usage:
#   sbatch scripts/advanced/perlmutter/job-rhea-transformers-perlmutter.sh
#
# Override at submit time:
#   MATSIM_MODEL_DIR=/path/to/Qwen3-32B \
#   MATSIM_WARMSTART_FIXTURES=Si_diamond \
#     sbatch scripts/advanced/perlmutter/job-rhea-transformers-perlmutter.sh
#
# Skip phases (set to 0):
#   SKIP_LLM=1 SKIP_QE=0 sbatch ...
# ---------------------------------------------------------------------------

set -euo pipefail

# ── paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../../.." 2>/dev/null && pwd)"
[[ ! -f "${REPO}/pyproject.toml" ]] && \
  REPO=/global/cfs/projectdirs/amsc001/cm2us/mlupopa/matsim-agents
PROJ="$(dirname "${REPO}")"

VENV=$PROJ/HydraGNN/installation_DOE_supercomputers/HydraGNN-Installation-Perlmutter/hydragnn_venv
HYDRAGNN_EXAMPLE=$PROJ/HydraGNN/examples/multidataset_hpo_sc26
LOGDIR=$HYDRAGNN_EXAMPLE/multidataset_hpo-BEST6-fp64
MLP_CHECKPOINT=$HYDRAGNN_EXAMPLE/mlp_branch_weights.pt

MODEL_DIR=${MATSIM_MODEL_DIR:-$PROJ/models/Qwen2.5-72B-Instruct}
MODEL_NAME=${MATSIM_MODEL_NAME:-$(basename "$MODEL_DIR")}

QE_LAUNCHER=$REPO/scripts/launchers/perlmutter/run-pw-gpu-perlmutter.sh
QE_PSEUDO_DIR=${MATSIM_QE_PSEUDO_DIR:-$REPO/external/quantum-espresso/src/pseudo}
QE_PW_BIN=$REPO/external/quantum-espresso/install-gpu/bin/pw.x

RUN_DIR=$PROJ/runs/rhea-hf-pm-${SLURM_JOB_ID:-$$}
OUTPUT_DIR=$RUN_DIR/outputs
WARMSTART_DIR=$RUN_DIR/qe-warmstart

mkdir -p "$RUN_DIR" "$OUTPUT_DIR" "$WARMSTART_DIR"

SKIP_LLM=${SKIP_LLM:-0}
SKIP_QE=${SKIP_QE:-0}

# ── modules & venv (HydraGNN-aligned stack) ──────────────────────────────────
source "$REPO/scripts/setup/perlmutter/perlmutter-module-stack.sh"
load_perlmutter_modules_gpu

# Conda activation (Miniforge from cudatoolkit-aware module set above)
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$VENV"

# Make HydraGNN example utilities importable (inference_fused, etc.)
export PYTHONPATH=$HYDRAGNN_EXAMPLE:$PROJ/HydraGNN:${PYTHONPATH:-}

# ── runtime environment ──────────────────────────────────────────────────────
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

# Compute nodes have no outbound internet → force fully offline HF stack.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Point matsim-agents at the local model directory
export MATSIM_LLM_PROVIDER=huggingface
export MATSIM_HF_MODEL_PATH=$MODEL_DIR

# CUDA / NCCL knobs
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

# QE integration env (consumed by warmstart_benchmark + integration test)
export MATSIM_QE_LAUNCHER=$QE_LAUNCHER
export MATSIM_QE_PSEUDO_DIR=$QE_PSEUDO_DIR
export MATSIM_HYDRAGNN_LOGDIR=$LOGDIR
export MATSIM_HYDRAGNN_MLP_CKPT=$MLP_CHECKPOINT
export MATSIM_QE_MLP_DEVICE=cuda
export MATSIM_QE_TIMEOUT_SEC=${MATSIM_QE_TIMEOUT_SEC:-3600}
# Restrict QE warmstart to fixtures whose elements have UPFs in QE_PSEUDO_DIR
# (Si_r.upf is present; the heavy-element/RHEA fixtures need extra UPFs).
export MATSIM_WARMSTART_FIXTURES=${MATSIM_WARMSTART_FIXTURES:-Si_diamond}

# ── diagnostics ──────────────────────────────────────────────────────────────
echo "=========================================="
echo "[Perlmutter discovery validation]"
echo "Date:          $(date)"
echo "Job ID:        ${SLURM_JOB_ID:-N/A}"
echo "Host:          $(hostname)"
echo "Run dir:       $RUN_DIR"
echo "Repo:          $REPO"
echo "Venv:          $VENV"
echo "LLM model:     $MODEL_NAME ($MODEL_DIR)"
echo "HydraGNN log:  $LOGDIR"
echo "MLP ckpt:      $MLP_CHECKPOINT"
echo "QE launcher:   $QE_LAUNCHER"
echo "QE pw.x:       $QE_PW_BIN"
echo "QE pseudos:    $QE_PSEUDO_DIR"
echo "QE fixtures:   $MATSIM_WARMSTART_FIXTURES"
echo "=========================================="

echo "[$(date)] Python: $(which python) ($(python --version 2>&1))"
python - <<'PY'
import torch
print(f"[torch] {torch.__version__}  cuda={torch.cuda.is_available()}  devices={torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
try:
    import hydragnn  # noqa
    import matsim_agents  # noqa
    print("[imports] hydragnn + matsim_agents OK")
except Exception as e:
    print(f"[imports] FAILED: {e}")
    raise
PY

echo "[$(date)] pw.x sanity: $(ls -l $QE_PW_BIN 2>&1 | head -1)"

# ── Phase A: LLM + HydraGNN discovery (RHEA) ─────────────────────────────────
if [[ "$SKIP_LLM" == "0" ]]; then
  echo ""
  echo "============================================================"
  echo "[Phase A] LLM-driven RHEA discovery + HydraGNN MLFF relaxation"
  echo "============================================================"

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

  echo "[$(date)] Phase A complete. Artifacts in $OUTPUT_DIR"
else
  echo "[$(date)] SKIP_LLM=1 → skipping Phase A."
fi

# ── Phase B: HydraGNN warm-start vs QE pw.x cold-start ───────────────────────
if [[ "$SKIP_QE" == "0" ]]; then
  echo ""
  echo "============================================================"
  echo "[Phase B] HydraGNN warm-start vs Quantum ESPRESSO cold-start"
  echo "============================================================"
  echo "[$(date)] Running tests/integration/test_qe_warmstart.py "
  echo "          (fixtures: $MATSIM_WARMSTART_FIXTURES)"

  pushd "$REPO" >/dev/null
  python -m pytest -xvs tests/integration/test_qe_warmstart.py \
    --basetemp="$WARMSTART_DIR" \
    2>&1 | tee "$RUN_DIR/qe-warmstart.log"
  popd >/dev/null

  echo "[$(date)] Phase B complete. Artifacts in $WARMSTART_DIR"
else
  echo "[$(date)] SKIP_QE=1 → skipping Phase B."
fi

echo ""
echo "[$(date)] All requested phases complete. Run dir: $RUN_DIR"
