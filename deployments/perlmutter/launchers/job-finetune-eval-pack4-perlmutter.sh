#!/bin/bash
#SBATCH -J fte-pack4
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.out
#SBATCH -t 06:00:00
#SBATCH -C gpu
#SBATCH -q premium
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH -c 128
#SBATCH --exclusive
# ---------------------------------------------------------------------------
# matsim-agents: pack up to FOUR finetune-eval campaigns onto ONE full GPU node
# (4x A100), one campaign per GPU, running concurrently. This turns the many
# single-GPU `-q shared` campaigns into a handful of full-node `-q premium`
# (or `-q regular`) jobs -- 2x charge but much higher scheduling priority and no
# wasted GPUs (all 4 are used).
#
# Each campaign REUSES the tested single-campaign launcher
# (job-finetune-eval-perlmutter.sh) unchanged: this script just pins it to a
# distinct GPU (CUDA_VISIBLE_DEVICES) and hands it per-campaign env overrides,
# so caches and output dirs stay identical to the shared-queue runs. Each
# campaign selects its own model, so one node can mix backends.
#
# Inputs (env)
# ------------
#   EVALUATION_CONFIG  default evaluation YAML for campaigns that set none
#   SPECS     up to 4 campaigns separated by ';'. Each campaign is a
#             space-separated list of VAR=VAL overrides passed to the single
#             launcher: CASE and MLIP_CONFIG, plus optional EVALUATION_CONFIG,
#             OUT_DIR, ... The matrix inputs under
#             $RUNS_ROOT/finetune-eval/<variant>/<case>/ (written by
#             finetune_eval_matrix.py) can be used directly. Example:
#
#     R=$RUNS_ROOT/finetune-eval
#     EVALUATION_CONFIG=$PWD/examples/evaluation/random-frame-interpolation.yaml SPECS="\
#       CASE=lifepo4-al-001 MLIP_CONFIG=$PWD/examples/mlip/mace-mp-medium.yaml; \
#       CASE=hea-bcc-al-001 MLIP_CONFIG=$R/mace-lora/hea-bcc-al-001/mlip.yaml \
#         EVALUATION_CONFIG=$R/mace-lora/hea-bcc-al-001/evaluation.yaml \
#         OUT_DIR=$R/mace-lora/hea-bcc-al-001" \
#       sbatch deployments/perlmutter/launchers/job-finetune-eval-pack4-perlmutter.sh
#
# Switch to the (free) regular queue with:  sbatch -q regular ...
# Fewer than 4 specs is fine (only the GPUs needed are used).
# ---------------------------------------------------------------------------
set -uo pipefail

# Slurm copies the batch script to its spool dir, so BASH_SOURCE points there;
# resolve the repo (and the sibling launcher) from PROJECT_ROOT / a known path.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
REPO_DEFAULT="$(cd "${SCRIPT_DIR}/../../.." 2>/dev/null && pwd)"
REPO="${PROJECT_ROOT:-${REPO_DEFAULT}}"
[[ ! -f "${REPO}/pyproject.toml" ]] && \
  REPO=${PROJECT_ROOT:?export PROJECT_ROOT}
SINGLE="${REPO}/deployments/perlmutter/launchers/job-finetune-eval-perlmutter.sh"
[[ ! -f "${SINGLE}" ]] && { echo "ERROR: single-campaign launcher not found: ${SINGLE}" >&2; exit 2; }

SPECS="${SPECS:?set SPECS='CASE=... MLIP_CONFIG=...; ...' (up to 4, ';'-separated)}"
if [[ -n "${EVALUATION_CONFIG:-}" ]]; then
  [[ -r "${EVALUATION_CONFIG}" ]] || { echo "ERROR: evaluation config not readable: ${EVALUATION_CONFIG}" >&2; exit 2; }
  export EVALUATION_CONFIG
fi
LOGDIR="${SLURM_SUBMIT_DIR:-$PWD}"

# Split SPECS on ';' into an array (trim surrounding whitespace per entry).
IFS=';' read -ra CAMPAIGNS <<<"${SPECS}"
n=0
for c in "${CAMPAIGNS[@]}"; do
  c="$(echo "${c}" | xargs)"      # trim
  [[ -z "${c}" ]] && continue
  CLEAN[$n]="${c}"
  n=$((n+1))
done
(( n == 0 )) && { echo "ERROR: no campaigns parsed from SPECS" >&2; exit 2; }
(( n > 4 ))  && { echo "ERROR: at most 4 campaigns per node (got ${n})" >&2; exit 2; }

echo "=========================================="
echo "[Perlmutter FT+eval PACK4]"
echo "Date:    $(date)"
echo "Job ID:  ${SLURM_JOB_ID:-N/A}   QOS: ${SLURM_JOB_QOS:-N/A}"
echo "Node:    $(hostname)   GPUs: ${SLURM_GPUS_PER_NODE:-?}"
echo "Campaigns (${n}):"
for i in $(seq 0 $((n-1))); do echo "  gpu${i}: ${CLEAN[$i]}"; done
echo "=========================================="

declare -a PIDS GPUOF
for i in $(seq 0 $((n-1))); do
  gpu_log="${LOGDIR}/${SLURM_JOB_NAME:-fte-pack4}-${SLURM_JOB_ID:-$$}-gpu${i}.out"
  echo "[launch] gpu${i} -> ${gpu_log}"
  # Reuse the single-campaign launcher, pinned to one GPU with per-spec env.
  CUDA_VISIBLE_DEVICES="${i}" CUDA_DEVICE_ORDER=PCI_BUS_ID \
    env ${CLEAN[$i]} bash "${SINGLE}" >"${gpu_log}" 2>&1 &
  PIDS[$i]=$!
  GPUOF[$i]="${CLEAN[$i]}"
done

rc_total=0
for i in $(seq 0 $((n-1))); do
  if wait "${PIDS[$i]}"; then
    echo "[done] gpu${i} OK   (${GPUOF[$i]})"
  else
    rc=$?
    echo "[done] gpu${i} FAILED rc=${rc}   (${GPUOF[$i]})" >&2
    rc_total=1
  fi
done

echo "[$(date)] PACK4 complete (${n} campaigns, overall rc=${rc_total})."
exit "${rc_total}"
