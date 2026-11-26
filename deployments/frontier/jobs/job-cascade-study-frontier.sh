#!/bin/bash
#SBATCH -J cascade-study
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -N 1
#SBATCH -p batch
#SBATCH --gpus-per-node=8
# ---------------------------------------------------------------------------
# Verification cascade study with MACE-MP-0 (scripts/cascade/run_cascade_study.py), one shard per
# GCD. The potential is unqualified; results are measurements of that model, not cascade statistics.
#
# Submit: CASCADE_OUT=<dir> CASCADE_ARGS="--study threshold --a 2.855 ..." \
#   sbatch -A <ACCOUNT> -t HH:MM:SS [-q debug|normal] deployments/frontier/jobs/job-cascade-study-frontier.sh
# ---------------------------------------------------------------------------

set -euo pipefail

REPO="${PROJECT_ROOT:-$(d="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." 2>/dev/null && pwd)"; [[ -f "$d/pyproject.toml" ]] && echo "$d" || echo "${SLURM_SUBMIT_DIR:?}")}"
OUT="${CASCADE_OUT:?set CASCADE_OUT}"
ARGS="${CASCADE_ARGS:?set CASCADE_ARGS}"
SHARDS="${CASCADE_SHARDS:-8}"

source /sw/frontier/miniforge3/23.11.0-0/etc/profile.d/conda.sh
source "$REPO/deployments/frontier/setup/frontier-module-stack.sh"
load_frontier_rocm72_modules
source "${MATSIM_MACE_VENV:-$REPO/.venv-mace}/bin/activate"

export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 XALT_EXECUTABLE_TRACKING=no
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=7
export MIOPEN_DISABLE_CACHE=1
export LD_LIBRARY_PATH="${VIRTUAL_ENV}/lib/python3.11/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "$OUT"
echo "[cascade-study] $(date) job=${SLURM_JOB_ID:-local} host=$(hostname) shards=$SHARDS"
echo "[cascade-study] args: $ARGS"

pids=()
for ((s = 0; s < SHARDS; s++)); do
  # shellcheck disable=SC2086
  HIP_VISIBLE_DEVICES=$s python "$REPO/scripts/cascade/run_cascade_study.py" $ARGS \
    --out "$OUT" --shard "$s" --shards "$SHARDS" --device cuda > "$OUT/shard-$s.log" 2>&1 &
  pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "[cascade-study] $(date) finished rc=$rc"
exit $rc
