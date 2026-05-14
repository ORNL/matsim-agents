#!/bin/bash
#SBATCH -A mat746
#SBATCH -J six-model-bench
#SBATCH -o /lustre/orion/mat746/proj-shared/runs/six-model-bench-%j/job-%j.out
#SBATCH -e /lustre/orion/mat746/proj-shared/runs/six-model-bench-%j/job-%j.out
#SBATCH -t 01:00:00
#SBATCH -N 1
#SBATCH -p batch
#SBATCH -q debug
# ---------------------------------------------------------------------------
# matsim-agents: submit the six-model benchmark pipeline as one Slurm job.
#
# This script wraps:
#   scripts/run_six_model_benchmark.sh
#
# Default behavior:
#   - Uses built-in six-model spec from eval_six_models_search_prompt.py
#   - Runs eval + ranking + plot
#
# Required at submission:
#   - BENCHMARK_PROMPT (the shared prompt for all models)
#
# Optional overrides:
#   - BENCHMARK_OUTPUT_PREFIX
#   - BENCHMARK_SPEC_FILE
#   - BENCHMARK_KEYWORDS
#   - BENCHMARK_TEMPERATURE
#   - BENCHMARK_TITLE
#   - BENCHMARK_SKIP_PLOT=1
#   - BENCHMARK_ARGS (extra raw args forwarded to wrapper)
#
# Example:
#   BENCHMARK_PROMPT="Search for a Pb-free halide double perovskite candidate and justify stability." \
#   BENCHMARK_SPEC_FILE="scripts/six_model_specs.example.json" \
#   sbatch scripts/frontier/job-six-model-benchmark-frontier.sh
# ---------------------------------------------------------------------------

set -euo pipefail

PROJ=/lustre/orion/mat746/proj-shared
REPO=$PROJ/matsim-agents
VENV=$PROJ/HydraGNN/installation_DOE_supercomputers/HydraGNN-Installation-Frontier-ROCm72/hydragnn_venv_rocm72
RUN_DIR=$PROJ/runs/six-model-bench-$SLURM_JOB_ID
mkdir -p "$RUN_DIR"

if [[ -z "${BENCHMARK_PROMPT:-}" ]]; then
  echo "Error: BENCHMARK_PROMPT is required." >&2
  echo "Example:" >&2
  echo "  BENCHMARK_PROMPT='Search for a Pb-free halide double perovskite candidate and justify stability.' sbatch scripts/frontier/job-six-model-benchmark-frontier.sh" >&2
  exit 2
fi

source /sw/frontier/miniforge3/23.11.0-0/etc/profile.d/conda.sh
source "$REPO/scripts/frontier/frontier-module-stack.sh"
load_frontier_rocm72_modules
source activate "$VENV"

cd "$REPO"

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1

# ── No outbound connections (HPC: no internet access) ────────────────────────
# Disable remote model downloads and all telemetry
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
# vLLM usage telemetry
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
# Ray telemetry
export RAY_USAGE_STATS_ENABLED=0
export RAY_DISABLE_IMPORT_WARNING=1
# Triton: disable remote kernel downloads
export TRITON_DISABLE_AUTOTUNE_CACHE=1
# Block all HTTP/S proxies
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ftp_proxy FTP_PROXY all_proxy ALL_PROXY
export no_proxy='*'
export NO_PROXY='*'

OUTPUT_PREFIX=${BENCHMARK_OUTPUT_PREFIX:-$RUN_DIR/six-model-benchmark}
SPEC_FILE=${BENCHMARK_SPEC_FILE:-}
KEYWORDS=${BENCHMARK_KEYWORDS:-stability,band gap,formation energy,synthesis}
TEMPERATURE=${BENCHMARK_TEMPERATURE:-0.0}
TITLE=${BENCHMARK_TITLE:-Six-Model matsim-agents Benchmark}
SKIP_PLOT=${BENCHMARK_SKIP_PLOT:-0}

CMD=(
  scripts/run_six_model_benchmark.sh
  --prompt "$BENCHMARK_PROMPT"
  --output-prefix "$OUTPUT_PREFIX"
  --keywords "$KEYWORDS"
  --temperature "$TEMPERATURE"
)

if [[ -n "$SPEC_FILE" ]]; then
  CMD+=(--spec-file "$SPEC_FILE")
fi
if [[ "$SKIP_PLOT" == "1" ]]; then
  CMD+=(--skip-plot)
else
  CMD+=(--title "$TITLE")
fi

if [[ -n "${BENCHMARK_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS=( ${BENCHMARK_ARGS} )
  CMD+=("${EXTRA_ARGS[@]}")
fi

echo "[$(date)] Running benchmark wrapper"
echo "[$(date)] Job ID: $SLURM_JOB_ID"
echo "[$(date)] Output prefix: $OUTPUT_PREFIX"
echo "[$(date)] Prompt: $BENCHMARK_PROMPT"
printf '[%s] Command:' "$(date)"
printf ' %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}" 2>&1 | tee "$RUN_DIR/benchmark.log"

echo "[$(date)] Done. Artifacts under: $RUN_DIR"
