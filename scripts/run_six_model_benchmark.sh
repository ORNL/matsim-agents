#!/usr/bin/env bash
set -euo pipefail

# One-command wrapper for the 6-model benchmark pipeline.
# Steps:
#   1) eval_six_models_search_prompt.py
#   2) rank_model_eval.py
#   3) plot_model_eval.py (optional; skipped when matplotlib is unavailable)

usage() {
  cat <<'EOF'
Usage:
  scripts/run_six_model_benchmark.sh --prompt "<text>" [options]

Required:
  --prompt TEXT                  Single prompt to run across all six models.

Optional:
  --output-prefix PATH           Prefix for artifacts (default: runs/six-model-benchmark-<timestamp>)
  --keywords CSV                 Keyword list for eval scoring
  --temperature FLOAT            Shared generation temperature (default: 0.0)
  --spec-file PATH               JSON model spec file (default: built-in six-model matrix)

  --weight-keyword FLOAT         Ranking weight (default: 0.45)
  --weight-composition FLOAT     Ranking weight (default: 0.20)
  --weight-length FLOAT          Ranking weight (default: 0.20)
  --weight-latency FLOAT         Ranking weight (default: 0.15)
  --min-chars INT                Full-score response length threshold (default: 600)
  --comp-target INT              Full-score composition count threshold (default: 3)
  --latency-pivot-sec FLOAT      Latency score pivot (default: 20.0)

  --title TEXT                   Plot title (default: Six-Model matsim-agents Benchmark)
  --skip-plot                    Skip PNG chart generation
  --python-bin CMD               Python executable (default: python3)
  -h, --help                     Show this message

Outputs:
  <prefix>.json                  Raw eval report
  <prefix>-leaderboard.csv       Ranked leaderboard
  <prefix>-ranked.json           Ranked JSON
  <prefix>-chart.png             Comparison chart (unless --skip-plot)
EOF
}

PROMPT=""
OUTPUT_PREFIX=""
KEYWORDS="stability,band gap,formation energy,synthesis"
TEMPERATURE="0.0"
SPEC_FILE=""
WEIGHT_KEYWORD="0.45"
WEIGHT_COMPOSITION="0.20"
WEIGHT_LENGTH="0.20"
WEIGHT_LATENCY="0.15"
MIN_CHARS="600"
COMP_TARGET="3"
LATENCY_PIVOT_SEC="20.0"
TITLE="Six-Model matsim-agents Benchmark"
SKIP_PLOT="0"
PYTHON_BIN="python3"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prompt)
      PROMPT="${2:-}"
      shift 2
      ;;
    --output-prefix)
      OUTPUT_PREFIX="${2:-}"
      shift 2
      ;;
    --keywords)
      KEYWORDS="${2:-}"
      shift 2
      ;;
    --temperature)
      TEMPERATURE="${2:-}"
      shift 2
      ;;
    --spec-file)
      SPEC_FILE="${2:-}"
      shift 2
      ;;
    --weight-keyword)
      WEIGHT_KEYWORD="${2:-}"
      shift 2
      ;;
    --weight-composition)
      WEIGHT_COMPOSITION="${2:-}"
      shift 2
      ;;
    --weight-length)
      WEIGHT_LENGTH="${2:-}"
      shift 2
      ;;
    --weight-latency)
      WEIGHT_LATENCY="${2:-}"
      shift 2
      ;;
    --min-chars)
      MIN_CHARS="${2:-}"
      shift 2
      ;;
    --comp-target)
      COMP_TARGET="${2:-}"
      shift 2
      ;;
    --latency-pivot-sec)
      LATENCY_PIVOT_SEC="${2:-}"
      shift 2
      ;;
    --title)
      TITLE="${2:-}"
      shift 2
      ;;
    --skip-plot)
      SKIP_PLOT="1"
      shift
      ;;
    --python-bin)
      PYTHON_BIN="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "$PROMPT" ]]; then
  echo "Error: --prompt is required." >&2
  usage
  exit 2
fi

if [[ -z "$OUTPUT_PREFIX" ]]; then
  TS="$(date +%Y%m%d-%H%M%S)"
  OUTPUT_PREFIX="runs/six-model-benchmark-${TS}"
fi

OUT_JSON="${OUTPUT_PREFIX}.json"
OUT_CSV="${OUTPUT_PREFIX}-leaderboard.csv"
OUT_RANKED_JSON="${OUTPUT_PREFIX}-ranked.json"
OUT_PNG="${OUTPUT_PREFIX}-chart.png"

mkdir -p "$(dirname "$OUT_JSON")"

echo "[1/3] Running six-model evaluation"
EVAL_CMD=(
  "$PYTHON_BIN" scripts/eval_six_models_search_prompt.py
  --prompt "$PROMPT"
  --keywords "$KEYWORDS"
  --temperature "$TEMPERATURE"
  --out "$OUT_JSON"
)
if [[ -n "$SPEC_FILE" ]]; then
  EVAL_CMD+=(--spec-file "$SPEC_FILE")
fi
"${EVAL_CMD[@]}"

echo "[2/3] Ranking evaluation report"
"$PYTHON_BIN" scripts/rank_model_eval.py \
  --input "$OUT_JSON" \
  --out-csv "$OUT_CSV" \
  --out-json "$OUT_RANKED_JSON" \
  --weight-keyword "$WEIGHT_KEYWORD" \
  --weight-composition "$WEIGHT_COMPOSITION" \
  --weight-length "$WEIGHT_LENGTH" \
  --weight-latency "$WEIGHT_LATENCY" \
  --min-chars "$MIN_CHARS" \
  --comp-target "$COMP_TARGET" \
  --latency-pivot-sec "$LATENCY_PIVOT_SEC"

if [[ "$SKIP_PLOT" == "1" ]]; then
  echo "[3/3] Plot generation skipped (--skip-plot)."
else
  echo "[3/3] Generating chart"
  if "$PYTHON_BIN" scripts/plot_model_eval.py --input-csv "$OUT_CSV" --out "$OUT_PNG" --title "$TITLE"; then
    :
  else
    echo "Plot step failed (likely missing matplotlib). Continuing with JSON/CSV outputs." >&2
  fi
fi

echo

echo "Benchmark pipeline completed."
echo "  Eval JSON:        $OUT_JSON"
echo "  Leaderboard CSV:  $OUT_CSV"
echo "  Ranked JSON:      $OUT_RANKED_JSON"
if [[ "$SKIP_PLOT" != "1" ]]; then
  echo "  Chart PNG:        $OUT_PNG"
fi
