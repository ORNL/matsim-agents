#!/bin/bash
# Shared LLM qualification steps, run under `matsim-agents resources serve`.
#
# Expects the client environment to be active and MATSIM_LLM_ENDPOINTS to name
# the endpoints file written by `resources serve` (exactly one server):
#   FACILITY=perlmutter RUN_DIR=... matsim-agents resources serve LAYOUT \
#     --output "$RUN_DIR/llm-servers" -- bash deployments/common/run-llm-check.sh
set -euo pipefail

: "${FACILITY:?set FACILITY (perlmutter, frontier, or aurora)}"
: "${RUN_DIR:?set RUN_DIR}"
: "${MATSIM_LLM_ENDPOINTS:?run under matsim-agents resources serve}"
REPO="${PROJECT_ROOT:?export PROJECT_ROOT}"
CHECK_ROOT="${RUN_DIR}/llm-check"
CHECK_CONFIG="${RUN_DIR}/llm-check-config.json"
mkdir -p "${CHECK_ROOT}"
cd "${REPO}"

CHECK_ROOT="${CHECK_ROOT}" CHECK_CONFIG="${CHECK_CONFIG}" \
CONCURRENCY="${MATSIM_LLM_CONCURRENCY:-2}" python3 - <<'PY'
import json, os, pathlib

endpoints = json.loads(pathlib.Path(os.environ["MATSIM_LLM_ENDPOINTS"]).read_text())
if len(endpoints) != 1:
    raise SystemExit(f"llm-check layouts serve exactly one model; got {sorted(endpoints)}")
config = {
    "server": next(iter(endpoints)),
    "output_root": os.environ["CHECK_ROOT"],
    "temperature": 0.0,
    "timeout_sec": 120,
    "concurrent_requests": int(os.environ["CONCURRENCY"]),
}
pathlib.Path(os.environ["CHECK_CONFIG"]).write_text(json.dumps(config, indent=2) + "\n")
PY

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
CHECK_RUN="${CHECK_RUN}" python3 -c \
  'import json, os, pathlib, sys; result=json.loads((pathlib.Path(os.environ["CHECK_RUN"])/"result.json").read_text()); required={"readiness","load","generation","structured","discussion","distributed"}; ok=result.get("status")=="complete" and required.issubset({k for k,v in result.get("stages",{}).items() if v}); sys.exit(0 if ok else 1)'
ln -sfn "${CHECK_RUN}" "${RUN_DIR}/successful-llm-check"
echo "[$(date)] Qualification passed: ${CHECK_RUN}"

if [[ "${MATSIM_RUN_PORTABILITY:-0}" == "1" ]]; then
  echo "[$(date)] Running live scientific portability suite"
  python3 benchmarks/portability/run.py \
    --facility "${FACILITY}" \
    --suite all \
    --backend qe \
    --execute \
    --live-llm \
    --llm-check-run "${CHECK_RUN}" \
    --output "${RUN_DIR}/portability-live"
fi
