#!/bin/bash
# ---------------------------------------------------------------------------
# Submit the Aurora active-learning loop for every paper case.
#
# Usage:
#   deployments/aurora/jobs/submit-active-learning-paper-cases.sh
#   CASES="hea_bcc lifepo4" deployments/aurora/jobs/submit-active-learning-paper-cases.sh
#   MLIP_BACKEND=uma deployments/aurora/jobs/submit-active-learning-paper-cases.sh
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
JOB="${SCRIPT_DIR}/job-active-learning-paper-cases-aurora.sh"

CASES="${CASES:-lifepo4 hea_bcc phosphorene phosphorene_qe}"

echo "Submitting Aurora AL paper-case jobs: ${CASES}"
for c in ${CASES}; do
  VARS="CASE=${c}"
  [[ -n "${MLIP_BACKEND:-}" ]] && VARS="${VARS},MLIP_BACKEND=${MLIP_BACKEND}"
  [[ -n "${DFT_BACKEND:-}" ]] && VARS="${VARS},DFT_BACKEND=${DFT_BACKEND}"
  jid=$(qsub -v "${VARS}" "${JOB}")
  echo "  ${c} -> job ${jid}"
done
echo "Done. Track with: qstat -u \$USER"
