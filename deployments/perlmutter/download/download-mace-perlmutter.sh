#!/bin/bash
# Cache one MACE foundation model for offline Perlmutter compute-node use.
set -euo pipefail

REPO="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
FAMILY="${1:-mace_mp}"
MODEL="${2:-medium}"

source "${REPO}/deployments/perlmutter/setup/model-artifacts-perlmutter.sh"
configure_mace_model_artifacts "${REPO}"
source "${REPO}/.venv-mace/bin/activate"

FAMILY="${FAMILY}" MODEL="${MODEL}" python - <<'PY'
import gc
import os

from matsim_agents.active_learning.calculator import build_mace_calculator
from matsim_agents.active_learning.config import MACEConfig

family = os.environ["FAMILY"]
model = os.environ["MODEL"]
calculator = build_mace_calculator(
    MACEConfig(family=family, model=model, device="cpu"),  # type: ignore[arg-type]
    enable_mc_dropout=False,
)
del calculator
gc.collect()
print(f"Cached MACE model {family}:{model} in {os.environ['MACE_CACHE']}")
PY