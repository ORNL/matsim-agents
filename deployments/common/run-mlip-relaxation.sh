#!/bin/bash
# Shared typed-relaxation runner. Source this after activating the site venv.
set -euo pipefail

: "${REPO:?set REPO}"
: "${RUN_DIR:?set RUN_DIR}"
: "${STRUCTURE:?set STRUCTURE}"
: "${LOGDIR:?set LOGDIR}"
: "${HYDRAGNN_BRANCH_MLP_CHECKPOINT:?set HYDRAGNN_BRANCH_MLP_CHECKPOINT}"

RELAX_CONFIG="${RUN_DIR}/relaxation-config.json"
REPO="${REPO}" RUN_DIR="${RUN_DIR}" STRUCTURE="${STRUCTURE}" LOGDIR="${LOGDIR}" \
HYDRAGNN_BRANCH_MLP_CHECKPOINT="${HYDRAGNN_BRANCH_MLP_CHECKPOINT}" \
RELAX_CONFIG="${RELAX_CONFIG}" python3 -c '
import json, os
from pathlib import Path
cfg = {
    "mode": "mlip",
    "structure_path": os.environ["STRUCTURE"],
    "output_root": str(Path(os.environ["RUN_DIR"]) / "outputs"),
    "geometry": {"relax_atoms": True, "relax_cell": False},
    "mlip": {
        "mlip_backend": "hydragnn",
        "logdir": os.environ["LOGDIR"],
        "hydragnn_branch_mlp_checkpoint": os.environ["HYDRAGNN_BRANCH_MLP_CHECKPOINT"],
        "mlp_device": os.environ.get("MATSIM_MLP_DEVICE", "cuda"),
    },
    "max_steps": int(os.environ.get("MATSIM_RELAX_MAX_STEPS", "200")),
    "force_tolerance_eV_per_A": float(os.environ.get("MATSIM_RELAX_FMAX", "0.02")),
}
Path(os.environ["RELAX_CONFIG"]).write_text(json.dumps(cfg, indent=2) + "\n")
'

matsim-agents relax "${RELAX_CONFIG}" 2>&1 | tee "${RUN_DIR}/single-relaxation.log"
