#!/usr/bin/env bash
# install-vllm-xpu-aurora.sh — Verify the vLLM-XPU stack on ALCF Aurora.
#
# UPDATE (May 2026): ALCF's `frameworks/2025.3.1` module ships a pre-built
# vLLM 0.15.0 + PyTorch 2.10 (XPU) + Ray + Triton-XPU.  No source build is
# required.  This script just sanity-checks the stack and reports versions.
#
# Run on a LOGIN NODE (xpu_avail will report False there — that's expected;
# the smoke-test PBS job will verify XPU works on a compute node).
#
# If you ever need to source-build a newer vLLM (e.g. for an unreleased
# feature), see the git history of this file for the previous full build
# recipe, or inspect Aurora docs at
#   https://docs.alcf.anl.gov/aurora/data-science/inference/vllm/

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../../.." 2>/dev/null && pwd)"
[[ ! -f "${REPO}/pyproject.toml" ]] && REPO=${PROJECT_ROOT:?export PROJECT_ROOT}

echo "==========================================================="
echo "vLLM XPU stack verification — Aurora"
echo "REPO:  $REPO"
echo "Date:  $(date)"
echo "Host:  $(hostname)"
echo "==========================================================="
echo

echo "=== Loading frameworks module ==="
module reset
module load frameworks
module list 2>&1 | head -30 || true
echo

echo "Python:  $(command -v python)  ($(python --version 2>&1))"
echo

echo "=== Stack versions ==="
python - <<'PY'
import importlib
def v(mod):
    try:
        m = importlib.import_module(mod)
        return getattr(m, "__version__", "unknown")
    except Exception as e:
        return f"MISSING ({e.__class__.__name__}: {e})"

for mod in (
    "torch",
    "intel_extension_for_pytorch",
    "oneccl_bindings_for_pytorch",
    "vllm",
    "ray",
    "triton",
):
    print(f"  {mod:36s} {v(mod)}")

import torch
xpu_avail = torch.xpu.is_available() if hasattr(torch, "xpu") else False
xpu_count = torch.xpu.device_count() if (hasattr(torch, "xpu") and xpu_avail) else 0
print()
print(f"  torch.xpu.is_available()     {xpu_avail}")
print(f"  torch.xpu.device_count()     {xpu_count}    (login node has 0 — expected)")
PY

echo
echo "==========================================================="
echo "Verification complete.  Next steps:"
echo "  1. Smoke test (1 PVC node, TP=2):"
echo "       qsub deployments/aurora/smoke-tests/smoke-vllm-singlenode-aurora.sh"
echo "  2. Multi-node serve (2 nodes, TP=24):"
echo "       SERVE_MODEL_PATH=\$PROJ/models/Mixtral-8x22B-Instruct-v0.1 \\"
echo "       qsub deployments/aurora/jobs/job-serve-multinode-vllm-aurora.sh"
echo "==========================================================="
