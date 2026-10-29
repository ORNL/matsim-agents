#!/usr/bin/env bash
# install-vllm-rocm72.sh — prepare and submit the Frontier vLLM ROCm build.
#
# Prerequisite: deployments/frontier/setup/install.sh has already built and
# verified the matsim-agents/HydraGNN environment. This script deliberately
# does not rebuild that base environment.
#
# Phase 1: Create an isolated vLLM environment with the matching ROCm Torch.
# Phase 2: Pre-download vLLM build dependencies into that environment so the
#          compute job can run fully offline.
#
# Phase 3: Submit build-vllm-rocm72.sh as a compute batch job.
#
# Usage (on a login node):
#   bash ${PROJECT_ROOT:?export PROJECT_ROOT}/deployments/frontier/setup/install-vllm-rocm72.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../../.." 2>/dev/null && pwd)"
[[ ! -f "${REPO}/pyproject.toml" ]] && REPO=${PROJECT_ROOT:?export PROJECT_ROOT}
PROJ="$(dirname "${REPO}")"
BASE_VENV="${BASE_VENV:-$REPO/.venv}"
VENV="${VLLM_VENV_PATH:-$REPO/.venv-vllm}"
SLURM_ACCOUNT="${SLURM_ACCOUNT:-lrn110}"
VLLM_SRC="${VLLM_SRC:-${REPO}/external/vllm/src}"
TRITON_SRC="${TRITON_SRC:-${REPO}/external/triton/src}"
PROTECTED_REQS=$REPO/deployments/frontier/setup/vllm-rocm72-protected-requirements.txt

[[ -x "${BASE_VENV}/bin/python" ]] || {
    echo "ERROR: HydraGNN/matsim environment not found: ${BASE_VENV}" >&2
    echo "Run first: bash ${REPO}/deployments/frontier/setup/install.sh" >&2
    exit 1
}
if [[ ! -x "${VENV}/bin/python" ]]; then
    echo "Creating isolated vLLM environment: ${VENV}"
    "${BASE_VENV}/bin/python" -m venv "${VENV}"
fi

echo "=== Phase 1: Install ROCm Torch into isolated vLLM environment ==="
module load miniforge3/23.11.0-0 2>/dev/null || true
source "${VENV}/bin/activate"
pip install --index-url https://download.pytorch.org/whl/rocm7.2 \
    --extra-index-url https://pypi.org/simple \
    "torch==2.14.0" "torchvision==0.29.0"
python - <<'PY'
import torch

assert torch.__version__.startswith("2.14.0+rocm7.2"), torch.__version__
assert torch.version.hip and torch.version.hip.startswith("7.2"), torch.version.hip
print("isolated ROCm Torch verified", torch.__version__, torch.version.hip)
PY

# ── Phase 2: Pre-download vLLM build deps into its isolated venv ─────────────
echo "=== Phase 2: Pre-install vLLM build dependencies ==="
pip install --upgrade \
    "setuptools==79.0.1" \
    "grpcio==1.78.0" \
    "grpcio-reflection==1.78.0"
pip install \
    "cmake>=3.26.1,<4" \
    ninja \
    "packaging>=24.2" \
    pybind11 \
    setuptools-scm \
    wheel \
    "jinja2>=3.1.6"

# Pre-clone vLLM source if not already present
VLLM_REF="${VLLM_REF:-v0.20.0}"
if [[ ! -d "$VLLM_SRC/.git" ]]; then
    echo "Cloning vLLM $VLLM_REF -> $VLLM_SRC"
    mkdir -p "$(dirname $VLLM_SRC)"
    git clone --depth=1 --branch "$VLLM_REF" \
        https://github.com/vllm-project/vllm.git "$VLLM_SRC"
else
    echo "vLLM source already at $VLLM_SRC"
fi

# Install vLLM requirements, filtering only numpy and Triton because Frontier's
# ROCm builds use pinned replacements for both.
if [[ -f "$VLLM_SRC/requirements/rocm.txt" ]]; then
    echo "Installing vLLM ROCm requirements (numpy + triton excluded)..."
    grep -viE '^[[:space:]]*(numpy|triton)([<>=!~[:space:]].*)?$|^[[:space:]]*-r[[:space:]]' "$VLLM_SRC/requirements/rocm.txt" > "$PROTECTED_REQS"
    grep -viE '^[[:space:]]*(numpy|triton)([<>=!~[:space:]].*)?$' "$VLLM_SRC/requirements/common.txt" >> "$PROTECTED_REQS"
    pip install -r "$PROTECTED_REQS"
else
    echo "Installing minimal vLLM build tooling..."
    pip install cmake ninja packaging pybind11 setuptools_scm setuptools wheel amdsmi
fi

# Install amdsmi from the ROCm 7.2 bundled package.
# vLLM v0.20.0 uses amdsmi (not torch.cuda.is_available()) to detect ROCm GPUs.
# The source directory is read-only so copy to /tmp first.
echo "Installing amdsmi from ROCm 7.2..."
(
    amd_smi_build_dir="$(mktemp -d /tmp/amd_smi_build.XXXXXX)"
    trap 'rm -rf -- "$amd_smi_build_dir"' EXIT
    cp -r /opt/rocm-7.2.0/share/amd_smi "$amd_smi_build_dir/amd_smi"
    pip install "$amd_smi_build_dir/amd_smi" --no-deps
)

# Pre-clone triton v3.6.0 for the compute-node build (no internet on compute nodes).
# build-vllm-rocm72.sh sets TRITON_KERNELS_SRC_DIR to bypass cmake FetchContent.
if [[ ! -d "$TRITON_SRC/.git" ]]; then
    echo "Cloning triton v3.6.0 -> $TRITON_SRC"
    mkdir -p "$(dirname $TRITON_SRC)"
    git clone --depth=1 --branch v3.6.0 \
        https://github.com/triton-lang/triton.git "$TRITON_SRC"
else
    echo "triton source already at $TRITON_SRC ($(cd $TRITON_SRC && git describe --tags))"
fi

# Pin NumPy and the ROCm Triton build in the isolated vLLM environment.
echo "Pinning numpy and triton-rocm..."
pip uninstall -y triton || true
pip install --no-deps --force-reinstall numpy==2.4.6
pip install --no-deps --force-reinstall \
    --extra-index-url https://download.pytorch.org/whl/rocm7.2 triton-rocm==3.6.0
python - <<'PY'
import importlib.metadata as md
import numpy

installed = {dist.metadata['Name'].lower(): dist.version for dist in md.distributions()}
print("numpy", numpy.__version__)
print("triton", installed.get("triton", "missing"))
print("triton-rocm", installed.get("triton-rocm", "missing"))
print("setuptools", installed.get("setuptools", "missing"))
print("grpcio", installed.get("grpcio", "missing"))
assert numpy.__version__ == "2.4.6",          f"numpy={numpy.__version__}"
assert installed.get("triton") in (None, "missing"), f"triton={installed.get('triton')}"
assert installed.get("triton-rocm") == "3.6.0",    f"triton-rocm={installed.get('triton-rocm')}"
print("All package version assertions passed.")
PY

# ── Install Ray (multi-node vLLM backend) with explicit pinned deps ───────────
# All packages installed with --no-deps to avoid disturbing the frozen vLLM /
# PyTorch dependency set. Versions verified against vLLM v0.20.0 on 2026-05-06.
echo "=== Installing Ray 2.55.1 and dependencies (pinned, --no-deps) ==="
pip install --no-deps \
    "ray==2.55.1" \
    "msgpack==1.1.2" \
    "aiohttp_cors==0.8.1" \
    "colorful==0.5.8" \
    "smart_open==7.6.0" \
    "opencensus==0.11.4" \
    "opencensus-context==0.1.3" \
    "opentelemetry-exporter-prometheus==0.62b1" \
    "py-spy==0.4.2" \
    "python-discovery==1.3.0" \
    "virtualenv==21.3.1" \
    "distlib==0.4.0"

python -c "import torch; print('isolated vLLM environment ready', torch.__version__, torch.version.hip)"

# ── Phase 3: Submit vLLM compute build job ────────────────────────────────────
echo "=== Phase 3: Submitting vLLM build job ==="
# sbatch runs the script from a spooled copy, so BASH_SOURCE-based REPO
# detection fails on the compute node; pass REPO through explicitly.
export PROJECT_ROOT="$REPO"
export VLLM_VENV_PATH="$VENV"
sbatch -A "$SLURM_ACCOUNT" --export=ALL,PROJECT_ROOT,VLLM_VENV_PATH \
    "$REPO/deployments/frontier/setup/build-vllm-rocm72.sh"
echo "Done. Monitor with: tail -f $PROJ/runs/build-vllm-rocm72-<jobid>.out"
