#!/usr/bin/env bash
# install-rocm72.sh  —  Run on the LOGIN NODE (has internet).
#
# Phase 1: Build the full HydraGNN ROCm 7.2 environment (pip installs, git
#          clones, PyG/mpi4py/ADIOS2 compilation).  SKIP_VLLM=1 here because
#          vLLM must be built on a compute node (needs GPU/hipcc at link time).
#
# Phase 2: Pre-download vLLM build dependencies into the new venv so the
#          compute job can run fully offline.
#
# Phase 3: Submit build-vllm-rocm72.sh as a compute batch job.
#
# Usage (on a login node):
#   bash /lustre/orion/mat746/proj-shared/matsim-agents/scripts/frontier/install-rocm72.sh

set -euo pipefail

PROJ=/lustre/orion/mat746/proj-shared
VENV=$PROJ/HydraGNN/installation_DOE_supercomputers/HydraGNN-Installation-Frontier-ROCm72/hydragnn_venv_rocm72
VLLM_SRC=$PROJ/cache/vllm-src/vllm
PROTECTED_REQS=$PROJ/matsim-agents/scripts/frontier/vllm-rocm72-protected-requirements.txt

# ── Phase 1: HydraGNN environment (login node, internet required) ─────────────
echo "=== Phase 1: HydraGNN ROCm 7.2 environment ==="
export SKIP_VLLM=1
cd $PROJ/HydraGNN/installation_DOE_supercomputers
bash hydragnn_installation_bash_script_frontier-rocm72.sh

# Pin to versions satisfying both HydraGNN Phase 1 AND vLLM Phase 2 constraints.
# Done here so Phase 2 never needs to reassert them.
#   setuptools: torch needs <82; vLLM needs >=77.0.3,<80.0.0  → 79.0.1
#   grpcio:     tensorflow needs >=1.24.3,<2.0; vLLM pins ==1.78.0 → 1.78.0
#   grpcio-reflection: vLLM pins ==1.78.0; not in Phase 1 at all → install now
#   protobuf:   Phase 1 installs 7.x (via tensorflow); vLLM needs >=5.29.6 → already fine
echo "=== Phase 1 post-step: vLLM compatibility pins ==="
module load miniforge3/23.11.0-0 2>/dev/null || true
source /sw/frontier/miniforge3/23.11.0-0/etc/profile.d/conda.sh
conda activate "$VENV"
pip install --force-reinstall \
    "setuptools==79.0.1" \
    "grpcio==1.78.0" \
    "grpcio-reflection==1.78.0"

# Install matsim-agents with all required extras.
# langgraph: required by matsim_agents package init (state.py imports it)
# langchain-huggingface: required for the huggingface provider
# accelerate: required by HuggingFacePipeline for device_map="auto"
echo "=== Phase 1 post-step: matsim-agents install ==="
pip install -e "$PROJ/matsim-agents[huggingface]" --no-deps
pip install langgraph langchain-huggingface accelerate
conda deactivate

# ── Phase 2: Pre-download vLLM build deps into the venv ──────────────────────
echo "=== Phase 2: Pre-install vLLM build dependencies ==="
conda activate "$VENV"

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

# Install vLLM requirements, filtering only packages with NO compatible
# intersection with Phase 1 (numpy and triton).
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
rm -rf /tmp/amd_smi_build
cp -r /opt/rocm-7.2.0/share/amd_smi /tmp/amd_smi_build
pip install /tmp/amd_smi_build --no-deps
rm -rf /tmp/amd_smi_build

# Pre-clone triton v3.6.0 for the compute-node build (no internet on compute nodes).
# build-vllm-rocm72.sh sets TRITON_KERNELS_SRC_DIR to bypass cmake FetchContent.
TRITON_SRC=$PROJ/cache/triton-src/triton
if [[ ! -d "$TRITON_SRC/.git" ]]; then
    echo "Cloning triton v3.6.0 -> $TRITON_SRC"
    mkdir -p "$(dirname $TRITON_SRC)"
    git clone --depth=1 --branch v3.6.0 \
        https://github.com/triton-lang/triton.git "$TRITON_SRC"
else
    echo "triton source already at $TRITON_SRC ($(cd $TRITON_SRC && git describe --tags))"
fi

# Reassert only numpy and triton-rocm (no compatible intersection with vLLM)
echo "Reasserting numpy and triton-rocm..."
pip uninstall -y triton || true
pip install --no-deps --force-reinstall numpy==1.26.4 \
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
assert numpy.__version__ == "1.26.4",         f"numpy={numpy.__version__}"
assert installed.get("triton") in (None, "missing"), f"triton={installed.get('triton')}"
assert installed.get("triton-rocm") == "3.6.0",    f"triton-rocm={installed.get('triton-rocm')}"
print("All package version assertions passed.")
PY

# ── Install Ray (multi-node vLLM backend) with explicit pinned deps ───────────
# All packages installed with --no-deps to avoid disturbing the frozen vLLM /
# PyTorch / HydraGNN dependency set. Versions verified against venv 2026-05-06.
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

# ── Phase 3: Submit vLLM compute build job ────────────────────────────────────
echo "=== Phase 3: Submitting vLLM build job ==="
sbatch "$PROJ/matsim-agents/scripts/frontier/build-vllm-rocm72.sh"
echo "Done. Monitor with: tail -f $PROJ/runs/build-vllm-rocm72-<jobid>.out"
