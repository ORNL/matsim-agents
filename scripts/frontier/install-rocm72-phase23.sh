#!/usr/bin/env bash
# Run Phase 2 (pre-download vLLM deps) and Phase 3 (sbatch build) only.
# Use after Phase 1 already completed successfully.
set -euo pipefail
PROJ=/lustre/orion/mat746/proj-shared
VENV=$PROJ/HydraGNN/installation_DOE_supercomputers/HydraGNN-Installation-Frontier-ROCm72/hydragnn_venv_rocm72
VLLM_SRC=$PROJ/cache/vllm-src/vllm
PROTECTED_REQS=$PROJ/matsim-agents/scripts/frontier/vllm-rocm72-protected-requirements.txt

module load miniforge3/23.11.0-0 2>/dev/null || true
source /sw/frontier/miniforge3/23.11.0-0/etc/profile.d/conda.sh
conda activate "$VENV"

# Ensure matsim-agents and its deps are installed.
pip install -e "$PROJ/matsim-agents[huggingface]" --no-deps
pip install langgraph langchain-huggingface accelerate

VLLM_REF="${VLLM_REF:-v0.20.0}"
if [[ ! -d "$VLLM_SRC/.git" ]]; then
    mkdir -p "$(dirname $VLLM_SRC)"
    git clone --depth=1 --branch "$VLLM_REF" \
        https://github.com/vllm-project/vllm.git "$VLLM_SRC"
fi

# Phase 1 post-step already pinned setuptools==79.0.1, grpcio==1.78.0,
# grpcio-reflection==1.78.0 to compatible intersections. Only numpy and triton
# have no compatible intersection and are filtered + reasserted here.
if [[ -f "$VLLM_SRC/requirements/rocm.txt" ]]; then
    echo "Installing vLLM ROCm requirements (numpy + triton excluded)..."
    grep -viE '^[[:space:]]*(numpy|triton)([<>=!~[:space:]].*)?$|^[[:space:]]*-r[[:space:]]' "$VLLM_SRC/requirements/rocm.txt" > "$PROTECTED_REQS"
    grep -viE '^[[:space:]]*(numpy|triton)([<>=!~[:space:]].*)?$' "$VLLM_SRC/requirements/common.txt" >> "$PROTECTED_REQS"
    pip install -r "$PROTECTED_REQS"
else
    pip install cmake ninja packaging pybind11 setuptools_scm setuptools wheel amdsmi
fi

# Install amdsmi from the ROCm 7.2 bundled package.
# vLLM v0.20.0 uses amdsmi (not torch.cuda.is_available()) to detect ROCm GPUs.
echo "Installing amdsmi from ROCm 7.2..."
rm -rf /tmp/amd_smi_build
cp -r /opt/rocm-7.2.0/share/amd_smi /tmp/amd_smi_build
pip install /tmp/amd_smi_build --no-deps
rm -rf /tmp/amd_smi_build

# Pre-clone triton v3.6.0 for the compute-node build (no internet on compute nodes).
TRITON_SRC=$PROJ/cache/triton-src/triton
if [[ ! -d "$TRITON_SRC/.git" ]]; then
    echo "Cloning triton v3.6.0 -> $TRITON_SRC"
    mkdir -p "$(dirname $TRITON_SRC)"
    git clone --depth=1 --branch v3.6.0 \
        https://github.com/triton-lang/triton.git "$TRITON_SRC"
else
    echo "triton source already at $TRITON_SRC ($(cd $TRITON_SRC && git describe --tags))"
fi

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

echo "=== Phase 3: Submitting vLLM build job ==="
sbatch "$PROJ/matsim-agents/scripts/frontier/build-vllm-rocm72.sh"
