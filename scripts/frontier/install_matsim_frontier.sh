#!/usr/bin/env bash
# =============================================================================
# install_matsim_frontier.sh
#
# Full two-phase install of matsim-agents + HydraGNN on Frontier (ROCm 7.1).
#
# Background
# ----------
# The matsim-agents setup_env.sh script, when run with PLATFORM=frontier-rocm71,
# delegates to HydraGNN's own supercomputer installer, which creates a conda
# environment and then exits (intentionally, exit code 1) asking you to
# activate the new env and re-run. The re-run path passes $VIRTUAL_ENV as the
# venv dir, but on Frontier the env is a conda env so $VIRTUAL_ENV is empty —
# causing the script to spin up a new .venv with the system Python 3.6.
#
# This script works around both issues:
#   Phase 1 – run HydraGNN's Frontier ROCm 7.1 installer directly
#   Phase 2 – activate the resulting conda env and pip-install matsim-agents
#   Phase 3 – submit a one-time SLURM job to prebuild the tvm-ffi ROCm addon
#
# Why is Phase 3 a SLURM job and not just another pip step?
# ---------------------------------------------------------
# Phases 1–2 succeed on the login node because everything they install is
# either pure Python or **prebuilt binary wheels** (torch+rocm7.1, the
# torch_scatter / torch_sparse / torch_cluster / torch_geometric wheels, etc.).
# None of those wheels invoke the system ROCm toolchain at install time —
# they ship with their own libtorch_hip.so etc., already linked.
#
# tvm-ffi is the one exception. vLLM imports tvm_ffi, which on first use calls
# torch.utils.cpp_extension.load() to JIT-compile a ~700-line C++ shared
# library against PyTorch headers AND the system ROCm toolchain (hipcc,
# /opt/rocm-7.1.1/include/*, libamdhip64). On Frontier:
#
#   • Login nodes:    /opt/rocm-7.1.1 is a 60-byte stub directory (empty).
#                     hipcc, ROCm headers, runtime libs are NOT present.
#   • Compute nodes:  /opt/rocm-7.1.1 is bind-mounted with the real install.
#
# So the build MUST run on a compute node. The output .so is written to a
# project-shared GPFS cache (Lustre), so every future job — on any node, by
# any collaborator — finds the prebuilt artifact and skips the ~5 min compile.
# Without prebuilding, every vLLM job pays that cost; worse, if a job is
# killed mid-build it leaves a stale FileLock that deadlocks every subsequent
# job sharing the cache.
#
# Usage
# -----
#   bash install_matsim_frontier.sh [--skip-hydragnn]
#
#   --skip-hydragnn   Skip Phase 1 (HydraGNN env build). Useful when the env
#                     already exists and only matsim-agents needs reinstalling.
#
# Configurable variables (override via environment before calling this script)
# ---------------------------------------------------------------------------
#   PROJECT_DIR      Root project directory       (default: directory of this script)
#   MATSIM_DIR       matsim-agents checkout path  (default: $PROJECT_DIR/matsim-agents)
#   HYDRAGNN_DIR     HydraGNN checkout path       (default: $PROJECT_DIR/HydraGNN)
#   HYDRAGNN_REPO    Git remote for HydraGNN      (default: https://github.com/allaffa/HydraGNN.git)
#   HYDRAGNN_BRANCH  Branch to use for runtime    (default: fix/structure-optimization-ase-fused)
#   MATSIM_REPO      Git remote for matsim-agents (default: https://github.com/ORNL/matsim-agents.git)
#   LLM_BACKENDS     pip extras to install        (default: vllm,dev)
#   VENV_PATH        Conda env path (auto-detected after Phase 1 if not set)
#   ROCM_VERSION     ROCm version to use: "7.1" (default) or "7.2"
#                    Equivalent to passing --rocm72 flag.
# =============================================================================
set -euo pipefail

# ── Configurable paths ────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-${SCRIPT_DIR}}"
MATSIM_DIR="${MATSIM_DIR:-${PROJECT_DIR}/matsim-agents}"
HYDRAGNN_DIR="${HYDRAGNN_DIR:-${PROJECT_DIR}/HydraGNN}"
HYDRAGNN_REPO="${HYDRAGNN_REPO:-https://github.com/allaffa/HydraGNN.git}"
HYDRAGNN_BRANCH="${HYDRAGNN_BRANCH:-fix/structure-optimization-ase-fused}"
MATSIM_REPO="${MATSIM_REPO:-https://github.com/ORNL/matsim-agents.git}"
LLM_BACKENDS="${LLM_BACKENDS:-vllm,dev}"

# ROCm version selection (default: 7.1; pass --rocm72 or set ROCM_VERSION=7.2)
ROCM_VERSION="${ROCM_VERSION:-7.1}"

# Auto-detect venv path: the HydraGNN Frontier installer places it here
# (overridden below once ROCM_VERSION is known if not explicitly set)
VENV_PATH_DEFAULT_71="${HYDRAGNN_DIR}/installation_DOE_supercomputers/HydraGNN-Installation-Frontier/hydragnn_venv"
VENV_PATH_DEFAULT_72="${HYDRAGNN_DIR}/installation_DOE_supercomputers/HydraGNN-Installation-Frontier-ROCm72/hydragnn_venv_rocm72"
VENV_PATH="${VENV_PATH:-}"

# ── Helpers ───────────────────────────────────────────────────────────────────
log()  { printf '\033[1;34m[install]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[install]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[install]\033[0m %s\n' "$*" >&2; exit 1; }

SKIP_HYDRAGNN=0
for arg in "$@"; do
    [[ "$arg" == "--skip-hydragnn" ]] && SKIP_HYDRAGNN=1
    [[ "$arg" == "--rocm72" ]] && ROCM_VERSION="7.2"
done

# Set VENV_PATH default based on selected ROCm version
if [[ -z "$VENV_PATH" ]]; then
    if [[ "$ROCM_VERSION" == "7.2" ]]; then
        VENV_PATH="$VENV_PATH_DEFAULT_72"
    else
        VENV_PATH="$VENV_PATH_DEFAULT_71"
    fi
fi

# ── Step 0: Load Frontier modules ─────────────────────────────────────────────
if [[ "$ROCM_VERSION" == "7.2" ]]; then
    log "Loading Frontier modules (ROCm 7.2)..."
    module reset
    ml cpe/24.07
    ml rocm/7.2.0
    ml amd-mixed/7.2.0
    ml PrgEnv-gnu
    ml miniforge3/23.11.0-0
    module unload darshan-runtime || true
else
    log "Loading Frontier modules (ROCm 7.1)..."
    module reset
    ml cpe/24.07
    ml rocm/7.1.1
    ml amd-mixed/7.1.1
    ml PrgEnv-gnu
    ml miniforge3/23.11.0-0
    module unload darshan-runtime || true
fi
log "Modules loaded (ROCm ${ROCM_VERSION})."

# ── Step 1: Clone repos if missing ────────────────────────────────────────────
if [[ ! -d "${MATSIM_DIR}/.git" ]]; then
    log "Cloning matsim-agents -> ${MATSIM_DIR}"
    git clone "${MATSIM_REPO}" "${MATSIM_DIR}"
else
    log "matsim-agents already present at ${MATSIM_DIR}"
fi

if [[ ! -d "${HYDRAGNN_DIR}/.git" ]]; then
    log "Cloning HydraGNN (${HYDRAGNN_BRANCH}) -> ${HYDRAGNN_DIR}"
    git clone --branch "${HYDRAGNN_BRANCH}" "${HYDRAGNN_REPO}" "${HYDRAGNN_DIR}" \
        || git clone "${HYDRAGNN_REPO}" "${HYDRAGNN_DIR}"
else
    log "HydraGNN already present at ${HYDRAGNN_DIR}"
fi

# Pre-clone vLLM source from the login node so the build job (which runs on a
# compute node without internet access) can find it under cache/vllm-src/vllm.
VLLM_SRC_DIR="${PROJECT_DIR}/cache/vllm-src/vllm"
VLLM_REF="${VLLM_REF:-v0.20.0}"
if [[ ! -d "${VLLM_SRC_DIR}/.git" ]]; then
    log "Pre-cloning vLLM ${VLLM_REF} -> ${VLLM_SRC_DIR} (compute nodes have no internet)"
    mkdir -p "${PROJECT_DIR}/cache/vllm-src"
    git clone --depth=1 --branch "${VLLM_REF}" \
        https://github.com/vllm-project/vllm.git "${VLLM_SRC_DIR}"
else
    log "vLLM source already present at ${VLLM_SRC_DIR}"
fi

# ── Phase 1: Build HydraGNN Frontier environment ─────────────────────────────
if [[ "$SKIP_HYDRAGNN" == "0" ]]; then
    if [[ "$ROCM_VERSION" == "7.2" ]]; then
        SC_INSTALLER="${HYDRAGNN_DIR}/installation_DOE_supercomputers/hydragnn_installation_bash_script_frontier-rocm72.sh"
        [[ -f "$SC_INSTALLER" ]] || die "HydraGNN Frontier ROCm 7.2 installer not found: ${SC_INSTALLER}"
        log "Running HydraGNN Frontier ROCm 7.2 installer (this takes ~1-2 hours)..."
        log "Installer: ${SC_INSTALLER}"
        ( cd "${HYDRAGNN_DIR}/installation_DOE_supercomputers" \
            && bash "hydragnn_installation_bash_script_frontier-rocm72.sh" )
    else
        SC_INSTALLER="${HYDRAGNN_DIR}/installation_DOE_supercomputers/hydragnn_installation_bash_script_frontier-rocm71.sh"
        [[ -f "$SC_INSTALLER" ]] || die "HydraGNN Frontier installer not found: ${SC_INSTALLER}"
        log "Running HydraGNN Frontier ROCm 7.1 installer (this takes ~1-2 hours)..."
        log "Installer: ${SC_INSTALLER}"
        ( cd "${HYDRAGNN_DIR}/installation_DOE_supercomputers" \
            && bash "hydragnn_installation_bash_script_frontier-rocm71.sh" )
    fi
    log "HydraGNN environment build complete."
else
    warn "--skip-hydragnn set: skipping Phase 1."
fi

# ── Verify the conda env exists ───────────────────────────────────────────────
[[ -d "${VENV_PATH}" ]] \
    || die "Expected conda env not found at: ${VENV_PATH}\nRun without --skip-hydragnn first."

# ── Phase 2: Activate the conda env and install matsim-agents ─────────────────
log "Activating conda env: ${VENV_PATH}"
# shellcheck disable=SC1091
source activate "${VENV_PATH}"

PYTHON_VER=$(python --version 2>&1)
log "Active Python: ${PYTHON_VER}"
[[ "$PYTHON_VER" == *"3.11"* ]] \
    || warn "Expected Python 3.11 — got ${PYTHON_VER}. Proceeding anyway."

log "Installing matsim-agents[${LLM_BACKENDS}] (editable) into ${VENV_PATH}..."
pip install --upgrade pip setuptools wheel
pip install -e "${MATSIM_DIR}[${LLM_BACKENDS}]"

# Pre-install the full vLLM ROCm dependency set on the login node.
# Frontier compute nodes do not have outbound internet, so the SLURM build job
# must run with dependencies already present in this environment.
VLLM_ROCM_REQ="${PROJECT_DIR}/cache/vllm-src/vllm/requirements/rocm.txt"
if [[ -f "${VLLM_ROCM_REQ}" ]]; then
    log "Pre-installing full vLLM ROCm dependencies from ${VLLM_ROCM_REQ}..."
    pip install -r "${VLLM_ROCM_REQ}"
else
    warn "vLLM ROCm requirements file not found at ${VLLM_ROCM_REQ}; falling back to minimal build tooling install."
    pip install -q cmake ninja packaging pybind11 setuptools_scm setuptools wheel amdsmi
fi

# accelerate is required for multi-GPU inference via HuggingFace Transformers
# (device_map="auto"), used by the smoke-transformers-frontier.sh smoke test.
log "Installing accelerate (required for Transformers device_map=auto)..."
pip install accelerate

# ── Phase 2.5: Patch flashinfer for ROCm (no libcudart, only libamdhip64) ─────
#
# flashinfer/comm/cuda_ipc.py defines its own CudaRTLibrary (copied from an
# older vLLM) and instantiates it at MODULE IMPORT TIME (line ~194).  The
# constructor only searches /proc/self/maps for "libcudart"; on ROCm there is
# no libcudart — only libamdhip64 — so the assertion fires in every vLLM
# worker subprocess before torch has been imported.
#
# vLLM's own copy of the same class (vllm/distributed/device_communicators/
# cuda_wrapper.py) already has the libamdhip64 + VLLM_CUDART_SO_PATH fallback,
# but flashinfer's copy does not.  We patch it here so the fix survives venv
# rebuilds without forking flashinfer.
FLASHINFER_CUDA_IPC="${VENV_PATH}/lib/python3.11/site-packages/flashinfer/comm/cuda_ipc.py"
if [[ -f "$FLASHINFER_CUDA_IPC" ]]; then
    if grep -q "libamdhip64" "$FLASHINFER_CUDA_IPC"; then
        log "flashinfer ROCm patch already applied; skipping."
    else
        log "Patching flashinfer/comm/cuda_ipc.py for ROCm (libamdhip64 fallback)..."
        python - "$FLASHINFER_CUDA_IPC" <<'PYEOF'
import sys, re

path = sys.argv[1]
with open(path) as f:
    src = f.read()

old = (
    '    def __init__(self, so_file: Optional[str] = None):\n'
    '        if so_file is None:\n'
    '            so_file = find_loaded_library("libcudart")\n'
    '            assert so_file is not None, "libcudart is not loaded in the current process"'
)
new = (
    '    # ROCm/HIP: libamdhip64 uses hipXxx names instead of cudaXxx\n'
    '    # (copied from vllm/distributed/device_communicators/cuda_wrapper.py)\n'
    '    cuda_to_hip_mapping = {\n'
    '        "cudaSetDevice": "hipSetDevice",\n'
    '        "cudaDeviceSynchronize": "hipDeviceSynchronize",\n'
    '        "cudaDeviceReset": "hipDeviceReset",\n'
    '        "cudaGetErrorString": "hipGetErrorString",\n'
    '        "cudaMalloc": "hipMalloc",\n'
    '        "cudaFree": "hipFree",\n'
    '        "cudaMemset": "hipMemset",\n'
    '        "cudaMemcpy": "hipMemcpy",\n'
    '        "cudaIpcGetMemHandle": "hipIpcGetMemHandle",\n'
    '        "cudaIpcOpenMemHandle": "hipIpcOpenMemHandle",\n'
    '    }\n'
    '\n'
    '    def __init__(self, so_file: Optional[str] = None):\n'
    '        import os\n'
    '        is_hip = False\n'
    '        if so_file is None:\n'
    '            so_file = find_loaded_library("libcudart")\n'
    '            if so_file is None:\n'
    '                so_file = find_loaded_library("libamdhip64")\n'
    '                if so_file is not None:\n'
    '                    is_hip = True\n'
    '            if so_file is None:\n'
    '                so_file = os.environ.get("VLLM_CUDART_SO_PATH")\n'
    '                if so_file is not None and "amdhip" in so_file:\n'
    '                    is_hip = True\n'
    '            assert so_file is not None, "libcudart is not loaded in the current process"\n'
    '        else:\n'
    '            is_hip = "amdhip" in so_file'
)

# Also patch the function-binding loop to use HIP symbol names
old2 = (
    '            for func in CudaRTLibrary.exported_functions:\n'
    '                f = getattr(self.lib, func.name)\n'
    '                f.restype = func.restype\n'
    '                f.argtypes = func.argtypes\n'
    '                _funcs[func.name] = f'
)
new2 = (
    '            for func in CudaRTLibrary.exported_functions:\n'
    '                sym = CudaRTLibrary.cuda_to_hip_mapping.get(func.name, func.name) if is_hip else func.name\n'
    '                f = getattr(self.lib, sym)\n'
    '                f.restype = func.restype\n'
    '                f.argtypes = func.argtypes\n'
    '                _funcs[func.name] = f'
)

if old not in src:
    print(f"WARNING: expected pattern not found in {path}; patch skipped.", file=sys.stderr)
    sys.exit(0)

src = src.replace(old, new, 1)
if old2 in src:
    src = src.replace(old2, new2, 1)

with open(path, 'w') as f:
    f.write(src)
print(f"Patched {path}")
PYEOF
        log "flashinfer patch applied."
    fi
else
    warn "flashinfer cuda_ipc.py not found at ${FLASHINFER_CUDA_IPC}; skipping ROCm patch."
fi

log "matsim-agents installed successfully."

# ── Step 3: Restore HydraGNN to the correct branch ───────────────────────────
# The HydraGNN installer resets the checkout to 'main'. Restore it here.
log "Restoring HydraGNN branch to: ${HYDRAGNN_BRANCH}"
git -C "${HYDRAGNN_DIR}" fetch origin "${HYDRAGNN_BRANCH}" || true
git -C "${HYDRAGNN_DIR}" checkout "${HYDRAGNN_BRANCH}"
log "HydraGNN branch: $(git -C "${HYDRAGNN_DIR}" branch --show-current)"

# ── Step 4: Prebuild tvm-ffi torch<->DLPack ROCm bridge (one-time) ───────────
#
# WHY THIS STEP IS DIFFERENT FROM PHASES 1–2
# ------------------------------------------
# Everything installed in Phases 1–2 (torch+rocm7.1, torch_geometric and its
# torch_scatter/sparse/cluster wheels, vllm, etc.) comes as PREBUILT WHEELS.
# Pip just unpacks them; no system toolchain is invoked. That is why the
# login node — which has only a 60-byte stub at /opt/rocm-7.1.1 — succeeds.
#
# tvm-ffi (pulled in by vllm) is the lone exception: on first use it calls
# torch.utils.cpp_extension.load() to JIT-compile a C++ addon against PyTorch
# headers AND the system ROCm toolchain (hipcc, /opt/rocm-7.1.1/include/*,
# libamdhip64). The login node lacks all of that, so the compile MUST run on
# a compute node where /opt/rocm-7.1.1 is bind-mounted with the real install.
#
# WHY WRITE THE OUTPUT TO LUSTRE
# ------------------------------
# The compute node performing the build writes the resulting .so into a
# project-shared GPFS cache. Every future job — on any compute node, by any
# collaborator — reads the same artifact via the same path and skips the
# ~5 min compile. Lustre is just the storage; only the build itself needs
# the compute node's ROCm bind-mount.
#
# WHY PREBUILD AT ALL
# -------------------
#  • Without prebuilding, every vLLM job pays the ~5 min JIT cost at startup.
#    During that compile vLLM appears "silently hung" — no log output.
#  • Worse: tvm-ffi guards the build with a FileLock. If a job is killed
#    mid-compile, the stale lock file deadlocks every subsequent job sharing
#    the same cache (this bit us with smoke job 4489500).
#  • Doing the compile once, in a known SLURM job we control, eliminates
#    both problems.
#
# Cross-node safety: Frontier compute nodes are homogeneous (MI250X, same
# OS image, same ROCm bind-mount). The .so is keyed by torch major.minor +
# the --build-with-rocm flag, so any compute node consuming it sees an ABI-
# compatible artifact as long as the conda env and rocm/X.Y.Z module pin
# match what produced it.
TVM_FFI_CACHE_DIR="${TVM_FFI_CACHE_DIR:-${PROJECT_DIR}/cache/tvm-ffi}"
mkdir -p "${TVM_FFI_CACHE_DIR}"
chmod g+ws "${TVM_FFI_CACHE_DIR}" 2>/dev/null || true
TORCH_MAJOR_MINOR=$(python -c 'import torch,sys; v=torch.__version__.split(".")[:2]; sys.stdout.write("".join(v))')
TVM_LIB_NAME="libtorch_c_dlpack_addon_torch${TORCH_MAJOR_MINOR}-rocm.so"

if [[ -f "${TVM_FFI_CACHE_DIR}/${TVM_LIB_NAME}" ]]; then
    log "tvm-ffi cache already populated (${TVM_LIB_NAME}); skipping prebuild submission."
else
    PREBUILD_SCRIPT="${MATSIM_DIR}/scripts/prebuild-tvm-ffi-frontier.sh"
    if [[ -x "$PREBUILD_SCRIPT" ]]; then
        log "Submitting one-time tvm-ffi prebuild SLURM job (debug queue, ~10 min)..."
        log "  sbatch ${PREBUILD_SCRIPT}"
        # Stale locks from previously killed jobs deadlock new builds; clean them.
        rm -f "${TVM_FFI_CACHE_DIR}"/*.lock "${HOME}/.cache/tvm-ffi"/*.lock 2>/dev/null || true
        ( cd "${PROJECT_DIR}" && sbatch "$PREBUILD_SCRIPT" ) || \
            warn "Failed to submit prebuild job; the first matsim-agents run will pay a ~5 min cold-start cost."
    else
        warn "Prebuild script not found at ${PREBUILD_SCRIPT}; the first matsim-agents run will pay a ~5 min cold-start cost."
    fi
fi

# ── Done ──────────────────────────────────────────────────────────────────────
log "================================================================"
log "Installation complete!"
log ""
log "To activate the environment in a new shell:"
log "  module reset"
log "  ml cpe/24.07 rocm/7.1.1 amd-mixed/7.1.1 PrgEnv-gnu miniforge3/23.11.0-0"
log "  module unload darshan-runtime"
log "  source activate ${VENV_PATH}"
log ""
log "Then verify:"
log "  python -c \"import torch; print(torch.__version__, torch.cuda.is_available())\""
log "  matsim-agents --help"
log ""
log "In job scripts, export the prebuilt tvm-ffi cache so vLLM starts fast:"
log "  export TVM_FFI_CACHE_DIR=${TVM_FFI_CACHE_DIR}"
log ""
if [[ "$ROCM_VERSION" == "7.2" ]]; then
    log "ROCm 7.2 vLLM note:"
    log "  No pre-built pip wheels exist for ROCm 7.2; vLLM was built from source."
    log "  In job scripts set: export VLLM_NCCL_SO_PATH=/opt/rocm-7.2.0/lib/librccl.so.1"
    log "  and:                export VLLM_CUDART_SO_PATH=/opt/rocm-7.2.0/lib/libamdhip64.so"
fi
log "================================================================"
