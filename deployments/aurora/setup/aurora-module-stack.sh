#!/bin/bash
# Shared Aurora module stacks. Source this file and call one function:
#   load_aurora_frameworks_modules  — ALCF `frameworks` (PyTorch/XPU, vLLM, oneCCL).
#                                     The matsim venvs are built on top of it by
#                                     setup/install.sh; use for Python, LLM, AL,
#                                     relaxation, and QE-driver jobs.
#   load_aurora_xpu_modules         — oneAPI + GPU user-mode driver + PTI stack used
#                                     by the qualified MLIP XPU inference path.
#   load_aurora_oneapi_modules      — oneAPI compilers/MKL + GPU UMD for native
#                                     builds (VASP). Honors ONEAPI_MODULE and
#                                     GPU_UMD_MODULE.
# Each function is a no-op when the `module` command is unavailable so that the
# scripts stay testable off-system.

load_aurora_frameworks_modules() {
    command -v module >/dev/null 2>&1 || return 0
    module reset
    module load "${FRAMEWORKS_MODULE:-frameworks}"
}

load_aurora_xpu_modules() {
    command -v module >/dev/null 2>&1 || return 0
    module load pti-gpu "${ONEAPI_MODULE:-oneapi/release/2026.1.0}" \
        "${GPU_UMD_MODULE:-intel_gpu_umd_aicoe}" hdf5
}

load_aurora_oneapi_modules() {
    command -v module >/dev/null 2>&1 || return 0
    module reset
    module load "${ONEAPI_MODULE:-oneapi/release/2026.1.0}" \
        "${GPU_UMD_MODULE:-intel_gpu_umd_aicoe}"
}
