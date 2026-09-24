#!/bin/bash

tvm_ffi_artifact_name() {
  local venv="$1"
  PYTHONNOUSERSITE=1 "${venv}/bin/python3" -c '
import hashlib
import sys
import torch

major, minor = torch.__version__.split(".")[:2]
if torch.cuda.is_available():
    device = "rocm" if torch.version.hip is not None else "cuda"
else:
    device = "cpu"
abi_id = f"{torch.__version__}|cxx11abi={int(torch.compiled_with_cxx11_abi())}"
abi_tag = hashlib.sha256(abi_id.encode()).hexdigest()[:8]
sys.stdout.write(f"libtorch_c_dlpack_addon_torch{major}{minor}-{device}-{abi_tag}.so")
'
}

configure_tvm_ffi_artifact() {
  local repo="$1" venv="$2"
  TVM_FFI_ARTIFACT_DIR="${TVM_FFI_ARTIFACT_DIR:-${repo}/external/tvm-ffi/lib}"
  TVM_FFI_CACHE_DIR="${TVM_FFI_ARTIFACT_DIR}"
  TVM_FFI_SO="${TVM_FFI_ARTIFACT_DIR}/$(tvm_ffi_artifact_name "${venv}")"
  export TVM_FFI_ARTIFACT_DIR TVM_FFI_CACHE_DIR TVM_FFI_SO
}

require_tvm_ffi_artifact() {
  configure_tvm_ffi_artifact "$1" "$2"
  if [[ ! -s "${TVM_FFI_SO}" ]]; then
    echo "[FAIL] Missing or empty installed tvm_ffi artifact: ${TVM_FFI_SO}" >&2
    echo "       Build with: deployments/frontier/setup/prebuild-tvm-ffi-frontier.sh" >&2
    return 1
  fi
}