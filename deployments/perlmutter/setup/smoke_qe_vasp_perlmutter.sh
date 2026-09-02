#!/bin/bash
# smoke_qe_vasp_perlmutter.sh
# Quick compute-node smoke checks for QE and VASP runtimes on Perlmutter.
#
# Usage examples:
#   srun -A <allocation> -q interactive -C gpu -N 1 -G 1 -t 00:10:00 \
#     bash deployments/perlmutter/setup/smoke_qe_vasp_perlmutter.sh
#
#   sbatch -A <allocation> -q debug -C gpu -N 1 -t 00:10:00 \
#     --output=%x-%j.out --error=%x-%j.err \
#     --wrap="bash deployments/perlmutter/setup/smoke_qe_vasp_perlmutter.sh"

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

source "${SCRIPT_DIR}/setup_matsim_perlmutter.sh" --gpu

# QE smoke can fail on minimal launch if MPICH GPU support is enabled but GTL
# is not linked in that launch context. Disable it for this quick smoke only.
export MPICH_GPU_SUPPORT_ENABLED=0

QE_BIN="${REPO_ROOT}/external/quantum-espresso/install-gpu/bin/pw.x"
VASP_STD_BIN="${REPO_ROOT}/external/vasp6/src/vasp.6.6.0/bin/vasp_std"

PASS=0
FAIL=0

print_header() {
    echo ""
    echo "================================"
    echo "$1"
    echo "================================"
}

mark_pass() {
    echo "PASS: $1"
    PASS=$((PASS + 1))
}

mark_fail() {
    echo "FAIL: $1"
    FAIL=$((FAIL + 1))
}

print_header "System"
echo "Host: $(hostname)"
echo "Workdir: ${REPO_ROOT}"
python - <<'PY'
import torch
print(f"torch_version={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"cuda_device_count={torch.cuda.device_count()}")
PY

print_header "QE Smoke"
if [[ ! -x "${QE_BIN}" ]]; then
    mark_fail "QE binary not found: ${QE_BIN}"
else
    timeout 20s "${QE_BIN}" -in /dev/null > /tmp/pw_smoke.out 2>&1 || true
    if grep -q "Program PWSCF" /tmp/pw_smoke.out; then
        mark_pass "QE startup banner detected"
    else
        mark_fail "QE startup banner missing"
        head -n 20 /tmp/pw_smoke.out || true
    fi
fi

print_header "VASP Smoke"
if [[ ! -x "${VASP_STD_BIN}" ]]; then
    mark_fail "VASP binary not found: ${VASP_STD_BIN}"
else
    timeout 20s "${VASP_STD_BIN}" > /tmp/vasp_smoke.out 2>&1 || true
    if grep -q "No INCAR found" /tmp/vasp_smoke.out; then
        mark_pass "VASP reached expected no-INCAR stop"
    else
        mark_fail "VASP did not reach expected no-INCAR stop"
        head -n 25 /tmp/vasp_smoke.out || true
    fi
fi

print_header "Summary"
echo "Passed: ${PASS}"
echo "Failed: ${FAIL}"

if [[ ${FAIL} -gt 0 ]]; then
    exit 1
fi

exit 0
