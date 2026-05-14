#!/bin/bash
# test_matsim_perlmutter.sh
# Comprehensive test script to validate matsim-agents setup on Perlmutter
#
# Usage:
#   source test_matsim_perlmutter.sh
#
# This script tests:
#   1. Module loading
#   2. Virtual environment activation
#   3. Python and CUDA availability
#   4. Required package imports
#   5. Basic functionality

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TEST_LOG="${SCRIPT_DIR}/test_matsim_perlmutter.log"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counter
TESTS_PASSED=0
TESTS_FAILED=0

# Helper functions
print_header() {
    echo ""
    echo "================================"
    echo "$1"
    echo "================================"
}

test_command() {
    local test_name="$1"
    local command="$2"
    
    echo -n "Testing: $test_name ... "
    
    if eval "$command" &>/dev/null; then
        echo -e "${GREEN}✓ PASS${NC}"
        ((TESTS_PASSED+=1))
    else
        echo -e "${RED}✗ FAIL${NC}"
        ((TESTS_FAILED+=1))
    fi
    return 0
}

test_output() {
    local test_name="$1"
    local command="$2"
    local expected_pattern="$3"
    
    echo -n "Testing: $test_name ... "
    
    if output=$(eval "$command" 2>&1) && echo "$output" | grep -q "$expected_pattern"; then
        echo -e "${GREEN}✓ PASS${NC}"
        echo "  Output: $(echo "$output" | head -n1)"
        ((TESTS_PASSED+=1))
    else
        echo -e "${RED}✗ FAIL${NC}"
        if [[ -n "$output" ]]; then
            echo "  Output: $(echo "$output" | head -n1)"
        fi
        ((TESTS_FAILED+=1))
    fi
    return 0
}

# Start testing
print_header "Setup Environment"
echo "Setting up matsim-agents environment..."
source "${SCRIPT_DIR}/setup_matsim_perlmutter.sh"

print_header "System Information"
echo "Hostname: $(hostname)"
echo "User: $(whoami)"
echo "Working Directory: $(pwd)"
echo "Python Executable: $(which python)"

print_header "Module Tests"
test_command "module command available" "command -v module"
test_output "PrgEnv-gnu loaded" "module list 2>&1" "PrgEnv-gnu"
test_output "CUDA toolkit loaded" "module list 2>&1" "cudatoolkit"
test_output "GCC native loaded" "module list 2>&1" "gcc-native"

print_header "Virtual Environment Tests"
test_command "Virtual environment activated" "[[ -n \${VIRTUAL_ENV:-} || -n \${CONDA_PREFIX:-} ]]"
test_command "venv path set" "[[ -n \${VIRTUAL_ENV:-} ]]"
test_output "Python version" "python --version" "Python"

print_header "Python Package Tests"
test_command "PyTorch installed" "python -c 'import torch'"
test_output "PyTorch version" "python -c 'import torch; print(torch.__version__)'" "[0-9]"
test_command "CUDA available in PyTorch" "python -c 'import torch; assert torch.cuda.is_available(), \"CUDA not available\"'"
test_output "CUDA device count" "python -c 'import torch; print(torch.cuda.device_count())'" "[0-9]"

test_command "HydraGNN installed" "python -c 'import hydragnn'"
test_command "HydraGNN import" "python -c 'from hydragnn import models'"

test_command "matsim-agents on PYTHONPATH" "python -c 'import sys; print(sys.path)' | grep -q matsim"
test_command "matsim-agents import" "python -c 'import matsim_agents'"

print_header "GPU Information"
if command -v nvidia-smi &>/dev/null; then
    test_command "nvidia-smi available" "command -v nvidia-smi"
    echo ""
    echo "GPU Devices:"
    nvidia-smi -L 2>/dev/null || echo "  No GPU info available"
    echo ""
    echo "GPU Memory:"
    nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null || echo "  No memory info available"
else
    echo -e "${YELLOW}⚠ nvidia-smi not available (may be normal on CPU-only jobs)${NC}"
fi

print_header "CUDA Compiler Test"
if command -v nvcc &>/dev/null; then
    test_output "NVCC available" "nvcc --version" "release"
else
    echo -e "${YELLOW}⚠ nvcc not found (compile tools may not be on compute nodes)${NC}"
fi

print_header "Test Summary"
echo ""
echo -e "Tests Passed: ${GREEN}${TESTS_PASSED}${NC}"
echo -e "Tests Failed: ${RED}${TESTS_FAILED}${NC}"

if [[ $TESTS_FAILED -eq 0 ]]; then
    echo ""
    echo -e "${GREEN}✓ All tests passed! Environment is ready.${NC}"
    return 0
else
    echo ""
    echo -e "${RED}✗ Some tests failed. Please review the output above.${NC}"
    return 1
fi
