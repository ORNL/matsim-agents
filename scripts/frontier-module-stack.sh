#!/bin/bash

# Shared Frontier module stack for ROCm jobs.
# Source this file and call load_frontier_rocm_modules.
load_frontier_rocm_modules() {
    module reset
    ml cpe/24.07 rocm/7.1.1 amd-mixed/7.1.1 PrgEnv-gnu miniforge3/23.11.0-0
    module unload darshan-runtime || true
}
