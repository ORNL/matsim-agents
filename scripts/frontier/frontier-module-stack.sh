#!/bin/bash

# Shared Frontier module stack for ROCm jobs.
# Source this file and call the appropriate function:
#   load_frontier_rocm711_modules      — ROCm 7.1.1
#   load_frontier_rocm72_modules       — ROCm 7.2.0 (may fix RCCL Generic_4 gfx90a bug)
load_frontier_rocm711_modules() {
    module reset
    ml cpe/24.07 rocm/7.1.1 amd-mixed/7.1.1 PrgEnv-gnu miniforge3/23.11.0-0
    module unload darshan-runtime || true
}

load_frontier_rocm72_modules() {
    module reset
    ml cpe/24.07 rocm/7.2.0 amd-mixed/7.2.0 PrgEnv-gnu miniforge3/23.11.0-0
    module unload darshan-runtime || true
}
