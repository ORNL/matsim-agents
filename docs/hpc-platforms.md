# HPC Platforms Documentation Index

This page centralizes HPC-oriented documentation for matsim-agents.

## Quick Start by Platform

Use the first-command patterns below from the repository root.

| Platform | First build command | First run command |
|---|---|---|
| Frontier (OLCF, MI250X) | `nohup bash scripts/setup/frontier/build-qe-gpu-frontier.sh > runs/build-qe-gpu-login/build-$(date +%Y%m%d-%H%M%S).log 2>&1 &` | `sbatch scripts/launchers/frontier/run-pw-gpu-frontier.sh path/to/your.in` |
| Aurora (ALCF, PVC) | `bash scripts/setup/aurora/build-qe-gpu-aurora.sh` | `bash scripts/launchers/aurora/run-pw-gpu-aurora.sh path/to/pw.in` |
| Perlmutter (NERSC, A100) | `nohup bash scripts/setup/perlmutter/build-qe-gpu-perlmutter.sh > runs/build-qe-gpu-login/build-$(date +%Y%m%d-%H%M%S).log 2>&1 &` | `./scripts/launchers/perlmutter/run-pw-gpu-perlmutter.sh path/to/pw.in` |

If you need full context (toolchain pins, overrides, troubleshooting), continue
to the detailed per-platform sections below.

## Quantum ESPRESSO (DFT) by platform

### Frontier (OLCF, AMD MI250X)

- Guide: [docs/quantum-espresso-frontier.md](quantum-espresso-frontier.md)
- Build script: [scripts/setup/frontier/build-qe-gpu-frontier.sh](../scripts/setup/frontier/build-qe-gpu-frontier.sh)
- Launcher: [scripts/launchers/frontier/run-pw-gpu-frontier.sh](../scripts/launchers/frontier/run-pw-gpu-frontier.sh)

Highlights:

- OpenMP target offload to gfx90a
- pinned module/toolchain recipe with Frontier-specific workarounds

### Aurora (ALCF, Intel PVC)

- Guide: [docs/quantum-espresso-aurora.md](quantum-espresso-aurora.md)
- Build script: [scripts/setup/aurora/build-qe-gpu-aurora.sh](../scripts/setup/aurora/build-qe-gpu-aurora.sh)
- Launcher: [scripts/launchers/aurora/run-pw-gpu-aurora.sh](../scripts/launchers/aurora/run-pw-gpu-aurora.sh)

Highlights:

- QE CMake GPU flags: `QE_GPU="openmp;oneapi"`, `QE_GPU_ARCHS=intel_gpu_pvc`
- validated repository run with full install to `external/quantum-espresso/install-gpu/`

## VASP by platform

### Aurora (ALCF, Intel PVC)

- Guide: [docs/vasp-aurora.md](vasp-aurora.md)
- Build script: [scripts/setup/aurora/build-vasp-gpu-aurora.sh](../scripts/setup/aurora/build-vasp-gpu-aurora.sh)

Highlights:

- documents the exact VASP 6.6.0 makefile lineage used on Aurora
- records that the build started from `arch/makefile.include.oneapi_omp_off`
- uses one Aurora script invocation to build `vasp_std`, `vasp_gam`, and `vasp_ncl`
- keeps only build provenance in git, not the proprietary VASP source tree

### Perlmutter (NERSC, NVIDIA A100)

- Guide: [docs/quantum-espresso-perlmutter.md](quantum-espresso-perlmutter.md)
- Build scripts overview: [scripts/setup/perlmutter/QE-BUILD-GUIDE.md](../scripts/setup/perlmutter/QE-BUILD-GUIDE.md)
- Setup docs: [scripts/setup/perlmutter/README.md](../scripts/setup/perlmutter/README.md)
- Launcher: [scripts/launchers/perlmutter/run-pw-gpu-perlmutter.sh](../scripts/launchers/perlmutter/run-pw-gpu-perlmutter.sh)

Highlights:

- NVIDIA-focused GPU build flow (A100/sm_80)
- Frontier-style phased setup for reproducibility

## vLLM inference serving by platform

### Aurora (ALCF, Intel PVC)

- Full guide (install, challenges, fixes): [docs/vllm-aurora.md](vllm-aurora.md)
- Stack verification script: [scripts/setup/aurora/install-vllm-xpu-aurora.sh](../scripts/setup/aurora/install-vllm-xpu-aurora.sh)
- Single-node smoke test: [scripts/smoke-tests/aurora/smoke-vllm-singlenode-aurora.sh](../scripts/smoke-tests/aurora/smoke-vllm-singlenode-aurora.sh)
- Multi-node serve job: [scripts/advanced/aurora/job-serve-multinode-vllm-aurora.sh](../scripts/advanced/aurora/job-serve-multinode-vllm-aurora.sh)

Highlights:

- `frameworks/2025.3.1` ships vLLM 0.15.0+xpu + PyTorch 2.10 (XPU) — no source build needed
- SIGSEGV in vLLM's model-registry subprocess fixed via `aurora_vllm_entrypoint.py`
  (sets `ONEAPI_DEVICE_SELECTOR=opencl:cpu` for the registry child process)
- Server must be launched via `mpiexec -n 1 --ppn 1` for PALS device-fabric permissions
- Do **not** pin numpy to 1.x on Aurora — the XPU stack requires numpy 2.2.6 (2.x ABI)
- Smoke test confirmed passing: job 8506887, May 25 2026

### Frontier (OLCF, AMD MI250X)

See the Frontier vLLM smoke test: [scripts/smoke-tests/frontier/smoke-vllm-singlenode-frontier.sh](../scripts/smoke-tests/frontier/smoke-vllm-singlenode-frontier.sh)

## LLM and model-serving docs on HPC

- Model download and resumable background transfer guide:
  [docs/model-download.md](model-download.md)
- Local backend comparison for HPC planning:
  [docs/llm-backends-comparison.md](llm-backends-comparison.md)

## Suggested starting points

- Building QE on your target machine: start with the platform QE guide above.
- Running matsim-agents with local models on HPC: read
  [docs/model-download.md](model-download.md) first, then use your platform
  setup scripts under `scripts/setup/<platform>/`.