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