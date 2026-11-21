# Running on DOE supercomputers

This guide explains how matsim-agents runs on each facility. For a list of
build scripts and launchers by facility and backend, see the
[HPC platforms index](../hpc-platforms.md).

## Portability across DOE supercomputers

`matsim-agents` is designed to run **the same Python code path on three
DOE leadership-class systems with three very different accelerators**.
All heavy backends (MLIP inference/training with HydraGNN, UMA, or MACE, vLLM model
serving, VASP, and Quantum ESPRESSO) have validated build + launcher
recipes per site, with all platform-specific gotchas (toolchains, MPI
GTL pins, ROCm/Cray cross-builds, CUDA-aware MPI) baked in.

| Capability                          | Frontier (OLCF) MI250X            | Aurora (ALCF) PVC                       | Perlmutter (NERSC) A100         |
|-------------------------------------|-----------------------------------|-----------------------------------------|---------------------------------|
| **Hardware**                        | AMD MI250X (gfx90a), 64-core EPYC | Intel Data Center GPU Max 1550 (PVC)    | NVIDIA A100 (40/80 GB), AMD EPYC|
| **HydraGNN venv**                   | ROCm 7.2.0 + PyTorch              | oneAPI + Intel Extension for PyTorch    | CUDA 12 + PyTorch               |
| **vLLM model server**               | ROCm 7.2.0, source build          | oneAPI                                  | CUDA                            |
| **VASP 6.6**                        | `build-vasp-gpu-frontier.sh`      | `build-vasp-gpu-aurora.sh` (`vasp_std`/`vasp_gam`/`vasp_ncl`) | `build-vasp-gpu-perlmutter.sh` (OpenACC) |
| **Quantum ESPRESSO `pw.x` (GPU)**   | OpenMP target offload to gfx90a   | `QE_GPU="openmp;oneapi"`, PVC arch      | CUDA build                      |
| **Setup entry point**               | `deployments/frontier/setup/install.sh` | `deployments/aurora/setup/install.sh` | `deployments/perlmutter/setup/install.sh` |
| **Active-learning launcher**        | `deployments/frontier/launchers/run-active-learning-frontier.sh` | (file-coupled via SLURM)                | (file-coupled via SLURM)        |
| **Per-platform docs**               | [docs/quantum-espresso-frontier.md](../quantum-espresso-frontier.md) | [docs/quantum-espresso-aurora.md](../quantum-espresso-aurora.md), [docs/vasp-aurora.md](../vasp-aurora.md) | [docs/quantum-espresso-perlmutter.md](../quantum-espresso-perlmutter.md), [docs/vasp-perlmutter.md](../vasp-perlmutter.md) |

Single entry-point index covering all three systems:
[`docs/hpc-platforms.md`](../hpc-platforms.md).

Design principles that keep the code portable:

- **DFT and Python/ML stacks are never co-loaded** in the same shell
  on any platform — each uses its own module set, and the active-
  learning loop couples them through SLURM steps + the filesystem.
  This avoids the pervasive ABI/toolchain conflicts (Cray MPI GTL
  SONAMEs on Frontier, oneAPI vs PyTorch CUDA stack on Perlmutter,
  etc.) that otherwise break shared builds.
- **Backend-agnostic active learning** — the same `matsim-agents al
  run` driver works whether the labeller is VASP or QE, and on any of
  the three platforms, because the DFT backend is selected by a single
  YAML field (`dft.backend: vasp | qe`).
- **Templated YAML configs** — `${VAR}` / `${VAR:-default}` /
  `${VAR:?msg}` substitution lets one config file follow you between
  Frontier scratch, Aurora flare, and Perlmutter pscratch without
  edits.

---

## Running on Frontier (OLCF)

> **⚠️ Frontier users — read this first:**
> See [`deployments/frontier/docs/README-frontier.md`](../../deployments/frontier/docs/README-frontier.md)
> for required setup and known issues. vLLM server jobs use the isolated
> `$REPO/.venv-vllm`; matsim-agents clients and HydraGNN continue to use
> `$REPO/.venv`. A matching prebuilt `tvm_ffi` shared library must exist under
> `$REPO/external/tvm-ffi/lib/` before starting a vLLM server. If it is missing,
> rebuild it with `sbatch deployments/frontier/setup/prebuild-tvm-ffi-frontier.sh`.

### ROCm versions on Frontier

The Frontier-targeted backends do **not** all use the same ROCm version:

| Backend | Module | Why this version |
|---|---|---|
| **HydraGNN environment** | `rocm/7.2.0` | Current Frontier-supported PyTorch + ROCm path; created by HydraGNN's installer at `matsim-agents/.venv/` |
| **vLLM model server** | `rocm/7.2.0` | Isolated Python `venv` at `matsim-agents/.venv-vllm/`; built from source via [`deployments/frontier/setup/build-vllm-rocm72.sh`](../../deployments/frontier/setup/build-vllm-rocm72.sh) |
| **Quantum ESPRESSO GPU** | `rocm/6.2.4` *(forced)* | Frontier's `cray-mpich/8.1.31` GTL `libmpi_gtl_hsa.so` is hard-linked against `libamdhip64.so.6` (rocm 6.x SONAME). rocm/7.x ships `.so.7` and breaks the MPI Fortran link probe at CMake configure. See [`docs/quantum-espresso-frontier.md`](../quantum-espresso-frontier.md). |

QE and the Python/ML stacks are deliberately never co-loaded in the same
shell; they couple through Slurm and the filesystem.

### Quantum ESPRESSO (DFT) backend on Frontier

The repo also ships a fully reproducible recipe for building Quantum
ESPRESSO `develop` with **AMD MI250X (gfx90a) OpenMP target offload**:

- Build script: [`deployments/frontier/setup/build-qe-gpu-frontier.sh`](../../deployments/frontier/setup/build-qe-gpu-frontier.sh)
- Run launcher: [`deployments/frontier/launchers/run-pw-gpu-frontier.sh`](../../deployments/frontier/launchers/run-pw-gpu-frontier.sh)
- Full docs: [`docs/quantum-espresso-frontier.md`](../quantum-espresso-frontier.md)
- Platform index: [`docs/hpc-platforms.md`](../hpc-platforms.md)

The build is cross-compiled on a login node and produces ~92 binaries
(`pw.x`, `cp.x`, `ph.x`, `pp.x`, `neb.x`, `epw.x`, `kcw.x`, `tddfpt`/
`turbo_*` suite, `pioud.x`, `all_currents.x`, …) under
`external/quantum-espresso/install-gpu/bin/` (gitignored). The recipe
includes baked-in workarounds for the cce/18.0.1 ftn-7991 ICE, the
PIOUD `etime()` link error (rewritten to F95 `cpu_time`), the
rocm/7.x cray-mpich SONAME mismatch, and a cce/18.0.1
uninitialized-descriptor segfault in `pw_init_qexsd_input` that (if
unpatched) both crashes `pw.x` and silently drops DFT+U/vdW restart
data — see [`docs/quantum-espresso-frontier.md`](../quantum-espresso-frontier.md#workarounds-baked-into-the-build-script)
for details.

QE uses a different module stack than matsim-agents' Python; the two
are deliberately kept isolated and coupled only through Slurm + files.

### VASP (DFT) backend on Frontier

VASP 6.6 is also wired up on Frontier MI250X for the active-learning
labeller path:

- Build script: [`deployments/frontier/setup/build-vasp-gpu-frontier.sh`](../../deployments/frontier/setup/build-vasp-gpu-frontier.sh)
- In-allocation step launcher (called by the AL loop):
  [`deployments/frontier/launchers/_vasp-step-frontier.sh`](../../deployments/frontier/launchers/_vasp-step-frontier.sh)

As with QE, the proprietary VASP source itself is **not** committed;
only the build recipe is. The repository assumes you have a licensed
VASP source tree under `external/vasp6/`.

---

## Running on Aurora (ALCF)

The repository also includes a validated build/run path for Quantum
ESPRESSO with Intel GPU offload on Aurora.

- Build script: [`deployments/aurora/setup/build-qe-gpu-aurora.sh`](../../deployments/aurora/setup/build-qe-gpu-aurora.sh)
- Run launcher: [`deployments/aurora/launchers/run-pw-gpu-aurora.sh`](../../deployments/aurora/launchers/run-pw-gpu-aurora.sh)
- Full docs: [`docs/quantum-espresso-aurora.md`](../quantum-espresso-aurora.md)
- Platform index: [`docs/hpc-platforms.md`](../hpc-platforms.md)

Validated outcome in this repo:

- successful CMake build + install (exit code 0)
- 106 installed executables in `external/quantum-espresso/install-gpu/bin/`
- core binaries verified: `pw.x`, `cp.x`, `ph.x`, `pp.x`, `epw.x`

Quick run pattern:

```bash
bash deployments/aurora/launchers/run-pw-gpu-aurora.sh path/to/pw.in
```

Aurora QE and the Python/ML environment are intentionally isolated and
typically coupled only via files and scheduler jobs.

For VASP on Aurora, the repository keeps only build provenance, not the vendor
source itself. The recorded makefile lineage is documented in
[`docs/vasp-aurora.md`](../vasp-aurora.md), including the upstream template
used (`arch/makefile.include.oneapi_omp_off`) and the local working makefile
path under `external/vasp6/`. The Aurora build entry point is
[`deployments/aurora/setup/build-vasp-gpu-aurora.sh`](../../deployments/aurora/setup/build-vasp-gpu-aurora.sh),
which defaults to building `vasp_std`, `vasp_gam`, and `vasp_ncl` in one run.

### vLLM on Aurora (Intel PVC)

Aurora supports vLLM-XPU serving and inference via the official ALCF `frameworks` module stack (Python 3.12, torch-xpu, ipex, vllm, ray, triton). The repo provides:

- **Single-node smoke test:** [`deployments/aurora/smoke-tests/smoke-vllm-singlenode-aurora.sh`](../../deployments/aurora/smoke-tests/smoke-vllm-singlenode-aurora.sh)
- **Advanced launchers:** [`deployments/aurora/jobs/job-serve-multinode-vllm-aurora.sh`](../../deployments/aurora/jobs/job-serve-multinode-vllm-aurora.sh) (multi-node Ray serve), plus single-relax, active-learning, and QE warmstart launchers

**Key requirements and gotchas:**

- **PVC visibility:** On Aurora compute nodes, bare `python` does NOT see the GPUs. Always wrap Python in `mpiexec -n 1 --ppn 1` (as in the smoke script) to expose XPUs via PALS.
- **Device mask:** Use `ZE_FLAT_DEVICE_HIERARCHY=FLAT` and a non-dotted `ZE_AFFINITY_MASK` (e.g., `0,1` for TP=2). In FLAT, each tile is a root device; dotted notation (`0.0,0.1`) is only valid in COMPOSITE and will result in `device_count()=0` in FLAT.
- **TMPDIR:** PBS sets `$TMPDIR` to a long path that exceeds the Unix socket limit for ZMQ IPC. Always set `export TMPDIR=/tmp` before launching vLLM.
- **oneCCL KVS:** Do NOT set `CCL_KVS_MODE=mpi` or `CCL_PROCESS_LAUNCHER=pmix` for vLLM. vLLM's multiproc_executor uses forked workers, not MPI ranks; oneCCL must use its default internal KVS over TCP.
- **Debug queue:** The default `debug` queue has a per-user limit of 1 queued job and short walltime. For parallel jobs, use `workq` or `prod`.
- **Model download:** Place models in `$PROJ/models/` (e.g., Mistral-Small-24B-Instruct-2501). Use the provided `hf_download.py` script if needed.

**To run the smoke test:**

1. Build the vLLM XPU venv (if not already):
  ```bash
  bash deployments/aurora/setup/install-vllm-xpu-aurora.sh
  ```
2. Download a supported model (e.g., Mistral-Small-24B):
  ```bash
  source /path/to/matsim-agents/.venv/bin/activate
  python deployments/aurora/setup/hf_download.py mistralai/Mistral-Small-24B-Instruct-2501
  ```
3. Submit the smoke test:
  ```bash
  qsub deployments/aurora/smoke-tests/smoke-vllm-singlenode-aurora.sh
  # or override model:
  qsub -v SMOKE_MODEL_PATH=$PROJ/models/Qwen2.5-32B-Instruct deployments/aurora/smoke-tests/smoke-vllm-singlenode-aurora.sh
  ```
4. Inspect results in `runs/smoke-vllm-singlenode-<jobid>/`.

If the job fails, check `vllm.log` for device mask, TMPDIR, or oneCCL errors. Each error layer is documented in the script comments.

For multi-node serving, see the advanced launchers in `deployments/aurora/jobs/`.

## Running on Perlmutter (NERSC)

Perlmutter (NERSC, NVIDIA A100) is supported as a first-class target
for the Python/ML stack, Quantum ESPRESSO GPU, and VASP GPU.

- Setup overview: [`deployments/perlmutter/setup/README.md`](../../deployments/perlmutter/setup/README.md)
- Matsim environment installer: [`deployments/perlmutter/setup/install.sh`](../../deployments/perlmutter/setup/install.sh)
- QE GPU build: [`deployments/perlmutter/setup/build-qe-gpu-perlmutter.sh`](../../deployments/perlmutter/setup/build-qe-gpu-perlmutter.sh)
  (CPU-only variant: [`build-qe-cpu-perlmutter.sh`](../../deployments/perlmutter/setup/build-qe-cpu-perlmutter.sh))
- QE detailed build guide: [`deployments/perlmutter/setup/QE-BUILD-GUIDE.md`](../../deployments/perlmutter/setup/QE-BUILD-GUIDE.md)
- Full QE docs: [`docs/quantum-espresso-perlmutter.md`](../quantum-espresso-perlmutter.md)
- VASP GPU build and runtime fixes: [`docs/vasp-perlmutter.md`](../vasp-perlmutter.md)
  (build script [`build-vasp-gpu-perlmutter.sh`](../../deployments/perlmutter/setup/build-vasp-gpu-perlmutter.sh))
- Launchers:
  - QE `pw.x` GPU: [`deployments/perlmutter/launchers/run-pw-gpu-perlmutter.sh`](../../deployments/perlmutter/launchers/run-pw-gpu-perlmutter.sh)
  - QE warm-start benchmark: [`deployments/perlmutter/launchers/run-qe-warmstart-benchmark-perlmutter.sh`](../../deployments/perlmutter/launchers/run-qe-warmstart-benchmark-perlmutter.sh)
  - Single-node / multi-node / all-models LLM smoke tests:
    `launch-test-singlenode-resume-perlmutter.sh`,
    `launch-test-multinode-perlmutter.sh`,
    `launch-test-all-models-perlmutter.sh`

Quick run pattern:

```bash
./deployments/perlmutter/launchers/run-pw-gpu-perlmutter.sh path/to/pw.in
```

As on Frontier and Aurora, the DFT module stack and the Python/ML
environment are intentionally isolated and coupled only through Slurm
+ files.

---

## Where to find platform files

[HPC platforms index](../hpc-platforms.md) lists every build script, launcher,
and per-backend guide by facility. [HPC validation](../hpc-validation.md)
records which machine and toolchain combinations have been validated.

---
