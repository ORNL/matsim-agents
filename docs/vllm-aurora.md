# vLLM-XPU on Aurora (ALCF Intel PVC)

This document records how to install, run, and troubleshoot vLLM on ALCF
Aurora.  It captures every challenge encountered during bring-up and the exact
fix applied, so future builds are reproducible.

---

## Quick-start checklist

```bash
# 1. Verify the stack (login node — XPU will report unavailable, that is expected)
bash deployments/aurora/setup/install-vllm-xpu-aurora.sh

# 2. Smoke-test on one compute node (TP=2, ~8 min walltime)
qsub deployments/aurora/smoke-tests/smoke-vllm-singlenode-aurora.sh

# 3. Multi-node serve (2 nodes, TP=24, adjust -l select=N as needed)
SERVE_MODEL_PATH=$PROJ/models/Mixtral-8x22B-Instruct-v0.1 \
qsub deployments/aurora/jobs/job-serve-multinode-vllm-aurora.sh
```

---

## Environment

### Module stack

```bash
module reset
module load frameworks   # currently frameworks/2025.3.1
```

`frameworks/2025.3.1` (as of May 2026) ships:

| Package | Version |
|---|---|
| Python | 3.12 |
| PyTorch | 2.10.0a0+git449b176 (XPU backend) |
| Intel Extension for PyTorch (IPEX) | 2.10.10 |
| vLLM | 0.15.0+xpu |
| Ray | 2.43.0 |
| Triton-XPU | (bundled with frameworks) |
| numpy | 2.2.6 |
| pandas | 3.0.0 |

**No vLLM source build is required.**  Everything needed for single- and
multi-node inference is bundled in the module.

### Python virtual environment

matsim-agents uses its `.venv`, built with `--system-site-packages` on top
of the frameworks module's Python 3.12.  This lets the venv inherit the XPU
PyTorch, IPEX, vLLM, Ray, and Triton-XPU from the module while adding
HydraGNN and matsim-agents on top.

Default venv path:

```
$REPO/.venv
```

---

## Known challenges and fixes

### Challenge 1 — SIGSEGV in vLLM's model-registry subprocess (node-specific)

**Symptom**

On some Aurora compute nodes, `vllm serve` crashes immediately with a
`SIGSEGV` (or `SIGABRT`) before loading the model.  The crash happens in a
short-lived child process that vLLM spawns to inspect the model's Python class
attributes (see `vllm/model_executor/models/registry.py:_run_in_subprocess`).
On other nodes the same command works fine.  The behaviour is
compute-node-specific.

**Root cause**

`_run_in_subprocess` spawns `[sys.executable, "-m",
"vllm.model_executor.models.registry"]` via `subprocess.run()` — a plain
`fork+exec` child, **not** an `mpiexec` rank.  On affected nodes, importing
IPEX inside this child process triggers Level Zero GPU device initialisation.
The child lacks the PALS device-fabric permissions that only `mpiexec` ranks
hold on those nodes, causing Level Zero to fault.

**Fix — `aurora_vllm_entrypoint.py`**

`deployments/aurora/smoke-tests/aurora_vllm_entrypoint.py` is a thin wrapper that:

1. Patches `_run_in_subprocess` before `api_server` is imported.
2. Sets `ONEAPI_DEVICE_SELECTOR=opencl:cpu` in the subprocess's environment
   so IPEX/SYCL routes to the CPU OpenCL backend instead of Level Zero.
   The registry subprocess only reads Python class attributes; it never
   executes GPU kernels, so this is safe.
3. On nodes where the plain subprocess already works, setting
   `ONEAPI_DEVICE_SELECTOR=opencl:cpu` in the child is harmless.

The server itself is launched with full Level Zero access because it is an
`mpiexec` rank (see Challenge 2).

**Usage** (already wired into the smoke-test and serve scripts):

```bash
mpiexec -n 1 --ppn 1 \
  env -u PMI_RANK -u PMI_SIZE -u PMI_FD \
      -u PALS_APID -u PALS_RANKID -u PALS_NODEID -u PALS_SPOOL_DIR \
      -u MPI_LOCALRANKID -u MPI_LOCALNRANKS \
      -u OMPI_COMM_WORLD_RANK -u OMPI_COMM_WORLD_SIZE \
  python deployments/aurora/smoke-tests/aurora_vllm_entrypoint.py \
    --model "$MODEL_PATH" ...
```

---

### Challenge 2 — vLLM server must be an mpiexec rank for PALS permissions

**Symptom**

When the vLLM API server process is started with a plain `python` invocation
(not via `mpiexec`), Level Zero initialisation may fail on some nodes because
the process is not registered with the PALS (Parallel Application Launch
Service) device-fabric layer.

**Fix**

Wrap the server launch with `mpiexec -n 1 --ppn 1` to make it a proper PALS
rank, then immediately unset all PMI/PALS/MPI rank-identity variables so vLLM
does not try to treat itself as an MPI job:

```bash
mpiexec -n 1 --ppn 1 \
  env -u PMI_RANK -u PMI_SIZE -u PMI_FD \
      -u PALS_APID -u PALS_RANKID -u PALS_NODEID -u PALS_SPOOL_DIR \
      -u MPI_LOCALRANKID -u MPI_LOCALNRANKS \
      -u OMPI_COMM_WORLD_RANK -u OMPI_COMM_WORLD_SIZE \
  python aurora_vllm_entrypoint.py --model ...
```

---

### Challenge 3 — `ONEAPI_DEVICE_SELECTOR` syntax differs between node driver versions

**Symptom**

Aurora compute nodes run different SYCL runtime versions depending on hardware
provisioning.  The `ONEAPI_DEVICE_SELECTOR` syntax that routes a process to the
CPU OpenCL backend is parsed differently:

| Value | Nodes where it fails | Error |
|---|---|---|
| `cpu` | x4310c4s0b0n0 and similar | `Incomplete selector! Try 'cpu:*'` |
| `cpu:*` | x4502c7s0b0n0 and similar | `Backend is required but missing from 'cpu:*'` |
| `opencl:cpu` | **none observed** | canonical `backend:device_type` form per Intel SYCL spec |

**Fix**

Always use `ONEAPI_DEVICE_SELECTOR=opencl:cpu`.  This is the canonical
`<backend>:<device_type>` form defined in the Intel SYCL specification and is
accepted by all observed Aurora node driver versions.

**Important**: Do **not** override `ONEAPI_DEVICE_SELECTOR` for the vLLM
server process itself.  The `frameworks` module sets it to
`opencl:gpu;level_zero:gpu`, which is required for Triton-XPU and vLLM GPU
execution.  The `opencl:cpu` override applies **only** to the short-lived
registry subprocess via `aurora_vllm_entrypoint.py`.

---

### Challenge 4 — numpy / pandas ABI conflict (do NOT pin 1.x on Aurora)

**Symptom**

After installing matsim-agents dependencies, importing `pyarrow`, `scipy`, or
`sklearn` fails with an ABI error like:

```
numpy.core._multiarray_umath failed to import ... numpy C-extension compiled
against numpy 1.x
```

or the reverse — installing a NumPy 1.x build causes `torch.xpu` or `ipex` to
fail at import because they were
compiled against the numpy 2.x ABI.

**Root cause**

`frameworks/2025.3.1` ships numpy **2.2.6** and pandas **3.0.0**.  Every C
extension in the stack (pyarrow 23.0, scipy 1.17, scikit-learn 1.8, torch
2.10/XPU, IPEX 2.10) is compiled against the **numpy 2.x** ABI.

The current cross-facility HydraGNN contract pins `numpy==2.4.6`. On Aurora,
downgrading that environment to NumPy 1.x would break the XPU stack.

**Fix (in the current HydraGNN Aurora installer)**

Keep the upstream `numpy==2.4.6` venv overlay. It remains on the NumPy 2.x ABI
and is now HydraGNN's qualified cross-facility version. Do not uninstall it to
fall back to the older frameworks copy.

Similarly, remove any venv-local CUDA PyTorch wheels (e.g. pulled in by
`transformers` or `accelerate`):

```bash
pip uninstall -y torch torchvision torchaudio triton \
    nvidia-cublas-cu12 nvidia-cuda-cupti-cu12 nvidia-cuda-nvrtc-cu12 \
    nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 \
    nvidia-curand-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 \
    2>/dev/null || true
```

A venv-local CUDA torch shadows the system-site-packages XPU torch and causes
`torchvision._meta_registrations` import errors and GPU unavailability.

---

### Challenge 5 — `set -eo pipefail` with `[[ -s file ]] && cmd` pattern

**Symptom**

PBS job scripts using `set -eo pipefail` exit silently (exit code 1) when a
pre-flight check file exists but is empty (no crash output).

**Root cause**

```bash
[[ -s "$file" ]] && echo "..." && cat "$file"   # BUG
```

`[[ -s "$file" ]]` returns exit code 1 when the file is 0 bytes (empty = no
crash = good).  Under `set -eo pipefail`, a non-zero exit from any command in
a statement terminates the script immediately.  The empty-file case (which is
the *success* path) kills the script.

**Fix**

Wrap the optional display block so it never propagates a non-zero exit:

```bash
{ [[ -s "$file" ]] && echo "..." && cat "$file"; } || true
```

---

## Smoke test

`deployments/aurora/smoke-tests/smoke-vllm-singlenode-aurora.sh` runs a series of
pre-flight checks and then starts vLLM on one node with TP=2 using
Mistral-Small-24B.

Pre-flight checks (in order):

| Tag | What it tests |
|---|---|
| `[pmi]` | Registry subprocess with PMI vars present (mimics a raw mpiexec rank) |
| `[nopmi]` | Registry subprocess with PMI vars unset (mimics the cleaned-up server env) |
| `[noipex]` | Registry subprocess with IPEX import blocked (`IPEX_IMPORT_ERROR=1`) |
| `[plain]` | Plain `subprocess.run()` with no selector override — this is what vLLM does without the patch; fails on affected nodes |
| `[cpudev]` | Plain `subprocess.run()` with `ONEAPI_DEVICE_SELECTOR=opencl:cpu` — this is the patched behaviour |

All five must pass before the server is started.  If `[plain]` fails and
`[cpudev]` passes, `aurora_vllm_entrypoint.py` is the required launch path.
If `[plain]` passes on the allocated node, the patch is still applied (it is
safe and harmless).

**Confirmed passing**: job `8506887` on Aurora, May 25 2026.
Chat-completion response to `"2+2="` → `"4"`.

---

## Multi-node serve

`deployments/aurora/jobs/job-serve-multinode-vllm-aurora.sh` bootstraps a Ray
cluster across all allocated nodes and starts vLLM with tensor parallelism
spanning every PVC tile:

- 6 PVC GPUs × 2 tiles = **12 ranks per node**
- Default TP = `NNODES × 12`

The vLLM server on the head node is launched with the same `mpiexec -n 1 --ppn 1`
+ `aurora_vllm_entrypoint.py` pattern from Challenge 1 above (already wired
into the script).

Ray workers are started on the remaining nodes via `mpiexec --hosts $node`.

### Environment variables for the client

After the server is ready the script prints:

```
export MATSIM_LLM_PROVIDER=vllm
export MATSIM_VLLM_BASE_URL=http://<head_node_ip>:8000/v1
export MATSIM_VLLM_API_KEY=EMPTY
```

Set these in the matsim-agents client job before running workflows.

---

## Relevant files

| File | Purpose |
|---|---|
| `deployments/aurora/setup/install-vllm-xpu-aurora.sh` | Stack verification (login node) |
| `deployments/aurora/smoke-tests/smoke-vllm-singlenode-aurora.sh` | Single-node PBS smoke test |
| `deployments/aurora/smoke-tests/aurora_vllm_entrypoint.py` | Registry subprocess patch + API server launcher |
| `deployments/aurora/jobs/job-serve-multinode-vllm-aurora.sh` | Multi-node Ray + vLLM serve job |
