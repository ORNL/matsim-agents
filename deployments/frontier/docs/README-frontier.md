# Running matsim-agents on Frontier (OLCF)

This document covers Frontier-specific setup, gotchas, and troubleshooting
for the vLLM-based agent backend.

> **Note on paths**: throughout this document, `$PROJ` refers to your
> project's proj-shared directory on Lustre (e.g.
> your project's shared filesystem). Set it once with
> `export PROJ=/path/to/project-shared` and all examples
> below will work as written.

## Frontier scripts directory layout

Frontier-specific assets are split by intent:

| Subdir | Purpose |
|---|---|
| `deployments/frontier/setup/` | Login-node installers and source builds (vLLM, Quantum ESPRESSO, tvm-ffi) |
| `deployments/frontier/launchers/` | sbatch-able launchers (vLLM model-server tests, `pw.x` GPU runner) |
| `deployments/frontier/smoke-tests/` | One-shot validation jobs (single-node, multi-node, HF fallback) |
| `deployments/frontier/docs/` | This document and the six-model benchmark notes |

**Other backends documented separately** (also Frontier-targeted):
- Quantum ESPRESSO `develop` with AMD MI250X (gfx90a) OpenMP target offload — see [`docs/quantum-espresso-frontier.md`](../../../docs/quantum-espresso-frontier.md). Build script: [`deployments/frontier/setup/build-qe-gpu-frontier.sh`](../setup/build-qe-gpu-frontier.sh); launcher: [`deployments/frontier/launchers/run-pw-gpu-frontier.sh`](../launchers/run-pw-gpu-frontier.sh).

---

## Prebuilt `tvm_ffi` artifact

The Torch DLPack addon is installed under
`<repo>/external/tvm-ffi/lib/` on project-shared Lustre. Its filename includes
the full Torch version, ROCm backend, and C++ ABI hash. Every Frontier job that
starts a vLLM server resolves and validates the artifact against `.venv-vllm`
before starting the server. Client-only jobs that call an existing endpoint do
not need the artifact.

### Symptoms

- `vllm.log` stays at 0 bytes for 10+ minutes (or the entire job time limit).
- Job stdout shows the script reaching `[vllm] Starting server ...` then nothing.
- `~/.cache/tvm-ffi/` contains a `*.so.lock` file but **no `.so` file**.

### Root cause

`vllm.entrypoints.openai.api_server` imports `xgrammar`, which imports `tvm_ffi`.
At import time, `tvm_ffi/_optional_torch_c_dlpack.py` calls `subprocess.run()`
to JIT-compile a small torch C++/HIP extension if its `.so` is missing.
On Frontier compute nodes (under srun's GPU binding), this build subprocess
hangs, and the parent Python waits forever in `selectors.select()`.

If a previous build was interrupted (Ctrl-C, OOM kill, job timeout), it leaves
a 0-byte `.so.lock` file and **no `.so`**, triggering this hang on every
subsequent run.

### How to recover

```bash
cd $PROJ/matsim-agents
sbatch deployments/frontier/setup/prebuild-tvm-ffi-frontier.sh    # rebuilds the .so
```

After it completes, verify:

```bash
ls -la external/tvm-ffi/lib/
# Expected: libtorch_c_dlpack_addon_torch<major><minor>-rocm-<abi>.so
```

### How it's prevented

Every job that starts a vLLM server sources the shared resolver and requires
the installed artifact, keyed to the isolated vLLM environment:

```bash
source "$REPO/deployments/frontier/setup/tvm-ffi-artifact.sh"
require_tvm_ffi_artifact "$REPO" "$VLLM_VENV"
```

`TVM_FFI_CACHE_DIR` is set internally because that is the variable name required
by tvm-ffi; it points to the durable `external/tvm-ffi/lib` install directory,
not an expendable cache.

**Do not remove these lines from any vLLM job script.**

---

## Python environments

Frontier keeps two Python environments separate:

- `$REPO/.venv` is HydraGNN's Conda environment. Use it for matsim-agents
  clients, HydraGNN inference/training, and jobs that call a vLLM endpoint.
- `$REPO/.venv-vllm` is a standard Python `venv` containing the ROCm 7.2 vLLM
  server stack. Use it for vLLM, Ray, and the matching `tvm_ffi` artifact.

Server scripts default to `.venv-vllm`; set `MATSIM_VLLM_VENV` to override it.
The `tvm_ffi` prebuild defaults to `.venv-vllm` and accepts `VLLM_VENV_PATH`.
Do not install vLLM dependencies into or replace the HydraGNN `.venv`.

## Required environment

All Frontier scripts source `frontier-module-stack.sh` which loads:

- `cpe/24.07`
- `rocm/7.2.0`
- `amd-mixed/7.2.0`
- `PrgEnv-gnu`
- `miniforge3/23.11.0-0`

The vLLM server jobs activate the isolated vLLM-on-ROCm-7.2 environment at:
`$REPO/.venv-vllm`. Client-side matsim-agents commands continue to use the
HydraGNN Conda environment at `$REPO/.venv`.

## Network policy

Compute nodes have **no outbound internet**. All scripts set:

- `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`
- `VLLM_NO_USAGE_STATS=1`, `DO_NOT_TRACK=1`
- `RAY_USAGE_STATS_ENABLED=0`, `RAY_ADDRESS=""`
- `TRITON_DISABLE_AUTOTUNE_CACHE=1`
- `unset http_proxy https_proxy ...; export NO_PROXY='*'`

Models must be pre-downloaded to `$PROJ/models/`
on the login node before submitting jobs.

## Required vLLM env vars (already set by scripts)

| Variable | Value | Why |
|---|---|---|
| `HSA_NO_SCRATCH_RECLAIM` | `1` | Prevents RCCL init crash on Frontier |
| `VLLM_NCCL_SO_PATH` | `/opt/rocm-7.2.0/lib/librccl.so.1` | Use system RCCL, not torch-bundled |
| `VLLM_CUDART_SO_PATH` | `/opt/rocm-7.2.0/lib/libamdhip64.so` | flashinfer cudart fallback |
| `PYTORCH_ROCM_ARCH` / `ROCM_ARCH` | `gfx90a` | Correct GPU arch (MI250X) |
| `MIOPEN_DISABLE_CACHE` | `1` | Avoid stale gfx900 kernels |
| `VLLM_CACHE_ROOT` | `$RUN_DIR/vllm-cache` | Per-run, wiped fresh |
| `NCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME` | `hsn` | Slingshot interconnect |
| `FI_CXI_ATS` | `0` | Disable address translation services |

## Launch command pattern

vLLM must be launched via `srun` (not bare `python`) for proper GPU binding:

```bash
srun -N1 -n1 -c56 --gpus-per-task=${GPUS_PER_NODE} --gpu-bind=closest \
  "$VLLM_VENV/bin/python" -m vllm.entrypoints.openai.api_server ...
```

Bare `python` will not see GPUs correctly even with `ROCR_VISIBLE_DEVICES`.

## Smoke tests

| Script | Purpose |
|---|---|
| `deployments/frontier/smoke-tests/smoke-vllm-singlenode-frontier.sh` | 1 node, configurable TP (default 8). **Run this first.** |
| `deployments/frontier/smoke-tests/smoke-vllm-multinode-frontier.sh` | 4 nodes, TP=32 via Ray, for DeepSeek-V4-Pro |
| `deployments/frontier/smoke-tests/smoke-transformers-frontier.sh` | HF Transformers + Accelerate fallback |

Submit with:

```bash
sbatch --nodes=1 deployments/frontier/smoke-tests/smoke-vllm-singlenode-frontier.sh
# or for TP=1 quick test:
GPUS_PER_NODE=1 sbatch --nodes=1 deployments/frontier/smoke-tests/smoke-vllm-singlenode-frontier.sh
```

## Common debugging commands

```bash
# Watch the active job in real time
tail -f $PROJ/runs/smoke-singlenode-<JOBID>/job-<JOBID>.out

# Watch vLLM server log
tail -f $PROJ/runs/smoke-singlenode-<JOBID>/vllm.log

# Check job status
squeue -u $USER

# Cancel a hung job
scancel <JOBID>
```

## When vLLM hangs with no output

1. Check that `external/tvm-ffi/lib/` contains a nonempty ABI-keyed `.so`.
2. If it is missing, run `prebuild-tvm-ffi-frontier.sh` on a compute node.
3. If it exists, use the faulthandler stack trace technique below.

### Generic Python hang stack trace

Add this to a script to dump all-thread stack traces after 45 seconds:

```bash
timeout --signal=ABRT --kill-after=10 60 srun ... \
  "$VLLM_VENV/bin/python" -u -X faulthandler -c "
import faulthandler, sys
faulthandler.dump_traceback_later(45, repeat=False, file=sys.stderr)
import vllm.entrypoints.openai.api_server  # or whatever import is suspected
"
```

This is how the `tvm_ffi` hang was originally diagnosed (May 6, 2026).
