# Running matsim-agents on Frontier (OLCF)

This document covers Frontier-specific setup, gotchas, and troubleshooting
for the vLLM-based agent backend.

---

## ⚠️ CRITICAL: prebuilt `tvm_ffi` `.so` MUST exist before launching ANY vLLM job

**TL;DR**: If the file
`/lustre/orion/mat746/proj-shared/cache/tvm-ffi/libtorch_c_dlpack_addon_torch211-rocm.so`
is missing or 0 bytes, **every vLLM job will silently hang forever** with no log output.

### Symptoms

- `vllm.log` stays at 0 bytes for 10+ minutes (or the entire job time limit).
- Job stdout shows the script reaching `[vllm] Starting server ...` then nothing.
- `~/.cache/tvm-ffi/` contains a `*.so.lock` file but **no `.so` file**.

### Root cause

`vllm.entrypoints.openai.api_server` imports `xgrammar`, which imports `tvm_ffi`.
At import time, `tvm_ffi/_optional_torch_c_dlpack.py:158` calls `subprocess.run()`
to JIT-compile a small torch C++/HIP extension if its cached `.so` is missing.
On Frontier compute nodes (under srun's GPU binding), this build subprocess
hangs, and the parent Python waits forever in `selectors.select()`.

If a previous build was interrupted (Ctrl-C, OOM kill, job timeout), it leaves
a 0-byte `.so.lock` file and **no `.so`**, triggering this hang on every
subsequent run.

### How to recover

```bash
cd /lustre/orion/mat746/proj-shared/matsim-agents
sbatch scripts/frontier/prebuild-tvm-ffi-frontier.sh    # rebuilds the .so
```

After it completes, verify:

```bash
ls -la /lustre/orion/mat746/proj-shared/cache/tvm-ffi/
# Expected: libtorch_c_dlpack_addon_torch211-rocm.so   (~200 KB)
```

### How it's prevented

Every vLLM job script in this directory has these two lines near the top:

```bash
export TVM_FFI_CACHE_DIR=$PROJ/cache/tvm-ffi
TVM_FFI_SO=$TVM_FFI_CACHE_DIR/libtorch_c_dlpack_addon_torch211-rocm.so
[[ ! -s "$TVM_FFI_SO" ]] && { echo "[FAIL] missing $TVM_FFI_SO"; exit 1; }
rm -f ~/.cache/tvm-ffi/*.lock 2>/dev/null || true
```

This causes jobs to **fail in 2 seconds with a clear error message** if the
`.so` is gone, instead of silently hanging for the whole job time limit.

**Do not remove these lines from any vLLM job script.**

---

## Required environment

All Frontier scripts source `frontier-module-stack.sh` which loads:

- `cpe/24.07`
- `rocm/7.2.0`
- `amd-mixed/7.2.0`
- `PrgEnv-gnu`
- `miniforge3/23.11.0-0`

Then activates the vLLM-on-ROCm-7.2 venv at:
`/lustre/orion/mat746/proj-shared/HydraGNN/installation_DOE_supercomputers/HydraGNN-Installation-Frontier-ROCm72/hydragnn_venv_rocm72`

## Network policy

Compute nodes have **no outbound internet**. All scripts set:

- `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`
- `VLLM_NO_USAGE_STATS=1`, `DO_NOT_TRACK=1`
- `RAY_USAGE_STATS_ENABLED=0`, `RAY_ADDRESS=""`
- `TRITON_DISABLE_AUTOTUNE_CACHE=1`
- `unset http_proxy https_proxy ...; export NO_PROXY='*'`

Models must be pre-downloaded to `/lustre/orion/mat746/proj-shared/models/`
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
  "$VENV/bin/python" -m vllm.entrypoints.openai.api_server ...
```

Bare `python` will not see GPUs correctly even with `ROCR_VISIBLE_DEVICES`.

## Smoke tests

| Script | Purpose |
|---|---|
| `smoke-vllm-singlenode-frontier.sh` | 1 node, configurable TP (default 8). **Run this first.** |
| `smoke-vllm-multinode-frontier.sh` | 4 nodes, TP=32 via Ray, for DeepSeek-V4-Pro |

Submit with:

```bash
sbatch --nodes=1 scripts/frontier/smoke-vllm-singlenode-frontier.sh
# or for TP=1 quick test:
GPUS_PER_NODE=1 sbatch --nodes=1 scripts/frontier/smoke-vllm-singlenode-frontier.sh
```

## Common debugging commands

```bash
# Watch the active job in real time
tail -f /lustre/orion/mat746/proj-shared/runs/smoke-singlenode-<JOBID>/job-<JOBID>.out

# Watch vLLM server log
tail -f /lustre/orion/mat746/proj-shared/runs/smoke-singlenode-<JOBID>/vllm.log

# Check job status
squeue -u $USER

# Cancel a hung job
scancel <JOBID>
```

## When vLLM hangs with no output

1. **First check**: is `~/.cache/tvm-ffi/libtorch_c_dlpack_addon_torch211-rocm.so` present and non-empty?
2. **Second check**: is `/lustre/orion/mat746/proj-shared/cache/tvm-ffi/libtorch_c_dlpack_addon_torch211-rocm.so` present and non-empty?
3. If the proj-shared `.so` is missing → run `prebuild-tvm-ffi-frontier.sh`.
4. If both `.so` files exist → look at the faulthandler stack trace technique below.

### Generic Python hang stack trace

Add this to a script to dump all-thread stack traces after 45 seconds:

```bash
timeout --signal=ABRT --kill-after=10 60 srun ... \
  "$VENV/bin/python" -u -X faulthandler -c "
import faulthandler, sys
faulthandler.dump_traceback_later(45, repeat=False, file=sys.stderr)
import vllm.entrypoints.openai.api_server  # or whatever import is suspected
"
```

This is how the `tvm_ffi` hang was originally diagnosed (May 6, 2026).
