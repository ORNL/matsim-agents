# Downloading LLM model weights for vLLM

The vLLM backend requires the model weights to be present on local (or
parallel) filesystem before starting the server. This page covers the
recommended model (`Qwen/Qwen2.5-72B-Instruct`) and the download
procedure on a workstation and on Frontier (OLCF).

---

## Prerequisites

`huggingface_hub >= 1.12` is already installed into the matsim-agents
environment (it is a transitive dependency). Starting from version 1.12
the `huggingface-cli` command was renamed to `hf`.

Verify your installation:
```bash
hf --version        # should print 1.12.x or newer
```

If it is missing or too old, install/upgrade it:
```bash
pip install -U "huggingface_hub"
```

> **Note**: if you see `huggingface-cli: error: invalid choice: 'download'`
> your installed version is ≤ 0.4.x (e.g. a system or user-local install
> that predates the conda env). Always use the `hf` binary from inside the
> activated conda/virtual environment.

---

## Recommended model

| Model | HuggingFace ID | Size | Notes |
|---|---|---|---|
| Qwen 2.5 72B Instruct | `Qwen/Qwen2.5-72B-Instruct` | ~150 GB (37 safetensor shards) | Default for matsim-agents on HPC. Requires ≥2 AMD MI250X GPUs with vLLM. |
| Qwen 2.5 14B Instruct | `Qwen/Qwen2.5-14B-Instruct` | ~28 GB | Fits on a single GPU; suitable for testing. |

---

## Workstation (interactive, foreground)

```bash
# Activate the matsim-agents env first
source .venv/bin/activate   # or: conda activate <env>

mkdir -p /path/to/models
hf download Qwen/Qwen2.5-72B-Instruct \
    --local-dir /path/to/models/Qwen2.5-72B-Instruct
```

The download is **resumable**: if interrupted, re-run the same command
and `hf` will skip already-complete shards.

---

## Frontier (OLCF) — background download from a login node

Frontier login nodes have outbound internet access but interactive
sessions time out. Use `nohup` to keep the download running in the
background.

### Step 1 — load miniforge and the matsim-agents conda env

```bash
ml miniforge3/23.11.0-0
```

### Step 2 — start the download as a background job

Use `conda run` to invoke `hf` inside the environment without needing
to activate it (activation in subshells can fail on Frontier because
`module` commands are not available there):

```bash
VENV=/lustre/orion/<project>/proj-shared/HydraGNN/installation_DOE_supercomputers/HydraGNN-Installation-Frontier-ROCm72/hydragnn_venv_rocm72
MODEL_DIR=/lustre/orion/<project>/proj-shared/models/Qwen2.5-72B-Instruct
LOG=/lustre/orion/<project>/proj-shared/models/qwen_download.log

mkdir -p "$MODEL_DIR"
nohup conda run -p "$VENV" hf download Qwen/Qwen2.5-72B-Instruct \
    --local-dir "$MODEL_DIR" \
    > "$LOG" 2>&1 &

echo "Download PID: $!"
```

> **Why `conda run -p <venv>` instead of activating?**  
> `source activate` and `conda activate` both rely on shell functions
> that are set up by `conda init`. On Frontier login nodes these
> functions are available in interactive shells, but not in subshells
> created by `nohup &`. `conda run -p <path>` side-steps this by
> running the command directly inside the environment's prefix without
> modifying the calling shell.

### Step 3 — monitor progress

```bash
# Disk usage (grows as shards arrive — expect ~150 GB total)
du -sh "$MODEL_DIR"

# Number of completed .safetensors shards (37 when done)
ls "$MODEL_DIR"/*.safetensors 2>/dev/null | wc -l

# Last lines of the log (shows "✓ Downloaded" when complete)
tail -5 "$LOG"

# Confirm the process is still running
ps -p <PID>
```

### Step 4 — verify completion

When finished the log ends with:
```
✓ Downloaded
  path: /lustre/orion/<project>/proj-shared/models/Qwen2.5-72B-Instruct
```
And the directory contains 37 `.safetensors` files (~133 GB on disk after
deduplication of HuggingFace cache blobs).

---

## Downloading other models

Replace `Qwen/Qwen2.5-72B-Instruct` with any public HuggingFace model ID:

```bash
nohup conda run -p "$VENV" hf download \
    meta-llama/Llama-3.1-8B-Instruct \
    --local-dir /lustre/orion/<project>/proj-shared/models/Llama-3.1-8B-Instruct \
    > /lustre/orion/<project>/proj-shared/models/llama_download.log 2>&1 &
```

For **gated models** (e.g. Llama 3 family) you must first accept the
license on HuggingFace and log in:
```bash
conda run -p "$VENV" hf auth login
# enter your HuggingFace token when prompted
```

---

## Starting the vLLM server

Once the weights are downloaded, start the server in a compute job or
interactive allocation. Example for Frontier (single node, 8 GPUs):

```bash
#!/bin/bash
#SBATCH -N 1 -t 02:00:00 -A <project> -p batch

module reset
ml rocm/7.2.0 amd-mixed/7.2.0 PrgEnv-gnu miniforge3/23.11.0-0
module unload darshan-runtime
conda activate /lustre/orion/<project>/proj-shared/HydraGNN/installation_DOE_supercomputers/HydraGNN-Installation-Frontier-ROCm72/hydragnn_venv_rocm72

pip install vllm  # if not already installed

vllm serve Qwen/Qwen2.5-72B-Instruct \
    --model /lustre/orion/<project>/proj-shared/models/Qwen2.5-72B-Instruct \
    --tensor-parallel-size 8 \
    --port 8000 \
    --host 0.0.0.0
```

Then point matsim-agents at it:
```bash
export MATSIM_LLM_PROVIDER=vllm
export MATSIM_VLLM_BASE_URL=http://<compute-node-hostname>:8000/v1
export MATSIM_LLM_MODEL=Qwen/Qwen2.5-72B-Instruct
matsim-agents chat ...
```

---

## UMA MLIP weights on Perlmutter (prefetch **required**)

The active-learning and warm-start jobs use the UMA foundation MLIP
(`facebook/UMA`, e.g. `uma-s-1p1`) via `fairchem-core`. On Perlmutter these
weights **must be pre-fetched before running any compute job** — a compute
job will *not* download them itself. There are two independent reasons:

1. **No outbound internet on compute nodes.** Perlmutter compute nodes cannot
   reach `huggingface.co`, so a lazy first-use download is impossible there.
2. **CFS does not support file locking on compute nodes.** The project
   filesystem (CFS, `/global/cfs/...`) is mounted over DVS on compute nodes,
   which does not implement `fcntl.flock`. `huggingface_hub` takes a per-file
   lock while writing to its cache, so a download that targets CFS from a
   compute node fails immediately with `OSError: [Errno 524] Unknown error 524`.

Because of this, the compute-side jobs read the cache in **offline mode**
(`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`), which resolves the cached path
directly and never takes a lock.

> ⚠️ **Caveat — the cache must be prefetched first.** Because reads are
> offline, if a model is **not** already in the cache the job fails fast with a
> "not found locally" / offline error instead of downloading it. This is the
> intended trade-off on compute nodes (which have no internet anyway). Run the
> prefetch step below once per model before submitting AL / warm-start jobs.

### Step 0 — one-time Hugging Face auth (gated repo)

`facebook/UMA` is gated. Accept the license at
<https://huggingface.co/facebook/UMA>, then log in once on a login node:
```bash
source $PROJ/HydraGNN/installation_DOE_supercomputers/HydraGNN-Installation-Perlmutter/fairchem_venv/bin/activate
hf auth login          # paste your hf_... token; writes ~/.cache/huggingface/token
hf auth whoami         # should print your username
```

### Step 1 — prefetch the weights (CPU job)

```bash
sbatch deployments/perlmutter/download/download-uma-perlmutter.sh
# multiple / alternate models:
UMA_MODELS="uma-s-1p1 uma-m-1p1" sbatch deployments/perlmutter/download/download-uma-perlmutter.sh
```

The download job **stages** the cache on a flock-capable filesystem
(`$SCRATCH` Lustre, else node-local `/tmp`), then copies the finished cache
into the persistent shared location on CFS
(`$PROJ/models/hf_cache`, override with `HF_HOME=...`). This is what side-steps
the errno-524 lock failure while still leaving the weights on CFS for reuse.

### Step 2 — verify the cache

```bash
ls $PROJ/models/hf_cache/hub/models--facebook--UMA
# should list blobs/, refs/, snapshots/ once the prefetch succeeded
```

### Step 3 — run compute jobs (offline reads)

The following jobs already set `HF_HOME=$PROJ/models/hf_cache` and
`HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`, so they read the prefetched
cache without any download or lock:

- `deployments/perlmutter/jobs/job-active-learning-paper-cases-perlmutter.sh`
- `deployments/perlmutter/jobs/job-uma-warmstart-perlmutter.sh`
- `deployments/perlmutter/jobs/job-uma-vasp-warmstart-perlmutter.sh`

If you point them at a different `HF_HOME`, prefetch into that directory first.
