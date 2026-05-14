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
VENV=/lustre/orion/<project>/proj-shared/HydraGNN/installation_DOE_supercomputers/HydraGNN-Installation-Frontier/hydragnn_venv
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
ml rocm/7.1.1 amd-mixed/7.1.1 PrgEnv-gnu miniforge3/23.11.0-0
module unload darshan-runtime
conda activate /lustre/orion/<project>/proj-shared/HydraGNN/installation_DOE_supercomputers/HydraGNN-Installation-Frontier/hydragnn_venv

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
