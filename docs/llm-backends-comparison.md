# LLM inference backends: vLLM vs HuggingFace Transformers

matsim-agents can serve a local LLM through two open-source backends.
This page compares them so you can choose the right one for your workflow.

---

## Quick summary

| | **vLLM** | **HuggingFace Transformers + Accelerate** |
|---|---|---|
| **Best for** | Production serving, multi-agent workloads | Validation, debugging, research |
| **Throughput** | Very high (continuous batching) | Low (one request at a time) |
| **Setup complexity** | High on ROCm (source build required) | Low (pure Python wheel) |
| **API** | OpenAI-compatible REST server | Python library only |
| **ROCm support** | ROCm wheel available (cp312) or build from source | Works out of the box |

---

## vLLM

### How it works

vLLM is a purpose-built LLM inference server. Its core innovation is
**PagedAttention**: instead of pre-allocating a fixed KV-cache block per
sequence, it manages cache memory in pages (like OS virtual memory). This
eliminates fragmentation and waste, allowing hundreds of concurrent requests
to share GPU memory efficiently.

On top of that, vLLM uses **continuous batching**: the server dynamically
fills each forward pass with a mix of requests at different generation steps,
rather than waiting for a full batch to be ready. The result is near-peak GPU
utilisation at all times.

### Pros

- **High throughput** — continuous batching keeps the GPU busy; throughput
  scales roughly linearly with the number of concurrent requests.
- **Low memory waste** — PagedAttention means a 72B model can serve many more
  simultaneous sessions than a naive implementation.
- **OpenAI-compatible REST API** — matsim-agents, LangChain, and any other
  OpenAI client talk to vLLM without modification.
- **Optimised HIP/CUDA kernels** — fused ops (`silu_and_mul`, flash-attention,
  etc.) are compiled for the target architecture (e.g. `gfx90a` on MI250X).
- **Built-in tensor parallelism** — `--tensor-parallel-size 8` splits a 72B
  model across all 8 GCDs with a single flag.
- **Quantisation support** — AWQ, GPTQ, FP8 reduce memory footprint without
  retraining.

### Cons

- **Complex ROCm setup** — no pre-built Python 3.11 wheel; must be compiled
  from source on a compute node with ROCm headers and `hipcc`. On Frontier
  this takes ~1–2 hours.
- **Heavyweight startup** — loads all model shards, warms up kernels, and
  compiles CUDA graphs before serving the first request (~2–5 min for 72B).
- **Less hackable** — the inference path is mostly in compiled C++/HIP; custom
  forward-pass modifications require forking the project.

### When to use it

Use vLLM for any workload where multiple agents (or users) fire requests
concurrently — e.g. a matsim-agents run with many hypothesis-generation steps
happening in parallel. The throughput advantage over Transformers grows with
the number of simultaneous requests.

---

## HuggingFace Transformers + Accelerate

### How it works

`AutoModelForCausalLM.from_pretrained(..., device_map="auto")` loads the
model weights and uses Accelerate's **naive pipeline parallelism** to spread
layers across available GPUs. Generation is done in pure PyTorch with no
special memory management — each call processes one prompt at a time.

### Pros

- **Zero build required** — both packages are pure Python wheels; `pip install
  accelerate` is all you need.
- **Works without a separate inference-server build** — Transformers uses the
  PyTorch 2.14 environment installed by the current HydraGNN recipe. Frontier
  accelerator execution must be qualified against the selected ROCm module.
- **Full model flexibility** — you can inspect activations, attach hooks,
  modify the forward pass, or inject custom layers with a few lines of Python.
- **Supports every architecture on day 0** — new models land in Transformers
  before any inference server.
- **Ideal for fine-tuning and debugging** — standard entry point for PEFT,
  LoRA, and custom training loops.

### Cons

- **Eager batching only** — one prompt is processed at a time; concurrent
  requests queue up instead of being interleaved.
- **5–10× lower throughput** than vLLM under multi-request load.
- **`device_map="auto"` is not optimised tensor parallelism** — it places
  whole layers on individual GPUs (pipeline parallel), so inter-GPU bandwidth
  is the bottleneck for large models rather than compute.
- **KV cache is not paged** — long contexts can OOM even when total GPU memory
  is sufficient, because a single large allocation is required upfront.
- **No built-in server** — you must wrap the model yourself to expose an API.

### When to use it

Use Transformers + Accelerate to verify that the GPU stack, model weights, and
Python environment are all working correctly — before investing in a vLLM
source build. It is also the right choice for one-off inference, interactive
debugging, and any research workflow that needs to inspect or modify model
internals.

---

## Choosing for Frontier (OLCF, ROCm 7.2)

| Scenario | Recommended backend |
|---|---|
| Validate GPU stack and model weights quickly | **Transformers + Accelerate** (`deployments/frontier/smoke-tests/smoke-transformers-frontier.sh`) |
| Run matsim-agents with many concurrent agent steps | **vLLM** (`deployments/frontier/smoke-tests/smoke-vllm-singlenode-frontier.sh`) |
| Fine-tune or debug model internals | **Transformers + Accelerate** |
| Production multi-node serving | **vLLM** (`deployments/frontier/smoke-tests/smoke-vllm-multinode-frontier.sh`) |

The smoke-test scripts cover both cases:

```bash
# HuggingFace Transformers (works today, no build needed)
sbatch deployments/frontier/smoke-tests/smoke-transformers-frontier.sh

# vLLM (requires source build first)
sbatch deployments/frontier/setup/build-vllm-rocm72.sh                 # one-time build
sbatch deployments/frontier/smoke-tests/smoke-vllm-singlenode-frontier.sh   # then serve
```

---

## Choosing for Perlmutter (NERSC, NVIDIA A100)

The Perlmutter matsim-owned `.venv` ships with `transformers` + `accelerate` but
**no vLLM and no DeepSpeed**, so all bundled smoke jobs use the
HuggingFace `transformers` backend. Multi-node sharding is done with
`transformers`' built-in `tp_plan="auto"` tensor-parallel planner over
`torch.distributed` (NCCL on Slingshot, one rank per GPU under `srun` —
no `torchrun` agent).

| Scenario | Recommended backend |
|---|---|
| Validate the GPU stack on a single node | **Transformers + Accelerate** ([deployments/perlmutter/smoke-tests/smoke-transformers-perlmutter.sh](../deployments/perlmutter/smoke-tests/smoke-transformers-perlmutter.sh)) |
| Sweep all locally-cached models | [deployments/perlmutter/launchers/launch-test-all-models-perlmutter.sh](../deployments/perlmutter/launchers/launch-test-all-models-perlmutter.sh) |
| Multi-node TP (Qwen2.5-72B, Mixtral-8x22B) | **Transformers `tp_plan="auto"`** ([deployments/perlmutter/smoke-tests/smoke-transformers-multinode-perlmutter.sh](../deployments/perlmutter/smoke-tests/smoke-transformers-multinode-perlmutter.sh)) |
| LLM + HydraGNN + QE end-to-end | [deployments/perlmutter/jobs/job-discovery-chat-perlmutter.sh](../deployments/perlmutter/jobs/job-discovery-chat-perlmutter.sh) |

```bash
# Single-node smoke (defaults to Qwen2.5-72B; override via MATSIM_MODEL_DIR)
sbatch deployments/perlmutter/smoke-tests/smoke-transformers-perlmutter.sh

# Multi-node TP smoke (2 nodes × 4 A100s = 8 ranks, NCCL + tp_plan="auto")
sbatch deployments/perlmutter/smoke-tests/smoke-transformers-multinode-perlmutter.sh

# End-to-end discovery validation
sbatch deployments/perlmutter/jobs/job-discovery-chat-perlmutter.sh
```

---

## Related docs

- [docs/model-download.md](model-download.md) — downloading Qwen2.5-72B-Instruct
- [deployments/frontier/smoke-tests/smoke-transformers-frontier.sh](../deployments/frontier/smoke-tests/smoke-transformers-frontier.sh)
- [deployments/frontier/smoke-tests/smoke-vllm-singlenode-frontier.sh](../deployments/frontier/smoke-tests/smoke-vllm-singlenode-frontier.sh)
- [deployments/perlmutter/smoke-tests/smoke-transformers-perlmutter.sh](../deployments/perlmutter/smoke-tests/smoke-transformers-perlmutter.sh)
- [deployments/perlmutter/smoke-tests/smoke-transformers-multinode-perlmutter.sh](../deployments/perlmutter/smoke-tests/smoke-transformers-multinode-perlmutter.sh)
- [deployments/perlmutter/setup/README.md](../deployments/perlmutter/setup/README.md)
