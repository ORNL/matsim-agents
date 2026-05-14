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
- **Works immediately on ROCm** — no source compilation; validated on Frontier
  with `torch 2.11.0+rocm7.1` and 8 MI250X GCDs.
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

## Choosing for Frontier (OLCF, ROCm 7.1)

| Scenario | Recommended backend |
|---|---|
| Validate GPU stack and model weights quickly | **Transformers + Accelerate** (`smoke-transformers-frontier.sh`) |
| Run matsim-agents with many concurrent agent steps | **vLLM** (`smoke-vllm-frontier.sh`) |
| Fine-tune or debug model internals | **Transformers + Accelerate** |
| Production multi-node serving | **vLLM** |

The smoke test scripts in `scripts/` cover both cases:

```bash
# HuggingFace Transformers (works today, no build needed)
sbatch scripts/smoke-transformers-frontier.sh

# vLLM (requires source build first)
sbatch scripts/build-vllm-from-source-frontier.sh   # one-time build
sbatch scripts/smoke-vllm-frontier.sh               # then serve
```

---

## Related docs

- [docs/model-download.md](model-download.md) — downloading Qwen2.5-72B-Instruct
- [scripts/smoke-transformers-frontier.sh](../scripts/smoke-transformers-frontier.sh)
- [scripts/smoke-vllm-frontier.sh](../scripts/smoke-vllm-frontier.sh)
- [scripts/build-vllm-from-source-frontier.sh](../scripts/build-vllm-from-source-frontier.sh)
