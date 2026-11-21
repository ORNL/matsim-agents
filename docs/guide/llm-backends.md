# LLM backends

## Providers

Set the provider at runtime via CLI flag, environment variable, or in
code. Local/open-source backends are the default.

For a detailed comparison of the two open-source local backends (vLLM vs
HuggingFace Transformers + Accelerate) — including pros, cons, and guidance
for Frontier (ROCm) — see [docs/llm-backends-comparison.md](../llm-backends-comparison.md).
For Genesis Mission hosted models through the Model Access Gateway, see the
[MAG setup and usage guide](../model-access-gateway.md).

| Provider | Install | Typical model | Notes |
|---|---|---|---|
| **`ollama`** *(library default)* | `brew install ollama && ollama pull llama3.1:8b` | `llama3.1:8b` *(default)*; `qwen2.5:14b`, `deepseek-r1:14b` are examples | Fully local, CPU/GPU/Metal. |
| **`vllm`** | Run a vLLM server (`vllm serve <model> --port 8000`) | `meta-llama/Llama-3.1-8B-Instruct` | OpenAI-compatible; great for HPC. |
| **`mag`** | `pip install matsim-agents[mag]` | Explicit ID from authenticated MAG `/v1/models` | Genesis Mission hosted models; set `AMSC_TOKEN`. Optional upstream route via `mag_provider` or `MATSIM_MAG_PROVIDER`. |
| **`copilot`** | `pip install matsim-agents[copilot]`, then `copilot login` once | Explicit ID from the account's Copilot catalog (`list_copilot_models()`) | Models included with a GitHub Copilot subscription, through the Copilot SDK; requests count against the account's Copilot usage. Uses the signed-in Copilot CLI account, `COPILOT_GITHUB_TOKEN`, or `api_key`. No sampling controls (temperature/top-p/seed), so replies are not guaranteed to be deterministic; `max_tokens` is not enforced by Copilot-hosted models (overruns are flagged in `provider_metadata`); tool-free sessions only. |
| **`openai`** | `pip install matsim-agents[openai]` | `gpt-4o-mini` | Hosted. Set `OPENAI_API_KEY`. |
| **`anthropic`** | `pip install matsim-agents[anthropic]` | `claude-3-5-sonnet-latest` | Hosted. Set `ANTHROPIC_API_KEY`. |
| **`huggingface`** | `pip install matsim-agents[huggingface]` | `Qwen/Qwen2.5-72B-Instruct` | Direct HF Transformers + Accelerate; no server needed. Ideal as fallback on HPC when vLLM is unavailable. Set `MATSIM_HF_MODEL_PATH` to a local model directory. |

Provider support can vary by workflow: active-learning prompt seed configuration
currently accepts only `ollama`, `vllm`, `openai`, `anthropic`, and
`huggingface`. The generated [capability matrix](../capabilities.md) lists
providers with registered defaults, so it does not list `mag` or `copilot`.

## Downloading models for vLLM

For the vLLM backend you need to download the model weights locally before
starting the server. The recommended model for matsim-agents on HPC is
`Qwen/Qwen2.5-72B-Instruct`. A quick one-liner using the `hf` CLI that ships
with `huggingface_hub>=1.12`:

```bash
hf download Qwen/Qwen2.5-72B-Instruct \
    --local-dir /path/to/models/Qwen2.5-72B-Instruct
```

For detailed instructions — including Frontier-specific steps, running the
download as a background job, and resuming interrupted downloads — see
[docs/model-download.md](../model-download.md).

## Environment variables and defaults

Configuration knobs:

```bash
export MATSIM_LLM_PROVIDER=ollama          # or vllm | mag | copilot | openai | anthropic | huggingface
export MATSIM_LLM_MODEL=llama3.1:8b         # provider-specific model ID or local path
export MATSIM_OLLAMA_BASE_URL=http://...    # optional
export MATSIM_VLLM_BASE_URL=http://node:8000/v1
export MATSIM_VLLM_API_KEY=EMPTY            # only if vLLM is auth-protected
export MATSIM_HF_MODEL_PATH=/path/to/model  # huggingface provider: local model dir
```

The exact environment-variable contract is:

| Variable | Allowed values / format | Default and scope |
|---|---|---|
| `MATSIM_LLM_PROVIDER` | `ollama`, `vllm`, `mag`, `copilot`, `openai`, `anthropic`, or `huggingface` | General library default: `ollama`. The portability `--live-llm` path defaults to `vllm` so it can address an HPC model server. `mag` and `copilot` require an explicit model ID. |
| `MATSIM_LLM_MODEL` | Provider-specific model identifier, Ollama tag, or local model path | Used by entry points that expose environment-selected models, including the live portability benchmark. If omitted there, the provider default below is used. |
| `MATSIM_VLLM_BASE_URL` | Full OpenAI-compatible vLLM API root, normally `http[s]://host:port/v1` | vLLM only; defaults to `http://localhost:8000/v1`. Include `/v1`. |
| `MATSIM_VLLM_MAX_TOKENS` | Positive integer completion-token limit | vLLM only; defaults to `4096`. |
| `MATSIM_VLLM_TIMEOUT_SECONDS` | Finite positive request timeout in seconds | vLLM only; defaults to `3600`. |

Provider model defaults implemented by `get_chat_model` are:

| Provider | Default model | Identifier convention |
|---|---|---|
| `ollama` | `llama3.1:8b` | Tag installed in the Ollama daemon |
| `vllm` | `meta-llama/Llama-3.1-8B-Instruct` | The model name exposed by the OpenAI-compatible server |
| `openai` | `gpt-4o-mini` | OpenAI model identifier; requires `OPENAI_API_KEY` |
| `anthropic` | `claude-3-5-sonnet-latest` | Anthropic model identifier; requires `ANTHROPIC_API_KEY` |
| `huggingface` | `Qwen/Qwen2.5-72B-Instruct` | Hugging Face ID or local path selected with `MATSIM_HF_MODEL_PATH` |

For vLLM, `MATSIM_LLM_MODEL` must match a model identifier accepted by the
running server; it is not necessarily the filesystem directory from which the
server loaded its weights. `MATSIM_VLLM_API_KEY` defaults to `EMPTY`, which is
appropriate for an unsecured local server. Set it when the endpoint requires
authentication. `MATSIM_VLLM_BASE_URL` is ignored by non-vLLM providers.
Explicit `max_completion_tokens` and `request_timeout` arguments override their
environment variables without parsing the overridden values. Both sources
use the same configuration validation; invalid limits fail before any request.

## Multi-model scientific debate

Use `matsim-agents debate debate.yaml` to assign one scientific hypothesis to
an enclave of two or more independently configured LLMs. The YAML specifies
each participant's name, scientific role, provider, model and optional endpoint,
plus the user-selected `rounds`. Models see and directly challenge prior peer
arguments; speaking order rotates each round. A final synthesis preserves both
consensus and unresolved disputes, while the complete transcript and model
provenance are retained in a uniquely named run directory. See
[`docs/scientific-workflows.md`](../scientific-workflows.md) for the complete
configuration and evidence limitations.

The dedicated [scientific debate guide](../scientific-debate.md) documents
all four equal/role-based and independent/designated modalities with runnable
YAML examples and their different outcome semantics.

The default is a debate between equals: roles do not alter prompts, all models
receive identical neutral instructions, and every model writes an independent
final verdict. Role-based debate and designated-model synthesis remain explicit
options for workflows that intentionally need an asymmetric review panel.

The same debate is available as the `multi_llm_debate` hypothesis-discussion
mode of `InvestigationConfig`. It replaces the single-LLM discussion before
phase exploration; it is not embedded in active learning. See
[the integration guide](../scientific-debate.md#using-debate-inside-the-agentic-materials-workflow).

Before using a real model in a scientific workflow, qualify its configuration,
endpoint or local artifacts, loading, structured output, multi-turn behavior,
resource visibility, and latency independently:

```bash
matsim-agents llm-check examples/llm_check/llm_check.example.yaml
```

See [docs/llm-readiness.md](../llm-readiness.md) for the stage contract,
artifacts, distributed checks, and portability integration.

The provider factory can address arbitrary compatible model identifiers.
“First-class supported” is narrower: it means a model is present in
`deployments/common/open-model-catalog.json` and therefore has catalog,
download, launch, and benchmark coverage on the supported facilities.

## First-class open-model support

The generic `vllm` and `huggingface` providers can address arbitrary model
identifiers, but a checkpoint is called **first-class supported** only when it
is present in
[`deployments/common/open-model-catalog.json`](../../deployments/common/open-model-catalog.json).
Every catalog entry has all of the following:

- a canonical Hugging Face checkpoint identifier;
- a named endpoint environment variable plus the shared
  `MATSIM_VLLM_BASE_URL` fallback;
- entries in both Frontier benchmark manifests;
- download support on Frontier, Aurora, and Perlmutter; and
- inclusion in the sequential model benchmark.

CI enforces this contract in `tests/test_open_model_catalog.py`. Adding a model
to the catalog without adding any required deployment path fails the test, so
the backend-compatible and first-class sets cannot silently drift apart.

The frontier open-model catalog currently covers Kimi K2.5, GLM-4.7 and
GLM-4.7-Flash, DeepSeek-V3.2, Mistral Large 3, Qwen3-235B-A22B Instruct and
Thinking, Gemma 4, and Devstral 2. These checkpoints vary enormously in size.
Catalog membership means the repository knows how to select, download, and
benchmark the model; it does not mean that every model fits on one node. Use
the multi-node serving launcher and choose tensor parallelism appropriate to
the checkpoint and facility.

Model-specific vLLM behavior continues to be controlled by the serving
environment. In particular, use a vLLM release that supports the checkpoint's
architecture and chat template, and pass any required reasoning/tool parser
arguments through the serving command. A successfully downloaded checkpoint
is not treated as a successful deployment until the smoke request completes.

---
