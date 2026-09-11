# LLM readiness and portability qualification

LLM deployment is qualified independently from materials workflows:

```bash
matsim-agents llm-check examples/llm_check/llm_check.example.yaml
```

This separation makes failures actionable. A model-download, endpoint,
credential, accelerator, or model-loading failure is reported as an LLM
deployment failure rather than as a relaxation or active-learning failure.

## Stages

The command runs and persists six required stages:

1. **Readiness:** validate local artifacts, credentials, or provider endpoint.
   vLLM queries `/v1/models`; Ollama queries `/api/tags`.
2. **Load:** construct the provider client or local model and record construction
   time and process memory. For server providers this does not measure server
   startup; preserve the separate server launch log for that measurement.
3. **Generation:** run a fixed, temperature-zero prompt and require the
   `PORTABILITY_OK` invariant.
4. **Structured output:** request JSON and validate it as a
   `ScientificHypothesis` Pydantic model.
5. **Discussion:** execute proposal, critique, and revision turns with prior
   context included.
6. **Distributed resources:** record client-visible devices. Local-model checks
   fail when `expected_accelerators` differs from the detected count. For a
   remote vLLM service, validate the declared tensor-parallel size and retain
   server launch evidence because client-side PyTorch cannot see remote GPUs.

Correctness checks use invariants and schemas, not identical prose. Construction
time, serial and concurrent generation latency, maximum resident memory,
model identity, endpoint metadata, and accelerator visibility are recorded
separately as performance/deployment evidence.

## Artifacts

Each run creates a collision-resistant directory containing:

```text
<output_root>/<UTC-timestamp>_<random-suffix>/
├── request.json
├── resolved_config.json
├── provenance.json
├── events.jsonl
├── health.json
├── model_identity.json
├── generation.json
├── structured_hypothesis.json
├── discussion.json
├── environment.json
├── performance.json
└── result.json
```

API keys are excluded from resolved configuration and provenance artifacts.

## Provider readiness rules

| Provider | Readiness requirement |
|---|---|
| `vllm` | `base_url` ends in `/v1`; `/models` contains the configured model ID |
| `ollama` | `/api/tags` contains the configured model tag |
| `huggingface` | Local paths contain `config.json`; remote IDs are validated when loading |
| `openai` | `OPENAI_API_KEY` or `api_key` is configured |
| `anthropic` | `ANTHROPIC_API_KEY` or `api_key` is configured |

For a multi-node vLLM service, set `expected_accelerators`,
`tensor_parallel_size`, and `context_length` to the deployed topology. Server
launch logs remain necessary evidence for worker membership and tensor/pipeline
parallel placement. For remote vLLM, the client-side check requires
`expected_accelerators == tensor_parallel_size`; it cannot independently prove
remote worker placement. For local providers it verifies the visible device
count directly.

When `concurrent_requests` is greater than one, the generation stage sends that
many simultaneous fixed-prompt requests and requires every response to satisfy
the invariant. This is a functional concurrency check and latency measurement,
not a saturation-throughput study.

## Scientific portability integration

A live-LLM scientific benchmark must reference the successful qualification:

```bash
python benchmarks/portability/run.py \
  --facility frontier --suite all --backend qe --execute --live-llm \
  --llm-check-run runs/llm-check/<successful-run> \
  --output runs/portability-live
```

The portability result embeds the qualification run ID and model identity.
Deterministic discussion remains available without a model server and is the
workflow-contract release gate.

## Perlmutter compute-job qualification

A login node cannot reach a vLLM endpoint bound inside a compute allocation.
Use the dedicated job, which owns both the server and qualification lifecycle:

```bash
PROJECT_ROOT=$PWD sbatch -A <allocation> \
  deployments/perlmutter/jobs/job-llm-check-perlmutter.sh
```

Override `MATSIM_MODEL_DIR`, `MATSIM_MODEL_NAME`, and `MATSIM_VLLM_TP` for the
checkpoint and topology being qualified. The job starts vLLM in its isolated
environment, polls `/v1/models` until the exact served identifier appears,
runs all six `llm-check` stages from the client environment, validates the
terminal `result.json`, and always stops the server through an exit trap. It
preserves the vLLM launch log, endpoint response, generated check configuration,
console log, and full qualification directory under one Slurm run directory.

Set `MATSIM_RUN_PORTABILITY=1` to run the live scientific portability suite
after qualification; that run receives the successful check directory through
`--llm-check-run`. The dedicated job is the authoritative deployment check.
The sequential model-quality benchmark may reuse the same pattern, but a model
evaluation failure should not be conflated with endpoint or deployment failure.
