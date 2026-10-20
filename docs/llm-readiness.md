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
├── contributions.jsonl
├── prompts/<id>.json
├── transcript.md
├── structured_hypothesis.json
├── environment.json
├── performance.json
└── result.json
```

Every model reply (`generation`, `concurrent-NNN`, `structured`, `proposal`,
`critique`, `revision`) is appended to `contributions.jsonl` with its answer,
any reasoning, finish reason, token counts, and latency; `prompts/<id>.json`
holds the messages sent and `transcript.md` the same exchange for reading.
`result.json` lists in `incomplete_replies` the replies the token limit cut off
or that contain no answer.

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

The model, its GPUs, and its tensor-parallel size come from a resource layout
(default `deployments/perlmutter/resources/llm-check.yaml`; override with
`MATSIM_RESOURCES` and `MODEL_ROOT`). The job runs
`matsim-agents resources serve <layout> -- deployments/common/run-llm-check.sh`:
`resources serve` starts vLLM through the facility launcher with the
placement's device mask, waits until `/v1/models` lists the exact served
identifier, runs the shared steps (all six `llm-check` stages against the
plan's server, validation of the terminal `result.json`, optional portability
run), and always stops the server. Server logs, `placement.json`, and
`llm-endpoints.json` are kept under `<run>/llm-servers`, next to the generated
check configuration, console log, and qualification directory.

`llm-check` can also manage the server itself: a configuration with a
`resources:` block (and optionally `server:`) starts the plan's server, takes
the model, endpoint, tensor-parallel size, device count, and context length
from the placement, and stops the server when the check ends.

The Frontier and Aurora jobs (`deployments/frontier/jobs/job-llm-check-frontier.sh`,
`deployments/aurora/jobs/job-llm-check-aurora.sh`) follow the same contract with
their own layouts and launchers (`vllm-server-frontier.sh`: ROCm vLLM on the
placed GCDs; `vllm-server-aurora.sh`: the `frameworks` vLLM build through
`mpiexec` on the placed PVC tiles).

Set `MATSIM_RUN_PORTABILITY=1` to run the live scientific portability suite
after qualification; that run receives the successful check directory through
`--llm-check-run`. The dedicated job is the authoritative deployment check.
The sequential model-quality benchmark may reuse the same pattern, but a model
evaluation failure should not be conflated with endpoint or deployment failure.
