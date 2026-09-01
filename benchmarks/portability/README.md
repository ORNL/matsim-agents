# Cross-facility portability benchmark

This directory defines one scientific problem and three deployment overlays so
Frontier, Aurora, and Perlmutter results can be compared without conflating a
machine-specific launch choice with a change to the science.

## Benchmark ladder

1. **Smoke:** imports the installed package and validates the selected DFT step
   launcher. It is dependency-light and is the first release gate.
2. **Relaxation contract:** runs the production relaxation workflow with a
   deterministic numerical adapter and verifies convergence, persistence, and
   artifact handoff without requiring model weights.
3. **Active learning contract:** a fixed four-candidate pool selects candidates
   1 and 3, validates their labels, writes the dataset and immutable manifest,
   and proves that retraining and promotion remain disabled.
4. **Phase exploration contract:** dispatches the fixed Si candidate through
   the same typed result used by the production phase workflow.
5. **LLM discussion contract:** records proposal, critique, and revision turns,
   dispatches the proposed Si composition, and persists the investigation. The
   default responses are deterministic; `--live-llm` tests the configured model
   endpoint instead.

The contract suite executes on every facility and tests workflow composition,
data governance, persistence, and deterministic selections. It deliberately
does not claim MLIP or DFT numerical equivalence: production model weights,
licensed VASP assets, QE pseudopotentials, and model servers remain separate
facility qualifications. Place their results in `scientific_summary.json` for
the cross-site tolerance comparison.

## Inputs and acceptance

- `manifest.yaml` identifies the immutable benchmark and tolerances.
- `structures/Si.vasp` is owned by this benchmark rather than borrowed from a
  test fixture.
- `config/science.yaml` contains every machine-independent scientific choice.
- `config/{frontier,aurora,perlmutter}.yaml` may contain only scheduler,
  accelerator, rank, and launcher settings. `run.py` rejects scientific keys in
  an overlay.
- Validation compares invariants and tolerances, never bitwise-identical
  floating-point results.

Every run records the Git commit, exact structure digest, resolved
configuration, scheduler allocation metadata, executable discovery, plan, and
status. A comparison fails when runs use different source commits or structure
bytes. Optional numerical summaries use the tolerances in `manifest.yaml`.

## Running

Plan locally without a scheduler:

```bash
python benchmarks/portability/run.py \
  --facility frontier --suite active-learning --backend qe \
  --output runs/portability-plan
python benchmarks/portability/validate.py runs/portability-plan
```

Submit the complete deterministic gate (allocation is supplied at submission,
never embedded):

```bash
sbatch -A <allocation> --export=ALL,PROJECT_ROOT="$PWD" \
  deployments/frontier/jobs/job-portability-benchmark-frontier.sh
sbatch -A <allocation> --export=ALL,PROJECT_ROOT="$PWD" \
  deployments/perlmutter/jobs/job-portability-benchmark-perlmutter.sh
qsub -A <allocation> -v PROJECT_ROOT="$PWD" \
  deployments/aurora/jobs/job-portability-benchmark-aurora.sh
```

Compare completed directories:

```bash
python benchmarks/portability/compare.py \
  runs/portability/frontier-* runs/portability/aurora-* \
  runs/portability/perlmutter-*
```

To include a real LLM server in the discussion benchmark, configure the same
model on each machine and add `--live-llm`. Supported environment variables are
`MATSIM_LLM_PROVIDER`, `MATSIM_LLM_MODEL`, and `MATSIM_VLLM_BASE_URL`. The
deterministic discussion remains the release gate because hosted/server
availability should not hide an orchestration regression.

### Live LLM configuration

The live portability path supports the same five providers as the library:
`ollama`, `vllm`, `openai`, `anthropic`, and `huggingface`. Unlike the general
library default (`ollama`), this benchmark defaults to `vllm`, matching the
model-serving pattern on Frontier, Aurora, and Perlmutter.

| Variable | Meaning | Live-benchmark default |
|---|---|---|
| `MATSIM_LLM_PROVIDER` | One of the five provider names above | `vllm` |
| `MATSIM_LLM_MODEL` | Model name understood by that provider/server | Provider-specific default |
| `MATSIM_VLLM_BASE_URL` | Full OpenAI-compatible API root; vLLM only | `http://localhost:8000/v1` |

Provider defaults are `llama3.1:8b` for Ollama,
`meta-llama/Llama-3.1-8B-Instruct` for vLLM, `gpt-4o-mini` for OpenAI,
`claude-3-5-sonnet-latest` for Anthropic, and
`Qwen/Qwen2.5-72B-Instruct` for Hugging Face. For vLLM, include `/v1` in the
base URL and ensure `MATSIM_LLM_MODEL` matches the name exposed by the running
server. Set `MATSIM_VLLM_API_KEY` when the endpoint is protected; otherwise it
defaults to `EMPTY`.

Example:

```bash
export MATSIM_LLM_PROVIDER=vllm
export MATSIM_LLM_MODEL=Qwen/Qwen2.5-72B-Instruct
export MATSIM_VLLM_BASE_URL=http://localhost:8000/v1

python benchmarks/portability/run.py \
  --facility frontier --suite all --backend qe --execute --live-llm \
  --output runs/portability-live
```

Use the same model identifier, prompt settings, source commit, and server
configuration on every facility before comparing responses. Live LLM output is
recorded for functional comparison, not required to be textually identical.

Paper cases, scaling sweeps, model catalog benchmarks, and warm-start studies
remain specialized benchmarks. They are intentionally not deleted or silently
redirected to this small portability gate.
