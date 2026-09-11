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
3. **Active learning contract:** runs a complete miniature production AL loop
   over four deterministic candidates, labels two through the DFT dispatch
   interface, writes the dataset and immutable manifest, and verifies that a
   restart neither repeats an iteration nor retrains/promotes the model.
4. **DFT allocation contract:** simulates a four-node scheduler allocation and
   verifies concurrent calculations receive four disjoint node groups.
5. **Phase exploration contract:** invokes phase exploration directly for the
   fixed Si candidate; it does not reuse or depend on the LLM discussion stub.
6. **LLM discussion contract:** records proposal, critique, and revision turns,
   dispatches the proposed Si composition, and persists the investigation. The
   default responses are deterministic; `--live-llm` tests the configured model
   endpoint instead.

The contract suite executes on every facility and tests workflow composition,
data governance, persistence, and deterministic selections. It deliberately
does not claim MLIP or DFT numerical equivalence: production model weights,
licensed VASP assets, QE pseudopotentials, and model servers remain separate
facility qualifications. The separate `--qualification compute` mode executes
the real production relaxation workflow and requires at least one MLIP-only
configuration and one QE-only configuration. Both modes always write
`scientific_summary.json`; contract metrics are explicitly marked as adapter
results and must never be presented as numerical backend evidence.

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

Every executed run records the Git commit, exact structure digest, resolved
configuration, scheduler allocation metadata, executable discovery, plan, and
status, and mandatory scientific summary. A comparison fails when runs use
different source commits, structure bytes, or qualification modes.

### Real compute qualification

Supply production `ScientificRelaxationConfig` YAML files. The benchmark owns
the Si input and output roots, so `structure_path` and `output_root` in those
files are overridden. All other model, pseudopotential, convergence, launcher,
and geometry choices remain explicit in the files. Repeat the option for each
MLIP model that should be qualified; at minimum one `mode: mlip` case and one
`mode: dft` case with `dft.backend: qe` are mandatory.

```bash
python benchmarks/portability/run.py \
  --facility frontier --qualification compute --execute \
  --relaxation-config /shared/configs/uma-si.yaml \
  --relaxation-config /shared/configs/qe-si.yaml \
  --output runs/portability/frontier-compute
python benchmarks/portability/validate.py runs/portability/frontier-compute
```

Use identical configuration bytes, model checkpoints, QE version, and
pseudopotentials at all sites. The facility jobs accept
`MATSIM_PORTABILITY_QUALIFICATION=compute` and a colon-separated
`MATSIM_PORTABILITY_RELAXATION_CONFIGS` list. Contract mode remains the default
CI/release gate because it does not require external model or DFT assets.

`config/relaxation/qe-si.yaml` and `config/relaxation/uma-si.yaml` are worked
Perlmutter examples (NERSC paths, `mlp_device: cuda` on the A100s).
`config/relaxation/qe-si-frontier.yaml` and
`config/relaxation/hydragnn-si-frontier.yaml` are the Frontier equivalents:
the QE case points `dft.launcher` at
[`deployments/frontier/launchers/run-pw-gpu-frontier.sh`](../../deployments/frontier/launchers/run-pw-gpu-frontier.sh)
(which performs its own module reset to the PrgEnv-cray/rocm-6.2.4 QE
toolchain) instead of a bare `pw.x` path, and the MLIP case uses the
HydraGNN backend because UMA/fairchem is not installed on Frontier. Submit
compute-mode qualification on Frontier with:

```bash
MATSIM_PORTABILITY_QUALIFICATION=compute \
MATSIM_PORTABILITY_RELAXATION_CONFIGS="$PWD/benchmarks/portability/config/relaxation/hydragnn-si-frontier.yaml:$PWD/benchmarks/portability/config/relaxation/qe-si-frontier.yaml" \
sbatch -A <allocation> --export=ALL,PROJECT_ROOT="$PWD",MATSIM_PORTABILITY_QUALIFICATION,MATSIM_PORTABILITY_RELAXATION_CONFIGS \
  deployments/frontier/jobs/job-portability-benchmark-frontier.sh
```

The job script loads the ROCm 7.2.0 module stack and HydraGNN venv for the
driver process (real MLIP inference); the QE step launcher it invokes for the
`dft`-mode case switches to the isolated QE toolchain itself, so the two
never share a shell (see
[Module-stack isolation](../../docs/quantum-espresso-frontier.md#module-stack-isolation-from-matsim-agents-python)).

### All-model scientific debate qualification

`all_model_scientific_debate.py` is the live portability test for the complete first-class
model catalog. It dynamically reads
`deployments/common/open-model-catalog.json`; therefore a newly cataloged model
automatically becomes required. Every model must contribute exactly once in
each of at least two rounds, directly respond to the accumulated peer dialogue,
and return non-empty text. Every participant receives identical neutral system
instructions and produces its own final verdict; the benchmark has no
privileged synthesizer. The assigned materials-design question is:

> What candidate material provides an optimal thermoelectric functional
> property—specifically a high dimensionless figure of merit ZT near 800 K—
> while remaining chemically stable and composed of reasonably abundant
> elements?

Thermoelectric `ZT` is a standard community metric that couples the Seebeck
coefficient, electrical conductivity, thermal conductivity, and temperature.
The benchmark tests scientific interaction and deployment portability; it does
not treat the proposed candidate or the debate consensus as validated evidence.

Each catalog entry declares a `base_url_env`. Export every corresponding
endpoint variable (or one `MATSIM_VLLM_BASE_URL` endpoint capable of routing
all catalog model IDs), then run:

```bash
python benchmarks/portability/all_model_scientific_debate.py \
  --rounds 2 --output runs/portability/all-model-scientific-debate
```

The required `dialogue.json` artifact contains the original user question,
every model argument in chronological dialogue order, and the final synthesis.
Every model argument and synthesis has a unique `contribution_id`, together
with its round, participant, provider, model identifier, and full text. The
machine-readable result fails for a missing catalog endpoint, missing model
turn, empty contribution, duplicate contribution ID, or empty synthesis.

Facility portability jobs run this additional test when
`MATSIM_RUN_ALL_MODEL_SCIENTIFIC_DEBATE=1`; use `MATSIM_DEBATE_ROUNDS` to request more than the
mandatory two rounds. The endpoints must already be live within the compute
allocation. Large catalog checkpoints will generally require separate or
multi-node servers; catalog membership does not imply simultaneous one-node
residency.

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
  --llm-check-run runs/llm-check/<successful-run> \
  --output runs/portability-live
```

Use the same model identifier, prompt settings, source commit, and server
configuration on every facility before comparing responses. Live LLM output is
recorded for functional comparison, not required to be textually identical.
`--live-llm` requires `--llm-check-run`; the portability result records that
qualification run ID and exact model identity. See
[`docs/llm-readiness.md`](../../docs/llm-readiness.md).

Paper cases, scaling sweeps, model catalog benchmarks, and warm-start studies
remain specialized benchmarks. They are intentionally not deleted or silently
redirected to this small portability gate.
