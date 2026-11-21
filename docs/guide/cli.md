# Command-line guide

The [README command overview](../../README.md#cli-reference) lists every
`matsim-agents` command in one line each. This page explains the commands used
to get started and the options of the agent commands `run`, `chat`, and
`supervisor-run`. Workflow-specific commands are documented with their
workflows; see [choosing a scientific workflow](workflows.md).

`matsim-agents COMMAND --help` always shows the options of the installed
version.

## Check the environment

```bash
matsim-agents doctor                     # Python, packages, and backend readiness
matsim-agents doctor --require mace      # exit non-zero unless MACE is usable
matsim-agents doctor --json              # machine-readable diagnostics
matsim-agents capabilities check         # which DFT, MLIP, and LLM backends can run here
```

Neither command loads a model or contacts a server. `--require` can be
repeated. `matsim-agents capabilities matrix` regenerates the
[backend capability matrix](../capabilities.md).

## Create starter configurations

```bash
matsim-agents config list                    # kinds, templates, and the command that reads each
matsim-agents config init explore explore.yaml
matsim-agents config schema explore -o explore.schema.json
```

Kinds are `relax`, `characterize`, `debate`, `llm-check`, `al`, `explore`, and
`workflow`. `config init` copies an annotated example and never overwrites an
existing file. `config schema` prints the JSON Schema for editor validation and
completion.

## Common options of the agent commands

Options accepted by `run`, `chat`, and `supervisor-run`:

| Option | Description |
|---|---|
| `--mlip-config FILE` | YAML with an `mlip:` block selecting the model; see [MLIP selection](configuration.md#mlip-selection). Required by `supervisor-run` and by `run`/`chat` when they relax structures. |
| `--output-dir PATH` | Root directory for artifacts (`chat` and `supervisor-run` default to `outputs`). |

LLM options of `run` and `chat` (`supervisor-run` does not call an LLM and has
none of these options):

| Option | Description |
|---|---|
| `--llm-provider NAME` | `ollama`, `vllm`, `mag`, `copilot`, `openai`, `anthropic`, or `huggingface` (default `ollama`). `mag` and `copilot` require `--llm-model`; see [LLM backends](llm-backends.md). |
| `--llm-model NAME` | Provider-specific model identifier. `chat` defaults to `qwen2.5:14b`. |
| `--llm-base-url URL` | Override the server URL (Ollama or vLLM `/v1` endpoint). |

Relaxation options shared by `chat` and `supervisor-run`:

| Option | Default | Description |
|---|---|---|
| `--ase-structure-optimizer` | `FIRE` | `FIRE`, `BFGS`, or `BFGSLineSearch`. |
| `--maxiter` | `200` | Maximum relaxation steps per structure. |
| `--fmax` | `0.02` | Converged when the maximum residual force (eV/Å) is below this value. |
| `--relative-increase-threshold` | `0.2` | Abort and roll back when the maximum force grows by more than this fraction in one step. Raise it (for example `10.0`) to let FIRE leave symmetric starting points. |
| `--n-random` | `50` | Target number of supplementary pyXtal structures per composition, added to every applicable AFLOW prototype. `0` disables them; fewer may be generated. |
| `--random-seed` | `0` | Seed for the pyXtal generator. |

`supervisor-run` also accepts `--maxstep` (default `0.01`).

Active-learning handoff options shared by all three commands:

| Option | Description |
|---|---|
| `--trigger-al-handoff / --no-trigger-al-handoff` | Evaluate branch-weight uncertainty after relaxations and optionally hand off to active learning. Enabled by default for `chat` and `supervisor-run`, disabled for `run`. |
| `--al-config PATH` | Base active-learning YAML used when the handoff fires. |
| `--al-dry-run / --al-run` | Only plan and record the handoff (default), or execute the loop. |
| `--uq-top-weight-threshold` | Trigger when the mean top branch weight is below this value (default `0.6`). |
| `--uq-min-unreliable-fraction` | Trigger when the low-confidence fraction exceeds this value (default `0.25`). |
| `--uq-min-relaxations-for-handoff` | Relaxations required before evaluation (default `1` for `run`, `3` otherwise). |
| `--al-handoff-audit-path PATH` | JSONL file for decisions (default `<output-dir>/discovery/al_handoff_events.jsonl`). |

Branch-weight uncertainty is produced only by fused HydraGNN inference.

## `run`: planner, executor, and analyst

```bash
matsim-agents plan "Relax examples/characterization/structures/MgO-rocksalt.vasp"
matsim-agents run "Relax examples/characterization/structures/MgO-rocksalt.vasp" \
  --mlip-config mlip.yaml
```

`plan OBJECTIVE [--json]` prints the planner's task list without executing it.
`run` executes the plan; see the [core agent graph](architecture.md#core-agent-graph).

| Option | Description |
|---|---|
| `--max-iterations` | Maximum executor iterations before analysis (default `5`). |
| `--mcp-config PATH` | External MCP servers and allowed tools; see [External MCP tools](../mcp.md). |
| `--mcp-audit-path PATH` | MCP audit log (default `<output-dir>/mcp/mcp_audit.jsonl`). |

## `chat`: hypothesis-driven discovery

`chat` is described in [hypothesis-driven discovery chat](discovery-chat.md).

| Option | Description |
|---|---|
| `--llm-only / --with-mlip` | Conversation only, without an MLIP, `/relax`, `/al`, or automatic exploration. |
| `--auto-confirm / --ask` | Explore every detected composition without asking (default: ask). |
| `--resume ID_OR_PATH` | Continue a saved session from `<output-dir>/chat/`. Models and tools come from the new command line. |
| `--resources PATH` | [Resource plan](../resource-dispatch.md) whose `llm` pool starts or names the models and whose `mlip` pool relaxes candidates concurrently. |
| `--llm-server NAME` | Proposer server from the plan's `llm` pool (default: its only server). |
| `--critic-llm-servers NAMES` | Comma-separated plan servers that act as critics. |
| `--llm-peer-review / --no-llm-peer-review` | Proposer/critic rounds that challenge and revise hypotheses (default off). |
| `--peer-review-rounds` | Number of revision rounds (default `1`). |
| `--critic-llm-provider`, `--critic-llm-model`, `--critic-llm-base-url` | A single critic (provider defaults to `--llm-provider`). |
| `--critic-panel-models`, `--critic-panel-providers`, `--critic-panel-base-urls` | Comma-separated, aligned lists for a panel of critics. |
| `--critic-cross-critique / --no-critic-cross-critique` | Let critics challenge each other before the proposer revises (default off). |

## `supervisor-run`: one composition

```bash
matsim-agents supervisor-run Li2MnO3 \
  --mlip-config mlip.yaml \
  --al-config examples/active_learning/al_config.example.yaml \
  --al-dry-run
```

The supervisor explores one composition, evaluates uncertainty, optionally hands
off to active learning, and summarizes. It uses the relaxation and handoff
options above and does not call an LLM.

## Package tools

`matsim-agents tools list` shows the training, dataset, and benchmark programs
shipped in the package, such as `finetune-mace`, `finetune-uma`,
`finetune-hydragnn`, `evaluate`, `splits`, `formation-labels`, and
`elemental-dft`. Run one with `matsim-agents tools run NAME [ARGS...]`;
`matsim-agents tools run NAME --help` lists its options. See
[evaluating fine-tuned MLIPs](../mlip-evaluation.md) and
[MLIP training and promotion](../mlip-training-and-promotion.md).

## Retained-artifact inventory

```bash
matsim-agents inventory dft runs/ --output dft-inventory.json
```

Reports the timers that VASP and Quantum ESPRESSO record in retained run
directories. `--accounting-csv FILE` adds allocations you supply as
`run_directory,elapsed_seconds,nodes,gpus`; the scheduler is not queried.

## E1 development questions

```bash
matsim-agents e1-development MANIFEST.json --output DIR [--max-new-requests N] [--retry-errors]
```

Runs the development questions of one candidate model from a manifest; held-out
questions are rejected. Rerunning with the same manifest and output reuses
committed answers. `--retry-errors` retries recorded errors within the
manifest's approved attempt limit. See
[run a workflow or campaign](../workflows/run-campaigns.md).
