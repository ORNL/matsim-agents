# Run a workflow

Choose a task first:

- A small MLIP-only search: [bounded exploration](simple-exploration.md).
- A campaign application (fusion BCC, perovskites, FCC hierarchy): see
  [examples/README.md](../../examples/README.md).
- The E3 benchmark streams: [E3 operations](../../benchmarks/e3/operations.md).
- An advanced workflow: the common interface and capability table below.
- After execution: [interpret results](interpret-results.md).
- To implement scientific changes: [extend the library](extend-library.md).

## The common interface

```bash
matsim-agents workflow list
matsim-agents workflow init explore ./search.yaml
# Edit the native config identified by config: in search.yaml.
matsim-agents workflow check ./search.yaml
matsim-agents workflow run ./search.yaml
# Use the invocation directory printed by run, not the native scientific directory.
matsim-agents workflow status WORKFLOW_RUN_DIRECTORY
matsim-agents workflow summary WORKFLOW_RUN_DIRECTORY
matsim-agents workflow resume WORKFLOW_RUN_DIRECTORY --native-run SCIENTIFIC_RUN_DIRECTORY
```

`init`, `status`, and `summary` do not calculate. `check` validates configuration,
native CLI arguments, and the relevant existing native configuration model.
It does **not** qualify authentication, model/device ABI, live services, or
scheduler readiness.

`run` delegates to the existing native workflow. It streams native output to
the console and retains it in an invocation log. Interactive chat retains stdin.
An invocation has its own configuration, status, JSON summary, and Markdown
report; the native workflow continues to own scientific results and checkpoints.
Two processes cannot execute the same invocation concurrently.

Relative `config` and `output_root` paths are resolved against the
wrapper YAML. Paths inside a native config retain **that workflow's** existing
semantics. The `arguments` list is passed literally without a shell; argument
paths retain the current working directory's semantics. Use absolute paths when
moving a wrapper between directories. Native execution records its working
directory and reuses it on resume, preserving those relative paths. Do not put credentials in configs or
arguments: invocation evidence records them.

### Scientific hypothesis revision records

When a workflow uses an LLM to revise scientific claims, recording the claims
is mandatory rather than a logging option. A revision separates the material
claim, falsifiable prediction, scientific status, reason for retaining or
changing it, and evidence citations from numerical convergence or sampling
plans. Earlier identities must remain visible, including retired claims.
MLIP and DFT evidence have separate fidelity scopes; peer opinions and
generation failures cannot establish material stability.

LLM-backed multi-formula campaign reviews now run an initial hypothesis stage,
a review after each calculation iteration, and a final evidence review.
The existing debate transcripts retain exact prompts and replies.
`hypothesis-reviews/initial.json`, `round-NNNN.json`, and `final-NNNN.json`
retain stage status, prior hypotheses, calculation records, and accepted
ledger revision IDs. The append-only `evidence-ledger.jsonl` retains each
scientific revision, including every participant's intermediate debate round.
Custom campaign debate responses must contain matching `scientific_hypotheses`
and ledger `revisions`; inconsistencies abort before the next model is called.
Use `matsim-agents ledger show PATH --hypothesis ID` to
inspect a claim's history. A rejected or interrupted mandatory review prevents
automatic continuation and must be reconciled, not treated as abstention.
Initial and final reviews add model calls: authorize their token and compute
budgets before running the campaign.

Deterministic exploration, MD sampling, and model-training iterations that
do not revisit scientific claims do not receive artificial LLM assessments.
Single-shot model checks and independent benchmark responses are not an
evolving hypothesis trajectory.

## Advanced workflow catalog

All entries expose `check`, `run`, `status`, and `summary` through the same
wrapper. `resume` is capability-based: a command is available, but rejects
workflows without a safe native restart contract instead of silently replaying.

| Workflow | Native command | Wrapper input | Native resume |
| --- | --- | --- | --- |
| `explore` | `explore run` | `config` | Version-2 scientific directory via `--native-run` |
| `relax` | `relax` | `config` | Unsupported |
| `characterize` | `characterize` | `config` | Same contract and native step journal |
| `debate` | `debate` | `config` | Unsupported |
| `llm-check` | `llm-check` | `config` | Unsupported |
| `al` | `al run` | `config` | Native loop with `loop.resume: true` |
| `objective` | `run` | Objective/options in `arguments` | Unsupported |
| `plan` | `plan` | Objective/options in `arguments` | Unsupported |
| `chat` | `chat` | Options in `arguments` | Native session via `--native-run` |
| `known-answers` | `benchmark known-answers` | Options in `arguments` | Unsupported |
| `agent-eval` | `benchmark agent-eval` | `config` | Unsupported |

Native approvals, backend selection, evidence policies, and compute budgets
remain in their existing native config/options.

For example, an objective wrapper is:

```yaml
workflow: objective
arguments:
  - "Find a candidate Nb oxide structure"
  - --mlip-config
  - examples/mlip/mace-mp-medium.yaml
output_root: ./workflow-records
```

Use native `--help` to choose options. Backend/service/scheduler utilities
(`capabilities`, `resources`, `inventory`, `ledger`, `coordination`) are operational
tools, not computational workflows; they retain their own interfaces. Native
CLI and public Python workflow APIs are still directly usable.
