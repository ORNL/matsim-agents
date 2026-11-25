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
matsim-agents config init explore ./search.yaml   # annotated starter config
# Edit search.yaml.
matsim-agents validate ./search.yaml              # no compute; exit 2 if invalid
matsim-agents plan ./search.yaml                  # what `run` would do and what blocks it
matsim-agents run ./search.yaml --runs-dir runs   # execute and record
matsim-agents status                              # list recorded runs
matsim-agents evidence RUN_ID                     # state, summary, blockers, native directory
matsim-agents resume RUN_ID                       # continue, or retry once blockers clear
```

The config's top-level keys select the workflow (`application`, `explore`, `al`,
`neb`, `relax`, `characterize`, `debate`, `llm-check`, `alloy-spec`, external
interfaces). `validate` checks configuration and the relevant native config model
without calculating. It does **not** qualify authentication, model/device ABI,
live services, or scheduler readiness; those appear as blockers in `plan` and
`run`.

`run` calls the native workflow in-process. Every invocation, including blocked
and failed ones, leaves `run-record.json` and `report.md` in a new directory under
`--runs-dir`; the native workflow continues to own scientific results and
checkpoints. Exit status: 0 finished, 1 failed or interrupted, 2 invalid input,
3 blocked on a missing backend, credential, approval or resource.

Relative paths inside a config keep that workflow's existing semantics. Do not
put credentials in configs: run records keep the resolved summary.

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

## Config kinds

| Kind | Native workflow | Resume |
| --- | --- | --- |
| `application` | Application runner (structure build, relaxation, campaign) | Blocked runs retry |
| `explore` | Bounded MLIP exploration | Checkpointed (`resume RUN_ID`) |
| `relax` | Provenance-tracked relaxation | Unsupported |
| `characterize` | Characterization contract | Native step journal |
| `debate` | Scientific debate | Unsupported |
| `llm-check` | LLM deployment qualification | Unsupported |
| `al` | Active-learning loop | Native loop with `loop.resume: true` |
| `neb` | QE NEB preparation and dispatch | Unsupported |
| `alloy-spec` | Structure generation | Unsupported |
| `external` | Optional external-software interfaces | Unsupported |

`resume` rejects kinds without a safe restart contract instead of replaying.
A natural-language goal uses `matsim-agents objective "TEXT"` and records a run of
kind `objective` the same way.

Native approvals, backend selection, evidence policies, and compute budgets
remain in their existing config.

Use native `--help` to choose options. Backend/service/scheduler utilities
(`capabilities`, `resources`, `inventory`, `ledger`, `coordination`) are operational
tools, not computational workflows; they retain their own interfaces. Native
CLI and public Python workflow APIs are still directly usable.
