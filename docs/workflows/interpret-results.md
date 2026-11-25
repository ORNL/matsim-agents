# Interpret workflow results and recover safely

Start with the human report, then inspect the authoritative evidence it names.
Do not infer scientific success from a zero process exit code.

## Common run records

`run` prints a run directory containing:

| Artifact | Meaning |
| --- | --- |
| `run-record.json` | Kind, source config, state, summary, blockers, failure reason, native run directory |
| `resolved_config.json` | Configuration summary used for this run |
| `report.md` | Human explanation, evidence links, and next actions |

```bash
matsim-agents status RUN_ID
matsim-agents evidence RUN_ID
```

States: `complete` and `partial` finished (partial means some candidates or
stages failed; read the report), `blocked` means nothing ran because a backend,
credential, approval or resource was missing (listed in `blockers`), `failed`
carries the error type and message, `interrupted` means Ctrl-C. The record is
the **last saved state**, not proof that a process or scheduler job is alive.
Consult the actual scheduler before recovery. A successfully submitted batch is
not a completed calculation, and a finished run is not scientific success: inspect
the native evidence the record names.

## Exploration: execution, convergence, and ranking are different

The native exploration directory has its own `report.md`,
`exploration-summary.json`, `rounds-state.json`, and `cumulative-ranking.json`.
Use native commands with **this** directory:

```bash
matsim-agents explore status SCIENTIFIC_RUN_DIRECTORY
matsim-agents explore summary SCIENTIFIC_RUN_DIRECTORY
```

The report gives generated/selected/deferred candidates, attempted and eligible
relaxations, rounds, backend loads, and observed stop reasons. Read it as follows:

- **Completed search:** the requested rounds and selected attempts have been
  committed. Backend failures and nonconverged attempts can still exist.
- **Eligible minimum:** a relaxation passed the configured convergence and
  eligibility criteria. Inspect forces, stress, and abort reasons, not only energy.
- **E3 hard deadline:** an overdue geometry task is forcibly stopped and labeled
  nonconverged with `optimization_time_limit`. Partial trajectories are forensic
  artifacts, not converged minima; no missing final metrics are fabricated.
- **Lowest observed candidate:** lowest energy among eligible attempted results
  under this model/protocol. It is not a proven global ground state.
- **Relative energy:** comparison within the sampled compatible candidates,
  not DFT accuracy, formation-energy validation, or chemical stability.
- **Diversity selection:** structural novelty among fresh candidates using a
  deterministic geometric fingerprint; not an uncertainty estimate or proof
  that distinct thermodynamic polymorphs were found.

Repeated minima remain auditable attempts; cumulative reporting does not count
them as independent competing polymorphs or assign chemical-stability proxies.
No eligible minima means investigate termination reasons and settings before
starting a new search. Do not disable scientific convergence requirements merely
to obtain a favorable ranking. See the [exploration guide](simple-exploration.md)
for thresholds and selection semantics.

## Resume deliberately

Read the [capability table](run-campaigns.md#advanced-workflow-catalog) first.
Unsupported restart operations fail explicitly. A fresh invocation is not a
resume and may repeat work.

For exploration, resume from the recorded run:

```bash
matsim-agents resume RUN_ID
# Or use the native checkpoint engine directly:
matsim-agents explore resume SCIENTIFIC_RUN_DIRECTORY
```

Only version-3 exploration checkpoints are supported: no old-format reader,
migration, or compatibility mode. Native resume verifies scientific config, the
model selection and the hashes of its local model files, runtime/dependency
identities, and committed artifact hashes. Completed attempts,
including failed/nonconverged ones, are not silently rerun. An uncommitted attempt
may restart. Completed runs are no-ops.

`resume` verifies the saved run record and the native config/manifest file.
These are **not** a replacement for native transitive scientific-input checks.
Changing scientific inputs requires a fresh run, not editing saved evidence.
Do not point a run at an unrelated native run/session. AL, characterization,
E1, chat, and E3 production each retain their own native restart policies.

## What to do next

1. Review run-record failure reasons and native termination reasons.
2. Verify convergence and the evidence level required for the scientific claim.
3. Resume only documented native checkpoints; otherwise prepare a fresh run.
4. For E3, preserve frozen protocols, approval gates, and resource accounting.
5. DFT verification or another campaign is a separate, explicitly authorized task;
   reading a report never starts it.
