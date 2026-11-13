# Interpret workflow results and recover safely

Start with the human report, then inspect the authoritative evidence it names.
Do not infer scientific success from a zero process exit code.

## Common invocation records

`workflow run` prints an invocation directory containing:

| Artifact | Meaning |
| --- | --- |
| `resolved_config.json` | Wrapper settings and paths used for this invocation |
| `workflow-state.json` | Last atomically saved native-command status and input identities |
| `workflow-summary.json` | JSON presentation of that invocation status |
| `workflow-output.log` | Native output, scientific stop reasons, and result locations |
| `report.md` | Human explanation, evidence links, and next actions |

```bash
matsim-agents workflow status WORKFLOW_RUN_DIRECTORY
matsim-agents workflow summary WORKFLOW_RUN_DIRECTORY
```

`completed` means the native command exited zero. `failed` means it failed;
`interrupted` means execution was interrupted. `running` is only the **last
saved record**, not proof that a process, service, or scheduler job is alive.
An abrupt termination may leave that record behind. Consult the actual
scheduler/service before recovery. A successfully submitted VASP batch is not
a completed VASP calculation.

Native output is streamed and logged, including interactive prompts. Ctrl-C
stops the wrapper's own local child process group, allowing a short graceful
shutdown before escalation. It does **not** cancel jobs already accepted by a
scheduler. Inspect native artifacts after any interruption.

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

For exploration, identify the native directory from the retained output:

```bash
matsim-agents workflow resume WORKFLOW_RUN_DIRECTORY \
  --native-run SCIENTIFIC_RUN_DIRECTORY
# Or use the native checkpoint engine directly:
matsim-agents explore resume SCIENTIFIC_RUN_DIRECTORY
```

Only version-2 exploration checkpoints are supported: no old-format reader,
migration, or compatibility mode. Native resume verifies scientific config,
runtime/dependency identities, and committed artifact hashes. Completed attempts,
including failed/nonconverged ones, are not silently rerun. An uncommitted attempt
may restart. Completed runs are no-ops.

The wrapper verifies saved wrapper settings and the native config/manifest file.
These are **not** a replacement for native transitive scientific-input checks.
Changing scientific inputs requires a fresh run, not editing saved evidence.
Do not point an invocation at an unrelated native run/session. AL, characterization,
E1, chat, and E3 production each retain their own native restart policies.

## What to do next

1. Review invocation errors and native termination reasons.
2. Verify convergence and the evidence level required for the scientific claim.
3. Resume only documented native checkpoints; otherwise prepare a fresh run.
4. For E3, preserve frozen protocols, approval gates, and resource accounting.
5. DFT verification or another campaign is a separate, explicitly authorized task;
   reading a report never starts it.
