# Extend a workflow without tracing the entire application

Choose the responsibility you need to change. See the complete
[package architecture](../architecture.md) for enforced import layers.

## Config-driven exploration

One checkpoint engine serves one and multiple rounds. There is no separate
single-round scientific implementation to keep synchronized.

| Change | Start here |
| --- | --- |
| Scientific option, YAML validation, relative paths, preflight | [exploration config](../../src/matsim_agents/workflows/exploration/config.py) |
| Generation/relaxation lifecycle | [round engine](../../src/matsim_agents/workflows/exploration/rounds.py) |
| Journal invariants, input/result identity, committed artifacts | [checkpoint persistence](../../src/matsim_agents/workflows/exploration/checkpoint.py) |
| Counts, ranking, readable scientific report | [reporting](../../src/matsim_agents/workflows/exploration/reporting.py) |
| Public Python entry points | [exploration facade](../../src/matsim_agents/workflows/exploration/api.py) |
| Terminal presentation and verbs | [exploration CLI](../../src/matsim_agents/cli/exploration.py) |

Add scientific options to the validated config and shared relaxation-request
builder. Preserve first-round parity and model reuse across rounds. Commit
candidate/attempt records atomically before publishing their identities. A
failed in-memory update must not become a partial resumable checkpoint.
Preserve final-ranking hash coverage and scientific resume identity checks.

## Common interface, not another numerical engine

The common `validate`, `plan`, `run` and `resume` commands delegate to the
native implementations:

- [Runner](../../src/matsim_agents/workflows/runner.py): kind detection, one
  `Handler` per config kind (no-compute preflight, execute, optional resume),
  blocker reporting for missing backends, credentials, approvals and resources.
- [Run records](../../src/matsim_agents/workflows/run_records.py): saved run
  state and human explanation, not scientific convergence.
- [CLI commands](../../src/matsim_agents/cli/runs.py): thin wrappers over the
  runner; exit 0 finished, 1 failed, 2 invalid input, 3 blocked.
- [Run directories](../../src/matsim_agents/execution/run_directory.py):
  collision-resistant run layout and the `events.jsonl` stream.

For a new advanced workflow, register a `Handler` for its config kind whose
preflight validates the native config without computation, and set `resume` only
if its native engine has a safe persistence contract. Do not implement
“resume” as blind replay. Keep scientific output and checkpoint handling native.
If the workflow needs a distinct typed preset, validate its parameters rather
than adding an unrestricted shell command.

Keep CLI dependencies downward: persistence/execution cannot import orchestration,
campaign, workflows, or CLI. Do not move machine-specific submission assumptions
into scientific workflow modules.

## Shared helpers in `core`

Use these instead of a private copy:

| Need | Use |
| --- | --- |
| Write a state or result file so a crash cannot leave it half written, or append one line to a log | `core.fileio.atomic_write_text`, `core.fileio.append_line` |
| Hash a file, canonical JSON or a JSON value | `core.hashing.sha256_file`, `canonical_json`, `sha256_json` |
| Read a YAML config that must be a mapping, with the file in the error | `core.yamlio.load_yaml_mapping`, `parse_yaml_mapping` |
| Skip work a restarted run already finished, keyed on everything that would change the result | `core.restart.KeyedResultLog` |

`backends/mcp/config.py` keeps its own YAML parse on purpose: PyYAML errors quote the offending source line, which can
be a secret. `execution/ledger.py` keeps its own writer because its entries form a hash chain.

## Tests to add when behavior changes

- Native arguments are validated without invoking callbacks or loading models.
- Invalid input fails before dispatch, with actionable CLI diagnostics.
- One-/multi-round parity, interruption, completed no-op, hashes, and config identity.
- Reports distinguish process completion, eligible convergence, and scientific claims.
- Locks prevent simultaneous execution of one invocation.
- E3 checks/preset initialization do not submit jobs; submission remains explicit.
- Native restart policy is honored, including error-retry and non-resumable cases.
- Atomic report writes preserve JSON storage behavior.
- Package-level API exports and dependency boundaries remain valid.

Focused validation:

```bash
PYTHONPATH="$PWD/src:$PWD" .venv/bin/python -m pytest -q \
  tests/test_workflow_interface.py tests/test_exploration_entrypoint.py \
  tests/test_exploration_rounds.py tests/test_scientific_run_contracts.py \
  tests/test_package_boundaries.py tests/test_documentation.py
bash scripts/check.sh lint
```

Do not substitute live E3/DFT launches for mocked dispatch/approval tests.
Update [running tasks](run-campaigns.md) and
[result interpretation](interpret-results.md) when the user-facing contract changes.
