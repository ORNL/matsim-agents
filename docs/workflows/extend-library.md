# Extend a workflow without tracing the entire application

Choose the responsibility you need to change. See the complete
[package architecture](../architecture.md) for enforced import layers.

## Config-driven exploration

One checkpoint engine serves one and multiple rounds. There is no separate
single-round scientific implementation to keep synchronized.

| Change | Start here |
| --- | --- |
| Scientific option, YAML validation, relative paths, preflight | [exploration config](../../src/matsim_agents/workflows/exploration_config.py) |
| Generation/relaxation lifecycle | [round engine](../../src/matsim_agents/workflows/exploration_rounds.py) |
| Journal invariants, input/result identity, committed artifacts | [checkpoint persistence](../../src/matsim_agents/workflows/exploration_checkpoint.py) |
| Counts, ranking, readable scientific report | [reporting](../../src/matsim_agents/workflows/exploration_reporting.py) |
| Public Python entry points | [exploration facade](../../src/matsim_agents/workflows/exploration.py) |
| Terminal presentation and verbs | [exploration CLI](../../src/matsim_agents/cli/exploration.py) |

Add scientific options to the validated config and shared relaxation-request
builder. Preserve first-round parity and model reuse across rounds. Commit
candidate/attempt records atomically before publishing their identities. A
failed in-memory update must not become a partial resumable checkpoint.
Preserve final-ranking hash coverage and scientific resume identity checks.

## Common interface, not another numerical engine

The common `workflow` commands delegate to the native implementations:

- [Catalog/configuration](../../src/matsim_agents/cli/workflow_catalog.py):
  typed E3 settings, validated wrapper inputs, native command and restart capability.
- [CLI dispatch](../../src/matsim_agents/cli/workflows.py): no-compute checks,
  explicit gates, native command construction, invocation locks and recovery.
- [Process execution](../../src/matsim_agents/execution/workflow_runner.py):
  live output, retained logs, interactive stdin, local child-group cleanup.
- [Invocation persistence/report](../../src/matsim_agents/execution/workflow_record.py):
  saved process state and human explanation, not scientific convergence.
- [Atomic run storage](../../src/matsim_agents/execution/run_directory.py):
  shared JSON/text artifact writes.

For a new advanced workflow, add a catalog adapter pointing to its existing
native CLI, validate its native config without computation, and declare restart
support only if its native engine has a safe persistence contract. Do not implement
“resume” as blind replay. Keep scientific output and checkpoint handling native.
If the workflow needs a distinct typed preset, validate its parameters rather
than adding an unrestricted shell command.

Keep CLI dependencies downward: persistence/execution cannot import orchestration,
campaign, workflows, or CLI. Do not move machine-specific submission assumptions
into scientific workflow modules.

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
