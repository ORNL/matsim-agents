# HPC Task Coordination

The current Git branch is the shared task channel for coordinating independent work on Perlmutter, Frontier, Aurora, and other HPC facilities. Task metadata is isolated under `.hpc-coordination/`; updates appear as ordinary commits in the current branch history. The tool does not dispatch jobs or modify scientific source files.

For the Aurora login-node read-only connectivity check, follow
[`docs/aurora-task-board-check.md`](aurora-task-board-check.md). It verifies
remote read access and visibility of the completed coordination smoke-test
record; it does not claim a task or submit compute.

## One-Time Setup

The remote must be accessible from each facility through an approved Git credential method. Initialize the board from a checkout with the same remote:

```bash
matsim-agents coordination init --repo . --remote origin
```

The CLI infers the currently checked-out branch; run it from the same named branch at each facility. Task updates are commits to that branch, so fetch/pull the normal branch to receive them. Use `--remote` to select a configured remote other than `origin`. Detached HEAD checkouts are not supported.

## Publish Work

Create one task per independently executable unit. Include facility, exact command, expected output location, acceptance criteria, method/configuration identifiers, resource ceiling, and dependencies in the details. Avoid adding tokens, passwords, private data, or large data artifacts to Git.

```bash
matsim-agents coordination add nbtao-audit-perlmutter \
  --title "Audit existing Nb-Ta-O DFT evidence" \
  --facility perlmutter \
  --command "python scripts/audit_nbtao.py --inventory /shared/path" \
  --artifact-uri "/shared/project/coordination/nbtao-audit" \
  --details "Read-only inventory first; report provenance, settings, convergence, and elemental/binary/ternary reference coverage. No new DFT without separate approval."
```

Task command text is documentation only. The coordinator never runs commands, allocates resources, submits jobs, or fetches arbitrary artifacts.

## Worker Loop

On each facility, fetch and review pending work; claim before starting. The claim is an expiring lease. Use a stable worker name for heartbeats and completion:

```bash
matsim-agents coordination list --status pending
matsim-agents coordination claim nbtao-audit-perlmutter --worker perlmutter-login03 --lease-minutes 120
# Review the task, satisfy local facility policy, then execute it manually or through the approved scheduler workflow.
matsim-agents coordination heartbeat nbtao-audit-perlmutter --worker perlmutter-login03 --lease-minutes 120
matsim-agents coordination complete nbtao-audit-perlmutter --worker perlmutter-login03 \
  --note "Inventory and audit summary: /shared/project/coordination/nbtao-audit/report.md"
```

Use `coordination fail TASK --worker NAME --note ...` to report a blocked or failed attempt. Requeue by creating a follow-up task rather than resetting history. If a worker disappears, another worker may reclaim the task after its lease expires; stale workers must check ownership before continuing. Heartbeat well before expiry.

## Consistency and Artifact Rules

Each mutation fetches the latest branch tip, makes a commit containing only `.hpc-coordination/`, and pushes. Concurrent writers that race are serialized by Git's fast-forward check; a rejected push is refetched and retried. A claim only succeeds if its status and lease remain claimable on the latest tip. If retries are exhausted, the operation fails visibly and must be retried after checking the board.

Git is the coordination channel, not a high-volume database or data plane. Store small task metadata, summaries, hashes and references in `.hpc-coordination/`. Put trajectories, checkpoints, DFT outputs and logs on approved shared storage or artifact systems, then record stable paths/URIs and checksums. Each facility must have access to the referenced artifact before it accepts a task.

The board provides cooperative leases, not scheduler-level fencing. A worker can continue running after network partition or lease expiry; therefore use facility scheduler job IDs, resource limits, idempotent task design, and explicit output paths to avoid duplicate expensive work. Do not use it to coordinate tasks that require strict mutual exclusion over physical resources.

Before publishing research tasks, identify an owner, a facility, resource caps, acceptance criteria, dependencies, and required approvals. Nb-Ta-O existing DFT must be audited for provenance, settings, convergence, and competing-phase coverage before authorizing additional calculations; incomplete reference coverage cannot support definitive thermodynamic-stability claims. The task board does not grant scientific, allocation, data-access, export-control, or facility approval.