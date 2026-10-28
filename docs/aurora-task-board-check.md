# Aurora Task-Board Read Check

## Purpose

Verify that an Aurora login-node checkout can read the shared Git-backed task
board using its configured Git credentials and network route. This is a
read-only connectivity check. It does not test write permission, claim work,
or validate scheduler, DFT, or accelerator workflows.

## Before You Start

- Log in to Aurora using the site's approved access method and use a login
  node. Do not request a compute allocation for this check.
- Use the authorized project checkout and branch
  `research/numerical-results-refresh`. Replace `/path/to/matsim-agents` below
  with that checkout's actual path.
- Use an existing project `.venv` that contains `matsim-agents`. Do not run the
  installer just for this check. The board command needs no Aurora accelerator
  module stack.
- Follow the project's and ALCF's Git credential policies. Never put tokens,
  passwords, or private keys in commands, task records, or the report.

## Run the Check

Run these commands in a shell on an Aurora login node:

```bash
cd /path/to/matsim-agents
hostname
git status --short --branch
git branch --show-current
git remote get-url origin
test -x .venv/bin/matsim-agents
```

Confirm the branch is `research/numerical-results-refresh` and `origin` is the
intended shared repository. If the checkout is detached, on a different branch,
or the executable is missing, stop and report that prerequisite; do not switch
branches or install packages as part of this check.

Fetch and display the shared board without changing it:

```bash
set -o pipefail
report="$HOME/aurora-task-board-read-$(date -u +%Y%m%dT%H%M%SZ).txt"
.venv/bin/matsim-agents coordination list --repo "$PWD" --remote origin \
  | tee "$report"
printf 'Saved read-only output to %s\n' "$report"
```

The expected result includes task `coordination-smoke-20261028` with status
`completed`, owner `local-pilot-worker`, and the `created`, `claimed`,
`heartbeat`, and `completed` events. Other tasks may also appear. An empty list
is a successful connection but does not confirm that this smoke-test record is
visible; report the distinction.

## What to Report

Send the project operator:

- UTC check time and Aurora login-node hostname.
- Checkout branch and whether `origin` matched the intended repository.
- Whether `coordination list` succeeded or failed. Include the task ID and
  status if the smoke-test record was visible.
- The saved report path, if the site's data-handling rules permit sharing it.
- Any concise error text, with credentials, tokens, and other sensitive values
  removed.

Do not paste the full task listing into public channels if it contains unrelated
project metadata. The operator needs the access result and smoke-test status,
not unrelated task details.

## Do Not Do During This Check

Do not run `coordination init`, `add`, `claim`, `heartbeat`, `complete`, or
`fail`. Do not edit or push board records, switch branches, submit a PBS job,
load an accelerator stack, run DFT, or launch model inference. A successful
read establishes only remote read access from this Aurora login-node checkout;
it does not grant facility approval or authorize scientific work.