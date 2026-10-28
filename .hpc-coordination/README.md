# HPC Task Coordination Board

This directory is a Git-backed coordination channel on the repository's
current branch. Each task is stored at `tasks/<task-id>.json`; commits are the
audit trail. Facility workers fetch/push the same branch through the configured
Git remote. A claim is protected by an expiring lease; do not run a task unless
the claim succeeded.

Task `command` and `artifact_uri` values are descriptive metadata only. This
board does not dispatch jobs, transfer scientific datasets, or store secrets.
Use shared facility storage or an approved artifact service for large outputs.
