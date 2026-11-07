# E4 resource estimate (not yet estimable)

No node-hour or dollar estimates are published at this stage. No representative E4 task has been timed under the proposed workflow, and facility allocation ceilings/rates have not been supplied.

Known plan only:

| Block | Allocations planned | Nodes per allocation | Resource estimate |
|---|---:|---|---|
| Aurora strong scaling | 18 (6 points x 3 repetitions) | 8, 16, 32, 64, 128, 256 | Awaiting Gate 2 measurements |
| Aurora weak scaling | 18 (6 points x 3 repetitions) | 8, 16, 32, 64, 128, 256 | Awaiting Gate 2 measurements |
| Cross-facility consistency | 9 (3 facilities x 3 repetitions) | provisional 3 each | Awaiting Gate 2/3 measurements |

Allocation node-hours must be calculated from actual scheduler allocation durations: sum(nodes x allocation_seconds / 3600) over all attempts, including failed attempts. Application stage timers and overlapping task durations are not allocation walltime. Convert costs to dollars only after verified facility billing units and account rates are provided.

Before estimates: measure one complete task and a bounded concurrency point; measure LLM startup/inference, MLIP and QE separately; establish per-task and tail durations, scheduler overhead, memory/device occupancy, sustainable pool concurrency, and cleanup time; obtain account ceilings and rate units. No allocations have been submitted.