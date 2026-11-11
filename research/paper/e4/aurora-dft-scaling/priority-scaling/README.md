# E4 Scaling Priority

For Perlmutter staging, VASP portability limits, and unfinished-case instructions,
see the [Perlmutter E3-MLIP and E4 execution runbook](../../../perlmutter-e3-e4-execution-runbook.md).

This package records the immediate BCC DFT scaling workload, its blockers, and
resource caps. The original BCC manifests and authorizations remain immutable.
The versioned six-class extension is
[`../six-class-workflow-v1.json`](../six-class-workflow-v1.json); it does not
authorize additional work.

## Six-Class Extension

The new manifest combines the existing frozen inputs with the currently
available, explicitly blocked candidate records. It distinguishes task-input
readiness from execution authorization and preserves workflow-level semantics:
each VASP task remains one node, and nodes execute independent calculations.
It does not measure solver-level scaling of one calculation across nodes.

| Class | Frozen/qualified inputs in this checkout | Strong input-backed tiers | Weak input-backed tiers | Current blocker |
|---|---|---|---|---|
| BCC refractory | 16 distinct frozen tasks | 1/2/4/8 nodes, 8 fixed tasks | 1/2/4/8 nodes, 2 tasks/node | Existing BCC authorization only; running measurements are preserved |
| FCC Cantor | 4 frozen first-batch cases | 1/2/4 nodes, 4 fixed tasks | 1/2 nodes, 2 tasks/node | 510 composition candidates are not audited structures; larger tiers need more frozen POSCARs |
| LiFePO4 | 8 frozen cases from the 510-entry antisite inventory | 1/2/4/8 nodes, 8 fixed tasks | 1/2/4 nodes, 2 tasks/node | Weak-8 requires 16 distinct inputs |
| Phosphorene | 2 small E3 seed cells only | blocked | blocked | Qualified E2 geometries are not staged; no E4 2D numerical protocol is frozen |
| Zn(HCOO)2 MOF | 2 example placeholders, formula C8O8Zn4 instead of C8H8O16Zn4 | blocked | blocked | Not valid DFT geometries; dispersion/reference protocol unverified |
| Nb-Ta-O | 14 symmetry-inequivalent NbTaO5 orderings; 5 designated pilot candidates; 1 converged E2 ordering | blocked | blocked | E2 VASP inputs need E4 staging and runtime/protocol qualification; weak-8 requires 16 |

Under the existing 24-hour-per-task ceiling, the current FCC tier set is a
432 node-hour cap and the LiFePO4 tier set is 1,104 node-hours. These are
proposals, not approvals. The E2 NbTaO5 2-hour workflow is a planning basis
only; the manifest's 66 node-hour cap (5 fixed strong tasks and up to 14 weak
inputs) assumes the missing E4 adapter and qualification. Phosphorene and MOF
have no defensible runtime budget until valid inputs and a pilot exist. The proposed additional request for currently
input-backed FCC and LiFePO4 scaling is 1,536 node-hours. NbTaO5's 66 node-hour
figure is a separate E2-derived planning ceiling, not a resource request until
the E4 adapter is validated. The existing BCC 1,488 node-hour authorization is
not counted again.

The six-class manifest also records 16/32/64/128-node weak-scaling points. The
capacity queue admits at most 16 nodes/job; 32/64/128 are explicitly marked
ineligible pending a different allocation/queue. At 2 calculations/node and a
24-hour/task cap, those hypothetical weak points require 64/128/256 distinct
inputs and 1,536/3,072/6,144 node-hours, respectively. Strong-scaling points
above 8 nodes are blocked because the fixed strong workload contains only eight
calculations; do not call idle-node runs strong scaling.

Generate/validate the plan and inspect a tier without submitting:

```bash
python -m benchmarks.e4.aurora_dft.scaling plan-six-class
python -m benchmarks.e4.aurora_dft.scaling validate-six-class
python -m benchmarks.e4.aurora_dft.scaling dispatch-six-class \
  --manifest research/paper/e4/aurora-dft-scaling/six-class-workflow-v1.json \
  --material-class fcc_hea --experiment strong --nodes 4 --dry-run
```

The six-class manifest has `execution_authorized=false`. No jobs were submitted
for this extension. In the last checked state, weak BCC scaling had completed
two one-node calculations and strong BCC job 9069491 was still running on eight
nodes. Do not edit or reuse their output roots.

After separately authorized work completes, `benchmarks/e4/analyze.py` reads
the dispatcher-written `measurements.jsonl` records, groups results by class,
and writes per-class CSV aggregates, a planned-versus-measured JSON summary, and
plots. For example:

```bash
find runs/e4-scaling -name measurements.jsonl -type f -print0 \
  | xargs -0 -r cat > runs/e4-scaling/measurements-all.jsonl
python -m benchmarks.e4.analyze \
  runs/e4-scaling/measurements-all.jsonl --kind scaling \
  --workload-manifest research/paper/e4/aurora-dft-scaling/six-class-workflow-v1.json \
  --output runs/e4-scaling/analysis/scaling.csv
```

The analyzer merges one-node continuation segments for the same fixed strong
workload before computing speedup. It leaves incomplete points without speedup
or efficiency and never interpolates unmeasured tiers.

## Proposed Submission Order

1. Preserve and finish the active BCC jobs; validate their dispatch and measurement records, then use the existing BCC authorization only for its frozen manifests.
2. After separate approval, run the input-backed FCC tiers (strong 1/2/4; weak 1/2) and LiFePO4 tiers (strong 1/2/4/8; weak 1/2/4). Extend the structure sets before requesting more tiers.
3. Qualify the NbTaO5 E2-derived inputs through the shared E4 case adapter and a one-node runtime pilot, then add unique orderings before weak-8.
4. Retrieve validated puckered phosphorene geometries, freeze its two-dimensional protocol, and prepare/adjudicate the E4 pool before estimating runtime.
5. Replace the incomplete Zn-formate placeholders with validated hydrogen-complete CIFs, qualify the dispersion protocol, and run a serial pilot before sizing tiers.

Each order step requires its own scientific and resource approval. The workload
manifest and dry-run commands alone authorize no production calculation.

## Workload Audit

The approved first-batch manifest pins eight BCC records to
`bcc-pbe-low-normal-v1-approved-ediffg-0.01`, with an 8-rank, 8-thread task
layout. Eight additional disorder realizations were frozen from the same
audited BCC parent and matched compositions. All 16 POSCAR hashes and
symmetry-canonical occupation IDs are distinct; per-case PAW component hashes,
protocol inputs, and assembled POTCAR hashes are recorded in
`weak-input-manifest.json`. This is a local structure-identity check only:
external archive duplicate screening remains incomplete.

The `capacity` queue currently reports a 16-node maximum, 168-hour job
walltime, and two running jobs per user. `benchmarks/e4/aurora_dft/scaling.py`
dispatches one existing E4 case runner per assigned node in task waves, records
per-task exit status and execution timing, and gives every task a fresh output
root. `deployments/aurora/jobs/job-e4-scaling.pbs` is the PBS entrypoint; it
does not submit jobs. The AL DFT scaling harness is unrelated because it
generates a different 32-case FCC workload through an active-learning loop.

The manifests reference approved input cases, not their existing output or
checkpoint directories. Any authorized scaling run must use fresh scaling run
and task IDs under the listed output roots; it must not resume or overwrite
first-batch or terminal-failure records. The initial 480 node-hour approval
does not authorize these new calculations.

## Blocker Corrections

The earlier GPU launch failure came from `--gpu-bind=closest`, which PALS
MPICH 1.8 does not support. The Aurora launcher now uses an explicit tile list.
The isolated capacity-queue probe (PBS 9067531) ran VASP through the pinned
`.51a9474` launcher; VASP reported 8 MPI ranks, 8 threads per rank, 8 GPUs
initialized, and accepted the frozen 128-ion BCC inputs. Its separate mask
subprobe ran under facility-default `.87e2045` and saw masks 0-7 once; that is
not treated as proof for the pinned module. A pinned-module follow-up on the
debug queue (PBS 9067675) failed before VASP because that node exposed only six
GPU devices (`Requested GPU 6 but only 6 exist`). Do not use that debug node for
the eight-rank placement check. The capacity run was deliberately interrupted
before convergence and is not a scaling measurement; PBS exit 1 followed the
intentional SIGINT, which the smoke script now recognizes explicitly.

The PAW hash failure was a missing-path issue: the new worktree lacked the
private/ignored `external/vasp6/potcar` files. The exact archived PAWs are
restored locally as read-only hardlinks. `submit_batch.py --dry-run --only
E4-BCC-1701` validates the frozen hashes and reports no qstat/qsub execution;
no hash checks were weakened.

## Resource Caps

All values are worst-case planning caps using the approved 24-hour per-task
walltime. They are not requests or approvals.

| Experiment | Nodes | Tasks | Max walltime | Node-hour cap | Status |
|---|---:|---:|---:|---:|---|
| Strong | 1 | 8 | 2 x 96 h | 192 | Two 4-task PBS segments; authorization required |
| Strong | 2 | 8 | 96 h | 192 | Authorization required |
| Strong | 4 | 8 | 48 h | 192 | Authorization required |
| Strong | 8 | 8 | 24 h | 192 | Authorization required |
| Weak | 1 | 2 | 48 h | 48 | Authorization required |
| Weak | 2 | 4 | 48 h | 96 | Authorization required |
| Weak | 4 | 8 | 48 h | 192 | Authorization required |
| Weak | 8 | 16 | 48 h | 384 | Authorization required; external duplicate audit incomplete |

The bounded request for one measurement at every listed point is **1,488
node-hours maximum**: 768 strong plus 720 weak. No repeats or larger tiers are
included. The root `strong-manifest.json` and `weak-manifest.json` remain
planning-only; corresponding copies under `authorized/` carry the existing
BCC-only authorization. The six-class manifest remains
`execution_authorized: false`. The dispatcher requires a separately authorized
manifest and matching `E4_SCALING_AUTHORIZATION` ID. For the 1-node strong point, request two
separate 96-hour jobs (`segment-1`, `segment-2`), each with four tasks. The
other per-job requests are 2 nodes/96 h, 4 nodes/48 h, 8 nodes/24 h for strong,
and 1/2/4/8 nodes/48 h for weak. All requests use `capacity` and the
`HydraGNN` allocation. Queue wait is excluded from task execution timing.

For weak scaling, each node runs its two tasks sequentially. Concurrently
running two unchanged 8-rank x 8-thread tasks on one node would oversubscribe
the node's CPU allocation and would change the resource layout.

## Path To 128 Nodes

Keep the DFT workflow study separate from MLIP device scaling. First qualify
the 1/2/4/8-node BCC workload, including the additional weak-scaling cases and
the multi-task dispatcher. Expand to 16 nodes only after the 8-node measurements
and resource review; the current queue ceiling is 16 nodes. The 32/64/128-node
weak-scaling points require a different approved allocation and respectively
64/128/256 unique frozen tasks, all under the same protocol. At two sequential
24-hour tasks per node, the 128-node point has a 48-hour walltime and a
6,144-node-hour cap. Do not extrapolate a curve to 128 nodes from the current
inventory.

## Existing Components

- The E4 per-case PBS wrapper, frozen-input validator, PAW hash checks, and
  corrected explicit Aurora tile-list launcher remain reusable. The scaling
  dispatcher runs the per-case wrapper concurrently on disjoint nodes and
  serially for each node's assigned weak-scaling tasks.
- The AL DFT scaling harness is an executable but different FCC/active-learning
  experiment; do not use it as E4 BCC evidence.
- `benchmarks/e4_mlip/run.py` is a separate single-node device-scaling runner.
  It still requires the reviewed 120-geometry manifest and frozen model
  metadata. Its timing outputs do not qualify HydraGNN energy/force accuracy.
- `benchmarks/e4/analyze.py` consumes recorded measurements only. Its strong
  baseline now uses the smallest measured node count; the focused planning
  tests pass (7 tests).

The isolated GPU-binding probe is
`deployments/aurora/jobs/e4-first-batch/smoke-vasp-gpu-binding.pbs`. It uses a
frozen BCC input copy in a unique temporary directory and is not a scaling or
recovery job. The initial 480 node-hour production-batch cap does not cover
the separately approved BCC scaling manifests or the proposed six-class tiers.