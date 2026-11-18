# E4 orchestration benchmark framework

E4.1 scalability, E4.2 portability and E4.3 heterogeneous workloads. This is a
measurement layer on top of the production execution stack: `ResourcePlan` and
presets for placement, `run_dft_batch` for node-group tasks, `MLIPWorkerPool` for
device-slot tasks, and `classify_failure` / `recovery-v1` for retries. There is
no second scheduler.

## Task states

Each attempt is appended to `attempts.jsonl` with `segment`, `attempt`,
`failure_class`, `retry_scheduled` and `terminal`. Summaries count tasks in these
states:

| State | Meaning |
|---|---|
| planned | Task is in the frozen manifest. |
| attempted | Task has at least one attempt record. |
| executed | Payload or science code ran and produced output, which may still be invalid. |
| converged | Code reported convergence. |
| qualified | Passed every acceptance check (finite output, artifact present, force or fmax rules). |
| failed | Terminal and not qualified. |
| retried | Task has more than one attempt. |

Only qualified results count toward throughput.

## Workloads

| Config | Experiment | Contents |
|---|---|---|
| `configs/smoke-local.json` | weak, strong | Local CPU smoke with injected `no-result` and `nonfinite` faults. |
| `configs/e4.1-orchestration.json` | weak 1–512 nodes, strong 1–128 nodes | Controlled one-node MPI steps: lognormal duration with a 30 s median, 1 MiB artifact, 2 % faults. Weak: 32 tasks/node. Strong: 1024 tasks. |
| `configs/e4.1-science.json` | weak, strong, 2–128 nodes | Real VASP Si SCF (`dft`) plus HydraGNN relaxations (`mlip-relax`) with the production qualification rules. |
| `configs/e4.3-heterogeneous.json` | fixed | Mixed long node tasks, short device tasks, VASP and HydraGNN. |
| `configs/e4.1-science-qe.json`, `configs/e4.3-heterogeneous-qe.json` | as above | Same workloads with the QE Si SCF class (`Si_r.upf`) instead of VASP. |

Workloads are deterministic in `(config, experiment, nodes)`:

- Strong scaling uses the same task set (same `workload_hash`) at every node count.
- Weak scaling uses `tasks_per_node × nodes`. Larger points extend the smaller ones.

Every string in a config can reference `${VAR}`. A variable that is not set is an error. The science configs need:

- `E4_VASP_BIN` and `E4_VASP_POTCAR_DIR`, or `E4_QE_PW_BIN` and `E4_QE_PSEUDO_DIR` for the `-qe` configs
- `E4_HYDRAGNN_LOGDIR`
- `E4_MLP_DEVICE` (job scripts default it to the facility accelerator)

HydraGNN inference uses only the OMat24 head (`hydragnn_inference_head`); the branch-weight MLP is not used.

On Aurora, `deployments/aurora/jobs/submit-e4-science-campaign.sh vasp|qe` prints the whole science campaign (E4.1 weak and strong, E4.3) with queues and paths filled in. Add `--validate` for the 2-node debug check and `--submit` to submit. Aurora's QE 7.6 offloads only EPW to the GPU, so the QE campaign runs the CPU `pw.x`.

`deployments/frontier/jobs/submit-e4-science-campaign.sh` (8 GCDs per node) and `deployments/perlmutter/jobs/submit-e4-science-campaign.sh` (4 A100 per node) are the Slurm counterparts. They take the same arguments and also need `E4_ACCOUNT`. Walltimes come from `cli.py jobs --facility <facility>`, which uses that facility's device count.

## Local smoke (no allocation)

```bash
PYTHONPATH=$PWD:$PWD/src python benchmarks/e4/framework/cli.py smoke --output-root /tmp/e4smoke
PYTHONPATH=$PWD:$PWD/src python -m pytest tests/test_e4_framework.py -q
```

## Launch

`jobs` prints one submission command per point, with a walltime and node-hour
estimate. It never submits anything.

The estimate is 300 s of startup plus 1.5 × the ideal makespan, rounded up to 10 minutes. Ideal makespan uses the frozen controlled durations, `estimate_seconds` for real classes, and the configured fault retries. Add your account and queue options yourself.

```bash
python benchmarks/e4/framework/cli.py jobs benchmarks/e4/framework/configs/e4.1-orchestration.json \
  --facility aurora --experiment weak --repetitions 1
# e.g. qsub -l select=8 -l walltime=00:40:00 -v E4_CONFIG=...,E4_EXPERIMENT=weak,... \
#        deployments/aurora/jobs/job-e4-framework-aurora.pbs
```

Job scripts take `PROJECT_ROOT`, `E4_CONFIG`, `E4_EXPERIMENT`, `E4_OUTPUT_ROOT`
and the optional `E4_REPETITION`, `E4_POLICY` and `E4_CLASSES`:

- `deployments/aurora/jobs/job-e4-framework-aurora.pbs`
- `deployments/frontier/jobs/job-e4-framework-frontier.sh`
- `deployments/perlmutter/jobs/job-e4-framework-perlmutter.sh`

Each script records the allocation start time and runs `cli.py run --auto-resume`.

### Estimated cost: Aurora, one repetition

These are budgeting estimates, not measurements.

| Experiment | Points | Node-hours |
|---|---|---|
| E4.1 orchestration, weak | 1–512 nodes, 40 min each | ≈ 518 |
| E4.1 orchestration, strong | 1–128 nodes, 13.5 h at 1 node down to 20 min | ≈ 128 |
| E4.1 science, weak | 2–128 nodes | ≈ 163 |
| E4.1 science, strong | 2–128 nodes, 13 h at 2 nodes | ≈ 182 |
| E4.3 at 8 nodes | mixed (4.7) + staged (4.7) + four homogeneous runs (8.2) | ≈ 18 |

The configs default to 3 repetitions. Check the job scripts first with the 1–2 node points.

## Monitoring and resume

Each run directory `<output-root>/<run_id>/` contains:

| File | Contents |
|---|---|
| `identity.json` | Run identity: workload and config hashes. |
| `manifest.json` | Frozen task list. |
| `resolved_config.json` | Config after variable expansion. |
| `placement-NN.json` | Placement for segment NN. |
| `environment-NN.json` | Git commit, software versions, modules, scheduler job IDs. |
| `attempts.jsonl` | Attempt checkpoint, one record per attempt. |
| `summary-NN.json` | Summary for segment NN. |
| `measurements.jsonl` | Measurement records. |

`tail -f attempts.jsonl` shows progress.

To continue an interrupted run, resubmit the same command. With `--auto-resume`, the run directory is reopened and these rules apply:

- Tasks with a terminal record are skipped.
- Device results already on disk are reconciled.
- Unrecorded node attempts are logged as `interrupted`.
- The run is rejected if the identity hashes changed.

## Analysis

```bash
python benchmarks/e4/framework/cli.py analyze ${RUNS_ROOT}/e4-framework --output ${RUNS_ROOT}/e4-analysis
```

Outputs:

- `runs.csv`: per-run throughput, timing, overheads, failures and retries, node-hours and device-hours.
- `scaling.csv` and `scaling-*.png`: speedup and efficiency for strong scaling, weak efficiency for weak scaling. Only fully qualified repetitions are used.
- `heterogeneous.csv`
- `report.json`

## E4.2 portability levels

```bash
python benchmarks/e4/framework/portability.py \
  --deployment aurora=<portability-run> frontier=<...> perlmutter=<...> \
  --stage      aurora=<framework-run>   frontier=<...> perlmutter=<...> \
  --workflow   aurora=<portability-run> ... \
  --output ${RUNS_ROOT}/e4.2
```

| Level | Pass criteria |
|---|---|
| deployment | A `benchmarks/portability` run passes `validate_run` and the smoke suite executed with return code 0. Installation alone does not count. |
| stage | A framework run of `e4.1-science.json` (for example `--experiment fixed --nodes 2`) has qualified results for every real class, with input and output hashes and complete provenance (commit, job ID, pinned DFT settings, structure hashes). |
| workflow | A portability run with `qualification == "compute"` that passed. Cross-facility checks use `compare_runs`. |

The numerical comparison takes per-structure medians of `energy_eV_per_atom` and
`max_force_eV_per_A` for each class and checks the spread across facilities against
`manifest.yaml` `acceptance.cross_facility`. Bitwise identity is not required.

## E4.3 protocol

Use the same node count and repetition for every run:

1. Mixed, concurrent: `--experiment fixed`.
2. One homogeneous run per class: `--classes <name>` (or `E4_CLASSES`).
3. Separate baseline policy: `--policy staged` (classes one after another in one allocation).

`heterogeneous.csv` reports, for each run, the makespan, the ratio to the sum of the homogeneous makespans, utilization, idle hours, queue waits, retries and recovery time.

## Not yet verified

All of these are still unverified:

- Facility runs.
- The step launchers inside real allocations.
- The real VASP and HydraGNN classes on any facility.

Only the local smoke and the unit tests have run.
