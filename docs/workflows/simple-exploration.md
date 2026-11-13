# A small phase search, without writing Python

This is the beginner-facing entry point for **bounded batches or rounds** of
MACE structure generation and relaxation. It reuses the same generators,
calculator, ranking, and evidence storage as the library workflows.
You need one YAML file, not a session-specific test driver.

This first interface supports a **trusted local MACE checkpoint**. It does
not yet reproduce the three-round Nb2O5 coordination test: there are no LLM
calls, SOAP filtering, adaptive decisions, DFT, or active learning.
Existing advanced commands remain available.

## 1. Prepare once

Use the [installation instructions](../../README.md#installation) or your
facility's prepared MACE environment. The examples below assume the repository
root and `.venv-mace/bin/python`. This interpreter must import matsim-agents,
MACE, ASE, PyTorch, pymatgen, and PyXtal if random generation is requested.
The CLI and Python API use the same validated configuration.

Copy [the Nb2O5 example](../../examples/workflows/explore-nb2o5.yaml) somewhere
convenient, then edit **`checkpoint`** to the local model file. The supplied hash
pins MACE-OMAT-0 medium; change the hash as well if deliberately using another
model. Only deserialize checkpoints you trust.

```bash
cp examples/workflows/explore-nb2o5.yaml ./explore.yaml
# Edit explore.yaml: set checkpoint to your trusted local model path.
```

Paths in the YAML are resolved relative to the YAML file, including the output
directory. There are no machine-specific Python paths to rewrite. The example
uses CPU and at most two relaxation attempts; use a compute allocation for
extended work. Nothing in these commands submits a scheduler job.

## 2. Check before spending compute

```bash
.venv-mace/bin/python -m matsim_agents.cli explore check explore.yaml
```

This validates the configuration, checks dependency availability in the
**current interpreter**, and verifies the checkpoint hash. It does not import
or load MACE, generate structures, contact an LLM, or create a run directory.
It is a static preflight, not a device/ABI or scientific qualification.
If dependencies are missing, use the correct prepared interpreter first;
do not mix packages from unrelated environments.

## 3. Run and read the result

Inside your compute allocation:

```bash
.venv-mace/bin/python -m matsim_agents.cli explore run explore.yaml
```

The command prints the run directory, generated/attempted/converged counts,
termination reasons, backend errors, and number of calculator loads. A single
calculator is reused across candidates; initial loading is not repeated for
each structure. Every run, including the default one-round search, uses the same
checkpointed engine and output layout:

| Output | Purpose |
| --- | --- |
| `resolved_config.json` | Complete validated configuration with absolute paths |
| `provenance.json` | Checkpoint identity and scientific runtime settings |
| `rounds-state.json` | Version-2 checkpoint: generated candidates, decisions, attempts, errors, hashes |
| `cumulative-ranking.json` | Relative-energy ranking, or `null` when nothing qualifies |
| `exploration-summary.json` | Counts, termination reasons, execution/scientific outcome |
| `round-0001/seeds/` | Generated input structures |
| `round-0001/attempt-0000/` | Relaxation output for candidate index zero |

Additional rounds use numbered directories in the same layout. Configuration
and checkpoint identity are recorded before generation, including when it fails.
There is no alternate single-batch `results.json` format.

Inspect it later without loading the model or repeating calculations:

```bash
.venv-mace/bin/python -m matsim_agents.cli explore summary /path/to/printed/run-directory
```

Running the YAML again starts a **fresh** run, not a resume. Keep the printed
directory when comparing results. Execution completion is not convergence;
nonconverged results are excluded from ranking. Ranking compares only
same-composition relaxed candidates, not thermodynamic stability against other
phases. Generated counts can be below the requested counts. Seeds do not
guarantee bit-for-bit reproducibility across generators or package versions.

To continue an interrupted run, even a one-round search:

```bash
.venv-mace/bin/python -m matsim_agents.cli explore resume /path/to/printed/run-directory
```

Exit codes: `0` means execution completed (possibly with no converged minimum),
`1` indicates preflight/backend/generation/ranking failure, and `2` indicates an
invalid configuration. Details remain in output and persisted evidence.

## Changing scientific settings

- `max_relaxations` caps attempts **per round**, including backend failures.
- `generation` controls AFLOW count, PyXtal count, formula units, seed, and
  optional `space_groups` (integers 1–230). At least one source is required.
- `relaxation` controls optimizer, steps, maximum step, force target, and a
  between-step wall-clock limit. A slow step can overrun the limit. The current
  shared ranking accepts force tolerances no looser than 0.05 eV/A, so this
  interface rejects larger `fmax` values rather than silently misranking them.
- The example is fixed-cell. For cell relaxation, explicitly set `relax_cell:
  true`, `require_native_stress: true`, and a positive
  `stress_tolerance_eV_per_A3` (for example 0.01). Set
  `max_cell_step_strain: 0.1` and `max_cell_total_strain: 0.5` if the revised
  E3 abort policy is intended. These are principal strains, not volume changes.
  Supply a suitable six-component `cell_mask` for slabs.
- Force-growth and strain aborts accept YAML `null` to disable them. They are
  screening safeguards, not convergence criteria. Disabling them can permit
  major structural collapse; inspect trajectories before scientific claims.

On shared HPC storage, dedicated MACE processes can optionally use
[the inference isolation launcher](../../scripts/mace-inference.sh) to avoid
unnecessary Transformers imports:

```bash
bash scripts/mace-inference.sh .venv-mace/bin/python -m matsim_agents.cli explore run explore.yaml
```

Read [the isolation cautions](../../research/paper/perlmutter-e3-e4-execution-runbook.md)
first. Do not use that opt-in for a process that needs Transformers.

## Python API

```python
from matsim_agents.workflows import (
    check_exploration,
    load_exploration_config,
    run_exploration,
)

config = load_exploration_config("explore.yaml")
check = check_exploration(config)
if not check.ready:
    raise RuntimeError("\n".join(check.errors))
summary = run_exploration(config)
print(summary.model_dump_json(indent=2))
```

## Next usability milestone

Add explicit LLM-guided generation, distinguishing advisory feedback from
validated decisions that actually steer generation. The rounds below use
fixed generation settings with incremented seeds, not adaptive LLM steering.

## Multiple rounds and safe resume

Set `iterations: 3` in the same YAML. Each round generates a fresh batch using
`random_seed + round_index` (the first round uses the configured seed).
`max_relaxations: 2` permits at most six attempts across three rounds.
All rounds share one calculator cache within the running process.
The default `iterations: 1` uses exactly the same engine, progress reporting,
selection options, checkpoint format, and resume command.
Deterministic AFLOW prototypes can repeat across rounds; incrementing the seed
does not guarantee distinct structures. The default `selection: sequential`
retains generation order and permits repeated structures. Opt in to the
structural screening policy below to avoid repeating equivalent structures.

```bash
.venv-mace/bin/python -m matsim_agents.cli explore run explore.yaml
.venv-mace/bin/python -m matsim_agents.cli explore summary /path/to/printed/run-directory
.venv-mace/bin/python -m matsim_agents.cli explore resume /path/to/printed/run-directory
```

All new runs store `rounds-state.json` atomically after generation, each completed attempt, and each
round. This contains the complete configuration, incremented-seed round
candidates, results/errors, checkpoint identity, and artifact hashes.
`cumulative-ranking.json` ranks converged results across completed work; a JSON
`null` means no eligible result. No LLM, DFT, or active learning is introduced.
The cumulative report preserves all eligible attempts, including equivalent
minima, but omits polymorph-degeneracy and chemical-stability proxy claims.
Its `ground_state` entry means only the lowest-energy observed candidate.

The summary reports round completion, cumulative candidate counts, stop reasons,
and errors even for an interrupted run. Execution that has not finished is
reported as `incomplete` with a nonzero summary exit status. Backend failures
consume the round budget, are logged and saved, and do not silently become
successful calculations.

**Resume guarantees and limits:**

- Completed attempts, including nonconverged results and backend failures, are
  not rerun or retried. Unselected candidates in a checkpointed round remain
  unattempted in that round. A completed run is a no-op on resume.
- The calculator is reused across rounds in one process. A resumed process
  loads its own calculator on its first outstanding attempt; it cannot recover
  GPU/CPU model memory from a terminated process.
- Only one writer may run/resume a directory at a time (an OS file lock).
- Missing/modified input or output artifacts, a changed checkpoint, changed
  workflow/optimizer/generator source, or changed scientific dependency versions
  block resume. Do not edit the journal to bypass checks. Start a fresh run for
  changed scientific settings; the stored configuration is authoritative.
- The final cumulative ranking is hashed in the same journal update that marks
  the run complete. A missing or changed final ranking blocks completed-run
  resume. Intermediate rankings are untracked and rebuilt after interruption.
- An interruption **before** an attempt's checkpoint commits can repeat that
  attempt from its original candidate, not its last optimizer step. Outputs for
  that uncommitted attempt may be replaced. The guarantee is checkpoint-level,
  not exactly-once execution.
- Generation interrupted before its checkpoint commits can run again; random
  generation is not guaranteed bit-for-bit reproducible. Once candidates are
  checkpointed, they are reused without regeneration.
- Interrupted checkpoint preparation discards in-memory partial hashes and
  results before recording the error; only the last committed work is retained.
- Ranking is rebuilt when resuming interrupted work from stored eligible results.
  These are relative phase energies, not thermodynamic stability or evidence of
  dynamic stability.

Use the same prepared interpreter and code checkout when resuming. Static
preflight and checkpoint verification still apply. The resume command never
changes the stored budget, submits jobs, or automatically retries backend errors.

## Select diverse, fresh structures

For checkpointed multi-round runs, add:

```yaml
iterations: 3
selection: diverse
max_relaxations: 2
generation:
  max_aflow_candidates: 4
  pyxtal_count: 4
  random_seed: 123
```

These settings supplement the formula, trusted checkpoint, and relaxation
settings in your YAML. Generate more candidates than the attempt budget to give
selection a meaningful choice. Generation itself can be costly; this policy does
not automatically increase candidate counts, replenish duplicates, or submit jobs.
`selection: diverse` works with either one round or multiple rounds.

Selection runs before each round's first relaxation:

1. Compare generated structures against **previously attempted input structures**
   (including backend failures and nonconvergence), previously **converged**
   optimized structures meeting the configured force target, and earlier unique
   candidates in the current batch. Unselected candidates from earlier rounds
   are not references: if generated again, they can be selected later.
2. Skip periodic equivalents using pymatgen `StructureMatcher`: `ltol=0.2`,
   `stol=0.3`, `angle_tol=5`, primitive-cell reduction, uniform-volume scaling,
   and no supercell search. Primitive/supercell representations that reduce to
   the same primitive structure can match. Uniformly expanded versions of the
   same topology are also treated as duplicates, **not** separate density trials.
   This is approximate geometric equivalence, not proof of a shared relaxed basin.
3. Among unique candidates, choose the farthest from the prior reference set,
   then iteratively choose the farthest from references plus current selections.
   Without prior references, seed selection with the first unique candidate.
   Input order breaks exact ties. The distance is RMS difference in angstroms
   between species-averaged first-eight periodic-neighbor distances. It includes
   periodic images, works for primitive cells, and needs no SOAP dependency.
   It is a coarse local-geometry heuristic: different structures can have equal
   descriptors. Structural matching, not this descriptor, determines duplicates.

All structures must be finite, nonempty, fully periodic, and match the requested
composition. Invalid inputs stop the run with persisted error evidence rather
than silently dropping structures. Skipped duplicates consume **no** relaxation
budget. Unique candidates beyond the per-round budget are **deferred**. A round
with only duplicates completes with zero attempts; later rounds still generate
their configured batches. There is no automatic retry of previous backend errors.

The canonical `rounds-state.json` records each round's `selection`:

- `method`, initial reference paths, and selected candidate indices **in attempt
  order** (not necessarily generation order).
- One decision per generated candidate: `selected`, `duplicate`, or `deferred`;
  a reason; the representative path for a duplicate; and the RMS novelty distance
  at selection time when references exist.

`explore summary` also reports cumulative `selected`, `duplicate`, and `deferred`
counts. Counts describe decisions across generated batches: the same structure
can be deferred in one round and selected in another. During interruption,
selected counts can exceed attempted counts. Resume reuses checkpointed decisions
without rerunning selection or generation for that round; artifact checks cover
both selected and skipped inputs as well as result references.

This uses previous results as a geometric coverage reference, **not** an energy
acquisition function. It neither biases toward low energies nor creates new
structures around minima. It adds no LLM steering, uncertainty calibration,
thermodynamic stability claims, or DFT calls.

## Run format and compatibility

This implementation requires **version 2** of the exploration checkpoint.
Older single-batch evidence (`results.json` without a checkpoint) and version-1
round journals are **not supported** by summary or resume. There is no migration,
legacy reader, or automatic conversion. Start a fresh run with your YAML; do not
relabel an old checkpoint's version.

The independent advanced phase-exploration workflow is not part of this
config-driven interface. Its results are not accepted as exploration checkpoints.
Changed code or scientific dependency versions also block resume of version-2
runs. The stored configuration and source/dependency identities are authoritative.
