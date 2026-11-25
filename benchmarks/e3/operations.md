# E3 benchmark operations

E3 is an application of the framework, not a framework workflow. It runs
through `python -m benchmarks.e3.production`; the `matsim-agents workflow`
interface does not register E3 presets. See the [E3 protocol](README.md).

## E3-MLIP: current three-round benchmark

```bash
PYTHONPATH="$PWD/src:$PWD" .venv/bin/python -m benchmarks.e3.production plan --help
```

The benchmark uses the
[current production manifest](manifests/e3-mlip-production.json).
Run these commands from a source checkout with `PYTHONPATH="$PWD/src:$PWD"`.
The benchmark runs one LLM-guided stream per class: three adaptive rounds,
seed `20261113`, and at most 64 candidate relaxations per round. It does not
include a rule-based arm or multiple seeds, so report descriptive results only.

For a separate 5,000-candidate ceiling per material per round, use the
[collaborator reproduction instructions](e3-mlip-5000-three-iteration-reproduction.txt).
That opt-in profile preserves the scientific settings, 64 GPU workers and
existing time/resource limits. It uses its own manifest and outputs; reaching
5,000 completed relaxations within those limits is not guaranteed. The default
64-candidate benchmark is unchanged.

All new LLM-guided E3-MLIP profiles require explicit scientific hypothesis
records before and after every completed round, including the last round.
Inspect `round-NN/acquisition/proposal-001.json` for the full acquisition prompt,
raw reply, and pre-round claims, and `round-NN/hypothesis-assessment.json` for
the post-round assessment and its exact numerical input. Missing records,
invalid citations, or omitted previous hypothesis IDs block progress and
resume. The protocol authorizes two calls per completed round (acquisition
plus assessment); old one-call manifests are not compatible.
In E3-QE, the assessment occurs after the current QE audit and training/promotion
decision, including the final iteration. See the
[E3 protocol](README.md) for record locations and scope.

For a separate DFT-backed version with periodic Quantum ESPRESSO force audits
and accuracy-triggered MACE fine-tuning, see the
[E3-QE/MACE reproduction instructions](../../examples/active_learning/e3-qe-mace-three-iteration-reproduction.txt).
This uses a dedicated entry point, not the frozen E3 launcher or generic MD
`al run`. It needs reviewed per-material QE settings, independent validation
and a separately approved compute budget.

- Materials: `bcc_refractory`, `fcc_cantor`, `lifepo4`, `phosphorene`,
  `nb_ta_o`, `zn_formate_mof`.
- Strategy: `llm_guided_adaptation`.
- Maximum total: 1,152 candidate relaxations across all six classes.
- Use the prepared inference environment and `MACE_ARTIFACT_DIR` from the
  [Perlmutter execution runbook](../../research/paper/perlmutter-e3-e4-execution-runbook.md).
  The native runner verifies the frozen model artifact.

Each material's allocation is limited to 16 premium-QOS GPU nodes for two
hours; it uses 64 workers, a five-minute parent-enforced geometry-task cutoff,
and a hard 30-minute round deadline. The six allocations reserve at most
768 physical GPU-hours (192 GPU-node-hours). Premium billing multipliers are
separate; verify current project charges. A five-element array concurrency
limit does not guarantee simultaneous start or continuous GPU utilization.
The round deadline includes generation and LLM work as well as GPU evaluation.
GPU workers may use up to 15 minutes of that round budget to initialize MACE;
the remaining time is still bounded by the 30-minute round deadline.

```bash
# Keep the existing manifest on resume. This performs no inference or submission.
PYTHONPATH="$PWD/src:$PWD" .venv/bin/python -m benchmarks.e3.production validate \
  --manifest benchmarks/e3/manifests/e3-mlip-production.json

# Example: actual CPU generation, then authorized GPU submission for LiFePO4 (index 2).
# This stops without submission if allocation, capacity, or real generation fails.
MATSIM_E3_MLIP_RESOURCE_AUTHORIZED=m5216:premium:768 \
  PROJECT_ROOT="$PWD" bash \
  deployments/perlmutter/jobs/submit-e3-mlip-production-perlmutter.sh 2
```

Use the [CPU-generation submission helper](../../deployments/perlmutter/jobs/submit-e3-mlip-production-perlmutter.sh),
not direct `sbatch`, for each stream. Index mapping is 0 BCC, 1 FCC, 2 LiFePO4,
3 phosphorene, 4 Nb-Ta-O, 5 Zn-formate. Do not relaunch completed streams.
For a partially completed stream, set `MATSIM_E3_MLIP_START_ROUND` to its first
incomplete round; all earlier round summaries must be complete.

The CPU gate executes the selected actions with production seeds, validates
composition and geometry, and applies the unchanged symmetry-aware deduplication.
Only a full 64-candidate workload receives `generation-preflight.json`; the GPU
launcher checks this receipt and all structure checksums before reservation.
Generation time is deducted from the round's 1,800-second budget, excluding queue
waiting time. Later adaptive rounds are generated only after the previous round's
optimization results exist, and before dispatching that round's GPU tasks.

Both acquisition policies use the same executable catalog and joint capacity
validator, although the production experiment remains LLM-only. The catalog
records installed AFLOW prototypes, template availability, perturbation bounds,
capacity-group aliases, and remaining operational capacity. Small four-atom
phosphorene reconstructions and 0.5% homogeneous MOF strains have a conservative
capacity of one per group/tier; this is not a physical limit on possible structures.
The existing affine-strain bounds and equivalence tolerances are unchanged.
Generator revision 2 adds explicitly recorded, bounded non-affine displacements
of at most 0.04 Angstrom to the strained-olivine family (small relative to
interatomic bond lengths), preserving composition and the olivine template.
Pure single-antisite actions have their template-symmetry orbit capacities
of 2, 4, 8, and 16 for tiers 4, 8, 16, and 32; aliases share those limits.

Round-local `acquisition/` records contain versioned catalogs, prompts, unmodified
LLM responses saved before parsing, parsed and accepted allocations, and precise
rejection/shortfall diagnostics. Partial generation artifacts are retained and
never counted as MLIP evaluations. The deterministic policy may make another
explicit decision using generation-shortfall evidence. Acquisition remains
limited to **one allocation call per round**; the mandatory post-round
hypothesis assessment is a separate call, not permission to retry a rejected
or exhausted allocation. Replacement decisions require scientific
authorization. Missing historical raw responses are recorded as unavailable,
never reconstructed or invented, and do not satisfy the new tracking contract.

The CPU submission helper reserves the full allocation in the shared resource
ledger before `sbatch`. The GPU launcher verifies the allocation's actual
account, QOS, nodes, time limit, and reservation; it does **not** reconcile
usage on exit. After Slurm reports the allocation as terminal, manually run
`budget-reconcile` with the helper's printed reservation ID and the same ledger
path, following the [accounting command](README.md).
Until verified terminal `sacct` usage is reconciled, the full reservation
remains charged against the 768-GPU-hour cap and can block further submissions.
This includes startup failures and unavailable or ambiguous scheduler records.
The launcher uses a private, unique node-local `COPILOT_HOME`
for Copilot sessions, initializing it from the user's CLI config when present
and removing it at exit. This keeps runtime metadata locks off CFS. Array
indices 0-5 map to the six classes listed above. The scheduler rejects all six
submitted tasks at once with its current five-task submit limit; submit the
last task only after a slot opens. Do not bypass these checks or start a
second writer in the same allocation.

### Hard task and round cutoffs

Each geometry task runs in a spawned process. At 300 seconds its controller
kills the process even if it is stuck inside a force evaluation/optimizer step,
then records it as terminal and nonconverged. Partial artifacts are retained;
no final energy, force, or convergence claim is invented.

The parent supervises each complete round for 30 minutes, including structure
generation, the remote LLM decision, GPU evaluation, and cleanup. Timeout
terminates the round process group and writes an incomplete campaign report.
The 2-hour Slurm allocation is the outer limit; the three 30-minute round
budgets leave up to 30 minutes for startup and finalization.

The run writes `report.md`, `campaign-summary.json`, round manifests/summaries,
and per-task structure/trajectory/result evidence under the configured output
root. Failed or nonconverged attempts are retained and not silently replayed.
The lowest MLIP energy is not a ground-state or DFT-validation claim.

## E3-DFT/VASP: validate the existing frozen Aurora batch

```bash
# Validate only (no qstat/qsub):
python3 deployments/aurora/jobs/e4-first-batch/submit_batch.py \
  --manifest deployments/aurora/jobs/e3-round1-bcc/batch-manifest.json
```

The DFT batch is specifically
`E3-round1-bcc-01`: three BCC calculations, Aurora/PBS, HydraGNN allocation,
and the frozen 72 node-hour cap. It is **not** a generic VASP campaign or a
Perlmutter/Slurm submission batch. Follow the
[batch staging and authorization guide](../../deployments/aurora/jobs/e3-round1-bcc/README.md)
for private inputs, PAWs, executable qualification, and authorized execution.
Validation does not stage proprietary inputs.

Submission requires setting `"execution_authorized": true` in the batch
manifest after independent authorization, then adding `--submit`.

The existing submission helper retains executable/PAW/input hashes, output
collision checks, and its accepted-job ledger. Successful invocation means the
validator/submission helper exited successfully, **not** that VASP converged or
that a submitted job completed. Accepted calculation IDs are never resubmitted
by the helper. There is no automatic resume for this batch; inspect the
native scheduler/jobs and use its documented same-ID restart procedure.

