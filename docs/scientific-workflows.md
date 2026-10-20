# Scientific workflow contracts

`matsim-agents` is organized as composable scientific services. Higher-level
workflows call lower-level workflows and consume their typed results; they do
not reproduce relaxation, labeling, ranking, validation, or provenance logic.

```text
relaxation
  └─ active learning
       └─ composition and phase exploration
            └─ property-driven agentic investigation
```

## Capability and execution status

| Workflow | Public implementation | Current execution status |
| --- | --- | --- |
| Structure relaxation | `matsim-agents relax CONFIG.yaml` | End-to-end for configured MLIP, QE, and VASP environments |
| Active learning | `matsim-agents al run CONFIG.yaml` | End-to-end candidate generation, acquisition, DFT labeling, dataset growth, and optional retraining |
| Phase exploration | `matsim_agents.workflows.run_phase_exploration` | Programmatic workflow; relaxation and optional AL are composed through callbacks |
| Agentic investigation | `matsim_agents.workflows.run_investigation` | Programmatic orchestration and persisted hypothesis revision; numerical work is delegated to phase exploration |
| Cross-facility benchmark | `benchmarks/portability/run.py` | Separates deterministic workflow contracts from `--qualification compute`, which executes real MLIP and QE relaxation configs and emits a mandatory scientific summary |
| Characterization | `matsim-agents characterize CONTRACT.yaml` | Deterministic relaxation, energy-ranking, and hull-stability recipes against an explicit contract; tested with stand-in operators, not yet qualified on HPC ([Characterization contracts](characterization.md)) |
| Scientific debate | `matsim-agents debate debate.yaml` | Runs independently configured LLMs that challenge a shared hypothesis for a user-selected number of rounds, then persists the transcript and synthesis |

## Multi-model scientific hypothesis debate

The debate workflow is separate from phase exploration and numerical evidence.
It assigns the same hypothesis to at least two LLM participants. Within each
round, every participant sees the prior transcript and must identify which peer
claims it supports or disputes, expose assumptions, and propose falsification
tests. Speaking order rotates between rounds, so one model does not permanently
receive either the first-turn or last-turn advantage. In equal mode, every
participant produces an independent final verdict; no designated model is
allowed to reinterpret the panel or manufacture consensus.

```yaml
hypothesis: "Pressure stabilizes a metastable silicon phase at room temperature."
rounds: 3
output_root: ./runs
debate_mode: equal
synthesis_method: independent_verdicts
participants:
  - name: theorist
    role: first-principles condensed-matter theorist
    provider: vllm
    model: Qwen/Qwen2.5-72B-Instruct
    base_url: http://localhost:8000/v1
  - name: experimentalist
    role: skeptical high-pressure experimentalist
    provider: ollama
    model: qwen2.5:14b
  - name: reviewer
    role: independent materials-science reviewer
    provider: openai
    model: gpt-4o-mini
```

Run `matsim-agents debate debate.yaml`. The run directory records resolved
participant identities, complete ordered transcript, synthesis, and provenance.
LLM statements remain hypothesis-level evidence until calculations or
experiments verify them. `max_transcript_chars` bounds the context sent to each
model (default 60,000) without truncating the transcript saved on disk.

`debate_mode: equal` is the default. Participant roles are ignored, all models
receive the same neutral system instructions, and
`synthesis_method: independent_verdicts` saves one conclusion per model. To run
an intentionally asymmetric panel instead, select `debate_mode: role_based`,
assign participant roles, and optionally use
`synthesis_method: designated_model` with `synthesis_participant`.

For all four supported combinations, complete configuration examples, output
semantics, and artifact definitions, see
[Scientific hypothesis debate](scientific-debate.md).

For deployment qualification across locally stored first-class catalog models,
use `benchmarks/portability/all_model_scientific_debate.py`. Unlike a
user-selected debate, this benchmark selects only checkpoint directories under
the configured parallel-filesystem model root, excludes entries temporarily
disabled by catalog policy, and fails closed unless every selected model
completes at least two rounds and the saved dialogue assigns a unique
contribution ID to every model turn.

“Supported” means that the workflow contract exists. It does not imply that a
licensed VASP binary, POTCAR library, QE pseudopotentials, or a particular MLIP
checkpoint is installed at a site.

### Bounded Perlmutter numerical qualification

`deployments/perlmutter/jobs/job-qualify-real-models-perlmutter.sh` runs
`qualify_real_models.py` inside an allocated GPU node. It uses the repository's
isolated UMA, MACE, and HydraGNN environments and local model/DFT assets.
Set `PROJECT_ROOT`, `QUALIFICATION_ROOT`, and `QUALIFICATION_STAGE`.
`QUALIFICATION_SOURCE_ROOT` can point to a frozen source snapshot so queued jobs
do not execute subsequent checkout edits.

- `bootstrap`, with `QUALIFICATION_DFT_BACKEND=qe` or `vasp`, computes a real
  Si elemental reference, exercises a one-attempt UMA candidate relaxation
  budget, and collects two DFT labels. Training must defer with one permanent
  training frame and one permanent held-out frame.
- `resume` collects two more two-label batches using reproducible, distinct MD
  seeds. It requires all six labels to accumulate without moving old partition
  members, real UMA training to complete, and finite incumbent/candidate
  held-out metrics. It depends on its matching bootstrap.
- `mace` uses the accumulated QE-labelled data to exercise a two-checkpoint
  force-disagreement ensemble, one-epoch fine-tuning, candidate reload, and
  held-out comparison. It depends on the QE resume stage.
- `hydragnn` independently exercises the explicit BEST6 epoch-97 checkpoint's
  pinned OMat24 head in fp64 and bf16, requiring finite energies and forces.

Keep QE and VASP output roots separate: labels and elemental references are
method-specific and must not be pooled. Each backend performs at most seven
DFT single-points (one elemental reference plus six AL labels). Training is
one epoch per eligible iteration, and individual DFT calls time out after
30 minutes. Specify Slurm walltimes/account/queue at submission.
When a QoS submission-count limit prevents a separate resume job, the matching
bootstrap and resume can run sequentially inside one allocation, with resume
launched only after bootstrap succeeds.

Only successful stages write `<stage>-verified.json`; submission or scheduler
completion alone does not qualify the assertions. A documented promotion
rejection is valid gate evidence, not a failure of this software-path test.
This tiny Si protocol is not a convergence study, an accuracy certification,
a full multi-LLM campaign, or a multi-component phase-hull validation.

## Structure relaxation

`ScientificRelaxationConfig` supports three modes:

- `mlip`: relax entirely with the selected machine-learned potential;
- `dft`: relax directly with Quantum ESPRESSO or VASP;
- `mlip-dft`: perform an MLIP warm start, then refine the resulting geometry
  with DFT.

The configuration declares the structure, output root, geometry controls,
force tolerance, maximum steps, backend settings, and approval policy. Atomic
and cell relaxation are independent choices. Fixed atoms, charge, spin,
pressure, symmetry preservation, and parent-run lineage are explicit inputs.

See `examples/relaxation/scientific_relaxation.example.yaml` for the complete
shape. DFT modes require a `dft` block and, by default, explicit approval:

```yaml
mode: mlip-dft
structure_path: structures/Si.vasp
output_root: runs
geometry:
  relax_atoms: true
  relax_cell: false
approvals:
  before_dft: true
dft_approved: false
dft:
  backend: qe
  pseudo_dir: /path/to/pseudopotentials
```

## Active learning: three separate decisions

DFT labeling, retraining, and model promotion are intentionally independent:

1. Acquisition selects uncertain structures and DFT labels them.
2. `trainer.enabled: true` trains a candidate model from the augmented dataset.
3. `trainer.promote_model: true` allows that candidate to drive the next
   iteration, but only when `promotion_approved: true` is also recorded.

The safe default only accumulates validated DFT labels:

```yaml
trainer:
  enabled: false
  promote_model: false
  promotion_approved: false
```

This is not a frozen or incomplete form of active learning. It is a valid data
acquisition workflow whose output can be reviewed and trained offline. Enabling
training does not silently replace the deployed model.
With `trainer.enabled: false`, comparison/promotion flags are inactive and
elemental-reference or held-out comparison metadata is not required. Labels
and dataset provenance are still validated and recorded for offline training.

New DFT frames are checked for finite energy and forces, correct force shape,
and duplicate geometry. Dataset manifests preserve hashes, backend identity,
energy-reference metadata, and validation outcomes. VASP and QE energies must
not be mixed without an explicit, recorded reference transformation.
For fully non-periodic structures, duplicate/held-out overlap identities use
species-labelled interatomic-distance environments rounded to six decimal
places in angstroms. This tolerates coordinate serialization noise and detects
reordered, translated, or rotated molecular copies independently of an
unused vacuum cell. Reflections also share this distance identity; it is a
conservative overlap guard, not a unique identifier for molecular chirality.
Periodic structures retain their existing cell-metric/translated-site identity.

The full-suite CI job installs CPU-only PyTorch pinned to the HydraGNN
dependency contract. Pinned-head precision/autocast and uncertainty/dropout
regressions are mandatory CPU tests, not skipped when PyTorch is missing.
For local full-suite validation, install PyTorch before running
`bash scripts/check.sh test` or `bash scripts/check.sh coverage`.
Real VASP/QE warm-start tests still require their facility executables, model
artifacts, and configured environment variables; CPU PyTorch alone does not
enable those scientific runs.

## Combined MLIP training and independent DFT ranking

A formula campaign can run both DFT tasks in a deliberate order:

1. Explore/rank structures with the incumbent MLIP.
2. Acquire single-point DFT labels through active learning and optionally train
   a candidate MLIP from that labelled dataset.
3. If a candidate is promoted, rerun MLIP exploration with the promoted model
   before selecting structures for final DFT refinement.
4. DFT-relax those structures and compatible competing-phase references, then
   recompute the convex-hull ranking.

The final refinement stage is an independent ranking check: it records its
candidate and reference inputs, relaxed structures, energies, method signature,
reference-set ID, and the model identifier that selected the candidates. It
does not append its calculations to the active-learning training dataset. The
workflow hashes that dataset before and after refinement and fails if it
changes. Each formula writes an ordered `campaign_stages.json` record so
completed, skipped, blocked, and interrupted stages are visible. When model
promotion and final DFT refinement are both configured, post-promotion MLIP
reevaluation is required.

The Perlmutter campaign supports either an external held-out file
(`MATSIM_CAMPAIGN_PROMOTION_VALIDATION_SET`) or a fresh-label holdout
(`MATSIM_CAMPAIGN_PROMOTION_VALIDATION_FRACTION=0.2`), not both. The latter
reserves 20% of each AL label batch before training; promotion still requires
explicit approval and passing the configured energy, force, improvement, and
minimum-frame thresholds. Final DFT ranking calculations are never used for
this split.
By default, a requested post-promotion reevaluation stops the formula if no model
is promoted. `MATSIM_CAMPAIGN_CONTINUE_ON_PROMOTION_REJECTION=1` explicitly allows
final DFT ranking from the unchanged incumbent exploration instead; the skipped
post-training stage and failed promotion metrics remain recorded.

Submit the bounded seven-LLM Nb-Ta-O run with:

```bash
bash deployments/perlmutter/jobs/submit-nb-ta-o-combined-bounded.sh
```

This requests 16 GPU nodes for six hours on `m5216_g`/premium, with at most
three formula attempts, three AL iterations, and 64 DFT calculations. Each
formula selects up to ten MD frames, trains UMA for five epochs with an 80/20
split, and independently DFT-refines up to two ranked structures plus the
included elemental/competing references. UMA, three MACE models, and HydraGNN
provide MLIP screening and proxy-hull cross-checks. QE uses explicitly pinned
80/640 Ry cutoffs, a 4x4x4 solid-state mesh, and a 0.01 eV/Angstrom relaxation
force tolerance; molecular O2 uses the separate triplet/Gamma reference setup.
The hull is bounded by its generated reference coverage, not an exhaustive
phase diagram. These settings are not yet convergence-tested: final DFT
ranking is higher-fidelity validation than the MLIP proxy, not a claim of
publication-grade converged thermodynamic stability. Walltime can interrupt
the run before all stages complete.

DFT labels collected before final ranking may train the candidate MLIP; the
DFT-refinement results remain separate. If those final results are later used
for another training round, that must be a new dataset/model version and needs
a new independent final-ranking check.

### Campaign names and identifiers

The Perlmutter seven-model launcher assigns several distinct names; the output
directory name is not the campaign ID or the reference-set ID.

| Name | How it is assigned | Bounded-run example |
| --- | --- | --- |
| Slurm job name | The launcher's `#SBATCH -J` default is `campaign-formula-e2e-all`; submission can override it with `--job-name`. The bounded submission script uses `nb-ta-o-combined-bounded`. | `nb-ta-o-combined-bounded` |
| Campaign ID | For fresh discovery, the launcher passes `--campaign-id "nb-ta-o-e2e-all-${SLURM_JOB_ID}"` to the discovery driver. This ID is persisted in the campaign state. | `nb-ta-o-e2e-all-59359342` |
| Output directory name | `nb-ta-o--<workflow>--<short-git-sha>--j<SlurmJobID>`, generated automatically. | `nb-ta-o--7llm-uma-al-qe-hull--c848c79--j59359342` (illustrative) |
| DFT reference-set ID | When creating a new reference-energy set, the campaign uses `campaign-<DFT-method-signature>`. The signature is supplied through `MATSIM_CAMPAIGN_DFT_METHOD_SIGNATURE`. | `campaign-qe-pbe-pslibrary-80-640-k4-o2-triplet-gamma-v1` |

Slurm assigns `SLURM_JOB_ID` when the job is submitted. Outputs are placed under
`${RUNS_ROOT}/portability/` when `RUNS_ROOT` is set; otherwise the root is
`runs/` in the parent directory of the checkout. The saved state is
`<output-directory>/campaign/campaign_state.json`. Slurm stdout and stderr use
`<Slurm-job-name>-<SlurmJobID>.out` and `.err`, respectively, in the submission
working directory.

The directory convention is mandatory for new launches: the naming helper
generates `MATSIM_CAMPAIGN_RUN_TAG`, replacing any inherited custom tag.
The bounded submission script captures the
Git revision at submission; direct launcher submissions capture it at job
startup unless submission revision metadata was explicitly exported.
The Slurm job name and persisted campaign ID remain separate identifiers.
When `MATSIM_CAMPAIGN_STATE_SOURCE` is supplied, the launcher copies the existing
state instead of running fresh discovery, preserving its campaign ID even
though the new output directory has a new Slurm job ID.

The default workflow label `7llm-uma-al-qe-hull` denotes seven-model debate,
UMA active learning, and QE hull refinement. VASP replaces `qe` with `vasp`;
disabled refinement omits `-hull`. MLIP-only runs use `7llm-uma-screen`,
debate-only runs use `7llm-debate`, and single-call runs use `1llm-debate`.
Other screening MLIPs and detailed budgets remain in configuration rather
than lengthening the directory name.

The revision component uses `git rev-parse --short=7` on the captured commit
(Git can lengthen it to disambiguate). `run_identity.txt` records the full
source revision and dirty status, plus the runtime revision and dirty status.
Dirty status includes untracked files. This records provenance but does not
freeze the checkout: queued jobs can execute later edits, so compare source
and runtime identities when auditing results.

Historical artifacts are not renamed. Previously submitted job `59359342`
used `nb-ta-o-combined-bounded-59359342`; the descriptive example above
illustrates the new convention, not a renamed historical run.

The bounded DFT signature spells out the intended setup: `qe` (Quantum
ESPRESSO), `pbe` (functional), `pslibrary` (pseudopotential family), `80-640`
(wavefunction/charge-density cutoffs in Ry), `k4` (4x4x4 solid-state mesh), and
`o2-triplet-gamma` (the molecular oxygen reference), followed by the setup
version `v1`. This is a manually assigned label, not an automatically generated
settings digest. Keep it consistent with the actual configuration.

The reference-set ID is also a label, not a content hash or an immutable
snapshot ID: compatible competing phases may accumulate under the same ID.
Existing reference sets retain their supplied identifiers. Hull snapshot
versions distinguish the evolving hull states; method signatures, structure
hashes, and provenance provide additional compatibility and audit evidence.
See [Competing-phase reference hulls](./reference-hulls.md).

## Formation-energy comparisons and campaign promotion

### Training references

Every AL fine-tune trains against elemental DFT references, so
`trainer.enabled: true` requires `trainer.training_references` or
`trainer.validation_reference_set`. The configuration is rejected when it
loads, and `run_active_learning` checks again before any MD or DFT runs (for
callers that edit a loaded configuration), instead of failing at the first
retrain after that iteration's DFT. HydraGNN trains on formation energies
relative to the references (below); UMA and MACE use them to align the labels
to their foundation model's energy scale
([Fine-tuning UMA and MACE](#fine-tuning-uma-and-mace)). The frozen reference
snapshot is shared by every backend.

#### HydraGNN formation energies

Both routed and new-head fine-tuners now require verified elemental DFT
references and train on physical formation energies plus unchanged forces.
Mixture-fitted composition offsets have been removed: a single-formula dataset
cannot uniquely identify physical elemental coefficients.

The shared formation-label preparation layer is implemented. Given a verified
native-total-energy dataset sidecar and compatible elemental DFT manifest, run:

```bash
python -m matsim_agents.datasets.formation_training \
  --dataset /path/to/raw-dft.extxyz \
  --elemental-reference-manifest /path/to/elemental-references.json \
  --output-dir /path/to/new-formation-snapshot
```

The output directory must not already exist. It contains `formation.extxyz`,
a method/hash/split-role sidecar, `energy-convention.json`, and a self-contained
copy of the elemental-reference geometries and manifest. Energy labels are
total-cell formation energies; forces, available stress, and geometries are
preserved. Each frame also records its original DFT total and elemental baseline
so total energy can be reconstructed. Raw input data and partition membership
are unchanged. Missing references/forces, incompatible methods, invalid labels,
and already-transformed input are rejected.

This command does not launch DFT, select elemental phases, or train a model.
The fine-tuners accept the **raw total-energy** dataset and the required
`--elemental-reference-manifest` argument; they create their own immutable
`training-reference/` snapshot before building graphs. Do not pass an already
converted dataset: double conversion is rejected. The standalone comparison
driver forwards its existing elemental manifest to both fine-tuners.
Force labels are read without applying ASE constraints, so constrained atoms
retain their raw DFT forces in both the snapshot and fine-tuning targets.

The shipped AL YAMLs use `finetune_hydragnn_newhead.py` and one process for
formation-aware retraining. Configurations that train with a branch-weighting
MLP should select `finetune_hydragnn.py` instead and provide that MLP checkpoint.
HydraGNN's `examples/mptrj/train.py` is a pretraining entry point, not an AL
fine-tuner, and is not used by these configurations.
AL forwards the configured charge/spin conditioning to both built-in trainers
and `mlip.hydragnn.ft_repo` to the new-head trainer, including through HPC launchers.

AL training requires references even with comparison and promotion disabled.
Configure a verified manifest explicitly:

```yaml
trainer:
  training_references:
    manifest: /path/to/elemental-references.json
```

Alternatively use an approved elemental-reference phase plan automatically:

```yaml
trainer:
  training_references:
    phase_plan: /path/to/approved-phases.yaml
    cache_dir: /path/to/elemental-dft-cache
    phases_approved: true
    dft_approved: true
    max_dft_calculations: 4
```

The reference calculation cap is **separate** from the compound-label cap.
Uncached calculations require explicit DFT approval; fully cached references
can be reused with `dft_approved: false` and a zero cap. The existing
`validation_reference_set` can supply training references when no explicit
training reference source is configured. AL freezes the selected reference
snapshot under its output directory and rejects changed reference inputs or
DFT protocols on subsequent iterations/restarts; use a new dataset for changes.
Dataset method metadata records the label-stage element scope, including
held-out species, so reference preparation uses the same potential set as
labeling rather than only the training split's elements. Relative reference
cache paths are resolved before launching relaxation or single-point jobs.

Saved checkpoints bind `energy-convention.json` to the checkpoint hash,
inference configuration, the exact bundled elemental-reference manifest, and
the verified training snapshot. AL also checks that the emitted bundle's source
dataset hash matches the requested dataset before accepting the candidate.
Neural outputs are
total-cell formation energies. ASE-facing calculators add the recorded
elemental baseline **exactly once**, returning DFT-reference total energies;
existing evaluation subtracts its own model elemental references once.
Forces/stress and dropout access are preserved. New-head checkpoints are
auto-detected, and routed checkpoints retain their trained-head restriction
and frozen routing MLP. Unsupported elements, incompatible/tampered artifacts,
and fine-tuned checkpoints without a verified energy convention fail explicitly;
unmarked foundation checkpoints are used unchanged.

Checkpoint reconstruction materializes HydraGNN's lazy graph-conditioning
projectors from saved weight shapes before loading weights. Single-head resume
and inference use strict loading; declared routed/selected-head inference reloads
strictly after the upstream loader so lazy backbone weights cannot be dropped.

Both fine-tuners honor the training configuration's resolved autocast dtype
for dry-run, optimization, and validation losses. BF16 keeps FP32 master
parameters and uses autocast for compute; single-head checkpoint inference
also honors that setting. FP32/FP64 paths do not enable autocast.

`trainer.train_script` must be one of the built-in fine-tuners,
`training/finetune_hydragnn.py` or `training/finetune_hydragnn_newhead.py`.
Launchers receive the manifest as positional argument eight, then optional
checkpoint, branch-MLP, ft_repo, charge, spin, and HydraGNN-root arguments, and
invoke the fine-tuner with `--output-dir`/`--gfm-logdir`. The fine-tuners are
single-process, so configure one node/one rank when using a launcher.

#### Approved elemental DFT prerequisite

Declared HydraGNN checkpoints with bundled routing artifacts can be used by
direct relaxation, chat, and the LangGraph executor without an external
BranchWeightMLP path. Artifact integrity is still verified when loading the
calculator. Branch-weight confidence is unavailable when fewer than two trained
branches are active; relaxation reports no `top_branch_weight` in that case.

`matsim_agents.active_learning.elemental_dft` calculates or reuses reference
energies for an approved phase list. Every phase explicitly declares whether its
input geometry is already verified as converged or requires relaxation. The
verified path remains a fixed-geometry single point; the other path performs a
DFT relaxation, requires the requested force threshold to be met, then performs
a production-protocol single point on the relaxed geometry. It does not search
for additional structures or magnetic states, apply fitted corrections, or
prove an elemental ground state: selection is the lowest energy per atom among
the declared phases at zero pressure.

The phase-plan YAML has this schema (paths are relative to the plan file):

```yaml
protocol: declared_phases_zero_pressure_v3
phases:
  - phase_id: nb-bcc
    element: Nb
    structure_path: references/Nb-bcc.extxyz
    kind: bulk
    geometry_status: requires_relaxation
    magnetic_state: nonmagnetic
    relaxation_force_tolerance_eV_per_A: 0.02
    relaxation_stress_tolerance_eV_per_A3: 0.01
    relaxation_max_steps: 200
  - phase_id: ta-bcc
    element: Ta
    structure_path: references/Ta-bcc.extxyz
    kind: bulk
    geometry_status: verified_converged
    magnetic_state: nonmagnetic
  - phase_id: oxygen-triplet
    element: O
    structure_path: references/O2-vacuum.extxyz
    kind: molecule
    geometry_status: requires_relaxation
    magnetic_state: triplet
    relaxation_force_tolerance_eV_per_A: 0.02
    relaxation_max_steps: 200
    qe_magnetic_settings:
      nspin: 2
      tot_magnetization: 2
    kpts: [1, 1, 1]
```

`geometry_status` is required: use `verified_converged` only when convergence
has been independently established and attested; otherwise use
`requires_relaxation`. Bulk relaxation defaults to relaxing atoms and cell
(`vc-relax`/VASP `ISIF=3`); set `relax_cell: false` to retain the input cell.
Bulk cell relaxation requires an explicit `relaxation_stress_tolerance_eV_per_A3`.
The maximum absolute final stress component must be within this threshold; VASP
and QE stress are checked in eV/Å³ and the measured stress is retained in the
relaxation cache record. The tolerance participates in the cache identity.
Molecules always retain their cell and vacuum while relaxing atomic positions.
The force tolerance and ionic-step limit are explicit per phase. The example is
a declaration format, not supplied physical reference data or a general
convergence recipe. Molecular cells need adequate vacuum. For VASP, replace
QE overrides with `vasp_magnetic_settings: {ISPIN: "2", NUPDOWN: "2"}` and
declare k-point settings in the shared VASP configuration instead of phase
`kpts`. Magnetic-state text documents intent; the actual DFT spin controls must
be supplied explicitly in the shared configuration or allowed phase overrides.
Wrong-backend overrides are errors.

Supply a standalone YAML `DFTConfig` block (the contents of AL `dft`, not a
whole AL configuration), with absolute executable, wrapper, pseudopotential,
and template paths. Generated QE inputs must explicitly pin both cutoffs,
k-points, and occupations to avoid composition-dependent automatic defaults.
QE templates cannot be combined with magnetic overrides because those overrides
would be ignored. Phase overrides are restricted to declared spin controls and
QE reference k-point sampling; they cannot change the functional or potentials.
Unverified QE geometries require generated QE inputs; template-based QE inputs
cannot be safely transformed into relaxations and are rejected before DFT work.
Trusted geometries can continue to use the fixed-geometry path with templates.
The common compound method signature and each reference's actual method
signature are both recorded, including the explicit state/sampling exceptions.

```bash
python -m matsim_agents.active_learning.elemental_dft \
  --phase-plan /path/to/approved-phases.yaml \
  --config /path/to/dft-and-resources.yaml \
  --elements Nb Ta O \
  --cache-dir /path/to/shared-reference-cache \
  --output-dir /path/to/new-reference-snapshot \
  --approve-reference-phases --approve-dft \
  --max-dft-calculations 5
```

The `--config` file holds a top-level `dft:` block and a `resources:` plan
with a `dft` pool ([resource dispatch](resource-dispatch.md)); the first DFT
job group of that pool runs every reference calculation.
Run on allocated compute resources with the backend's wrapper/environment,
not a login node. Phase-list approval is always required. Without
`--approve-dft`, all phases must already have valid compatible cached results.
The calculation cap counts missing unique DFT invocations and fails before
launching if the full requested reference set exceeds it. An uncached phase
requiring relaxation normally costs two invocations: the relaxation and its
final single point; a verified geometry costs one. Cache identities include
the input/relaxed geometry, method/executable/potential/template hashes, and
declared phase and relaxation controls; per-entry file locks serialize shared
cache calculations. Malformed cache entries fail explicitly rather than
silently recalculating. Only force-converged relaxations and successful finite
single-points are cached; failures expose their work directories. This stage's
DFT calls are separate from AL label budgets and must be budgeted with its own
cap.

The output directory must be new. Its `elemental-references.json` is compatible
with the formation-label preparation command and includes all tested phases,
selected phases, calculation locations, actual reference method signatures,
and both source and final geometry hashes, plus relaxation provenance.
Selected final geometries are copied into the snapshot.
All input formats supported by ASE, including CIF and POSCAR, are serialized
as extxyz before caching the single point. The source hash remains separate
from the hash of the serialized geometry used by the cache and snapshot.
Formation-label snapshots retain this selection audit. Cache reuse does not
change an already-created reference/training snapshot or choose new endpoints
silently. QE and VASP caches remain method-separated.

The implemented replacement workflow:

- Require an approved list of pure-element reference phases covering all
  training species, independently of comparison or promotion being enabled.
- Reuse verified compatible DFT calculations or, with DFT approval, calculate
  missing references before training. References must declare their structures,
  magnetic states, DFT method, and any corrections. Molecular references such
  as triplet O2 must be explicitly declared rather than treated as bulk crystals.
- Select the lowest DFT energy per atom among the declared compatible phases
  for each element under a zero-pressure energy protocol. This is the lowest
  among tested phases, not a claim of a global ground-state search. Other
  pressure or temperature conditions need an explicit corresponding protocol.
- Preserve raw DFT totals and derive total-cell formation-energy labels as
  `E_formation = E_DFT - sum(N_element * e_element_reference)`. Forces remain
  unchanged. Actual cell atom counts must be used, including supercells.
- Apply one reference set to all polymorphs and compatible formulas; do not
  independently zero each polymorph. Keep elemental reference selection fixed
  within a training snapshot and record its provenance.
- Remove mixture-fitted offsets from both HydraGNN trainers and record the new
  declared DFT formation-energy convention in each fine-tuned checkpoint.
  Adapting to this convention does not assert that the pretrained head used it.
- Make checkpoint reload, materials ranking, evaluation, and hull analysis
  honor that convention. Reconstruct DFT-reference total energies only where
  needed, and prevent a second elemental subtraction from formation outputs.

The formation-energy convention is HydraGNN's; UMA and MACE keep their
foundation's total-energy convention and align the labels to it instead
([Fine-tuning UMA and MACE](#fine-tuning-uma-and-mace)). Per-formula
cumulative collection and permanent train/held-out membership remain unchanged.
Validation must cover multiple polymorphs, cell multiplicity, reference reuse,
missing or incompatible references, approval gates, unchanged forces, and
checkpoint reload without double referencing. CPU contract tests cover the
reference/training/inference wiring; end-to-end real GPU model and DFT
qualification of this new workflow remains pending.

### Fine-tuning UMA and MACE

A foundation model's energies follow its pretraining DFT setup (UMA OMat and
MACE-MP: VASP PBE). Labels from Quantum ESPRESSO, or VASP with other settings,
differ from it by large per-element constants; training on raw totals would make
the model absorb them. The UMA and MACE fine-tuners
(`matsim_agents.training.finetune_uma` / `finetune_mace`, both requiring
`--elemental-reference-manifest`) therefore shift every label by the elemental
reference offsets:

```text
E_label = E_DFT - sum over atoms i of (e_DFT[el_i] - e_model[el_i])
```

where `e_DFT` is the DFT energy per atom of each declared elemental reference
and `e_model` the base model's energy per atom on the same geometry. DFT
formation energies relative to the references are unchanged, and the fine-tuned
model stays on its foundation's scale. The alignment is written to
`energy-alignment.json` beside the checkpoint, with the hashes of the training
dataset and reference manifest; `retrain_uma` / `retrain_mace` reject a model
whose recorded hashes do not match what they asked for. Fine-tuning a model that
has an `energy-alignment.json` reuses its recorded model references, so repeated
fine-tunes keep one scale. Promotion comparisons are unaffected: each model's
formation energies use its own predicted elemental references.

Other details:

- **UMA**: the energy loss is per atom (`(E_err / N_atoms)^2`, so large cells do
  not dominate) plus `force_weight` times the force MSE. 10% of the frames (at
  least one) are held out, scored after every epoch, and the weights of the best
  epoch are kept.
- **MACE**: `mace_run_train` uses the foundation's atomic energies
  (`--E0s foundation`) instead of refitting them from the training set
  (`--E0s average`), which cannot separate the elements when the AL data cover
  few compositions; multihead replay of pretraining data is off (compute nodes
  are offline). When every frame carries a DFT stress, stress is trained too
  (`--loss stress`). The retrain fails if no `mace_finetuned.model` is written.
- **Adapters**: `trainer.finetune_mode: lora` (default) trains low-rank adapters
  and the heads, which keeps more of the pretraining on small AL datasets;
  `full` trains all weights. HydraGNN uses its fine-tuner's own strategy.
- **Start point**: by default each retrain starts from the current model (the
  last promoted one). `trainer.start_from: foundation` starts every retrain from
  the model the run was configured with, on the cumulative dataset, so
  successive fine-tunes do not compound.
- **Stress**: the UMA and HydraGNN fine-tuners train on energies and forces only.
- **Launchers**: a UMA or MACE `train_launcher` receives
  `<train_script> <dataset> <out_model_dir> <base model fields> <epochs> <nodes>`
  `<ranks> <reference_manifest> <finetune_mode>` and must write the model and
  `energy-alignment.json` to `<out_model_dir>`
  (see `deployments/perlmutter/launchers/_mace-train-step-perlmutter.sh`).

### Keeping the surrogate loaded across iterations

The AL loop loads the surrogate on the driver for scoring and in one worker per
mlip device for MD. When the resource plan gives the `mlip` pool devices that
neither the `dft` nor the `training` pool uses, these models and the MD workers
stay loaded across iterations and are reloaded only when a candidate is
promoted; the promotion comparison then reuses the loaded incumbent (unless it
was loaded with MC dropout for acquisition). When the devices are shared, the
models are released before DFT and reloaded each iteration, as before. The run
log states which applies. The retrained HydraGNN candidate is checked from its
recorded hashes rather than loaded; the promotion comparison loads it.

### Current promotion and evaluation behavior

Offline evaluation of fine-tuned models (which split, which partition selects,
and which is the untouched test) is described in
[MLIP evaluation](./mlip-evaluation.md). In the AL loop below, promotion uses
held-out validation labels only; those are held out frame by frame, so they are
correlated with training frames of the same trajectories.

Promotion requires both held-out formation-energy MAE (eV/atom) and
force-component MAE (eV/Å) to meet their absolute limits and not worsen relative
to the incumbent, apart from an absolute `1e-12` numerical tolerance in each
metric's units. At least one must also improve by
`promotion_min_relative_improvement` (default `0.05`, meaning a 5% relative
MAE reduction). Improvement in one metric cannot compensate for regression
in the other. Equal performance, two worse metrics, or improvements below the
configured threshold retain the incumbent, with explicit rejection reasons.
Finite, nonnegative MAEs and the existing minimum evaluated-frame counts are
still required.

Configure the threshold in AL `trainer` or campaign `retraining` YAML:

```yaml
promotion_min_relative_improvement: 0.05
```

The Perlmutter CLI exposes `--promotion-min-relative-improvement`; the campaign
launcher uses `MATSIM_CAMPAIGN_PROMOTION_MIN_RELATIVE_IMPROVEMENT` (default
`0.05`). Values must be finite and between zero and one. Zero still requires a
strict reduction larger than the numerical tolerance. An exact-zero incumbent
MAE cannot improve, but the other metric may qualify. This threshold is a policy
choice, not statistical significance or broad scientific qualification.

Promotion never accepts an MAE regression on either metric. Training,
cumulative collection, and permanent held-out partitions are unchanged;
rejected candidates do not replace the incumbent.

Scientific MLIP-versus-DFT energy comparisons use formation energies, not
raw energy zeros or offsets fitted to training/test compounds. Every energy
test must provide DFT calculations for its pure-element reference geometries
using the same DFT setup as its compound labels. Each MLIP evaluates those
exact fixed geometries first and subtracts its own elemental energies; DFT
subtracts its own references. Molecular references such as O2 are normalized
by their actual atom count. Missing elements, impure references, changed
geometry hashes, non-finite energies, and missing manifests are errors.

Supply a JSON manifest, with paths relative to the manifest:

```json
{
  "backend": "qe",
  "method_signature": "qe-pbe-reference-protocol-v1",
  "references": {
    "Nb": {
      "structure_path": "Nb.extxyz",
      "structure_sha256": "<SHA-256 of the fixed geometry file>",
      "energy_eV": -20.0
    },
    "O": {
      "structure_path": "O2.extxyz",
      "structure_sha256": "<SHA-256 of the fixed geometry file>",
      "energy_eV": -10.0
    }
  }
}
```

The energies above are illustrative total-cell values, not physical reference
data. Populate them from converged DFT calculations on the declared geometries,
with full coverage of the elements in the test. The manifest is not a request
to launch DFT: the caller must supply those calculations and ensure method
compatibility with the compound dataset. Use a separate manifest for each
DFT protocol being compared. Baseline errors remain available as diagnostics.
File-based energy evaluation, fine-tune/evaluation, and promotion require a
`*.extxyz.manifest.json` compound-dataset sidecar. They check its `sha256`,
`dft_backend`, and `method_signature` against the data and elemental manifest
before inference. Missing or incompatible metadata is an error, not a warning.
Campaign promotion checks supplied validation metadata before starting the
loop. Fine-tune/evaluation writes matching sidecars for newly generated splits;
fraction-based active learning also maintains separate training and held-out
sidecars on each iteration, checking the held-out hash and method before append.
New split sidecars explicitly record `split_role: training_pool` for training
and `split_role: validation` for held-out data, including AL-generated splits.
Training accumulates within each formula's AL dataset across iterations and
restarts; this does not pool labels across formulas. `validation_fraction`
targets the cumulative held-out fraction by assigning only newly accepted
frames. Existing assignments are permanent: training frames never move to
validation, held-out frames never enter training, and duplicate labels are
rejected against both partitions. The target fraction is approximate for
small batches and cannot retroactively repartition prior data.

Readiness is checked against the cumulative data, not the latest batch.
Training needs at least two accumulated training frames; energy comparison
also needs at least `promotion_min_evaluated_frames` energy-labelled and
force-labelled held-out frames. Valid small batches are saved with
`training_status: deferred` and explicit `training_deferred_reasons` until
those requirements are met. Iteration state records both new-frame counts and
`n_training_frames_total` / `n_validation_frames_total`. No candidate is trained
or promoted during deferral, and the incumbent remains active. The
campaign skips post-training reevaluation when training is deferred; this is
distinct from rejection of an evaluated candidate, which still follows the
configured promotion-rejection policy. A one-iteration
run acquiring two fresh labels with a 20% split therefore retains one training
and one validation frame and defers training; extend the iteration count or
resume the same formula dataset to accumulate more data. DFT failures and
duplicate rejection can delay readiness further. These are software minimums,
not evidence that such a small validation set scientifically qualifies a model.

Eval-only reuse requires existing, hash-matching split sidecars. Data without a
recorded DFT method signature is rejected; do not infer compatibility from the
elemental manifest alone. The caller
remains responsible for convergence of the supplied DFT calculations.

Energy evaluation and fine-tune/evaluation require
`--elemental-reference-manifest`; force-only evaluation does not require
elemental references or model energy predictions. Campaign promotion uses
`--promotion-validation-reference-set` / `trainer.validation_reference_set`
for this JSON manifest, not for training-partition offset fitting. Perlmutter
retraining submissions that request model promotion or enable
`trainer.compare_after_training` in the selected AL config require
`MATSIM_CAMPAIGN_PROMOTION_VALIDATION_REFERENCE_SET` before submission. The
all-model launcher checks this before starting model servers.
Fine-tune/evaluation launchers require `ELEMENTAL_REFERENCE_MANIFEST`, or an
existing `elemental_references.json` in each dataset's case directory.
Promotion energy thresholds now apply to
`formation_energy_mae_eV_per_atom`; raw total-energy errors remain diagnostic.
Evaluation outputs no longer contain `energy_*_per_atom_shifted` fields; use
`formation_energy_*_per_atom`. Reassess historical thresholds under this
convention. Force comparisons and within-method polymorph/hull
ranking are unchanged; final DFT ranking results still never enter training.

Nb-Ta-O reference preparation merges curated phases before unary structure
deduplication, retaining curated entries ahead of equivalent bootstrap,
AFLOW, or pyXtal candidates so duplicate unary phases are not relaxed twice.

Campaigns use this comparison for held-out incumbent/candidate evaluation and
promotion only. Active-learning acquisition, force metrics, same-composition
polymorph ranking, and method-specific hull construction retain their existing
semantics. Training without energy comparison does not require elemental
references. Before training a candidate that will be compared, the active-learning
loop checks element coverage and requires the reference backend/method signature
to match the generated compound labels. Use the campaign's recorded DFT method
signature, not a descriptive label from an unrelated hull protocol.

For method `m`, the formation energy per atom is
`(E_compound_m - sum_i(n_i * mu_i_m)) / N`, where `mu_i_m` is that method's
elemental energy per atom on the declared geometry. Different MLIPs and DFT
protocols retain separate baselines; no cross-method totals are pooled.
This removes additive elemental energy zeros, not approximation errors.
If the fixed reference phases are not elemental ground states, describe the
result as formation energy relative to those declared phases rather than
claiming a ground-state thermodynamic formation energy.

Pure-element baseline errors remain visible because cancellation can yield
good formation energies even when individual elemental predictions are poor.
Formation-energy accuracy does not establish hull stability, force accuracy,
or accurate same-composition energy differences. Promotion retains force
and incumbent-regression gates rather than relying on formation energy alone.

For Codabench, the same convention applies to Task 1 and the energy inputs to
Task 5. See the [participant guide](../benchmarks/codabench/starting_kit/README.md#elemental-reference-manifest)
for the reference schema, inference commands, outputs, and release requirements.

## Campaign benchmark comparisons

Campaign benchmark comparisons must contain observations from a single
pre-registered protocol digest. Both paired differences and statistical
comparisons reject mixed digests, including incomplete or metric-missing
observations. Compare each protocol separately rather than pooling effects
across different candidate pools, budgets, models, or DFT methods.
Treatment and control must be distinct arms, and paired metric values and
their differences must be finite before statistical analysis.
Paired sign-flip tests enumerate all assignments for up to 16 pairs. Larger
comparisons sample assignments and report the finite-sample-corrected
Monte Carlo p-value `(exceedances + 1) / (samples + 1)`, rather than a
potentially zero raw exceedance fraction.

## Phase exploration

`PhaseExplorationPolicy` controls relaxation, label collection, training,
promotion, and post-promotion reevaluation:

```yaml
relax_structures: true
active_learning: false
retrain_mlip: false
promote_model: false
reevaluate_after_retraining: false
ranking_mode: relative_phase_ranking
```

Retraining requires active learning. Re-evaluation requires `promote_model: true`
(which requires retraining) at configuration time, and successful model promotion
at runtime unless `continue_on_promotion_rejection: true` explicitly retains
incumbent results without reevaluation. Compute budgets may cap candidates, MLIP
relaxations, DFT calculations, AL iterations, and node-hours.
Candidate-count limits stop between formula attempts. The campaign orchestrator
passes remaining DFT and candidate-MLIP relaxation allowances to formula runners.
`n_random` remains the requested random-seed count per composition, in addition
to prototype seeds; it is never silently lowered to fit a relaxation budget.
`max_mlip_relaxations` caps candidate relaxation attempts across formulas,
including failures and both initial and post-promotion exploration. Each
budgeted attempt is checkpointed before launch, so interrupted attempts remain
charged on restart. Custom budgeted formula runners must honor `mlip_allowance`
and call `on_mlip_relaxation_attempt` before each attempt.
Generated seeds beyond the allowance remain recorded in
`unrelaxed_candidate_paths`, with `relaxation_budget_exhausted: true`; rankings
cover only the attempted subset, not the full generated search space.
This budget does not count optimizer steps, MD sampling, perturbation robustness
trials, or model-specific unary reference qualification. Those separate stages
retain their own configuration controls. Node-hour limits remain between-formula
stopping thresholds rather than interruptible hard caps; use scheduler wall-time
limits for allocation control. Historical campaign `n_mlip_relaxations` counts
recorded successful selected-stage results, whereas new attempts count both
exploration stages and failures; old missing attempt counts cannot be recovered
from that field alone.
Completion progress-callback errors are logged and stored in exploration
`callback_failures`, separately from relaxation `failures`. They do not
invalidate a successful relaxation or add an extra attempted/failed candidate
to the audit counts; the exploration continues with later candidates.

The AL callback receives `(composition, output_dir, retrain, promote_model,
promotion_approved)`. It must honor the promotion request and approval before
any model-changing side effects; the campaign adapter rejects mismatched controls
before invoking AL. A post-call guard additionally rejects unauthorized promotion
reported by the callback.

`relative_phase_ranking` compares converged candidates within one exploration.
It is not a convex-hull claim. `convex_hull_ranking` additionally requires
method-compatible elemental and competing-phase references. Residual forces
filter unconverged structures; they are not added to formation energies.
Relaxed DFT references, including unary endpoints, must match the declared
reduced composition before their energies enter the hull.
Campaign hull snapshots include only convex-hull reports matching the active
reference-set identifier. Incompatible reports remain stored for provenance,
but do not enter snapshot energies, vertices, or decomposition products.

## Agentic investigation

The investigation layer stores the original user request, the LLM-generated
scientific hypothesis, explicit property tasks, each phase-exploration result,
and subsequent hypothesis revisions. A new interaction can consume a previous
result without overwriting it. Run identifiers combine a UTC timestamp and a
random suffix, preventing concurrent studies from sharing a directory.

LLM proposals have `hypothesis` evidence. They do not become DFT or
experimental claims merely because a lower-level workflow was dispatched.
Formula merging rejects undeclared elements outside the campaign's fixed
element set even for directly supplied proposals and when charge-balance
screening is disabled. Rejected proposals remain recorded as inactive
candidates with their rejection reasons and participant attribution.

## Approval, evidence, and failure semantics

`ApprovalPolicy` exposes gates before DFT, retraining, and model promotion.
`EvidenceLevel` distinguishes hypotheses, MLIP predictions/relaxations,
low-fidelity DFT, converged DFT, higher-accuracy DFT, and experiment.
Stability reports carry their evidence level; only DFT convex-hull reports
enter a campaign's hull, and only converged DFT can establish thermodynamic
stability in a [characterization](characterization.md).

Every failed or rejected result must include a reason. Unconverged jobs remain
in the run record and are excluded from rankings rather than silently dropped.

Related documentation:

- [Characterization contracts](characterization.md)
- [Distributed DFT dispatch](distributed-dft-dispatch.md)
- [Run artifacts and restarts](run-artifacts-and-restarts.md)
- [Cross-facility portability benchmark](../benchmarks/portability/README.md)
