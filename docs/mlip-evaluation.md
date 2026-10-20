# Evaluating fine-tuned MLIPs: splits, roles, and test domains

How well a fine-tuned MLIP does depends on what it is asked to do. Interpolating
between frames of the trajectories it was trained on is easier than predicting
a new trajectory, a new structure, a new composition, or a new temperature. This
guide describes how `matsim-agents` splits DFT-labelled data so that an
evaluation answers one stated question, keeps model selection away from the final
test, and records what the result does and does not show.

The pieces are:

- `matsim_agents.datasets.splits`: split strategies, partitions, leakage checks,
  and the split manifest;
- `matsim_agents.active_learning.finetune_eval`: fine-tuning and evaluation of
  one backend with those partitions (`--evaluation-config`);
- `matsim_agents.active_learning.eval_report`: per-group reports, the backend x
  strategy table, and active-learning learning curves;
- `matsim_agents.active_learning.uncertainty.calibrate_uncertainty`: how well
  ensemble or MC-dropout disagreement tracks the actual error, on validation data.

## Split strategies

| Strategy | Question it answers | Groups | Needs on every frame | Limitation |
|---|---|---|---|---|
| `random_frame` | Accuracy on frames like the training frames (interpolation) | each frame | nothing | Held-out frames are correlated with training frames of the same trajectory; this says nothing about new trajectories, structures, or compositions. |
| `temporal_block` | Accuracy later in the same trajectories | trajectory, split in time | `trajectory_id`, `time_index` | Extrapolation in time within trajectories, not transfer to new ones. A `purge_gap_frames` gap between blocks reduces correlation across the boundary. |
| `trajectory_holdout` | Accuracy on independent trajectories | `trajectory_id` | `trajectory_id` | Same compositions and conditions as training. |
| `structure_holdout` | Accuracy on new initial structures or structure families | `initial_structure_id` (default) or `structure_family_id` (`group_by`) | that field | Only as independent as the declared families. |
| `composition_holdout` | Accuracy on new compositions | reduced formula (`composition.mode: exact`) or atomic-fraction bins (`mode: bin`, `bin_width`) | nothing (from the atoms) | Exact formulas treat `Nb2O5` and `Nb4O10` as one composition; bins group nearby alloys. |
| `condition_holdout` | Accuracy under new physical conditions | values of the declared `conditions` (each with a `unit`, optionally binned with `bin_width`) | each condition field | Only the declared conditions are held out. |
| `al_iteration_holdout` | Accuracy on later acquisition batches | `al_iteration`, in chronological order | `al_iteration` | Earlier iterations train, later ones test: never a random permutation of iterations. |
| `generic_group` | Holdout of any declared metadata field | `group_by` | `group_by` | As meaningful as the field. |

The active-learning loop records `config_id`, `candidate_id`, `trajectory_id`
(one trajectory per seed and iteration), `time_index` (the MD step),
`initial_structure_id` (the seed), `al_iteration`, `temperature_K`, and
`source_campaign` on every label it adds. Other datasets must carry the fields
their strategy needs; a missing field is an error that names the frames, never a
silent fallback to a random split.

A held-out group is held out from fine-tuning only. Foundation models may have
been pretrained on similar structures, so no split here is a strict
out-of-distribution test; every manifest says so.

## Roles

- `train` updates the weights. Anything fitted from labels is fitted here; the
  energy references used by the fine-tuners come from elemental DFT
  calculations, not from these labels.
- `validation` serves early stopping (each fine-tuner takes it with
  `--validation-dataset` instead of holding out training frames at random),
  backend and hyperparameter comparison, uncertainty calibration, and the choice
  between the foundation and the fine-tuned model.
- `test` is evaluated only after that choice has been written and frozen.

Fractions are configurable and need not be 70/15/15; a role may be empty (fraction
0). Group strategies assign whole groups, so frame counts only approximate the
fractions. A role with a nonzero fraction but no group is an error; fewer than
`min_groups_per_role` independent groups is reported as a limitation.
`explicit_groups` assigns named groups to roles instead of drawing them; it must
name every group exactly once.

## Test domains

`test_domains` are external, immutable datasets evaluated alongside `test`, each
with a membership rule: under `membership: composition_holdout`, for example, no
composition of the domain may appear in train or validation. `random_frame`
membership requires only distinct configurations. Domains may share frames with
each other only when one lists the other in `allow_overlap_with`. A domain must
carry a dataset manifest with the same DFT method as the split.

## The split manifest

A split writes `train.extxyz`, `validation.extxyz`, and `test.extxyz` (the roles
in use), each with a dataset sidecar, and `split-manifest.json`:

- the source dataset path, SHA-256, frame count, and its DFT provenance;
- the strategy, the declared objective, the full configuration and seed;
- per role: the partition file and its hash, frame and independent-group counts,
  and the configuration ID and group of every frame;
- every frame's trajectory, time index, AL iteration, initial structure,
  composition, conditions, and source campaign, where present;
- purged frames and why (temporal gaps);
- test domains, their hashes, groups, and overlap permissions;
- checks with `pass`, `warn`, or `fail`, and the resulting limitations;
- the schema version, the code commit, and a content hash.

The manifest is content-addressed: writing a different split into the same
directory is an error, and `load_split` refuses a manifest, partition, or test
domain that changed after the split. Every backend uses the same partition files,
so HydraGNN, UMA, and MACE are evaluated on identical frames: run the first
backend with a split configuration and the others with `split_manifest:`
pointing at its `split/split-manifest.json`.

### Checks

| Check | Result |
|---|---|
| A configuration ID in two roles | fail |
| A group in two roles (group strategies) | fail |
| A domain configuration or (under its membership) group in train or validation | fail |
| Two domains sharing configurations without permission | fail |
| A domain with a different DFT method | fail |
| Later roles not strictly after earlier ones (`temporal_block`, `al_iteration_holdout`), or a purge gap shorter than declared | fail |
| No group in a role | fail |
| Fewer than `min_groups_per_role` groups in a role | warn |
| Trajectories with frames in several roles under `random_frame` | warn (correlated frames) |
| Cross-role pairs closer than `near_duplicate_tolerance_A` | warn |

The near-duplicate check compares frames of the same composition and (rounded)
volume by their sorted per-atom nearest-neighbour distances under periodic
boundary conditions, so it ignores atom order and translation; it is a
diagnostic with a stated tolerance (0.01 A RMS by default), not a proof that two
structures differ. A split that fails a check is written for inspection and then
rejected.

To make a split without fine-tuning:

```bash
python -m matsim_agents.datasets.splits --dataset dataset.extxyz \
  --config examples/evaluation/trajectory-holdout.yaml --output-dir splits/trajectory
```

## Selection and the final test

`finetune_eval` runs one backend per invocation:

1. It fine-tunes the foundation model on `train`, with `validation` for early
   stopping.
2. It evaluates the foundation ("baseline") and the fine-tuned model on
   `validation` and applies `selection` (`min_relative_improvement`, absolute
   limits, `min_evaluated_frames`): the fine-tuned model is selected only if it
   is not worse in formation-energy or force MAE and improves one of them enough.
   Without a validation partition the fine-tuned model is reported without
   selection, and the report says so.
3. It writes `selection.json` (the decision, the rule, whether it was satisfied,
   and the hashes of both models and of the split) before any test frame is read.
4. It evaluates both models on `test` and on every test domain. Test results never
   change the selection.

In the active-learning loop, model promotion uses the held-out validation
labels only (`trainer.validation_set` or `trainer.validation_fraction`); the loop
has no test partition. Its validation labels are held out frame by frame, so they
are correlated with training frames of the same trajectories.

## Reports

`eval/report.json` holds, per evaluation (`validation`, `test`,
`domain:<name>`) and model:

- the metrics, defined unambiguously in `metric_definitions`: force MAE over
  Cartesian components, force MAE of the per-atom error vector norm, raw energy
  MAE per atom, and formation-energy MAE per atom;
- per-frame quantiles (median, 90th, 95th percentile, maximum), per-group means,
  and the worst group;
- 95% intervals from a bootstrap that resamples independent groups, not frames;
- paired fine-tuned minus baseline differences on identical groups, with the
  number of groups that improved;
- the split summary and limitations, the selection, and the fine-tuning cost.

The test partition's results are also written to `eval/iter0.json` (baseline) and
`eval/iter<N>.json` (fine-tuned, `N` = AL iterations in the dataset), as before.

`repetitions` repeats the evaluation over `split_seeds` (one split each) and
`training_seeds` (fine-tuning seeds on the same split), in
`split-seed-<s>/train-seed-<t>/`. `repetitions.json` gives the mean, standard
deviation, and range of the selected model's metrics, and the number of
independent test groups of each repetition. Repetitions reuse the same data, so
they show sensitivity to the split and seed, not independent experiments.

To tabulate backend x split strategy for several runs:

```bash
python -m matsim_agents.active_learning.eval_report matrix runs/finetune-eval/* \
  --metric force_mae_eV_per_A --domain test --csv matrix.csv
```

## Active-learning comparisons and learning curves

`loop.benchmark_sets` lists frozen, DFT-labelled evaluation sets. After every
iteration's training and promotion decisions, the model in use is scored on each,
and `iteration_*/state.json` records the metrics with the cumulative number of
DFT labels. Benchmarks never feed acquisition, training, or promotion; frames
also acquired as labels are counted and logged. They require elemental
references (`trainer.validation_reference_set` or
`trainer.training_references.manifest`). Learning curves:

```bash
python -m matsim_agents.active_learning.eval_report learning-curve runs/al-ensemble > ensemble.csv
```

To compare acquisition strategies (random, ensemble, MC dropout, adaptive), start
each from the same initial model and seeds, give each its own `loop.out_dir` so
labels and models never mix, score all of them on the same benchmark sets at the
same label budgets, repeat with controlled acquisition seeds and independently
generated source trajectories, and report error against DFT labels together with
DFT and GPU cost. Compare against deterministic acquisition baselines before
attributing a gain to an LLM planner or debate.

## Uncertainty calibration

Ensemble and MC-dropout disagreement rank candidates for labelling; they are not
error probabilities. `calibrate_uncertainty(scores, errors, error_threshold=...,
partition="validation")` and `validation_uncertainty_report(...)` measure, on
labelled validation frames, the rank correlation between disagreement and actual
force error, the error per disagreement bin, and a screening threshold with its
precision and recall. They refuse test data, and their output states that the
relation is empirical.

## Migration

- `finetune_eval` without `--evaluation-config` uses `random_frame` with
  `--test-fraction` and `--seed`, and selects the same test frames for the same
  dataset, fraction, and seed as before. Its partitions are now
  `split/train.extxyz` and `split/test.extxyz` (the held-out file was
  `split/test_set.extxyz` and was labelled `validation` in its sidecar).
- Splits now reject identical configurations in two roles; deduplicate datasets
  that contain repeated frames.
- `--eval-only` needs the run's `split/split-manifest.json`. To re-score a run made
  before split manifests, split the dataset again with the same
  `random_frame` fraction and seed (the test frames are the same) and fine-tune or
  re-score into a new output directory.
- Datasets labelled before this change lack trajectory metadata; only
  `random_frame`, `composition_holdout`, and `al_iteration_holdout` (which needs
  `al_iteration`, recorded since the first AL versions) apply to them.

Examples: [random-frame interpolation](../examples/evaluation/random-frame-interpolation.yaml),
[trajectory holdout](../examples/evaluation/trajectory-holdout.yaml),
[composition holdout with test domains](../examples/evaluation/composition-holdout.yaml),
and [chronological AL-iteration holdout](../examples/evaluation/al-iteration-holdout.yaml).
