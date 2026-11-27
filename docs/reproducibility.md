# Reproducibility and random seeds

Every random draw in MATSIM-AGENTS comes from one module,
[`matsim_agents.core.seeding`](../src/matsim_agents/core/seeding.py),
and every seed is a configured value. The same configuration, code checkout,
and package versions therefore regenerate the same structures, trajectories,
selections, and splits.

## Streams

A draw is identified by a configured seed and its purpose, for example the
MD of seed structure `k` in iteration `i`. `stream_seed(seed, "md", i, k)`
derives its seed with SHA-256, so:

- the same seed reproduces every draw, independent of platform or of the order
  in which concurrent work runs;
- streams of different purposes, iterations, or items are independent, and a
  run seeded `S + 1` does not replay part of the run seeded `S` (as a
  `seed + i` scheme would);
- adding a stream does not shift existing ones.

The bare seed (no purpose) is the root stream: for a non-negative seed it equals
`numpy.random.default_rng(seed)` and `random.Random(seed)`, so seeded pyXtal
generation and frame-level dataset splits keep their established results.
Derived seeds are 32-bit, which every seeding interface accepts. Negative seeds
are valid and never alias positive ones.

## What each workflow seeds

| Workflow | Configured seed | Streams |
|---|---|---|
| Exploration (`explore`, chat, composition graph) | `generation.random_seed` / `--random-seed` | pyXtal space groups and atom placement; round `r > 0` uses stream `("round", r)` |
| Active learning | top-level `random_seed` (required) | `seed-source` (pyXtal, per formula), `("md", i)` then per seed structure, `("acquisition", i)` (random scores, MC-dropout masks), `("validation-split", i)`, `("training", i)` (fine-tuner `--seed`) |
| Campaign | the AL template's `random_seed` | `("formula-acquisition", iteration)`; each formula attempt gets `("formula", formula, attempt)`, and within it `generation`, the AL streams, `candidate-uncertainty`, `perturbation`, and `dft-refinement` |
| Fine-tuning (`finetune_*`) | `--seed` (required) | split from the root stream; global Python, NumPy, and PyTorch state from `training`; `mace_run_train --seed` from `mace_run_train` |
| Fine-tune evaluation | split `seed`, training seeds | group-bootstrap intervals from `bootstrap` |
| Held-out evaluation (`evaluate`, fine-tune evaluation) | the AL config's `random_seed` / the evaluation `seed` | parity-plot atoms from `parity-plot` (whole atoms, so x/y/z stay balanced) |
| Dataset splits | `split.seed` | root stream |
| Relaxation with `random_displacement` | `displacement_seed` (required) | root stream |
| LAMMPS ML-IAP export check (AL loop) | the AL config's `random_seed`, passed as `seed` | `("lammps-export", "rattle")`, `("lammps-export", "velocities")` |
| LAMMPS ML-IAP scripts and Frontier jobs | `--seed` / `MATSIM_MLIAP_SEED` (required) | `rattle`, `velocities`, `finetune-inputs`; the seed is also passed to the fine-tune |

A campaign retry of an inconclusive formula is a new attempt with a new seed
and its own directory (`formulas/<formula>/attempt-NNNN`), so it explores new
structures and never resumes the previous attempt's active-learning artifacts.
Resuming an interrupted attempt keeps its number, seed
(`FormulaRunRecord.random_seed`), and directory, and replays it. The campaign
state records the root seed of its first run (`CampaignState.random_seed`);
resuming with a different seed is rejected. Each new attempt is charged its own
DFT calculations, node hours, and active-learning iterations, also when it fails.
A promoted checkpoint is kept on the formula record (`promoted_checkpoint`), so
a later attempt that promotes nothing does not erase it; a promotion committed
before a later stage fails is used by the next formula. A checkpoint with prior
attempts but no recorded root seed predates seed and promotion recording and is
rejected rather than resumed. An interrupted attempt from a checkpoint
without attempt seeds cannot be recovered reproducibly and is rejected before
the checkpoint is modified.

A standalone active-learning run records its root seed in
`<out_dir>/random-seed.json`. Before any seed structure is resolved or written,
the loop rejects `loop.resume` with a different seed, a fresh run
(`loop.resume: false`) in a directory that already holds a run, and any prior
artifacts (iterations, datasets, seed structures) without a recorded seed, so
retained labels are never reused under a seed that did not produce them.

Exploration checkpoints include `seeding.py` in their code identity, so a
change to stream derivation blocks resuming an interrupted run.

A configured `generation_action` receives the attempt's `generation` seed on a
copy of the action. Public entry points take the seed explicitly:
`explore_composition(..., random_seed=...)` accepts `None` only when no pyXtal
structure is drawn.

## Rules for code

`tests/test_seeding.py` fails when code under `src/` or `deployments/`:

- builds a generator or seeds global state outside `seeding.py`
  (`default_rng`, `random.Random`, `np.random.*`, `torch.manual_seed`, ...);
- passes a literal seed (`seed=0`) or derives one by arithmetic
  (`seed=base + i`); use `stream_seed(base, "purpose", i)` instead;
- gives a seed parameter a default.

Campaign and active-learning configurations reject unknown keys, so a removed
seed field (for example `md.random_seed` or `perturbation_seed`) fails
validation instead of being ignored.

`seeding` is the lowest import layer, so every package, including `datasets`
and `execution`, can use it.

`benchmarks/` and `scripts/` are exempt: they hold frozen benchmark and study
protocols (E3/E4 manifests, blinded assessments) whose recorded seeds must keep
regenerating the published artifacts.

Use `numpy_rng`/`python_rng` for generators, `seeded_torch` for code that draws
from PyTorch's global generator, and
`seed_process` at the start of a training process. `seeded_torch(seed, ...,
devices=...)` seeds and afterwards restores, also on error, the CPU generator
and only the given accelerators (MC dropout passes the devices of the model's
parameters), so GPUs reserved for DFT, training, or an LLM are not touched.

## Limits

Seeds do not make these deterministic:

- LLM sampling, unless the provider honors a decoding `seed` (agent evaluation
  passes one through `decoding`);
- nondeterministic GPU kernels (for example scatter-add in message passing) and
  multi-rank reductions;
- results across different versions of pyXtal, NumPy, PyTorch, or the MLIP
  packages.
