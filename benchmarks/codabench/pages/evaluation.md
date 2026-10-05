# Evaluation

## Submission format

Submit a single zip whose top level may contain any subset of:

```
task1.csv            # structure_id,formation_energy_eV_per_atom
task2.zip            # one MATS-XXXX.npy per structure, shape (N,3), eV/Å
task3.zip            # relaxed/  — MATS-XXXX.xyz
task4.zip            # task4_relaxed/ — MATS-XXXX.xyz  (+ optional task4_energies.csv)
task5.csv            # structure_id,formation_energy_eV_per_atom
```

The scorer reads your submission from `res/` and the reference labels from
`ref/`, matching on `MATS-XXXX` keys.

## Units & normalisation (read carefully)

- **Energies are per atom** (eV/atom) — total cell energy divided by the
  number of atoms `N`. Submit per-atom values, not total eV.
- The **elemental-reference geometries are fixed** and published through
  `public_data/elemental_references.json`, with converged DFT total energies.
  Evaluate these pure-element geometries with your model before the compounds
  and subtract your model's own elemental predictions. The DFT formation
  labels subtract the corresponding DFT energies. Do not mix DFT elemental
  energies with raw MLIP totals. The scorer compares supplied formation energies;
  it does not fit offsets or perform this subtraction on your behalf.
  Reference energies are normalized by the actual reference atom count,
  including molecules. Baseline/raw errors remain runner diagnostics, not
  additional leaderboard tasks.
- **Forces** are per component, in eV/Å.

## Metrics

| Task | Metric | Direction |
|------|--------|-----------|
| 1 | Formation energy MAE (eV/atom) | lower better |
| 2 | Force MAE (eV/Å) | lower better |
| 3 | Relaxation RMSD (Å) vs DFT-relaxed | lower better |
| 4 | Relaxation RMSD (Å) + energy MAE (eV/atom) | lower better |
| 5 | mean Spearman ρ across formula groups | higher better |

Per-task scores are normalised and combined into `overall_score ∈ [0, 1]`
(see the **Overview** tab for weights).

Subtracting a common elemental baseline within a formula group does not change
Task 5's within-method ranking. Neither Task 1 nor Task 5 establishes convex-hull
stability. Formation energies are relative to the declared elemental phases;
ground-state formation-energy claims require appropriate ground-state references.
Separate DFT setups require separate matching reference conventions, not a
mixture of raw total energies.

## Public / private leaderboard split

- The leaderboard shows metrics computed on the **public** partition only
  (~30 % of structures), so you can iterate.
- The remaining **private** partition (~70 %) is held out and only revealed
  for the **final ranking** when the competition closes.
- The split is deterministic and stratified by formula group
  (`reference_data/create_split.py`, `SEED=42`, `PUBLIC_FRACTION=0.30`).
- **Submissions are rate-limited to 3 per day** to prevent reconstructing
  private labels by repeated probing.

> **Tip:** optimise on the public score but do not overfit — it is only 30 %
> of the final evaluation.

## Anti-cheating policy

This is an **ML-potential** benchmark. Submitting values obtained by running
DFT (or any first-principles calculation) on the released geometries is **not
allowed**. Every submission is automatically screened for accuracy that is
physically implausible for an ML potential (per-structure errors below DFT
noise floors). Organizer-supplied pure-element DFT labels are provided to
define the reference convention, not as compound predictions. Compound DFT
labels remain protected. Participants must subtract their own MLIP elemental
predictions, and may not replace ML compound predictions with DFT results.
Flagged submissions are reviewed and may be disqualified.
