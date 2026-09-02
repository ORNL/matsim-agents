# Matsim-Agents Atomistic Simulation Competition — Starting Kit

## Overview

This competition benchmarks machine-learning interatomic potentials (MLIPs) and
AI-accelerated DFT workflows on **159 atomistic test structures** spanning
11 material classes:

| Class | Examples |
|---|---|
| 2D Monolayer | hexagonal monolayers |
| BCC HEA | equiatomic (128 atoms) |
| FCC HEA | equiatomic (108 atoms) |
| Catalysis | close-packed metal slabs |
| Critical Minerals | oxides, carbides, phosphates |
| High-Entropy Ceramics | carbide / nitride / boride / oxide |
| Intermetallics | ordered binary prototypes |
| MAX Phases | Mₙ₊₁AXₙ phases |
| Nuclear | actinide / zirconia oxides |
| Perovskites | oxide / halide perovskites |
| Thermoelectrics | telluride thermoelectrics |

Each structure comes in ideal, supercell, vacancy, antisite, interstitial,
and/or alloyed variants.

> **Note on structure IDs**: all test structures are identified by opaque
> `MATS-XXXX` keys (e.g. `MATS-0023`, `MATS-0038`).  The mapping from these
> keys to compound names is intentionally kept private to prevent participants
> from looking up or reproducing the DFT reference values from external
> databases.  `public_data/structures_metadata.csv` lists only
> `structure_id,file_path` — the material class, chemical formula, and the
> specific variant (ideal / vacancy / antisite / interstitial) are **not**
> disclosed.  Determine composition, if needed, directly from the provided
> structure files.

---

## Tasks and Submission Formats

### Task 1 — Formation Energy Prediction (CSV)

Predict the DFT formation energy per atom (eV/atom) for each structure.

$$\Delta H_f / N = E_{\text{compound}} / N - \sum_i x_i \cdot E_{\text{ref}}[i]$$

The elemental reference energies $E_{\text{ref}}[i]$ (from DFT ground-state
solids / molecules, same PBE settings) are published in
`reference_data/elemental_energies.json`.  Use these same references when
converting your ML total energies to formation energies.

> **Units & normalisation (read carefully).** All energies are reported **per
> atom** (eV/atom), i.e. the total cell energy divided by $N$, the number of
> atoms in that structure. This makes the metric intensive and comparable
> across structures of very different size — submit per-atom values, not total
> eV. The elemental-reference convention above is **fixed**; the scorer uses
> exactly these references, so you must use them too or your formation energies
> will be offset. Forces (Task 2) are likewise per-component in eV/Å.

File: `task1.csv`
```
structure_id,formation_energy_eV_per_atom
MATS-0023,-0.234
MATS-0038,-0.412
...
```

Metric: MAE vs DFT reference (eV/atom). Lower is better.

---

### Task 2 — Force Prediction (ZIP of NPY files)

Predict the DFT forces on each atom (eV/Å).

File: `task2.zip` — contains one `.npy` file per structure:
```
task2.zip
  MATS-0023.npy   # shape (128, 3)  float64
  MATS-0038.npy   # shape (N_atoms, 3)  float64
  ...
```

Forces are the DFT forces on the **as-generated (unrelaxed) geometry**.
Metric: MAE over all force components (eV/Å). Lower is better.

---

### Task 3 — ML Structure Relaxation (ZIP of XYZ files)

Relax each structure using a pure ML potential (no DFT calls).

File: `task3.zip` — contains one extended-XYZ file per structure:
```
task3.zip
  MATS-0023.xyz
  MATS-0038.xyz
  ...
```

Metric: RMSD vs DFT-relaxed geometry (Å). Lower is better.

---

### Task 4 — AI-Accelerated DFT Relaxation (ZIP + optional CSV)

Run a DFT relaxation **guided by an ML potential** (e.g. ML-based initialisation,
ML-preconditioned BFGS, or on-the-fly active learning) and submit the final
DFT-relaxed structures and energies.

File: `task4.zip` — must contain two entries:

1. `task4_relaxed/` — directory of `.xyz` files (same naming as Task 3)
2. `task4_energies.csv` (optional) — final DFT formation energies:
   ```
   structure_id,formation_energy_eV_per_atom,n_atoms
   MATS-0023,-0.245,128
   ...
   ```

Metrics: RMSD vs reference DFT geometry (Å) **and** energy MAE (eV/atom) if
`task4_energies.csv` is provided. Lower is better.

---

### Task 5 — Phase Stability Ranking (CSV)

Same format as Task 1. The scorer groups structures by chemical formula and
computes Spearman ρ between your predicted energy ordering and the DFT ordering
within each group.

File: `task5.csv`
```
structure_id,formation_energy_eV_per_atom
MATS-0023,-0.234
MATS-0022,-0.201
...
```

Metric: mean Spearman ρ across formula groups. Higher is better.

---

## Overall Score

The overall score is a weighted average of normalised per-task scores, mapped
to [0, 1] where 1 = perfect:

| Task | Weight |
|---|---|
| Task 1 formation energy MAE | 1.0 |
| Task 2 force MAE | 1.0 |
| Task 3 relaxation RMSD | 0.5 |
| Task 4 relaxation RMSD | 0.5 |
| Task 4 energy MAE | 0.5 |
| Task 5 Spearman ρ | 0.5 |

Tasks with no submission are excluded from the average (not penalised).

---

## Leaderboard — public / private split

The 159 test structures are divided into two partitions:

| Partition | Structures | Purpose |
|-----------|-----------|---------|
| **Public (~30 %)** | 51 structures | Visible on the leaderboard *during* the competition |
| **Private (~70 %)** | 108 structures | Used for the **final ranking** at competition close |

The split is deterministic: SEED=42, stratified by chemical formula so every
formula has at least one structure in each partition.

**What you see during the competition**: all leaderboard columns report metrics
on the public partition only (prefixed `public_` in the scores file).  The
private partition scores are computed at every submission but are hidden until
the competition closes.

**Final ranking**: at close, the organizers reconfigure the leaderboard to show
`private_*` metrics, which are scored on the 108 held-out structures you could
not probe during the competition.

**Submission rate limit**: 3 submissions per day.  This is enforced by
Codabench to prevent participants from reconstructing private labels by
exhaustive probing.

> **Tip**: optimise your model on the public score, but do not overfit to
> it — the public partition is only 30 % of the final evaluation.

> **Anti-cheating — predictions must come from your model, not DFT.** This is
> an ML-potential benchmark. Submitting values obtained by running DFT (or any
> first-principles calculation) on the released geometries is not allowed. The
> scorer automatically screens every submission for accuracy that is physically
> implausible for an ML potential (per-structure errors below DFT noise floors).
> Note that the exact reference DFT protocol (pseudopotentials, k-mesh, INCAR,
> elemental references) is **not** published, so independently-run DFT will not
> match the reference anyway. Flagged submissions are reviewed and may be
> disqualified.

---

## Provided Baselines

Four baselines are provided in `baselines/`:

| Baseline | Architecture | Tasks | Notes |
|----------|-------------|-------|-------|
| **MACE-MP-0** | Equivariant GNN (MACE) | 1–3, 5 | Universal MLIP, no extra auth needed |
| **HydraGNN** | Multi-headed graph NN | 1–3, 5 | ORNL model |
| **UMA** (`uma-s-1p2`) | Transformer-based universal model | 1–3, 5 | Requires `fairchem-core ≥2.20` and HF model card acceptance |
| **AllScAIP** (`allscaip-md-conserving-all-omol`) | Message-passing NN (OMol102M) | 1–3, 5 | Requires `fairchem-core ≥2.20` and HF model card acceptance |

Run with:

```bash
python run_baselines.py --model mace        # MACE-MP-0
python run_baselines.py --model hydragnn    # HydraGNN
python run_baselines.py --model uma         # UMA
python run_baselines.py --model allscaip    # AllScAIP
python run_baselines.py --model all --relax # all baselines including relaxation (Tasks 3 & 4)
```

`run_baselines.py` writes raw model totals and numerical artifacts beneath
`predictions/<model>/`. Raw total energies are not formation energies. After
applying the published elemental-reference convention into a file containing
`structure_id,formation_energy_eV_per_atom`, create a submission with:

```bash
python package_submission.py predictions/<model> submission/ \
  --formation-energies path/to/formation_energies.csv
```

The packager refuses to label a raw `energy_eV` file as Task 1 or Task 5.
Forces and available relaxed structures can be packaged independently.

The HydraGNN reference baseline also requires an installed `matsim-agents`
checkout and its separately installed HydraGNN runtime. The MACE, UMA, and
AllScAIP baselines do not import `matsim-agents`.

To use UMA or AllScAIP, accept the model-card licenses on HuggingFace first:

- UMA: <https://huggingface.co/facebook/UMA>
- AllScAIP (OMol25): <https://huggingface.co/facebook/OMol25>

> **Note on elemental references**: participants must apply the same DFT
> elemental reference energies as the competition (provided in
> `reference_data/elemental_energies.json`) to convert ML total energies to
> formation energies before submission.
