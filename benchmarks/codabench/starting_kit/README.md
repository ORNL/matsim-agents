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

The reference geometries (declared elemental solids / molecules) and DFT
total-cell labels are published in `public_data/elemental_references.json`.
Evaluate these same fixed geometries with your model, then use its own
per-atom elemental predictions as $E_{\text{ref}}[i]$ when converting ML totals
to formation energies. The protected DFT labels subtract the corresponding
DFT elemental values using the same DFT setup as the compounds.

> **Units & normalisation (read carefully).** All energies are reported **per
> atom** (eV/atom), i.e. the total cell energy divided by $N$, the number of
> atoms in that structure. This makes the metric intensive and comparable
> across structures of very different size — submit per-atom values, not total
> eV. The elemental-reference geometry convention above is **fixed**; subtract
> each method's own predictions on those geometries, not another method's
> numerical energy zeros. Forces (Task 2) are likewise per-component in eV/Å.

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
> The organizer-provided pure-element DFT labels are an explicit exception:
> they define the reference convention, not compound predictions. Use your
> model's own predictions on those reference geometries when forming its
> energies. Compound DFT labels remain protected; an elemental manifest does
> not authorize participants to replace ML predictions with compound DFT.
> Flagged submissions are reviewed and may be disqualified.

---

## Provided Baselines

Four baselines are provided in `baselines/`:

| Baseline | Architecture | Tasks | Notes |
|----------|-------------|-------|-------|
| **MACE foundations** | Equivariant GNN (MACE) | 1–3, 5 | MP, MPA, OMAT, and MATPES crystal models; no extra auth needed |
| **HydraGNN** | Multi-headed graph NN | 1–3, 5 | ORNL model |
| **UMA** (`uma-s-1p2`) | Transformer-based universal model | 1–3, 5 | Requires `fairchem-core ≥2.20` and HF model card acceptance |
| **AllScAIP** (`allscaip-md-conserving-all-omol`) | Message-passing NN (OMol102M) | 1–3, 5 | Requires `fairchem-core ≥2.20` and HF model card acceptance |

Run with:

```bash
REF=public_data/elemental_references.json
python run_baselines.py --model mace --mace-variant mace_omat_medium --elemental-reference-manifest "$REF"
python run_baselines.py --model mace --mace-variant materials --elemental-reference-manifest "$REF"  # 14 crystal models
python run_baselines.py --model hydragnn --elemental-reference-manifest "$REF"  # also provide --hydragnn-logdir
python run_baselines.py --model uma --elemental-reference-manifest "$REF"
python run_baselines.py --model allscaip --elemental-reference-manifest "$REF"
python run_baselines.py --model all --relax --elemental-reference-manifest "$REF"
```

`--mace-variant` accepts every MACE model exposed by the workflow. Use
`materials` for the Codabench bulk-crystal matrix (MP, MPA, OMAT, and
MATPES), or `all` to include all 22 catalog variants. The latter also includes
OFF, OMOL, Polar, and ANI-CC molecular models; those are selectable for explicit
diagnostics but are not general-purpose bulk-crystal potentials. MACE-OFF and
MACE-OMOL use the non-commercial Academic Software License; verify each model's
upstream terms before publishing or redistributing results.

MACE-MH (`mh-0` and `mh-1`) variants are excluded until explicit inference-head
selection is supported through the adapters.

Install backend dependencies separately: `requirements-mace.txt` in the MACE
environment and `requirements-fairchem.txt` in the HydraGNN/FairChem
environment. Aggregate runs use `.venv-mace/bin/python` and `.venv/bin/python`
by default; set `MATSIM_MACE_PYTHON` and `MATSIM_BASE_PYTHON` when the
environments live elsewhere.

`run_baselines.py` and `evaluate.py` first predict energies of the fixed
pure-element structures in `--elemental-reference-manifest`, then subtract
the model's own reference energies from compound totals. They write
`formation_energies.csv`, raw-energy diagnostics, and
`elemental_reference_predictions.json` beneath their prediction directory.
The manifest supplies the same geometries and their DFT total-cell energies,
with structure hashes, DFT backend, and method signature. All test elements
must be covered. Models must not relax or replace these reference geometries.
Create a submission with:

```bash
python package_submission.py predictions/<model> submission/
```

The packager refuses to label a raw `energy_eV` file as Task 1 or Task 5.
Forces and available relaxed structures can be packaged independently.

The HydraGNN reference baseline also requires an installed `matsim-agents`
checkout and its separately installed HydraGNN runtime. The MACE, UMA, and
AllScAIP baselines do not import `matsim-agents`.

To use UMA or AllScAIP, accept the model-card licenses on HuggingFace first:

- UMA: <https://huggingface.co/facebook/UMA>
- AllScAIP (OMol25): <https://huggingface.co/facebook/OMol25>

> **Note on elemental references**: use the same declared reference
> geometries as the competition, but subtract each model's own predictions
> on those geometries from its compound predictions. DFT reference formation
> energies subtract DFT elemental values. Do not subtract DFT energies from
> raw MLIP totals. Organizers must provide converged pure-element DFT
> calculations and `public_data/elemental_references.json` before release.

## Elemental reference manifest

Each test must cover all its elements, using one fixed geometry per declared
elemental reference. Paths resolve relative to the JSON manifest:

```json
{
  "backend": "qe",
  "method_signature": "declared-dft-protocol",
  "references": {
    "Nb": {
      "structure_path": "elemental_structures/Nb.extxyz",
      "structure_sha256": "<SHA-256 of the geometry file>",
      "energy_eV": -20.0
    },
    "O": {
      "structure_path": "elemental_structures/O2.extxyz",
      "structure_sha256": "<SHA-256 of the geometry file>",
      "energy_eV": -10.0
    }
  }
}
```

These numbers are illustrative, not physical DFT data. `energy_eV` is the
DFT **total-cell** energy; the runner divides by the reference atom count,
including two for O2. A reference keyed by `"O"` must contain only oxygen.
Missing coverage, impure geometries, hash mismatches, non-finite labels, and
multiple geometries per reference are errors. The runner consumes existing
DFT labels; it does not launch their calculations.

For a custom submitted calculator:

```bash
python evaluate.py \
  --submission /path/to/model_submission \
  --structures public_data/structures_metadata.csv \
  --struct-dir public_data/structures \
  --elemental-reference-manifest public_data/elemental_references.json \
  --output predictions/custom
python package_submission.py predictions/custom submission/
```

Each method subtracts its own reference predictions. Elemental-baseline errors
and raw totals are retained in the prediction directory for audit, while the
scorer consumes the packaged formation energies. Preserve these diagnostics
with your experiment results; the packager does not add them as leaderboard
tasks. The scorer does not independently verify your reference calculations.

Formation subtraction removes additive energy-zero differences, not bonding
errors or differences between approximation theories. Identical fixed
geometries prevent model-dependent reference relaxation from confounding
comparison. If these are not the appropriate elemental ground states, report
energies relative to the declared phases, not ground-state formation energies.
Good formation-energy accuracy does not imply accurate forces or hull stability.
Task 5's same-formula ordering is unchanged by this subtraction.

### Organizer release checklist

- Supply converged pure-element DFT calculations for all test elements,
  using the compound-label DFT setup, with declared bulk/molecular phases.
- Publish their fixed geometries, geometry hashes, total-cell DFT energies,
  backend, and method signature in `public_data/elemental_references.json`.
  Reference paths must be relative and remain inside the bundled public data.
- Form protected compound formation labels by subtracting the matching DFT
  elemental values, normalized per atom. Protected
  `reference_data/elemental_energies.json` must contain the same per-atom values.
- Use separate matching manifests for different DFT protocols; do not pool
  incompatible raw energies. Matching elemental numbers alone does not verify
  compound-label convergence or method provenance.
- Build the release with the repository's `build_bundle.py`; it validates
  coverage, hashes, reference packaging, and protected/public elemental
  agreement before creating the competition ZIP.
- Qualify real checkpoint inference and DFT data separately. Synthetic
  reference tests and mocked model adapters validate software, not scientific
  accuracy.
