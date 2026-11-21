# Codabench competition

The `benchmarks/codabench/` directory contains a fully self-contained
[Codabench](https://www.codabench.org/) challenge called the
**Matsim-Agents Materials Discovery Challenge**.

## What is tested

121 completed atomistic test structures from a dataset originally designed
across 11 material classes — 2D
monolayers, intermetallics, BCC/FCC high-entropy alloys, catalysis slabs,
critical minerals, high-entropy ceramics, MAX phases, nuclear oxides,
perovskites, thermoelectrics — each available in ideal, vacancy, antisite,
and interstitial variants. Tasks cover:

| # | Task | Metric |
|---|------|--------|
| 1 | Formation energy prediction | MAE (eV/atom) ↓ |
| 2 | Atomic force prediction | MAE (eV/Å) ↓ |
| 3 | ML structure relaxation | RMSD vs DFT geometry (Å) ↓ |
| 4 | AI-accelerated DFT relaxation | RMSD + energy MAE ↓ |
| 5 | Phase stability ranking | Mean Spearman ρ ↑ |

The overall score is a weighted average mapped to [0, 1]; tasks with no
submission are excluded (not penalised).

## Leaderboard — public / private split

To prevent participants from reverse-engineering the reference labels by
repeatedly probing the leaderboard, the 121 completed structures are split into
two partitions:

| Partition | Size | When visible |
|-----------|------|-------------|
| **Public** | 37 structures (~30 %) | Always — during the competition |
| **Private** | 84 structures (~70 %) | Only at competition close (final ranking) |

The split is deterministic and reproducible (SEED=42, stratified by chemical
formula within the published benchmark metadata). Singleton groups remain
private-only; larger groups have structures in both partitions. The
`reference_data/public_ids.txt` and `reference_data/private_ids.txt` files
record which structure IDs belong to each partition.

The scoring program (`scoring_program/score.py`) computes metrics for both
partitions and emits `public_*` and `private_*` keys to `scores.json`. The
Codabench leaderboard is configured to display only `public_*` columns during
the competition. To switch to final ranking, change the key prefix from
`public_` → `private_` in `competition.yaml`.

**Submission rate limit**: 3 submissions per day, enforced via
`max_submissions_per_day: 3` in `competition.yaml`.

## Baselines

Two baseline adapters are provided in `benchmarks/codabench/baselines/`:
`mlip/` builds any matsim-agents MLIP from an `mlip:` YAML block, and
`allscaip/` wraps AllScAIP. Curated MLIP selections live in
`benchmarks/codabench/mlip/`:

| Selection | Architecture | Source |
|----------|-------------|--------|
| `mlip/materials/*.yaml` (14), `mlip/molecular/*.yaml` (8) | Equivariant GNN (MACE) | MACE foundation models (Cambridge) |
| `mlip/hydragnn.yaml` | Multi-headed graph NN | This repo / ORNL |
| `mlip/uma.yaml` (`uma-s-1p2`) | Transformer-based universal model | Meta / fairchem |
| `--allscaip` (`allscaip-md-conserving-all-omol`) | Message-passing NN | Meta / OMol25 |

Run any or all baselines. Every energy run requires the organizer-supplied
elemental manifest. `--mlip-config FILE` is repeatable and accepts several
files; each writes `predictions/<file stem>/` and runs on its block's
`device` (the curated files read `MLIP_DEVICE`, default `cpu`):

```bash
cd benchmarks/codabench
REF=public_data/elemental_references.json
python run_baselines.py --mlip-config mlip/materials/*.yaml --elemental-reference-manifest "$REF"
HYDRAGNN_LOGDIR=/path/to/model python run_baselines.py --mlip-config mlip/hydragnn.yaml --elemental-reference-manifest "$REF"
python run_baselines.py --mlip-config mlip/uma.yaml --elemental-reference-manifest "$REF"
python run_baselines.py --allscaip --elemental-reference-manifest "$REF"
HYDRAGNN_LOGDIR=/path/to/model python run_baselines.py --relax --allscaip \
    --mlip-config mlip/materials/*.yaml mlip/hydragnn.yaml mlip/uma.yaml --elemental-reference-manifest "$REF"
```

UMA and AllScAIP require the `fairchem-core` package and the model checkpoints
(downloaded on first use from HuggingFace — the relevant model cards must be
accepted before use at <https://huggingface.co/facebook/UMA> and
<https://huggingface.co/facebook/OMol25>).

## Reference convention and release requirements

Task 1 compares formation energies, not raw total-energy zeros. Both baseline
and submitted-model runners evaluate the declared pure-element geometries
first, then subtract each MLIP's own elemental energies. Protected DFT
formation labels subtract DFT references from the same declared geometries.
This removes additive elemental offsets, not genuine bonding/method errors.
Task 5 ranks structures within formula groups; this subtraction does not
change their within-method ordering or establish convex-hull stability.

Runners retain `energies.csv` and `elemental_reference_predictions.json` for
diagnostics and write submission-ready `formation_energies.csv`. Force and
relaxation metrics remain separate; a good formation-energy score does not
by itself qualify a model's forces, reference-phase accuracy, or hull stability.
Reference ground-state claims require appropriate elemental ground states;
otherwise these energies are relative to the explicitly declared phases.

Organizers must provide real, converged elemental DFT calculations, fixed
geometry files, and `public_data/elemental_references.json` before release.
Release validation checks full element coverage, geometry hashes, bundled
reference paths, and agreement with protected per-atom
`reference_data/elemental_energies.json`. Organizers remain responsible for
matching the compound-label DFT setup and forming the protected compound
labels correctly; validation does not independently rerun DFT.

```bash
python benchmarks/codabench/build_bundle.py \
  --public-data /path/to/public_data \
  --reference-data /path/to/protected_reference_data \
  --output /path/to/competition.zip
```

The builder validates the assembled release before creating the ZIP.
Synthetic unit tests and mocked model-catalog tests are software checks, not
evidence of real MLIP/DFT scientific accuracy or completed competition data.

## Directory layout

```
benchmarks/codabench/
├── competition.yaml             # Codabench bundle manifest & leaderboard config
├── run_baselines.py             # entry point: --mlip-config FILE... and/or --allscaip
├── evaluate.py                  # submitted-model inference + reference subtraction
├── requirements.txt             # backend-neutral competition dependencies
├── requirements-mace.txt        # MACE process dependencies (e3nn 0.4.4)
├── requirements-fairchem.txt    # UMA/AllScAIP process dependencies
├── baselines/
│   ├── mlip/model.py            # any mlip: block (MACE, HydraGNN, UMA)
│   └── allscaip/model.py        # AllScAIP (fairchem) baseline
├── mlip/                        # curated mlip: selections (materials/, molecular/, hydragnn, uma)
├── scoring_program/
│   └── score.py                 # Codabench scorer (public + private partitions)
├── reference_data/
│   ├── public_ids.txt           # 37 structure IDs in the public partition
│   ├── private_ids.txt          # 84 structure IDs in the private partition
│   ├── create_split.py          # reproducible split generator (SEED=42)
│   ├── formation_energies.csv   # DFT reference energies (server-side, not public)
│   ├── elemental_energies.json  # protected per-atom DFT references; must match public manifest
│   └── forces/                  # per-structure force arrays (server-side, not public)
├── public_data/
│   ├── structures_metadata.csv  # opaque MATS-XXXX IDs and geometry file paths
│   ├── elemental_references.json # public geometry paths, hashes, DFT total-cell labels
│   └── structures/              # XYZ files of all test structures
└── starting_kit/
    ├── README.md                # participant guide (tasks, formats, scoring)
    └── MODEL_INTERFACE.md       # how to write a custom MLIP adapter
```

See [`benchmarks/codabench/starting_kit/README.md`](../../benchmarks/codabench/starting_kit/README.md)
for the full participant guide including submission formats.

---
