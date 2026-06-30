# Matsim-Agents Materials Discovery Challenge

A multi-task benchmark for atomistic machine learning. Given a set of
crystalline and defect structures (identified only by opaque `MATS-XXXX`
keys), your model must predict DFT-quality properties **without running
DFT itself**.

## Tasks

| Task | What you predict | Metric |
|------|------------------|--------|
| **1. Formation energy** | Formation energy per atom (eV/atom) | MAE (lower better) |
| **2. Forces** | Per-atom forces (eV/Å) | MAE (lower better) |
| **3. Structure relaxation** | Relaxed geometry | RMSD to DFT-relaxed (Å) |
| **4. AI-accelerated DFT relaxation** | DFT-relaxed geometry + energy, guided by an ML potential | RMSD (Å) + energy MAE |
| **5. Phase-stability ranking** | Energy ordering within each formula group | mean Spearman ρ (higher better) |

The test set spans five diverse material classes — 2D monolayers,
intermetallics, a BCC high-entropy alloy, critical minerals, and catalysis
slabs — each including ideal crystals and defect variants (vacancies,
antisites, interstitials). **Compound identities and exact variants are
withheld**; structures carry only `MATS-XXXX` keys.

## How scoring works

A single submission bundles all tasks you wish to enter. Tasks you omit are
simply excluded from your overall score (not penalised). The **overall score**
is a weighted average of normalised per-task scores mapped to `[0, 1]`
(1 = perfect):

| Task | Weight |
|------|--------|
| Task 1 formation energy MAE | 1.0 |
| Task 2 force MAE | 1.0 |
| Task 3 relaxation RMSD | 0.5 |
| Task 4 relaxation RMSD | 0.5 |
| Task 4 energy MAE | 0.5 |
| Task 5 Spearman ρ | 0.5 |

See the **Evaluation** and **Data** tabs for submission formats and the
public/private leaderboard split. The full participant guide lives in the
starting kit (`starting_kit/README.md`).
