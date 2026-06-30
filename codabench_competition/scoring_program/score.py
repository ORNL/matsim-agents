#!/usr/bin/env python3
"""
score.py — Codabench scoring program for the Matsim-Agents competition.

Called by Codabench as:
    python score.py <input_dir> <output_dir>

Expected layout of input_dir/:
    res/
        formation_energies.csv — structure_id (MATS-XXXX), formation_energy_eV_per_atom, n_atoms
        forces/               — MATS-XXXX.npy  (float64, shape (N,3), eV/Å)
        relaxed/              — MATS-XXXX.xyz   (Task 3: ML-relaxed structures)
        task4_relaxed/        — MATS-XXXX.xyz   (Task 4: AI-DFT-relaxed structures)
        task4_energies.csv    — structure_id (MATS-XXXX), formation_energy_eV_per_atom, n_atoms (optional)
    ref/
        formation_energies.csv — same columns, DFT reference (MATS-XXXX keys)
        forces/               — same layout, DFT reference
        relaxed/              — DFT-relaxed structures
        elemental_energies.json — DFT elemental reference energies (published for participants)
        structures_metadata.csv  — structure_id=MATS-XXXX, material_class, formula

Outputs <output_dir>/scores.json with:
    {
        "Task1_energy_MAE_eV_per_atom":  ...,
        "Task2_forces_MAE_eV_per_A":     ...,
        "Task3_relaxation_RMSD_A":       ...,  (if relaxed/ present)
        "Task4_relaxation_RMSD_A":       ...,  (if task4_relaxed/ present)
        "Task4_energy_MAE_eV_per_atom":  ...,  (if task4_energies.csv present)
        "Task5_phase_spearman_rho":      ...,
        "overall_score":                 ...,
        "dft_cheating_suspect":          0/1,  (heuristic anti-cheat flag)
        "dft_cheating_reason":           "...", (present only when flagged)
    }
(metrics are additionally emitted with public_ / private_ prefixes per the
public/private leaderboard split).

Anti-cheating: detect_dft_cheating() flags submissions whose per-structure
errors fall below physically-implausible "DFT noise floors" — a strong tell
that the participant ran DFT on the released geometries rather than predicting
with an ML potential.  The flag is advisory (for organiser review); it does not
change the computed scores.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_energies(path: Path) -> dict[str, dict]:
    """Return {structure_id: {formation_energy_eV_per_atom, n_atoms}}.

    Expects column: formation_energy_eV_per_atom.
    Use dft_reference/compute_formation_energies.py to produce this file
    from raw VASP/QE total energies and elemental references.
    """
    result = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if "formation_energy_eV_per_atom" not in row:
                raise KeyError(
                    f"Column 'formation_energy_eV_per_atom' not found in {path}.\n"
                    f"Run dft_reference/compute_formation_energies.py to produce "
                    f"formation energies from total energies + elemental references."
                )
            result[row["structure_id"]] = {
                "formation_energy_eV_per_atom": float(row["formation_energy_eV_per_atom"]),
                "n_atoms":                      int(row.get("n_atoms", 0)),
            }
    return result


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def prefix_keys(d: dict, prefix: str) -> dict:
    """Return a copy of `d` with every key prefixed by `prefix`."""
    return {f"{prefix}{k}": v for k, v in d.items()}


def rmsd_structures(pred_path: Path, ref_path: Path, structure_ids: list[str]) -> float:
    """Mean RMSD (Å) between predicted and reference relaxed structures."""
    from ase.io import read
    rmsds = []
    for sid in structure_ids:
        p = pred_path / f"{sid}.xyz"
        r = ref_path  / f"{sid}.xyz"
        if not p.is_file() or not r.is_file():
            continue
        try:
            pred_pos = read(str(p), format="extxyz").get_positions()
            ref_pos  = read(str(r), format="extxyz").get_positions()
            if pred_pos.shape != ref_pos.shape:
                continue
            diff = pred_pos - ref_pos
            rmsds.append(float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))))
        except Exception:
            pass
    return float(np.mean(rmsds)) if rmsds else float("nan")


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation coefficient."""
    from scipy.stats import spearmanr
    rho, _ = spearmanr(x, y)
    return float(rho)


def _load_force_errors(pred_dir: Path, ref_dir: Path,
                       structure_ids: list[str]) -> tuple[np.ndarray, int]:
    """Return (concatenated pred-minus-ref force errors, n_structures_matched)."""
    errs: list[np.ndarray] = []
    n_ok = 0
    for sid in structure_ids:
        p = pred_dir / f"{sid}.npy"
        r = ref_dir  / f"{sid}.npy"
        if not p.is_file() or not r.is_file():
            continue
        try:
            fp = np.load(str(p)).reshape(-1)
            fr = np.load(str(r)).reshape(-1)
            if fp.shape != fr.shape:
                continue
            errs.append(fp - fr)
            n_ok += 1
        except Exception:
            pass
    arr = np.concatenate(errs) if errs else np.array([])
    return arr, n_ok


# ---------------------------------------------------------------------------
# DFT-cheating detection
# ---------------------------------------------------------------------------
# A genuine ML interatomic potential cannot match the reference DFT below these
# per-structure "noise floors" on a diverse, out-of-distribution test set.
# Errors far below them across (almost) all structures indicate the participant
# likely ran DFT on the released geometries and submitted those values instead
# of true ML predictions.  Because the exact reference DFT protocol (INCAR,
# pseudopotentials, k-mesh, elemental references) is withheld, even careful
# independent DFT should leave a residual offset above these floors — so falling
# below them is a strong tell.
#
# These are HEURISTICS: they raise a flag for manual organiser review.  They do
# NOT by themselves change the score or disqualify a team.
ENERGY_DFT_FLOOR_eV_per_atom = 0.003   # 3 meV/atom
FORCES_DFT_FLOOR_eV_per_A    = 0.010   # 10 meV/Å
RELAX_DFT_FLOOR_A            = 0.050   # 50 mÅ mean RMSD vs DFT-relaxed geometry
CHEAT_FRACTION_THRESHOLD     = 0.90    # ≥90 % of structures below the floor → flag
CHEAT_MAE_MULTIPLE           = 1.0     # aggregate MAE below 1× the floor → flag


def _below_floor_fraction(errors: np.ndarray, floor: float) -> float:
    if errors.size == 0:
        return float("nan")
    return float(np.mean(np.abs(errors) < floor))


def detect_dft_cheating(energy_errors: np.ndarray,
                        force_errors: np.ndarray,
                        relax_rmsd: float) -> dict:
    """Heuristic screen for submissions that look like raw DFT rather than ML.

    Returns diagnostic fields plus an integer ``dft_cheating_suspect`` flag
    (0/1) and a human-readable ``dft_cheating_reason``.  Merged into the
    per-partition score dict so organisers can review flagged submissions.
    """
    out: dict = {}
    reasons: list[str] = []
    signals = 0

    if energy_errors.size:
        emae  = float(np.mean(np.abs(energy_errors)))
        efrac = _below_floor_fraction(energy_errors, ENERGY_DFT_FLOOR_eV_per_atom)
        out["energy_frac_below_DFT_floor"] = round(efrac, 4)
        if efrac >= CHEAT_FRACTION_THRESHOLD or emae < ENERGY_DFT_FLOOR_eV_per_atom * CHEAT_MAE_MULTIPLE:
            signals += 1
            reasons.append(
                f"energy MAE {emae*1e3:.2f} meV/atom, {efrac*100:.0f}% of "
                f"structures < {ENERGY_DFT_FLOOR_eV_per_atom*1e3:.0f} meV/atom"
            )

    if force_errors.size:
        fmae  = float(np.mean(np.abs(force_errors)))
        ffrac = _below_floor_fraction(force_errors, FORCES_DFT_FLOOR_eV_per_A)
        out["forces_frac_below_DFT_floor"] = round(ffrac, 4)
        if ffrac >= CHEAT_FRACTION_THRESHOLD or fmae < FORCES_DFT_FLOOR_eV_per_A * CHEAT_MAE_MULTIPLE:
            signals += 1
            reasons.append(
                f"forces MAE {fmae*1e3:.2f} meV/Å, {ffrac*100:.0f}% of "
                f"components < {FORCES_DFT_FLOOR_eV_per_A*1e3:.0f} meV/Å"
            )

    # Relaxation RMSD ≈ 0 means the submitted geometry is essentially the
    # DFT-relaxed one (i.e. the participant ran DFT relaxation themselves).
    if relax_rmsd == relax_rmsd and relax_rmsd < RELAX_DFT_FLOOR_A:  # not nan
        signals += 1
        reasons.append(f"relaxation RMSD {relax_rmsd*1e3:.1f} mÅ ≈ DFT-relaxed")

    out["dft_cheating_signals"] = signals
    out["dft_cheating_suspect"] = int(signals >= 1)
    if reasons:
        out["dft_cheating_reason"] = "; ".join(reasons)
    return out


# ---------------------------------------------------------------------------
# Task scorers
# ---------------------------------------------------------------------------

def score_energy(pred: dict, ref: dict) -> dict[str, float]:
    """Task 1: MAE of DFT formation energy per atom (eV/atom)."""
    common = sorted(set(pred) & set(ref))
    if not common:
        return {"Task1_energy_MAE_eV_per_atom": float("nan"), "Task1_n_structures": 0}
    pred_e = np.array([pred[s]["formation_energy_eV_per_atom"] for s in common])
    ref_e  = np.array([ref[s]["formation_energy_eV_per_atom"]  for s in common])
    return {
        "Task1_energy_MAE_eV_per_atom": mae(pred_e, ref_e),
        "Task1_n_structures": len(common),
    }


def score_forces(pred_dir: Path, ref_dir: Path, structure_ids: list[str]) -> dict[str, float]:
    """Task 2: MAE of atomic forces per component (eV/Å)."""
    err, n_ok = _load_force_errors(pred_dir, ref_dir, structure_ids)
    if err.size == 0:
        return {"Task2_forces_MAE_eV_per_A": float("nan"), "Task2_n_structures": 0}
    return {
        "Task2_forces_MAE_eV_per_A": float(np.mean(np.abs(err))),
        "Task2_n_structures": n_ok,
    }


def score_task3(pred_relaxed: Path, ref_relaxed: Path,
                structure_ids: list[str]) -> dict[str, float]:
    """Task 3: Mean RMSD of ML-relaxed structures vs DFT reference (Å)."""
    rmsd = rmsd_structures(pred_relaxed, ref_relaxed, structure_ids)
    return {"Task3_relaxation_RMSD_A": rmsd}


def score_task4(pred_dir: Path, ref_dir: Path, ref_energies: dict,
                structure_ids: list[str]) -> dict[str, float]:
    """Task 4: AI-accelerated DFT relaxation.

    Scores RMSD of relaxed structures (from pred_dir/task4_relaxed/) and,
    if task4_energies.csv is present, the final DFT energy MAE.
    """
    out: dict[str, float] = {}

    t4_relaxed  = pred_dir  / "task4_relaxed"
    ref_relaxed = ref_dir   / "relaxed"
    if t4_relaxed.is_dir() and ref_relaxed.is_dir():
        out["Task4_relaxation_RMSD_A"] = rmsd_structures(
            t4_relaxed, ref_relaxed, structure_ids
        )

    t4_energies_file = pred_dir / "task4_energies.csv"
    if t4_energies_file.is_file():
        pred_e4 = load_energies(t4_energies_file)
        common  = sorted(set(pred_e4) & set(ref_energies))
        if common:
            pred_arr = np.array([pred_e4[s]["formation_energy_eV_per_atom"]    for s in common])
            ref_arr  = np.array([ref_energies[s]["formation_energy_eV_per_atom"] for s in common])
            out["Task4_energy_MAE_eV_per_atom"] = mae(pred_arr, ref_arr)
            out["Task4_n_structures"] = len(common)

    return out


def score_phase_stability(pred: dict, ref: dict,
                           metadata_path: Path) -> dict[str, float]:
    """Task 5: Spearman rho on energy ranking within each phase group.

    Phase groups are defined by structures sharing the same formula (column 3)
    in structures_metadata.csv but having different variants.
    """
    if not metadata_path.is_file():
        return {"Task5_phase_spearman_rho": float("nan")}

    # Build groups: formula → list of structure_ids with ideal variants only
    groups: dict[str, list[str]] = {}
    with open(metadata_path, newline="") as f:
        for row in csv.DictReader(f):
            sid = row["structure_id"]
            formula = row["formula"]
            if sid in pred and sid in ref:
                groups.setdefault(formula, []).append(sid)

    rhos = []
    for formula, sids in groups.items():
        if len(sids) < 2:
            continue
        p_e = np.array([pred[s]["formation_energy_eV_per_atom"] for s in sids])
        r_e = np.array([ref[s]["formation_energy_eV_per_atom"]  for s in sids])
        try:
            rhos.append(spearman_rho(p_e, r_e))
        except Exception:
            pass

    return {
        "Task5_phase_spearman_rho": float(np.mean(rhos)) if rhos else float("nan"),
        "Task5_n_groups": len(rhos),
    }


def overall_score(scores: dict) -> float:
    """Aggregate score: average of normalised per-task scores (higher = better).

    Each metric is inverted/normalised so that the overall score is in [0, 1]
    where 1 = perfect. Weights can be adjusted by the organisers.
    """
    weights = {
        "Task1_energy_MAE_eV_per_atom":  ("lower",  1.0),
        "Task2_forces_MAE_eV_per_A":     ("lower",  1.0),
        "Task3_relaxation_RMSD_A":       ("lower",  0.5),
        "Task4_relaxation_RMSD_A":       ("lower",  0.5),
        "Task4_energy_MAE_eV_per_atom":  ("lower",  0.5),
        "Task5_phase_spearman_rho":      ("higher", 0.5),
    }
    # Soft normalisation: use a reference scale for lower-is-better metrics
    ref_scale = {
        "Task1_energy_MAE_eV_per_atom":  1.0,   # 1 eV/atom = bad
        "Task2_forces_MAE_eV_per_A":     1.0,   # 1 eV/Å   = bad
        "Task3_relaxation_RMSD_A":       2.0,   # 2 Å RMSD  = bad (pure ML)
        "Task4_relaxation_RMSD_A":       1.0,   # 1 Å RMSD  = bad (AI-DFT)
        "Task4_energy_MAE_eV_per_atom":  0.1,   # 0.1 eV/atom = bad (real DFT)
    }
    total_w, total_s = 0.0, 0.0
    for key, (direction, w) in weights.items():
        v = scores.get(key, float("nan"))
        if v != v:   # nan
            continue
        if direction == "lower":
            s = max(0.0, 1.0 - v / ref_scale[key])
        else:
            s = max(0.0, (v + 1) / 2)   # Spearman in [-1,1] → [0,1]
        total_s += w * s
        total_w += w
    return total_s / total_w if total_w > 0 else float("nan")


# ---------------------------------------------------------------------------
# Partition scoring
# ---------------------------------------------------------------------------

def score_partition(
    pred_dir: Path,
    ref_dir: Path,
    pred_energies: dict,
    ref_energies: dict,
    sids: list,
    meta_path: Path,
) -> dict:
    """Compute all task scores for *sids* (a subset of all structure IDs).

    Returns a flat dict with keys like Task1_energy_MAE_eV_per_atom,
    overall_score, etc.  The caller is responsible for prefixing keys.
    """
    # Filter energy dicts to this partition's structures
    p_e = {s: pred_energies[s] for s in sids if s in pred_energies}
    r_e = {s: ref_energies[s]  for s in sids if s in ref_energies}
    part_sids = sorted(set(p_e) & set(r_e))

    out = {}

    # Task 1
    out.update(score_energy(p_e, r_e))

    # Task 2 — compute force errors once and reuse for cheat detection
    force_err, n_force = _load_force_errors(
        pred_dir / "forces", ref_dir / "forces", part_sids
    )
    if force_err.size:
        out["Task2_forces_MAE_eV_per_A"] = float(np.mean(np.abs(force_err)))
        out["Task2_n_structures"] = n_force
    else:
        out["Task2_forces_MAE_eV_per_A"] = float("nan")
        out["Task2_n_structures"] = 0

    # Task 3
    pred_relaxed = pred_dir / "relaxed"
    ref_relaxed  = ref_dir  / "relaxed"
    if pred_relaxed.is_dir() and ref_relaxed.is_dir():
        out.update(score_task3(pred_relaxed, ref_relaxed, part_sids))

    # Task 4
    out.update(score_task4(pred_dir, ref_dir, r_e, part_sids))

    # Task 5
    out.update(score_phase_stability(p_e, r_e, meta_path))

    # DFT-cheating screen (flags submissions whose accuracy is physically
    # implausible for an ML potential — see detect_dft_cheating).
    e_common = sorted(set(p_e) & set(r_e))
    energy_err = np.array([
        p_e[s]["formation_energy_eV_per_atom"] - r_e[s]["formation_energy_eV_per_atom"]
        for s in e_common
    ]) if e_common else np.array([])
    out.update(detect_dft_cheating(
        energy_err, force_err, out.get("Task3_relaxation_RMSD_A", float("nan")),
    ))

    # Overall for this partition
    out["overall_score"] = overall_score(out)

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    input_dir  = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_dir = input_dir / "res"
    ref_dir  = input_dir / "ref"

    # Load energies
    pred_e_file = pred_dir / "formation_energies.csv"
    ref_e_file  = ref_dir  / "formation_energies.csv"
    if not pred_e_file.is_file():
        sys.exit(f"Submission formation_energies.csv not found: {pred_e_file}")
    if not ref_e_file.is_file():
        sys.exit(f"Reference formation_energies.csv not found: {ref_e_file}")

    pred_energies = load_energies(pred_e_file)
    ref_energies  = load_energies(ref_e_file)
    all_sids      = sorted(set(pred_energies) & set(ref_energies))
    meta_path     = ref_dir / "structures_metadata.csv"

    scores = {}

    # ── Public / private split ────────────────────────────────────────────
    # reference_data/public_ids.txt lists the ~30 % of structures whose
    # scores are shown on the live leaderboard.  The remaining ~70 % form
    # the private partition, whose scores are only revealed after the
    # competition closes — preventing participants from reverse-engineering
    # the hidden labels via repeated leaderboard queries.
    public_ids_file = ref_dir / "public_ids.txt"
    if public_ids_file.is_file():
        public_ids   = set(public_ids_file.read_text().split())
        public_sids  = [s for s in all_sids if s     in public_ids]
        private_sids = [s for s in all_sids if s not in public_ids]
        partitions   = [("public_", public_sids), ("private_", private_sids)]
        print(f"Split: {len(public_sids)} public, {len(private_sids)} private structures")
    else:
        # No split file: score everything without prefix (backward-compatible)
        partitions = [("", all_sids)]
        print("No public_ids.txt — scoring all structures (no partition prefix)")

    for prefix, sids in partitions:
        if not sids:
            continue
        part_scores = score_partition(
            pred_dir, ref_dir, pred_energies, ref_energies, sids, meta_path,
        )
        scores.update(prefix_keys(part_scores, prefix))

    # Write output
    scores_file = output_dir / "scores.json"
    with open(scores_file, "w") as f:
        json.dump(scores, f, indent=2)

    # ── DFT-cheating summary ──────────────────────────────────────────────
    # Surface any flag prominently in the scoring log so organisers reviewing
    # the leaderboard backend can audit suspicious submissions.  The flag is
    # advisory only — it does not alter the computed scores above.
    flagged = [k for k, v in scores.items()
               if k.endswith("dft_cheating_suspect") and v]
    if flagged:
        print("\n" + "=" * 70)
        print("⚠  POSSIBLE DFT CHEATING — submission flagged for manual review")
        for k in flagged:
            prefix = k[: -len("dft_cheating_suspect")]
            reason = scores.get(prefix + "dft_cheating_reason", "(no detail)")
            print(f"   [{prefix or 'all'}] {reason}")
        print("=" * 70)

    print(json.dumps(scores, indent=2))
    print(f"\nScores written to {scores_file}")


if __name__ == "__main__":
    main()

