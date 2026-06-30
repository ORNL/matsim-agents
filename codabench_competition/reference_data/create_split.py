#!/usr/bin/env python3
"""
create_split.py — Generate the public/private test-set split for the competition.

Stratified by formula: every formula group has at least one structure in each
partition.  This prevents participants from reconstructing DFT labels by
memorising a per-formula baseline and ensures Task 5 (phase stability) is
scorable in both partitions.

  Public set  (~30%) — scored live on the leaderboard during the competition.
  Private set (~70%) — used for the final ranking, revealed only after the
                       competition closes.  This is the anti-cheating partition:
                       even if a team overfits to the public scores, the private
                       ranking cannot be gamed without access to hidden labels.

Outputs (written to this script's directory = reference_data/):
    public_ids.txt   — one structure_id per line
    private_ids.txt  — one structure_id per line

Stratification reads the *private* anonymisation table (which still carries the
`formula` column alongside the opaque `anon_id`).  The public
structures_metadata.csv only exposes `structure_id,file_path`, so the formula
grouping is intentionally not derivable from the public competition bundle.
Provide the private table server-side via the MATSIM_PRIVATE_METADATA environment
variable, or place it at
reference_data/private_reference/anon_to_descriptive.csv (gitignored).

Run once, after DFT calculations are complete but BEFORE uploading reference_data
to Codabench:
    python reference_data/create_split.py

Re-running with the same SEED always produces the same split.
"""
from __future__ import annotations

import csv
import os
import random
from collections import defaultdict
from pathlib import Path

# ── Configuration ────────────────────────────────────────────────────────────
SEED             = 42          # fixed seed → reproducible, auditable split
PUBLIC_FRACTION  = 0.30        # ~30 % of each formula group goes to public set

# ── Paths ────────────────────────────────────────────────────────────────────
HERE          = Path(__file__).parent
# Private anonymisation table (maps anon_id → descriptive, and carries `formula`).
# NEVER committed to the public repo; supplied server-side only.
METADATA_PATH = Path(
    os.environ.get(
        "MATSIM_PRIVATE_METADATA",
        str(HERE / "private_reference" / "anon_to_descriptive.csv"),
    )
)
PUBLIC_FILE   = HERE / "public_ids.txt"
PRIVATE_FILE  = HERE / "private_ids.txt"


def main() -> None:
    if not METADATA_PATH.exists():
        raise SystemExit(
            f"Private anonymisation table not found at {METADATA_PATH}.\n"
            "Set MATSIM_PRIVATE_METADATA to the server-side anon_to_descriptive.csv "
            "(the public structures_metadata.csv no longer carries `formula`)."
        )
    rows = list(csv.DictReader(open(METADATA_PATH)))
    print(f"Total structures: {len(rows)}")

    # The public-facing structure id is the opaque MATS key (`anon_id`); fall
    # back to `structure_id` if running against a table that lacks it.
    id_col = "anon_id" if rows and "anon_id" in rows[0] else "structure_id"

    # Group structure_ids by formula (= phase-stability group)
    groups: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        groups[row["formula"]].append(row[id_col])

    rng = random.Random(SEED)

    public_ids: list[str]  = []
    private_ids: list[str] = []

    for formula, sids in sorted(groups.items()):
        # Sort first so the split is reproducible regardless of the input row
        # order of the private table.
        shuffled = sorted(sids)
        rng.shuffle(shuffled)

        # At least 1 public, at least 1 private in every group
        n_pub = max(1, round(len(shuffled) * PUBLIC_FRACTION))
        n_pub = min(n_pub, len(shuffled) - 1)

        public_ids.extend(shuffled[:n_pub])
        private_ids.extend(shuffled[n_pub:])

        print(f"  {formula:30s}  total={len(shuffled):3d}  "
              f"public={n_pub}  private={len(shuffled)-n_pub}")

    # Sort for determinism and readability
    public_ids.sort()
    private_ids.sort()

    PUBLIC_FILE.write_text("\n".join(public_ids) + "\n")
    PRIVATE_FILE.write_text("\n".join(private_ids) + "\n")

    print(f"\nPublic  : {len(public_ids):3d} structures → {PUBLIC_FILE}")
    print(f"Private : {len(private_ids):3d} structures → {PRIVATE_FILE}")
    print("\nDone.  Commit these files to reference_data/ before uploading to Codabench.")
    print("Do NOT commit raw DFT energies (formation_energies.csv) to any public repo.")


if __name__ == "__main__":
    main()
