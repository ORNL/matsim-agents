#!/usr/bin/env python3
"""Analyze E4 raw JSONL measurements; never synthesize missing observations."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.e4.consistency import compare_consistency_records  # noqa: E402
from benchmarks.e4.planning import analyze_scaling  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("measurements", type=Path)
    parser.add_argument("--kind", choices=("scaling", "consistency"), default="scaling")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = [
        json.loads(line) for line in args.measurements.read_text().splitlines() if line.strip()
    ]
    rows = (
        analyze_scaling(records)
        if args.kind == "scaling"
        else [compare_consistency_records(records)]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise SystemExit("no strong/weak scaling records found")
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} aggregates from {len(records)} raw runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
