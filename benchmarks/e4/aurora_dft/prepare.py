#!/usr/bin/env python3
"""Write planning-only E4 Aurora candidate and run manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarks.e4.aurora_dft.planning import write_manifests


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[3]
        / "research/paper/e4/aurora-dft-scaling/manifests",
    )
    args = parser.parse_args()
    summary = write_manifests(args.output.resolve())
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
