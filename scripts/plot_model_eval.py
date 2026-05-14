#!/usr/bin/env python3
"""Create a compact comparison chart from the model leaderboard CSV.

Input: CSV from scripts/rank_model_eval.py
Output: PNG with three aligned panels:
  - overall score
  - latency (seconds, lower is better)
  - keyword coverage
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _load_rows(csv_path: Path) -> list[dict[str, Any]]:
    with csv_path.open("r", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("Input CSV has no rows.")

    for row in rows:
        row["rank"] = int(_safe_float(row.get("rank"), 0))
        row["score"] = _safe_float(row.get("score"), 0.0)
        row["latency_sec"] = _safe_float(row.get("latency_sec"), 0.0)
        row["keyword_cov"] = _safe_float(row.get("keyword_cov"), 0.0)

    rows.sort(key=lambda r: r["rank"])
    return rows


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot score/latency/keyword-coverage from model leaderboard CSV."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="Leaderboard CSV from scripts/rank_model_eval.py",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG path. Defaults to <input-stem>-chart.png.",
    )
    parser.add_argument(
        "--title",
        default="Model Comparison",
        help="Chart title.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(
            "matplotlib is required for plotting. Install with 'pip install matplotlib'."
        ) from exc

    rows = _load_rows(args.input_csv)
    names = [str(r.get("name", "")) for r in rows]
    scores = [float(r["score"]) for r in rows]
    latencies = [float(r["latency_sec"]) for r in rows]
    keyword_cov = [float(r["keyword_cov"]) for r in rows]

    out_path = args.out
    if out_path is None:
        out_path = args.input_csv.with_name(f"{args.input_csv.stem}-chart.png")

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 6), constrained_layout=True)

    # Panel 1: overall score
    axes[0].barh(names, scores, color="#2E86AB")
    axes[0].invert_yaxis()
    axes[0].set_title("Overall Score")
    axes[0].set_xlabel("Score")
    axes[0].set_xlim(0.0, max(1.0, max(scores) * 1.1 if scores else 1.0))

    # Panel 2: latency (lower is better)
    axes[1].barh(names, latencies, color="#F18F01")
    axes[1].invert_yaxis()
    axes[1].set_title("Latency")
    axes[1].set_xlabel("Seconds (lower is better)")

    # Panel 3: keyword coverage
    axes[2].barh(names, keyword_cov, color="#6A994E")
    axes[2].invert_yaxis()
    axes[2].set_title("Keyword Coverage")
    axes[2].set_xlabel("Coverage fraction")
    axes[2].set_xlim(0.0, 1.0)

    fig.suptitle(args.title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)

    print(f"Chart written: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
