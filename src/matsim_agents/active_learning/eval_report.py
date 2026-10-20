"""Per-group error reports and comparisons of MLIP evaluations.

Errors are summarised per frame, per independent group (as the dataset split
defines groups), and overall, with bootstrap intervals that resample groups,
not frames: frames of one group are correlated, so resampling frames would
overstate the evidence. Command line::

    python -m matsim_agents.active_learning.eval_report matrix RUN_DIR... [--domain test]
    python -m matsim_agents.active_learning.eval_report learning-curve AL_OUT_DIR

``matrix`` tabulates backend x split strategy for the selected models of
finetune_eval runs; ``learning-curve`` tabulates benchmark errors against
cumulative DFT labels from an active-learning run (``loop.benchmark_sets``).
Both print measured values only.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np

METRIC_DEFINITIONS = {
    "force_mae_eV_per_A": "Mean absolute error over all Cartesian force components (eV/A).",
    "force_vector_mae_eV_per_A": (
        "Mean Euclidean norm of the per-atom force error vector (eV/A); larger than the "
        "component MAE by roughly sqrt(3) for isotropic errors."
    ),
    "energy_mae_eV_per_atom": (
        "Mean of |E_model - E_DFT| / N_atoms over frames (eV/atom), raw total energies: "
        "includes any constant offset between the model's and the DFT energy scale."
    ),
    "formation_energy_mae_eV_per_atom": (
        "Mean absolute error of the formation energy per atom (eV/atom); model and DFT each "
        "subtract their own energies of the same declared elemental reference geometries."
    ),
}
# Per-frame arrays of evaluate_frames' parity output, by metric.
FRAME_ERRORS = {
    "force_mae_eV_per_A": "frame_force_mae_eV_per_A",
    "force_vector_mae_eV_per_A": "frame_force_vector_mae_eV_per_A",
    "energy_mae_eV_per_atom": "frame_energy_abs_err_eV_per_atom",
    "formation_energy_mae_eV_per_atom": "frame_formation_abs_err_eV_per_atom",
}
FRAME_SUMMARY_NOTE = (
    "Frame and group summaries weight every frame equally (a frame's force error is the "
    "mean over its atoms' components), so they differ slightly from the overall metrics, "
    "which weight every force component equally."
)


def _bootstrap(group_values: dict[str, list[float]], *, seed: int, n: int) -> list[float] | None:
    """95% interval of the frame-weighted mean, resampling groups with replacement."""
    groups = list(group_values)
    if len(groups) < 2:
        return None
    rng = np.random.default_rng(seed)
    sums = np.array([np.sum(group_values[g]) for g in groups])
    counts = np.array([len(group_values[g]) for g in groups])
    picks = rng.integers(0, len(groups), size=(n, len(groups)))
    means = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def group_report(
    parity: dict[str, np.ndarray],
    group_ids: Sequence[str],
    *,
    seed: int = 0,
    n_bootstrap: int = 2000,
) -> dict:
    """Per-frame, per-group, and overall summaries of one evaluation.

    ``group_ids`` gives each evaluated frame's group, in frame order.
    """

    report: dict = {"n_frames": len(group_ids), "n_groups": len(set(group_ids)), "metrics": {}}
    for metric, key in FRAME_ERRORS.items():
        values = np.asarray(parity.get(key, np.full(len(group_ids), np.nan)), dtype=float)
        if values.shape != (len(group_ids),):
            raise ValueError(f"{key} has {values.shape} values for {len(group_ids)} frames")
        by_group: dict[str, list[float]] = defaultdict(list)
        for group, value in zip(group_ids, values, strict=True):
            if np.isfinite(value):
                by_group[group].append(float(value))
        finite = values[np.isfinite(values)]
        if not finite.size:
            continue
        per_group = {
            g: {"n_frames": len(v), "mean": float(np.mean(v))} for g, v in by_group.items()
        }
        worst = max(per_group, key=lambda g: per_group[g]["mean"])
        report["metrics"][metric] = {
            "frame_mean": float(np.mean(finite)),
            "frame_quantiles": {
                q: float(np.quantile(finite, level))
                for q, level in (("p50", 0.5), ("p90", 0.9), ("p95", 0.95), ("max", 1.0))
            },
            "per_group": per_group,
            "worst_group": {"group": worst, **per_group[worst]},
            "group_bootstrap_95": _bootstrap(by_group, seed=seed, n=n_bootstrap),
        }
    report["note"] = FRAME_SUMMARY_NOTE
    return report


def paired_difference(after: dict, before: dict, *, seed: int = 0, n_bootstrap: int = 2000) -> dict:
    """``after - before`` per metric on identical groups, with a group bootstrap interval."""
    result = {}
    for metric in FRAME_ERRORS:
        a = after.get("metrics", {}).get(metric)
        b = before.get("metrics", {}).get(metric)
        if a is None or b is None:
            continue
        groups = sorted(set(a["per_group"]) & set(b["per_group"]))
        if not groups:
            continue
        diffs = np.array([a["per_group"][g]["mean"] - b["per_group"][g]["mean"] for g in groups])
        weights = np.array([a["per_group"][g]["n_frames"] for g in groups], dtype=float)
        interval = None
        if len(groups) >= 2:
            rng = np.random.default_rng(seed)
            picks = rng.integers(0, len(groups), size=(n_bootstrap, len(groups)))
            means = (diffs[picks] * weights[picks]).sum(axis=1) / weights[picks].sum(axis=1)
            interval = [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]
        result[metric] = {
            "mean_difference": float(np.average(diffs, weights=weights)),
            "n_groups": len(groups),
            "groups_improved": int(np.sum(diffs < 0)),
            "group_bootstrap_95": interval,
        }
    return result


def summarize_repetitions(summaries: list[dict], *, n_test_groups: list[int]) -> dict:
    """Mean and spread of the selected model's metrics over repetitions."""
    by_metric: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for summary in summaries:
        for domain, metrics in summary["selected_metrics"].items():
            for metric, value in metrics.items():
                if value is not None and np.isfinite(value):
                    by_metric[domain][metric].append(float(value))
    return {
        "n_repetitions": len(summaries),
        "repetitions": [
            {"split_seed": s["split_seed"], "training_seed": s["training_seed"]} for s in summaries
        ],
        "test_groups_per_repetition": n_test_groups,
        "metrics": {
            domain: {
                metric: {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }
                for metric, values in metrics.items()
            }
            for domain, metrics in by_metric.items()
        },
        "independence": (
            "Repetitions with different split seeds reuse the same frames and groups in "
            "different partitions, and training seeds reuse the same split: the spread "
            "describes sensitivity to these choices, not independent experiments. The "
            "number of independent test groups is listed per repetition."
        ),
    }


# --------------------------------------------------------------------------- #
# Aggregation across runs                                                     #
# --------------------------------------------------------------------------- #


def collect_runs(roots: Sequence[Path]) -> list[dict]:
    """One row per (run, evaluation domain) from finetune_eval ``eval/report.json`` files."""
    rows = []
    for root in roots:
        for path in sorted(Path(root).rglob("eval/report.json")):
            report = json.loads(path.read_text())
            selected = report["selection"]["selected"]
            for domain, models in report["evaluations"].items():
                metrics = models.get(selected, {}).get("metrics", {})
                groups = models.get(selected, {}).get("groups", {})
                rows.append(
                    {
                        "run": str(path.parent.parent),
                        "backend": report["backend"],
                        "strategy": report["split"]["strategy"],
                        "objective": report["split"]["objective"],
                        "domain": domain,
                        "selected": selected,
                        "n_groups": groups.get("n_groups"),
                        **{metric: metrics.get(metric) for metric in METRIC_DEFINITIONS},
                        **{
                            f"{metric}_ci95": groups.get("metrics", {})
                            .get(metric, {})
                            .get("group_bootstrap_95")
                            for metric in METRIC_DEFINITIONS
                        },
                    }
                )
    return rows


def matrix(rows: list[dict], metric: str, domain: str = "test") -> str:
    """Markdown table of ``metric`` for backend (rows) x split strategy (columns)."""
    cells: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in rows:
        if row["domain"] != domain or row.get(metric) is None:
            continue
        interval = row.get(f"{metric}_ci95")
        text = f"{row[metric]:.4g}"
        if interval:
            text += f" [{interval[0]:.3g}, {interval[1]:.3g}]"
        text += f" ({row['n_groups']} groups)"
        cells[(row["backend"], row["strategy"])].append(text)
    backends = sorted({b for b, _ in cells})
    strategies = sorted({s for _, s in cells})
    lines = [
        f"{metric} on `{domain}`, selected model; [group-bootstrap 95% interval]",
        "",
        "| backend | " + " | ".join(strategies) + " |",
        "|---|" + "---|" * len(strategies),
    ]
    for backend in backends:
        lines.append(
            f"| {backend} | "
            + " | ".join("<br>".join(cells.get((backend, s), ["-"])) for s in strategies)
            + " |"
        )
    return "\n".join(lines) + "\n"


def learning_curve(al_out_dir: Path) -> list[dict]:
    """Benchmark errors against cumulative DFT labels, one row per iteration and benchmark."""
    rows = []
    for state_path in sorted(Path(al_out_dir).glob("iteration_*/state.json")):
        state = json.loads(state_path.read_text())
        for name, entry in sorted((state.get("benchmark") or {}).items()):
            rows.append(
                {
                    "iteration": state["iteration"],
                    "n_dft_labels_total": entry["n_dft_labels_total"],
                    "model": entry["model"],
                    "benchmark": name,
                    **{metric: entry["metrics"].get(metric) for metric in METRIC_DEFINITIONS},
                }
            )
    return rows


def _write_csv(rows: list[dict], stream) -> None:
    if not rows:
        return
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    table = commands.add_parser("matrix", help="Backend x split strategy table.")
    table.add_argument("runs", nargs="+", type=Path)
    table.add_argument("--metric", default="force_mae_eV_per_A", choices=sorted(METRIC_DEFINITIONS))
    table.add_argument("--domain", default="test")
    table.add_argument("--csv", type=Path, help="Also write every row as CSV.")
    curve = commands.add_parser("learning-curve", help="Benchmark error vs DFT labels.")
    curve.add_argument("al_out_dir", type=Path)
    args = parser.parse_args(argv)
    if args.command == "matrix":
        rows = collect_runs(args.runs)
        sys.stdout.write(matrix(rows, args.metric, args.domain))
        if args.csv:
            with args.csv.open("w", newline="") as stream:
                _write_csv(rows, stream)
    else:
        _write_csv(learning_curve(args.al_out_dir), sys.stdout)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
