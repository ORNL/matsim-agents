"""Expand the fine-tune + evaluation matrix into per-run input files.

For every selected (variant, case), write ``<out-root>/<variant>/<case>/mlip.yaml``
(the foundation model) and ``evaluation.yaml`` (the base evaluation with the
variant's ``fine_tune`` recipe), after validating both, and print one
tab-separated line per run: ``variant case backend out_dir``.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from matsim_agents.active_learning.finetune_eval import EvaluationConfig  # noqa: E402
from matsim_agents.backends.mlip.config import parse_mlip_config  # noqa: E402
from matsim_agents.execution.variables import expand_variables, variable_defaults  # noqa: E402


def load_matrix(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    raw = yaml.safe_load(text) or {}
    defaults = variable_defaults(raw.get("vars"), source=path)
    matrix = yaml.safe_load(expand_variables(text, defaults, source=path))
    matrix.pop("vars", None)
    for key in ("cases", "variants"):
        if not isinstance(matrix.get(key), dict) or not matrix[key]:
            raise ValueError(f"{path}: `{key}` must be a non-empty mapping")
    return matrix


def _selected(names: list[str], requested: str | None, kind: str) -> list[str]:
    if not requested:
        return names
    wanted = [name.strip() for name in requested.split(",") if name.strip()]
    unknown = sorted(set(wanted) - set(names))
    if unknown:
        raise ValueError(f"unknown {kind}: {', '.join(unknown)} (known: {', '.join(names)})")
    return [name for name in names if name in wanted]


def expand(
    matrix: dict,
    evaluation: dict,
    out_root: Path,
    *,
    variants: str | None = None,
    cases: str | None = None,
) -> list[tuple[str, str, str, Path]]:
    runs = []
    for variant in _selected(list(matrix["variants"]), variants, "variants"):
        spec = matrix["variants"][variant]
        for case in _selected(list(matrix["cases"]), cases, "cases"):
            mlip = copy.deepcopy(spec["mlip"])
            overrides = (matrix["cases"][case] or {}).get("mlip", {})
            mlip.update(overrides.get(mlip["backend"], {}))
            selection = parse_mlip_config(mlip)
            config = copy.deepcopy(evaluation)
            if "fine_tune" in spec:
                config["fine_tune"] = spec["fine_tune"]
            EvaluationConfig.model_validate(config).recipe_for(selection)
            out_dir = out_root / variant / case
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "mlip.yaml").write_text(yaml.safe_dump({"mlip": mlip}, sort_keys=False))
            (out_dir / "evaluation.yaml").write_text(
                yaml.safe_dump({"evaluation": config}, sort_keys=False)
            )
            runs.append((variant, case, selection.backend, out_dir))
    return runs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--evaluation-config", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--variants", help="Comma-separated variant names (default: all).")
    parser.add_argument("--cases", help="Comma-separated case names (default: all).")
    args = parser.parse_args(argv)
    raw = yaml.safe_load(args.evaluation_config.read_text(encoding="utf-8")) or {}
    evaluation = raw.get("evaluation", raw)
    try:
        runs = expand(
            load_matrix(args.matrix),
            evaluation,
            args.out_root,
            variants=args.variants,
            cases=args.cases,
        )
    except ValueError as error:
        parser.error(str(error))
    for variant, case, backend, out_dir in runs:
        print(f"{variant}\t{case}\t{backend}\t{out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
