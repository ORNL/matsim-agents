#!/usr/bin/env python3
"""Validate one portability result against structural benchmark invariants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def validate_run(path: Path) -> list[str]:
    errors: list[str] = []
    for name in ("environment.json", "resolved_config.json", "result.json"):
        if not (path / name).is_file():
            errors.append(f"missing {name}")
    if errors:
        return errors
    result = json.loads((path / "result.json").read_text(encoding="utf-8"))
    config = json.loads((path / "resolved_config.json").read_text(encoding="utf-8"))
    if result.get("status") == "failed":
        errors.append("benchmark stage failed")
    policy = config["science"]["active_learning"]
    if policy["retrain"] or policy["promote_model"]:
        errors.append("portability run enabled forbidden model mutation")
    if result.get("facility") != config["deployment"].get("facility"):
        errors.append("result facility differs from resolved deployment")
    execution = result.get("execution", {})
    if result.get("suite") == "all":
        required = {
            "smoke",
            "relaxation",
            "active-learning",
            "phase-exploration",
            "llm-discussion",
        }
        missing = required - set(execution)
        if missing:
            errors.append(f"all-suite result missing executions: {sorted(missing)}")
        active_learning = execution.get("active-learning", {})
        if active_learning.get("selected_candidate_ids") != [1, 3]:
            errors.append("fixed active-learning selection changed")
        if active_learning.get("retrain") or active_learning.get("promote_model"):
            errors.append("portability active learning mutated the model")
        discussion = execution.get("llm-discussion", {})
        if discussion.get("discussion_turns") != 3:
            errors.append("LLM discussion did not complete proposal, critique, and revision")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    errors = validate_run(args.run)
    if errors:
        print("\n".join(f"ERROR: {error}" for error in errors))
        return 1
    print(f"PASS: {args.run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
