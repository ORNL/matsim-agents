"""Read-only hypothesis audit tables for an E3-MLIP campaign root.

Writes an LLM interaction index, a per-record hypothesis evolution table, review flags
and a provenance manifest. Revision types are mechanical field comparisons; whether a
change is a scientific revision still needs human review of the flagged rows.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

RESOLVED = {"supported_by_mlip", "challenged_by_mlip"}
SKIP_DIRS = {"tasks", "generated"}
HISTORY = re.compile(r"Prior strategy-local outcomes: (\[.*?\])\n", re.S)

INTERACTION_FIELDS = [
    "material",
    "round",
    "interaction_type",
    "model",
    "prompt_version",
    "status",
    "error",
    "prompt_chars",
    "raw_response_chars",
    "prompt_outcome_rows",
    "hypotheses",
    "path",
    "sha256",
]
EVOLUTION_FIELDS = [
    "material",
    "round",
    "record_kind",
    "hypothesis_id",
    "status_before",
    "status_after",
    "revision_type",
    "round_converged",
    "evidence_formulas",
    "claim_before",
    "claim_after",
    "prediction_before",
    "prediction_after",
    "reason",
    "path",
]
FLAG_FIELDS = ["material", "round", "record_kind", "hypothesis_id", "flag", "detail", "path"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _history_rows(prompt: str) -> int | None:
    match = HISTORY.search(prompt)
    return len(json.loads(match.group(1))) if match else None


def _revision_type(before: dict | None, after: dict) -> str:
    if before is None:
        return "new"
    changed = [k for k in ("claim", "prediction", "status") if before.get(k) != after.get(k)]
    if not changed:
        return "unchanged"
    if changed == ["status"]:
        return "status_only"
    return "+".join(k for k in changed if k != "status") + "_changed"


def _streams(root: Path):
    for stream in sorted(root.glob("*/*/seed-*")):
        if stream.is_dir() and any(stream.glob("round-*")):
            yield stream.parts[-3], stream


def _records(round_dir: Path):
    for path in sorted((round_dir / "acquisition").glob("proposal-*.json")):
        yield "acquisition proposal", path
    for path in sorted(round_dir.glob("hypothesis-assessment*.json")):
        yield "hypothesis assessment", path


def audit(root: Path) -> dict[str, list[dict[str, Any]]]:
    interactions, evolution, flags = [], [], []
    for material, stream in _streams(root):
        formulas: dict[str, str] = {}
        converged: dict[int, int] = {}
        for summary_path in sorted(stream.glob("round-*/round-summary.json")):
            summary = _load(summary_path)
            converged[int(summary_path.parent.name.split("-")[1])] = summary["outcomes"].get(
                "converged", 0
            )
            for row in summary.get("action_outcomes", []):
                formulas[row["evidence_id"]] = row["formula"]
        # Failed or timed-out rounds carry their terminal results only in the assessment.
        for path in sorted(stream.glob("round-*/hypothesis-assessment-*.json")):
            n = int(path.parent.name.split("-")[1])
            partial = [row for row in _load(path).get("additional_history", []) if "formula" in row]
            if partial and n not in converged:
                converged[n] = sum(row.get("converged", 0) for row in partial)
            formulas.update((row["evidence_id"], row["formula"]) for row in partial)
        for round_dir in sorted(stream.glob("round-*")):
            n = int(round_dir.name.split("-")[1])
            for kind, path in _records(round_dir):
                record = _load(path)
                hypotheses = record.get("scientific_hypotheses")
                interactions.append(
                    {
                        "material": material,
                        "round": n,
                        "interaction_type": kind,
                        "model": record.get("model_id"),
                        "prompt_version": record.get("prompt_version") or record.get("version"),
                        "status": record.get("status"),
                        "error": record.get("error") or "",
                        "prompt_chars": len(record.get("prompt") or ""),
                        "raw_response_chars": len(record.get("raw_response") or ""),
                        "prompt_outcome_rows": _history_rows(record.get("prompt") or ""),
                        "hypotheses": None if hypotheses is None else len(hypotheses),
                        "path": str(path),
                        "sha256": _sha256(path),
                    }
                )
                if record.get("status") != "accepted":
                    flags.append(
                        {
                            "material": material,
                            "round": n,
                            "record_kind": kind,
                            "hypothesis_id": "",
                            "flag": "record_not_accepted",
                            "detail": f"status {record.get('status')}: {record.get('error') or ''}"[
                                :300
                            ],
                            "path": str(path),
                        }
                    )
                    continue
                if not hypotheses:
                    continue
                short = "proposal" if kind == "acquisition proposal" else "assessment"
                # A proposal sees the same outcomes as the previous round's assessment.
                new_data = converged.get(n) if short == "assessment" else 0
                previous = {h["id"]: h for h in record.get("previous_hypotheses") or []}
                for h in hypotheses:
                    before = previous.get(h["id"])
                    revision = _revision_type(before, h)
                    cited = sorted(
                        {formulas[e] for e in h.get("evidence_ids", []) if e in formulas}
                    )
                    base = {
                        "material": material,
                        "round": n,
                        "record_kind": short,
                        "hypothesis_id": h["id"],
                        "path": str(path),
                    }
                    evolution.append(
                        {
                            **base,
                            "status_before": before and before.get("status") or "",
                            "status_after": h.get("status"),
                            "revision_type": revision,
                            "round_converged": converged.get(n, ""),
                            "evidence_formulas": ";".join(cited),
                            "claim_before": before and before.get("claim") or "",
                            "claim_after": h.get("claim"),
                            "prediction_before": before and before.get("prediction") or "",
                            "prediction_after": h.get("prediction"),
                            "reason": h.get("reason", ""),
                        }
                    )
                    status_changed = before is not None and before.get("status") != h.get("status")
                    if before is None and h.get("status") in RESOLVED:
                        flags.append(
                            {
                                **base,
                                "flag": "resolved_on_introduction",
                                "detail": f"new hypothesis with status {h['status']}",
                            }
                        )
                    if status_changed and h.get("status") in RESOLVED and not new_data:
                        flags.append(
                            {
                                **base,
                                "flag": "status_change_without_new_converged_data",
                                "detail": f"{before['status']} -> {h['status']}",
                            }
                        )
                    if (
                        (status_changed or before is None)
                        and h.get("status") in RESOLVED
                        and len(cited) > 1
                    ):
                        flags.append(
                            {
                                **base,
                                "flag": "resolved_citing_multiple_compositions",
                                "detail": f"{h['status']} citing {';'.join(cited)}",
                            }
                        )
                    if revision not in ("new", "unchanged", "status_only"):
                        flags.append(
                            {
                                **base,
                                "flag": "text_changed",
                                "detail": f"{revision}; round converged {converged.get(n, 'n/a')}",
                            }
                        )
            if n in converged and converged[n] == 0:
                flags.append(
                    {
                        "material": material,
                        "round": n,
                        "record_kind": "round",
                        "hypothesis_id": "",
                        "flag": "round_no_converged_structures",
                        "detail": "check task termination reasons before reading outcomes",
                        "path": str(round_dir / "round-summary.json"),
                    }
                )
    return {"interactions": interactions, "evolution": evolution, "flags": flags}


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)


def write(root: Path, output: Path) -> dict[str, int]:
    tables = audit(root)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "interaction-index.csv", INTERACTION_FIELDS, tables["interactions"])
    _write_csv(output / "hypothesis-evolution.csv", EVOLUTION_FIELDS, tables["evolution"])
    _write_csv(output / "review-flags.csv", FLAG_FIELDS, tables["flags"])
    output_resolved = output.resolve()
    run_files = {}
    for directory, dirs, files in os.walk(root):
        dirs[:] = sorted(
            d
            for d in dirs
            if d not in SKIP_DIRS and (Path(directory) / d).resolve() != output_resolved
        )
        for name in sorted(files):
            path = Path(directory) / name
            run_files[str(path)] = _sha256(path)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    manifest = {
        "campaign_root": str(root),
        "generated_utc": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "repository_head": head,
        "excluded": sorted(SKIP_DIRS),
        "outputs": {p.name: _sha256(p) for p in sorted(output.glob("*.csv"))},
        "run_files": run_files,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return {key: len(rows) for key, rows in tables.items()} | {"run_files": len(run_files)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_root", type=Path)
    parser.add_argument("--output", type=Path, help="default: <campaign_root>/hypothesis-audit")
    args = parser.parse_args(argv)
    counts = write(args.campaign_root, args.output or args.campaign_root / "hypothesis-audit")
    print(json.dumps(counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
