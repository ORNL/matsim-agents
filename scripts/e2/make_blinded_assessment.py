#!/usr/bin/env python3
"""Build a blinded second-assessment package from a frozen E2 agent-eval run.

The package holds the exact question+evidence packet per case and the shuffled,
anonymized assessor answers. Arm names, model names, grades and answer keys are
excluded. The unblinding key is written outside the package; only its SHA-256
is recorded inside.
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import random
import re
from pathlib import Path

MODEL_NAMES = re.compile(r"\b(claude[-_ ]?sonnet|gemini[-_ ]?(?:2\.5[-_ ]?)?flash|claude|gemini|sonnet)\b", re.I)

RUBRIC = """# Blinded scientific assessment: {suite}

You are scoring {n_items} anonymized answers to {n_cases} evidence-assessment questions.
Each `packets/<case>.md` file shows the exact question and evidence that every assessor received.
`items.jsonl` holds the answers in random order. Several items share a case.

Do not try to identify which system produced an item. Score only what is written.
Length is not quality: a short answer that is correct and well grounded can earn full marks.

## Fields to fill in `scores.csv` (one row per item)

| Field | Values | Meaning |
|---|---|---|
| verdict_correct | Y / N | Is `status` the scientifically correct reading of the packet (supported / contradicted / unresolved)? Use your own judgement, not a key. |
| physical_reasoning | 0-2 | 0 = wrong or missing physics; 1 = correct but generic or partly wrong; 2 = correct, specific, and quantitatively checked where the packet allows |
| evidence_attribution | 0-2 | 0 = cites wrong or no records, or relies on outside knowledge as evidence; 1 = correct records but loose mapping to the claim; 2 = every decisive statement traces to the cited record values |
| uncertainty | 0-2 | 0 = none stated, or overstated or misapplied (e.g. calling a decidable claim unresolved); 1 = caveats present but generic; 2 = caveats are specific, correctly weighted, and do not change a decidable verdict |
| follow_up | 0-2 | 0 = none when one is needed, or irrelevant; 1 = relevant but not decisive or not consistent with `request_dft`; 2 = the decisive test, consistent with `request_dft` (or correctly no request) |
| notes | text | Short justification, required for any 0 or N |

Rater ID and date go in `rater.json`. Return `scores.csv` and `rater.json` unchanged in structure.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--key-out", type=Path, required=True, help="unblinding key, kept outside the package")
    ap.add_argument("--seed", type=int, default=20261104)
    a = ap.parse_args()

    records = [json.loads(line) for line in open(a.run_dir / "answers.jsonl")]
    suite = json.load(open(a.run_dir / "summary.json"))["suite_id"]
    cases = list(dict.fromkeys(r["case_id"] for r in records))
    (a.out / "packets").mkdir(parents=True, exist_ok=False)
    for case in cases:
        req = glob.glob(str(a.run_dir / "arms/single_llm/seed-0" / case / "debate-*/request.json"))
        if len(req) != 1:
            raise SystemExit(f"expected one single_llm request for {case}, found {len(req)}")
        packet = json.load(open(req[0]))["hypothesis"]
        (a.out / "packets" / f"{case}.md").write_text(f"# {case}\n\n```text\n{packet}\n```\n")

    rng = random.Random(a.seed)
    order = list(range(len(records)))
    rng.shuffle(order)
    items, key = [], []
    for n, idx in enumerate(order, 1):
        r = records[idx]
        ans = r["answer"]
        item_id = f"item-{n:03d}"
        items.append({
            "item_id": item_id, "case": r["case_id"], "status": ans["status"],
            "request_dft": ans.get("request_dft"), "proposed_test_ids": ans.get("proposed_test_ids", []),
            "evidence_ids": ans.get("evidence_ids", []),
            "rationale": MODEL_NAMES.sub("[participant]", ans.get("rationale", "")),
        })
        key.append({"item_id": item_id, "case_id": r["case_id"], "arm": r["arm"], "random_seed": r["random_seed"]})

    with open(a.out / "items.jsonl", "w") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")
    with open(a.out / "scores.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "case", "verdict_correct", "physical_reasoning", "evidence_attribution",
                    "uncertainty", "follow_up", "notes"])
        for item in items:
            w.writerow([item["item_id"], item["case"], "", "", "", "", "", ""])
    (a.out / "rater.json").write_text(json.dumps({"rater_id": "", "affiliation": "", "date": "",
                                                  "saw_unblinded_results": False}, indent=2) + "\n")
    a.key_out.parent.mkdir(parents=True, exist_ok=True)
    key_text = json.dumps(key, indent=1) + "\n"
    a.key_out.write_text(key_text)
    (a.out / "README.md").write_text(RUBRIC.format(suite=suite, n_items=len(items), n_cases=len(cases)))
    (a.out / "provenance.json").write_text(json.dumps({
        "suite_id": suite, "source_run": str(a.run_dir),
        "answers_sha256": hashlib.sha256((a.run_dir / "answers.jsonl").read_bytes()).hexdigest(),
        "shuffle_seed": a.seed,
        "unblinding_key_sha256": hashlib.sha256(key_text.encode()).hexdigest(),
        "redaction": "model names replaced by [participant]; arm, model, grade and answer key removed",
        "residual_unblinding_risk": "template-style deterministic rationales and references to peers "
                                    "or critique rounds can reveal the arm family",
    }, indent=2) + "\n")
    print(f"{len(items)} items, {len(cases)} packets -> {a.out}; key -> {a.key_out}")


if __name__ == "__main__":
    main()
