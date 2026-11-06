#!/usr/bin/env python3
"""Consolidate frozen E2 suites into coverage, comparison, cost, and provenance artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import yaml

from matsim_agents.campaign.agent_eval import (
    assessment_packet,
    deterministic_evidence_assessment,
    load_suite,
)
from matsim_agents.execution.ledger import EvidenceLedger

ROOT = Path(__file__).resolve().parents[2]
E2 = ROOT / "research/paper/e2"
OUT = E2 / "final"
CONFIG = E2 / "configs/lgea-dea-mag-v1.yaml"
ARMS = ["no_llm", "single_llm", "multi_llm_no_critique", "multi_llm_cross_critique"]
RATERS = {"opus": "rater-llm-claude-opus-5-5_google", "argon": "rater-llm-gemini-4-argon_google"}
FIELDS = ["physical_reasoning", "evidence_attribution", "uncertainty", "follow_up"]
SUITES = [
    ("lfp-phosphorene-v1", "raw/lfp-phosphorene-v1-hvk5dot2", "historical"),
    ("nbtao-v1", "raw/nbtao-v1-pgbpfthh", "historical"),
    ("nbtao-v2", "raw/nbtao-v2/agent-eval-37jize3g", "manifest-frozen"),
    ("zn-formate-v1", "raw/zn-formate-v1/agent-eval-vhjqyznq", "manifest-frozen"),
    ("nbtavhfrz-bcc-v1", "raw/nbtavhfrz-bcc-v1/agent-eval-bj8rbngk", "manifest-frozen"),
]
MATERIAL = {"lfp": "LiFePO4", "phos": "phosphorene", "nbtao": "Nb-Ta-O",
            "zn": "Zn-formate", "bcc": "NbTaVHfZrTi BCC"}
QUALIFICATION = {
    "lfp-phosphorene-v1": "archived VASP 6.6.0 OUTCAR values (private archive 4e18697); raw OUTCARs not mounted here",
    "nbtao-v1": "raw-output verified (scripts/verify_vasp_run.py), evidence/nbtao5-q1,q2",
    "nbtao-v2": "raw-output verified Q1/Q2/Q3 static energies, evidence/nbtao5-q1..q3",
    "zn-formate-v1": "raw-output verified relax/static (verify-*.json), PBE-D3(BJ)",
    "nbtavhfrz-bcc-v1": "raw-output verified relax/static, ISPIN=1; no k-point or spin sensitivity test",
}
LINKS = [("nbtao-formation-energy-negative", "nbtao-formation-energy-negative-v2")]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def material(case_id: str) -> str:
    return MATERIAL[case_id.split("-")[0]]


def unblind(raw: Path, prov: dict) -> dict[str, dict]:
    answers = [json.loads(line) for line in (raw / "answers.jsonl").read_text().splitlines()]
    order = list(range(len(answers)))
    random.Random(prov["shuffle_seed"]).shuffle(order)
    key = [{"item_id": f"item-{n:03d}", "case_id": answers[i]["case_id"], "arm": answers[i]["arm"],
            "random_seed": answers[i]["random_seed"]} for n, i in enumerate(order, 1)]
    if hashlib.sha256((json.dumps(key, indent=1) + "\n").encode()).hexdigest() != prov["unblinding_key_sha256"]:
        raise SystemExit(f"unblinding key mismatch for {raw}")
    return {k["item_id"]: k for k in key}


def evidence_refs(node, found: set[str]) -> None:
    if isinstance(node, dict):
        for k, v in node.items():
            if isinstance(v, str) and ("sha256" in k or "record" in k or k == "calculation"):
                found.add(f"{k}={v}")
            elif isinstance(v, list) and ("record" in k or "sha256" in k):
                found.update(f"{k}={x}" for x in v)
            else:
                evidence_refs(v, found)
    elif isinstance(node, list):
        for v in node:
            evidence_refs(v, found)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    evaluator = ROOT / "src/matsim_agents/campaign/agent_eval.py"
    coverage, records, scores, manifest = [], [], defaultdict(dict), {"suites": []}
    structures: dict[str, set[str]] = {}
    for name, raw_rel, freeze in SUITES:
        suite_dir = ROOT / "benchmarks/agent_eval_e2" / name
        raw, blind = E2 / raw_rel, E2 / "blinded" / name
        suite = load_suite(suite_dir)
        answers = [json.loads(line) for line in (raw / "answers.jsonl").read_text().splitlines()]
        prov = json.loads((blind / "provenance.json").read_text())
        key = unblind(raw, prov)
        summary = json.loads((raw / "summary.json").read_text())
        stored = {(a["case_id"], a["arm"]): a for a in answers}
        repro = []
        with tempfile.TemporaryDirectory() as tmp:
            for case in suite.cases:
                ledger = EvidenceLedger(Path(tmp) / f"{case.case_id}.jsonl")
                recs = {lab: ledger.append(spec.record(suite.suite_id, case.case_id, lab))
                        for lab, spec in case.evidence.items()}
                ans = deterministic_evidence_assessment(assessment_packet(case, recs), ledger)
                old = stored[(case.case_id, "no_llm")]["answer"]
                repro.append(ans.status == old["status"] and ans.request_dft == old["request_dft"])
        raw_yaml = yaml.safe_load((suite_dir / "cases.yaml").read_text())
        for case, case_yaml in zip(suite.cases, raw_yaml["cases"]):
            labels = {v.structure for e in case.evidence.values() for v in e.values if v.structure}
            structures[case.case_id] = labels
            refs: set[str] = set()
            evidence_refs(case_yaml.get("evidence", {}), refs)
            row = {"material": material(case.case_id), "suite_id": suite.suite_id, "case_id": case.case_id,
                   "claim_type": case.claim_type, "reference_verdict": case.expected_status,
                   "dft_request_justified": case.dft_request_justified,
                   "evidence_qualification": QUALIFICATION[name], "answer_key_freeze": freeze,
                   "included_in_aggregate": True}
            for arm in ARMS:
                a = stored[(case.case_id, arm)]
                row[f"{arm}_verdict"] = a["answer"]["status"]
                row[f"{arm}_correct"] = a["grade"]["correct"]
            coverage.append(row)
        for a in answers:
            records.append({"material": material(a["case_id"]), "suite": name, **a})
        for label, folder in RATERS.items():
            for r in csv.DictReader((blind / folder / "scores.csv").open(newline="")):
                k = key[r["item_id"]]
                scores[(k["case_id"], k["arm"])][label] = r
        frozen = E2 / f"{name}.FROZEN.sha256"
        manifest["suites"].append({
            "suite_id": suite.suite_id, "cases": len(suite.cases), "responses": len(answers),
            "cases_yaml_sha256": sha(suite_dir / "cases.yaml"),
            "frozen_record": str(frozen.relative_to(ROOT)) if frozen.exists() else None,
            "answer_key_freeze": freeze, "protocol_digest": summary["protocol_digest"],
            "run_dir": str(raw.relative_to(ROOT)), "answers_sha256": sha(raw / "answers.jsonl"),
            "summary_sha256": sha(raw / "summary.json"), "blinded_package": str(blind.relative_to(ROOT)),
            "blinded_answers_sha256_matches": prov["answers_sha256"] == sha(raw / "answers.jsonl"),
            "unblinding_key_sha256": prov["unblinding_key_sha256"],
            "rater_scores_sha256": {lab: sha(blind / f / "scores.csv") for lab, f in RATERS.items()},
            "no_llm_reproduced_by_pinned_evaluator": f"{sum(repro)}/{len(repro)}",
            "evidence_sources": sorted({ref for case_yaml in raw_yaml["cases"]
                                        for ref in (lambda s: (evidence_refs(case_yaml.get('evidence', {}), s), s)[1])(set())}),
        })

    parent = {c: c for c in structures}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    ids = list(structures)
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            if structures[a] & structures[b]:
                parent[find(a)] = find(b)
    for a, b in LINKS:
        parent[find(a)] = find(b)
    for row in coverage:
        row["dependency_cluster"] = find(row["case_id"])
    coverage.append({"material": "Cantor FCC", "suite_id": "none", "case_id": "none",
                     "evidence_qualification": "not qualified (PBS 9040206, 2026-11-06T02:49Z): ideal/random1/random2 relax "
                     "in progress, Fmax 0.055/0.206/0.232 > 0.02 eV/A, some ionic steps at NELM=200; random3 killed "
                     "(exit 143) at step 6; no static",
                     "included_in_aggregate": False})
    with (OUT / "e2-six-material-coverage.csv").open("w", newline="") as f:
        cols = list(dict.fromkeys(k for r in coverage for k in r))
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(coverage)

    groups: dict[tuple[str, str], list] = defaultdict(list)
    for r in records:
        groups[(r["material"], r["arm"])].append(r)
    comp, cost = [], []
    for (mat, arm), rs in sorted(groups.items()):
        n = len(rs)
        row = {"material": mat, "arm": arm, "cases": n,
               "verdict_correct": sum(r["grade"]["correct"] for r in rs) / n,
               "evidence_attribution": sum(r["grade"]["evidence_attribution"] for r in rs) / n,
               "dft_request_correct": sum(r["grade"]["dft_request_correct"] for r in rs) / n}
        inc = [r["grade"]["correctly_inconclusive"] for r in rs if r["grade"]["correctly_inconclusive"] is not None]
        row["correctly_inconclusive"] = sum(inc) / len(inc) if inc else ""
        row["unresolved_reference_cases"] = len(inc)
        for lab in RATERS:
            s = [scores[(r["case_id"], arm)][lab] for r in rs]
            row[f"{lab}_verdict_Y"] = sum(x["verdict_correct"] == "Y" for x in s) / n
            for fld in FIELDS:
                row[f"{lab}_{fld}"] = sum(int(x[fld]) for x in s) / n
        row["rater_verdict_agreement"] = sum(scores[(r["case_id"], arm)]["opus"]["verdict_correct"]
                                             == scores[(r["case_id"], arm)]["argon"]["verdict_correct"] for r in rs) / n
        comp.append(row)
        c = {k: sum(r[k] for r in rs) for k in ("calls", "prompt_tokens", "completion_tokens", "latency_s", "parse_failures")}
        cost.append({"material": mat, "arm": arm, "cases": n, "responses": n, **c,
                     "total_tokens": c["prompt_tokens"] + c["completion_tokens"],
                     "tokens_per_case": (c["prompt_tokens"] + c["completion_tokens"]) / n,
                     "latency_s_per_case": round(c["latency_s"] / n, 3)})
    for arm in ARMS:
        rows = [r for r in comp if r["arm"] == arm]
        macro = {"material": "MACRO (5 materials, equal weight)", "arm": arm, "cases": sum(r["cases"] for r in rows)}
        for k in rows[0]:
            if k not in macro and k not in ("unresolved_reference_cases",):
                vals = [r[k] for r in rows if r[k] != ""]
                macro[k] = sum(vals) / len(vals)
        comp.append(macro)
        crows = [r for r in cost if r["arm"] == arm]
        cost.append({"material": "ALL", "arm": arm, **{k: sum(r[k] for r in crows) for k in
                     ("cases", "responses", "calls", "prompt_tokens", "completion_tokens", "total_tokens", "latency_s", "parse_failures")}})
    cluster_of = {r["case_id"]: r["dependency_cluster"] for r in coverage if r.get("included_in_aggregate") is True}
    n_clusters = len(set(cluster_of.values()))
    for arm in ARMS:
        by_cluster: dict[str, list] = defaultdict(list)
        for r in records:
            if r["arm"] == arm:
                by_cluster[cluster_of[r["case_id"]]].append(r)

        def cmean(fn):
            return sum(sum(fn(r) for r in rs) / len(rs) for rs in by_cluster.values()) / len(by_cluster)

        row = {"material": f"CLUSTER-WEIGHTED ({n_clusters} evidence groups)", "arm": arm,
               "cases": sum(len(v) for v in by_cluster.values()),
               "verdict_correct": cmean(lambda r: r["grade"]["correct"]),
               "evidence_attribution": cmean(lambda r: r["grade"]["evidence_attribution"]),
               "dft_request_correct": cmean(lambda r: r["grade"]["dft_request_correct"])}
        for lab in RATERS:
            row[f"{lab}_verdict_Y"] = cmean(lambda r: scores[(r["case_id"], arm)][lab]["verdict_correct"] == "Y")
            for fld in FIELDS:
                row[f"{lab}_{fld}"] = cmean(lambda r, f=fld: int(scores[(r["case_id"], arm)][lab][f]))
        comp.append(row)
    for path, rows in (("e2-four-arm-comparison.csv", comp), ("e2-inference-costs.csv", cost)):
        with (OUT / path).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
            w.writeheader()
            w.writerows(rows)

    manifest.update({
        "config": str(CONFIG.relative_to(ROOT)), "config_sha256": sha(CONFIG),
        "consolidation_commit": head, "pinned_evaluator": str(evaluator.relative_to(ROOT)),
        "pinned_evaluator_sha256": sha(evaluator),
        "historical_evaluator_note": "Raw runs record no evaluator revision. Commit 38110e2 hardcodes no_llm to "
            "unresolved/request_dft, contradicting stored lfp-phosphorene-v1 answers; stored no_llm answers are "
            "reproduced by the pinned evaluator (see no_llm_reproduced_by_pinned_evaluator). LLM-arm answers are "
            "historical outputs and are not re-executed.",
        "decoding": {"temperature": 0.0, "random_seeds": [0]},
        "models": {"assessors": ["claude-sonnet-5-5@google", "gemini-2.5-flash@google"],
                   "raters": ["claude-opus-5-5@google", "gemini-4-argon@google"],
                   "rater_independence": "same vendors (Anthropic, Google) as assessor panel; served via MAG/Google"},
        "excluded": [{"material": "Cantor FCC", "reason": "no relaxation reached EDIFFG=-0.02 or produced a static; "
                       "PBS 9040206 still running as of 2026-11-06T02:49Z; random3 killed (exit 143)"}],
    })
    (OUT / "e2-provenance-manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps({s["suite_id"]: s["no_llm_reproduced_by_pinned_evaluator"] for s in manifest["suites"]}))


if __name__ == "__main__":
    main()
