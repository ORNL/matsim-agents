#!/usr/bin/env python3
"""One-off blinded LLM rating of an E2 blinded package via MAG (OpenAI-compatible).

Sends, per case: the rubric, the exact packet, and that case's anonymized items.
Writes raw responses, applied-gateway headers, and a filled scores CSV.
"""

import csv
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

PKG = Path(sys.argv[1])
MODEL = sys.argv[2]
OUT = PKG / f"rater-llm-{MODEL.replace('@', '_')}"
OUT.mkdir(exist_ok=True)
URL = os.environ.get("MATSIM_MAG_BASE_URL", "https://mag.genesis.americansciencecloud.org/v1") + "/chat/completions"
rubric = (PKG / "README.md").read_text()
items = [json.loads(line) for line in open(PKG / "items.jsonl")]
FIELDS = ["verdict_correct", "physical_reasoning", "evidence_attribution", "uncertainty", "follow_up", "notes"]
INSTR = (
    "Score every item below using the rubric. Judge the science from the packet yourself; there is no answer key. "
    "Return only a JSON object: {\"scores\": [{\"item_id\": ..., \"verdict_correct\": \"Y\"|\"N\", "
    "\"physical_reasoning\": 0-2, \"evidence_attribution\": 0-2, \"uncertainty\": 0-2, \"follow_up\": 0-2, "
    "\"notes\": \"...\"}], \"correct_verdict_in_your_judgement\": \"supported|contradicted|unresolved\", "
    "\"case_notes\": \"...\"}"
)

scores = {}
for case in dict.fromkeys(i["case"] for i in items):
    raw_path = OUT / f"{case}.response.json"
    if not raw_path.exists():
        case_items = [i for i in items if i["case"] == case]
        content = (f"{rubric}\n\n## Packet\n{(PKG / 'packets' / f'{case}.md').read_text()}\n\n## Items\n"
                   + "\n".join(json.dumps(i) for i in case_items) + f"\n\n{INSTR}")
        body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": content}],
                           "response_format": {"type": "json_object"}}).encode()
        req = urllib.request.Request(URL, data=body, headers={
            "Authorization": f"Bearer {os.environ['AMSC_TOKEN']}", "Content-Type": "application/json"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=900) as resp:
            payload = json.load(resp)
            applied = resp.headers.get("X-AmSC-MAG-Applied")
        raw_path.write_text(json.dumps({"request_sha256_inputs": case, "latency_s": round(time.time() - t0, 1),
                                        "mag_applied": applied, "response": payload}, indent=1))
    data = json.loads(raw_path.read_text())["response"]
    content = data["choices"][0]["message"]["content"].strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1].rsplit("```", 1)[0]
    parsed = json.loads(content)
    for s in parsed["scores"]:
        scores[s["item_id"]] = s
    print(case, "->", parsed.get("correct_verdict_in_your_judgement"), flush=True)

with open(OUT / "scores.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["item_id", "case"] + FIELDS)
    for i in items:
        s = scores.get(i["item_id"], {})
        w.writerow([i["item_id"], i["case"]] + [s.get(k, "") for k in FIELDS])
(OUT / "rater.json").write_text(json.dumps({
    "rater_id": MODEL, "type": "llm", "model_version": MODEL, "provider": "MAG",
    "vendor_independent": False,
    "independence": "Not vendor-independent; rater models share model vendors with the assessor panel.",
    "saw_unblinded_results": False, "date": time.strftime("%Y-%m-%d")}, indent=2) + "\n")
print(f"scored {len(scores)}/{len(items)} items -> {OUT}")
