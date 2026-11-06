# Blinded scientific assessment: e2-nbtao-v2

You are scoring 4 anonymized answers to 1 evidence-assessment questions.
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
