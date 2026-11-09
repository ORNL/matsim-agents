# E3 Round 1 BCC acquisition batch (`E3-round1-bcc-01`)

Three DFT calculations for the frozen E3 Round 1 selections
([`round1-bcc-manifest.json`](../../../../research/paper/e3/round1-bcc-manifest.json)).
They use the existing E4 BCC runner and wrapper
([`../e4-first-batch/job-bcc.pbs`](../e4-first-batch/job-bcc.pbs)) with the frozen
E4 production protocol `bcc-pbe-low-normal-v1-approved-ediffg-0.01`. `START` and
`KPOINTS` are byte-identical to the E4 BCC production inputs.

| E3 calculation | Pool structure | Composition | Requested by | Logical attempts |
|---|---|---|---|---:|
| E3-R1-BCC-1077 | E4-BCC-1077 | Hf10 Ta46 Ti32 Zr40 | LLM-guided adaptation, Rule-based adaptation | 2 |
| E3-R1-BCC-0394 | E4-BCC-0394 | Hf40 Nb46 Ta10 Ti32 | Rule-based adaptation | 1 |
| E3-R1-BCC-0582 | E4-BCC-0582 | Hf20 Nb37 Ti36 V35 | LLM-guided adaptation | 1 |

Each job runs on one node, in the `capacity` queue, under the HydraGNN
allocation, for at most 24 h. The cap is **72 node-hours**. The distinct
`E3-R1-` IDs prevent collisions with E4 scaling jobs. The structures are
regenerated from the frozen E3 pool, staged with
`benchmarks/e3/stage_round1.py`, and checked against the decision structure
hashes. Private inputs, including POTCARs, live in
`../../../../../e3-round1-bcc-01/inputs/` outside the repository. The batch
manifest stores repository-relative paths, including the sibling input
directory, for resolution by the submission helper.

```bash
# Validate (default; no qstat/qsub).
python3 deployments/aurora/jobs/e4-first-batch/submit_batch.py \
  --manifest deployments/aurora/jobs/e3-round1-bcc/batch-manifest.json
# After authorization: set "execution_authorized": true, then add --submit.
```

**Accounting:** each strategy is charged two logical attempts once `qsub`
accepts the jobs. E3-R1-BCC-1077 runs once and is referenced by both
strategies. Restarts keep the calculation ID and consume no new selection slot.
Their node-hours are recorded by the runner under the same ID. Accepted job IDs
are appended to `submissions.jsonl` in this directory.
