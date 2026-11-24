# Hierarchical transfer campaign (W-Ta-Cr-V-Hf-Mo)

Software for planning a cost-aware, evidence-gated DFT campaign that reuses lower-component data
to support the six-component target **W640Ta128Cr64V96Hf32Mo64** (8x8x8 BCC parent, 1024 sites;
reference: Lupo Pasini et al., Comput. Mater. Sci. 257 (2025) 113908).

> Status: software and synthetic-data tests only. No Quantum ESPRESSO job has been submitted by
> this campaign and nothing is numerically validated. Real-alloy transferability is
> `NOT_YET_ESTABLISHED`.

## Components (`src/matsim_agents/campaign/`)

| Step | Module | Role |
| --- | --- | --- |
| H1 | `registry.py` | Hierarchy subspaces, dataset ingestion, structure and calculation de-duplication, provenance |
| H2 | `coverage.py` | Interaction coverage, explicit missing interactions, coverage-gain ranking |
| H3 | `transfer.py`, `synthetic.py` | Held-out transfer experiments (2->3 ... 2,3,4,5->6), leakage checks, per-property verdicts |
| H4 | `acquisition.py` | Cost-aware candidate selection; falls back to coverage + uncertainty without transfer evidence |
| H5 | `stages.py` | Stages 0-5 with evidence gates and a next-action decision |
| H6 | `qe_plan.py` | Full-size (1024 / 1023 / 1025 / 1022 atom) records, method compatibility, `READY` / `RESOURCE_BLOCKED` / `BUDGET_BLOCKED` / `INPUT_BLOCKED` |
| H7 | `budget.py` | Hard node-hour ledger with a 15% validation reserve |
| H8 | `strategies.py` | Strategies A-E compared per node-hour on identical held-out rows |
| H9 | `hierarchy_config.py`, `hierarchy_campaign.py`, `cli/campaign.py` | Validated config, resumable planner, CLI, report |

Cells are never shrunk to fit a queue. A job that does not fit is `RESOURCE_BLOCKED` and its
full-size input is kept. Historical smaller cells are prior evidence only.

## Configuration

Every section uses `extra="forbid"`. Sections: `campaign`, `target`, `data`,
`chemical_hierarchy`, `sampling`, `transfer_evaluation`, `qe`, `budget`, `execution`.

`execution.dry_run` defaults to `true`. Setting it to `false` requires `approved_by` and a bounded
`approved_node_hours` that does not exceed `budget.total_node_hours`. `prevent_duplicate_jobs` can
only be `true`. `max_jobs_per_dispatch` bounds each `approve` call.

```yaml
campaign: {campaign_id: wtacrvhfmo-hierarchy}
data:
  datasets: [{name: lower-bcc, path: lower.json, method: {...}}]
  transfer_data: {energy: energy.npz, magnetic_moment: moments.npz}
  candidates: candidates.json
  reserved_validation_ids: [val-001]
qe: {method: {...}, resources: {nodes: 64, walltime_hours: 12, memory_gb_per_node: 400},
     limits: {max_nodes: 9408, max_walltime_hours: 2, node_memory_gb: 512}, input_ready: false}
budget: {total_node_hours: 5000}
execution: {dry_run: true}
```

## CLI

```
matsim-agents campaign check           CONFIG
matsim-agents campaign inventory       CONFIG -w DIR   # ingest datasets, check method compatibility
matsim-agents campaign coverage        CONFIG -w DIR   # covered and missing interactions
matsim-agents campaign transferability CONFIG -w DIR   # held-out verdicts per property
matsim-agents campaign preview         CONFIG -w DIR [--acquisition policy|deterministic_baseline]
matsim-agents campaign cost            CONFIG -w DIR
matsim-agents campaign approve         CONFIG -w DIR --approved-by NAME   # bounded dispatch
matsim-agents campaign monitor         CONFIG -w DIR   # charge actual node-hours
matsim-agents campaign review          CONFIG -w DIR   # report table and next decision
```

All commands except `approve` and `monitor` never contact the scheduler. `approve` refuses under
`dry_run` or when `--approved-by` differs from the config. It submits with
`sbatch --parsable -q <qos> -A <account>`, no shell parsing.

## Restart safety

State (`state.json`) and ledger (`ledger.json`) are written atomically. A submission is recorded
before the scheduler is contacted. If a run dies in between, the record is marked
`needs_manual_check` on resume and is never resubmitted automatically. A calculation ID (structure
hash + backend + settings + magnetic initialisation) with any ledger entry is never submitted
again; reruns need an authorised `BudgetLedger.retry`.

## Evidence rules

- A dataset whose functional, pseudopotentials or magnetic convention differ from the target
  method is excluded from coverage.
- Magnetic and defect transferability are `pending` until labelled data for that property exists.
  An energy verdict is never reused for them.
- Any negative verdict makes a property `unsupported`.
- Coverage without explicit interactions is composition-level and is labelled as such.
- Reserved validation examples are never selected; held-out rows never enter training.

## Tests

`tests/test_campaign_h9.py` holds the 10-item synthetic campaign: transferable lower data, useless
lower data, a bridge that supplies a missing interaction, the policy finding the useful data, no
repeated redundant picks, hard budget, reserved examples never chosen, resume without duplicate
submission, policy beating the deterministic baseline, and detection of failed transfer. They show
the software behaves correctly on synthetic data, not that real alloys transfer.

## Open blockers

- Pseudopotentials (pslibrary 1.0.0 PBE PAW, six elements, checksums and provenance beside the files
  under `/lustre/orion/lrn110/proj-shared/cache/pseudopotentials/pslibrary-1.0.0-pbe-paw`) are on disk and
  wired into `deployments/frontier/campaigns/qe-1024/campaign.yaml`. Five full-size SCF inputs and three
  NEB records now prepare; `pristine_ferromagnetic` stays `INPUT_BLOCKED` until a starting magnetization
  is derived from the UPF valence.
- The default cutoffs (45 / 360 Ry) are below the UPF-header suggestions (up to 50.4 Ry for W, 645 Ry
  density for V). No cutoff or k-point convergence test has been run, so results would not be validated.
- The debug QoS (2 h, one job) cannot run 1024-atom QE, so such jobs are `RESOURCE_BLOCKED` until a
  queue and node-hour budget are authorised.
