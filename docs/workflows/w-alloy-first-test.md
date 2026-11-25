# First end-to-end test of the W-alloy campaign on Frontier

Target: BCC W640Ta128Cr64V96Hf32Mo64, 8x8x8 conventional cells, 1,024 sites, a = 3.16 A.
Application: [`examples/fusion_bcc_1024/application.yaml`](../../examples/fusion_bcc_1024/application.yaml).
Workflow config: [`campaign.yaml`](../../examples/fusion_bcc_1024/campaign.yaml).

Steps 1 to 3 were run on a Frontier login node. Steps 4 to 9 follow the repository scripts and
docs and have not been run end to end.

## 0. What this test can and cannot show

- No single command runs the whole campaign. `matsim-agents run examples/fusion_bcc_1024/application.yaml`
  is a dry check: it reports `complete` but writes only empty `structures/`, `calculations/`,
  `datasets/` and `models/` directories.
- Everything is software-verified only. Cutoffs (51/650 Ry) and k-points (2x2x2) are unconverged.
- No converged 1,024-atom QE result exists. The debug QoS cannot run that cell (`RESOURCE_BLOCKED`).
- `pristine_ferromagnetic` is `INPUT_BLOCKED` until the starting magnetization is tied to verified UPF valence.
- The 128-atom SCF has not converged yet.
- The SPECTRA-PKA and OpenMC runs are smoke runs, and the OpenMC job phase has not run on a compute node.
- A Slurm `COMPLETED` state is not a QE success. Check `pw.out` for `convergence has been achieved` and `JOB DONE`.

## 1. Setup

Compute nodes have no internet. Run downloads and builds on a login node, from the repository root.

```bash
export PYTHONPATH=$PWD/src XALT_EXECUTABLE_TRACKING=no
export MATSIM_PSEUDO_DIR=/lustre/orion/lrn110/proj-shared/cache/pseudopotentials/pslibrary-1.0.0-pbe-paw
```

- Create the environment with `deployments/frontier/setup/install.sh` (the shared checkout already has `.venv`).
- `XALT_EXECUTABLE_TRACKING=no` avoids an exit-time segfault after importing torch.
- `MATSIM_PSEUDO_DIR` is required: `campaign.yaml` reads it for `qe.pseudo_dir`.
- `matsim-agents capabilities check` may show `dft qe  missing (env MATSIM_QE_LAUNCHER)`. The steps below submit QE with `sbatch`, so this does not block them.

## 2. Offline checks

```bash
.venv/bin/matsim-agents validate examples/fusion_bcc_1024/application.yaml
.venv/bin/matsim-agents plan     examples/fusion_bcc_1024/application.yaml
.venv/bin/python -m pytest tests/test_applications.py tests/test_full_supercell_campaign.py \
  tests/test_radiation_e2e_synthetic.py -q -p no:cacheprovider
```

`validate` builds the 1,024-site parent and checks composition, repeats and lattice (exit 0, "nothing
blocks `run`"). The synthetic test runs alloy, transmutation, gas complex, PKA and ranking on mock
inputs; it establishes no physical result.

## 3. Generate the 1,024-atom QE inputs

```python
from pathlib import Path
from matsim_agents.workflows.full_supercell_campaign import FullSupercellCampaign, prepare_campaign

cfg = FullSupercellCampaign.from_yaml("examples/fusion_bcc_1024/campaign.yaml")
entries = prepare_campaign(
    cfg, Path("runs/campaigns/wtacrvhfmo-1024-first-test"), git_revision="<git rev-parse HEAD>"
)
for e in entries:
    print(e["calc_id"], e["kind"], e["qe_atoms"], e["status"])
```

Expected result (nothing is submitted):

| Calculation | Atoms | Status |
| --- | --- | --- |
| `pristine_nonmagnetic` | 1024 | PREPARED |
| `pristine_ferromagnetic` | 1024 | INPUT_BLOCKED |
| `vacancy` | 1023 | PREPARED |
| `interstitial` | 1025 | PREPARED |
| `frenkel_pair` | 1024 | PREPARED |
| `divacancy` | 1022 | PREPARED |
| `neb_vacancy` | 1023 | PREPARED |
| `neb_interstitial` | 1025 | PREPARED |
| `neb_recombination` | 1024 | PREPARED |

Each directory holds `pw.in` (or the NEB endpoints) and `structure.extxyz`; `campaign_manifest.json`
records `input_sha256`. `INPUT_BLOCKED: missing pseudopotentials` means `MATSIM_PSEUDO_DIR` is wrong.

## 4. First real DFT test: 128 atoms

1. Check `external/quantum-espresso/install-gpu/bin/pw.x`. If it is missing, build it with
   `deployments/frontier/setup/build-qe-gpu-frontier.sh` (see [QE on Frontier](../quantum-espresso-frontier.md)).
2. Prepare the 4x4x4 test cell. Its meshes match the k-spacing of the full cell at 2x2x2, 3x3x3 and 4x4x4:

   ```bash
   .venv/bin/python scripts/qe_kpoint_convergence.py prepare \
     --campaign examples/fusion_bcc_1024/campaign.yaml --out runs/campaigns/wtacrvhfmo-kconv
   ```

3. Submit. The account is given on the command line, never with `#SBATCH -A`:

   ```bash
   sbatch --parsable -A LRN110 -q debug --export=ALL,KCONV_DIR=$PWD/runs/campaigns/wtacrvhfmo-kconv \
     deployments/frontier/campaigns/qe-1024/qe_kconv_job.sbatch
   ```

4. Analyze energy, forces and pressure against the densest mesh:

   ```bash
   .venv/bin/python scripts/qe_kpoint_convergence.py analyze --out runs/campaigns/wtacrvhfmo-kconv
   ```

Pass criterion for this step: each `pw.out` has `convergence has been achieved` and `JOB DONE`.
Record wall time and any failure as found.

## 5. Full 1,024-atom SCF

Run this only after step 4 converges and a non-debug QoS and node-hour budget are approved. The debug
QoS allows 2 h and one job per user, which is not enough for this cell.

```bash
CALC_DIR=$PWD/runs/campaigns/wtacrvhfmo-1024-first-test/pristine_nonmagnetic \
  sbatch --parsable -A LRN110 -q <approved-qos> --export=ALL,CALC_DIR \
  deployments/frontier/campaigns/qe-1024/qe_1024_job.sbatch
```

The script refuses a `pw.in` that is not a full-size cell and fails unless `pw.out` has both success
strings. Repeat for `vacancy`, `interstitial`, `frenkel_pair` and `divacancy`.

## 6. Defect migration

Relax the defect endpoints first. Then use `matsim-agents neb prepare|validate|inspect|resume`
(see [Run a workflow](run-campaigns.md)). The prepared `neb_*` entries require relaxed endpoints.
A barrier from this chain is at most `unqualified`.

## 7. Radiation inputs

- SPECTRA-PKA: build with `deployments/common/setup/build-spectra-pka.sh`. The TENDL data is in
  `/lustre/orion/lrn110/proj-shared/cache/nuclear-data/spectra-pka`. Use
  `matsim-agents external spectra-pka check|prepare|run -c examples/external/spectra-pka/campaign.yaml`.
  `run` is a dry run unless `--execute` is given. See [SPECTRA-PKA](../external/spectra-pka.md).
- OpenMC activation: build with `deployments/frontier/setup/build-openmc-frontier.sh`, then run
  `examples/external/openmc_activation.py`. See [OpenMC](../external/openmc.md).

## 8. Higher-level stages

All are dry runs by default.

- Active learning: `matsim-agents active-learning --help` and
  `deployments/frontier/launchers/run-active-learning-frontier.sh`.
- Hierarchical transfer: `matsim_agents.workflows.hierarchy.engine`. Dispatch needs `approved_by` and a node-hour budget.
- Radiation-damage chain: `matsim_agents.workflows.radiation_damage.run_radiation_damage`. Dispatch needs `dispatch=True`.
  Kinetics are blocked without NEB evidence and are never labelled qualified.

## 9. Check results

- `matsim-agents status` and `matsim-agents evidence RUN_ID`. Exit codes: 0 finished, 1 failed, 2 invalid, 3 blocked.
- Read `pw.out` directly. Record the git revision and `input_sha256` from the manifest.
- Treat nothing as a scientific result until cutoff and k-point convergence are shown.
  See [Interpret results](interpret-results.md).

## Rules

- Do not commit `*.err` files or run outputs.
- Scripts under `deployments/` must not contain `#SBATCH -A`, absolute `-o/-e` paths or `/lustre/orion/` literals.
- Do not submit anything larger than the 128-atom test without approval.
