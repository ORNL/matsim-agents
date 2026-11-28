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
- Everything is software-verified only. Cutoffs (51/650 Ry) and k-points (2x2x2) are unconverged. The measured evidence for better values is in [dft-parameterization.md](../../examples/fusion_bcc_1024/dft-parameterization.md).
- No converged 1,024-atom QE result exists. The debug QoS cannot run that cell (`RESOURCE_BLOCKED`).
- `pristine_ferromagnetic` is `INPUT_BLOCKED` until the starting magnetization is tied to verified UPF valence.
- The 128-atom SCF has not converged yet.
- The SPECTRA-PKA and OpenMC runs are smoke runs, and the OpenMC job phase has not run on a compute node.
- A Slurm `COMPLETED` state is not a QE success. Check `pw.out` for `convergence has been achieved` and `JOB DONE`.

## 1. Setup

Compute nodes have no internet. Run downloads and builds on a login node, from the repository root.

```bash
export PYTHONPATH=$PWD/src XALT_EXECUTABLE_TRACKING=no
export PROJECT_ROOT=$PWD
export MATSIM_PSEUDO_DIR=/lustre/orion/lrn110/proj-shared/cache/pseudopotentials/pslibrary-1.0.0-pbe-paw
```

- Create the environment with `deployments/frontier/setup/install.sh` (the shared checkout already has `.venv`).
  The installer does not activate it. In every new shell, run
  `source deployments/frontier/setup/setup_matsim_frontier.sh --rocm72` (it loads the ROCm 7.2 modules and
  puts `.venv/bin` on `PATH`), or use `.venv/bin/matsim-agents` explicitly. The bare `matsim-agents`
  commands below need one of the two.
- `PROJECT_ROOT` is required for the QE submissions in steps 4 and 5. Both scripts default to the shared
  checkout when it is unset, so without it they would run the shared `pw.x` instead of the one in your checkout.
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

Run this only after step 4 converges and a non-debug QoS, wall time and node-hour budget are approved.
The debug QoS allows 2 h and one job per user, which is not enough for this cell. The script declares
`#SBATCH -t 02:00:00`, so the command line must override the wall time as well as the QoS.

```bash
CALC_DIR=$PWD/runs/campaigns/wtacrvhfmo-1024-first-test/pristine_nonmagnetic \
  sbatch --parsable -A LRN110 -q <approved-qos> -t <approved-walltime> --export=ALL,CALC_DIR \
  deployments/frontier/campaigns/qe-1024/qe_1024_job.sbatch
```

The script refuses a `pw.in` that is not a full-size cell and fails unless `pw.out` has both success
strings. Repeat for `vacancy`, `interstitial`, `frenkel_pair` and `divacancy`.

## 6. Defect migration

Relax the defect endpoints first. The prepared `neb_*` entries require relaxed endpoints.

`matsim-agents neb prepare` cannot use them. It builds its own vacancy hop from the `parent` and
`defect` keys of its config and rejects the interstitial and recombination mechanisms
([`cli/neb.py`](../../src/matsim_agents/cli/neb.py)), so it ignores the prior relaxation and covers
only `neb_vacancy`. For all three paths, build a `matsim_agents.science.neb.endpoints.MigrationPair`
in Python from the relaxed structures with their endpoint provenance and call
`matsim_agents.workflows.neb.workflow.run_neb_workflow`.
Then use `matsim-agents neb validate|inspect|resume` on the resulting workflow directory
(see [Run a workflow](run-campaigns.md)).

A barrier from this chain is at most `unqualified`.

## 7. Radiation inputs

- SPECTRA-PKA: build with `deployments/common/setup/build-spectra-pka.sh`. The TENDL archive is in
  `/lustre/orion/lrn110/proj-shared/cache/nuclear-data/spectra-pka`. The template
  [`campaign.yaml`](../../examples/external/spectra-pka/campaign.yaml) is not runnable as shipped.
  Before `check|prepare|run`:
  - Export `SPECTRA_PKA_EXECUTABLE_PATH` (the build script only prints this export), `PKA_DATA_DIRECTORY`
    (extract the matrices with `external spectra-pka fetch-data`) and `NEUTRON_SPECTRUM_FILE`. The neutron
    spectrum is always supplied by you.
  - Copy the template and replace its `REPLACE-WITH-...` values: `nuclear_data_identity` and `flux.identity`.
    Check `flux.normalization` and `norm_type` against your spectrum. The composition in it is an
    invented W-Ta example, not this alloy.

  Then run `matsim-agents external spectra-pka check|prepare|run -c <your-campaign.yaml>`.
  `run` is a dry run unless `--execute` is given. See [SPECTRA-PKA](../external/spectra-pka.md).
- OpenMC activation: build with `deployments/frontier/setup/build-openmc-frontier.sh`, then run
  `examples/external/openmc_activation.py`. See [OpenMC](../external/openmc.md).

## 8. Higher-level stages

Hierarchical transfer and the radiation-damage chain are dry runs by default. Active learning is not.

- Hierarchical transfer: `matsim_agents.workflows.hierarchy.engine`. Dispatch needs `approved_by` and a node-hour budget.
- Radiation-damage chain: `matsim_agents.workflows.radiation.damage.run_radiation_damage`. Dispatch needs `dispatch=True`.
  Kinetics are blocked without NEB evidence and are never labelled qualified.
- Active learning: `matsim-agents active-learning --help` (alias `al`). `active-learning run` calls
  `run_active_learning` with no dry-run gate. `deployments/frontier/launchers/run-active-learning-frontier.sh`
  requests 64 nodes for 12 h and runs DFT and training. It is live execution and needs approval.
  - Generate your own `al.yaml` with `active-learning init-campaign`. `run` refuses a config under
    `examples/` unless `--allow-template` or `MATSIM_ALLOW_TEMPLATE_CONFIG=1` is set, and it stops if the
    training references miss an element in the seeds (see [Active learning](../guide/active-learning.md)).
  - `loop.resume` defaults to true. A restart continues from the last iteration and reuses finished DFT
    labels instead of recomputing them.

## 9. Check results

- The QE jobs in steps 3 to 5 are not recorded runs. Use the saved Slurm job IDs, the scheduler state
  (`sacct -j <jobid>`), `campaign_manifest.json` and `pw.out`. Record the git revision and `input_sha256`
  from the manifest.
- `matsim-agents status` and `matsim-agents evidence RUN_ID` read `run-record.json`, which
  `matsim-agents run`, `resume` and `objective` write. Use them for those runs only (see
  [Run a workflow](run-campaigns.md) for the `run` exit codes).
- Treat nothing as a scientific result until cutoff and k-point convergence are shown.
  See [Interpret results](interpret-results.md).

## Rules

- Do not commit `*.err` files or run outputs.
- Scripts under `deployments/` must not contain `#SBATCH -A`, absolute `-o/-e` paths or `/lustre/orion/` literals.
- Do not submit anything larger than the 128-atom test without approval.
