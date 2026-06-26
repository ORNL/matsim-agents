# Paper Run Inventory (2026-06-19)

## Scope
This inventory summarizes what has already been executed for the manuscript and what still needs to be run to complete the reported benchmark set.

Data sources:
- docs/paper/results/paper_results_master.json
- docs/paper/main.tex
- Per-run log files under /global/cfs/projectdirs/amsc001/cm2us/mlupopa/runs

Local mirror (copied on 2026-06-19):
- /global/cfs/projectdirs/m5216/mlupopa/matsim-agents/runs/imported_from_amsc001
- Mirrored directories:
  - active-learning-uq-54352943
  - qe-warmstart-54511140
  - qe-warmstart-54654938
  - singlepass-paper-54683468

## Completed Runs

### 1) UQ-Gated Relaxation Benchmark
- Status: Complete
- Job: 54352943
- Run dir: /global/cfs/projectdirs/amsc001/cm2us/mlupopa/runs/active-learning-uq-54352943
- Output summary: /global/cfs/projectdirs/amsc001/cm2us/mlupopa/runs/active-learning-uq-54352943/outputs/uq_summary.csv
- Structures completed:
  - Si.vasp
  - MgO.vasp
  - MoNbTaW_HEA.vasp
- Result state:
  - 3/3 converged
  - 0/3 flagged unreliable

### 2) QE Warm-Start Benchmark (Si)
- Status: Complete
- Job: 54511140
- Run dir: /global/cfs/projectdirs/amsc001/cm2us/mlupopa/runs/qe-warmstart-54511140
- Comparison JSON:
  - /global/cfs/projectdirs/amsc001/cm2us/mlupopa/runs/qe-warmstart-54511140/pytest-tmp/test_hydragnn_warmstart_helps_0/Si_diamond/comparison.json
- Result state:
  - Passed
  - Cold and warm both converged
  - BFGS steps: 4 vs 4
  - Total SCF: 32 vs 32

### 3) Single-Pass Paper Sweep (MLIP-only)
- Status: Complete for configured set
- Job: 54683468
- Run dir: /global/cfs/projectdirs/amsc001/cm2us/mlupopa/runs/singlepass-paper-54683468
- Log: /global/cfs/projectdirs/amsc001/cm2us/mlupopa/runs/singlepass-paper-54683468/job-54683468.out
- Cases completed with OK status:
  - lifepo4
  - hea_bcc
  - hea_fcc
  - phosphorene
  - zn_formate
- Note:
  - cu_bht not included in this completed run set (requires supplied CIF).

## Incomplete / Needs Re-Run

### 4) QE Warm-Start Benchmark (MoNbTaW HEA)
- Status: Failed, must be re-run
- Job: 54654938
- Run dir: /global/cfs/projectdirs/amsc001/cm2us/mlupopa/runs/qe-warmstart-54654938
- Failure cause:
  - Missing pseudopotential for Nb in MATSIM_QE_PSEUDO_DIR.
- Error signature in job log:
  - FileNotFoundError: No pseudopotential file found for 'Nb' ... looked for Nb.*.UPF
- Current implication:
  - No comparison.json produced for HEA warm-start.
  - Manuscript HEA warm-start subsection remains provisional.

## Required Inputs Before Next Submission

1. Complete QE pseudopotential set for HEA elements in MATSIM_QE_PSEUDO_DIR:
- Mo
- Nb
- Ta
- W

2. Use m5216 allocation and premium queue at submit time (script defaults are currently amsc001 and regular).

## Run Plan (m5216 + premium)

### A) Re-run HEA warm-start benchmark (required)
Command:

sbatch -A m5216_g -q premium \
  --export=ALL,PROJECT_ROOT=/global/cfs/projectdirs/m5216/mlupopa/matsim-agents,RUNS_ROOT=/global/cfs/projectdirs/m5216/mlupopa/runs,MATSIM_WARMSTART_FIXTURES=MoNbTaW_HEA,MATSIM_QE_PSEUDO_DIR=/global/cfs/projectdirs/m5216/mlupopa/matsim-agents/external/quantum-espresso/src/pseudo \
  scripts/advanced/perlmutter/job-qe-warmstart-perlmutter.sh

### B) Optional reproducibility reruns (recommended)
1) UQ benchmark rerun on m5216 premium:

sbatch -A m5216_g -q premium \
  --export=ALL,PROJECT_ROOT=/global/cfs/projectdirs/m5216/mlupopa/matsim-agents,RUNS_ROOT=/global/cfs/projectdirs/m5216/mlupopa/runs \
  scripts/advanced/perlmutter/job-active-learning-uq-perlmutter.sh

2) Si warm-start rerun on m5216 premium:

sbatch -A m5216_g -q premium \
  --export=ALL,PROJECT_ROOT=/global/cfs/projectdirs/m5216/mlupopa/matsim-agents,RUNS_ROOT=/global/cfs/projectdirs/m5216/mlupopa/runs,MATSIM_WARMSTART_FIXTURES=Si_diamond,MATSIM_QE_PSEUDO_DIR=/global/cfs/projectdirs/m5216/mlupopa/matsim-agents/external/quantum-espresso/src/pseudo \
  scripts/advanced/perlmutter/job-qe-warmstart-perlmutter.sh

## Post-Run Update Steps

After any new run completes:

1) Rebuild consolidated results:
python scripts/diagnostics/collect_paper_results.py

2) Regenerate LaTeX fragments:
python scripts/diagnostics/render_paper_tables.py

3) Update manuscript narrative text where HEA section currently says results are pending.
