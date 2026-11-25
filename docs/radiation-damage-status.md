# Radiation-damage extension: implementation status

This page separates what is implemented and unit-tested from what is scientifically established.
Nothing here is numerically validated: no production DFT or MLIP result, no validated SPECTRA-PKA
result (one software smoke run only), and no kinetic prediction exists. Unit tests establish software behaviour only.

## Qualification levels

| Level | Meaning |
| --- | --- |
| `IMPLEMENTED_TESTED` | Code exists and passes synthetic unit tests. |
| `INPUT_BLOCKED` | Code exists but required external inputs are missing. |
| `EXTERNAL_UNVERIFIED` | Interface to external software; no real run made. |
| `EXTERNAL_SMOKE_RUN` | External software installed and run once end to end; output not validated. |
| `NOT_ESTABLISHED` | No scientific claim is made. |

## Items

| # | Item | Module | Level | Caveat |
| --- | --- | --- | --- | --- |
| 1 | W to Re/Os substitutional transmutation | `discovery/transmutation.py` | `IMPLEMENTED_TESTED` | Counts are scenario inputs, not predicted compositions. |
| 2 | Time-dependent inventory ingestion | `discovery/transmutation_source.py` | `IMPLEMENTED_TESTED` | Reads external inventories; does not compute them. |
| 3 | `V_n He_m H_p` complexes | `discovery/gas_defects.py` | `IMPLEMENTED_TESTED` | Geometry only; nothing relaxed. |
| 4 | Species, spin and MLIP capability gate | `discovery/species_support.py` | `IMPLEMENTED_TESTED` | Re/Os/H/He are `MLIP_UNSUPPORTED` or undeclared. |
| 5 | Binding and insertion energy contracts | `discovery/gas_energetics.py` | `IMPLEMENTED_TESTED` | No energies computed. |
| 6 | Mechanism registry and NEB preparation | `kinetics/gas_mechanisms.py` | `IMPLEMENTED_TESTED` | No barriers computed. |
| 7 | Cluster-dynamics bridge | `kinetics/cluster_dynamics.py` | `IMPLEMENTED_TESTED` | Bubble physics rejected as future work; bubble kinetics `NOT_ESTABLISHED`. |
| 8 | Transfer-driven gas acquisition | `campaign/gas_transfer.py` | `IMPLEMENTED_TESTED` | Not shown to beat random selection on any real data. |
| 9 | 1,024-site QE campaign | `campaign/gas_qe_campaign.py` | `IMPLEMENTED_TESTED` | Re/Os/H/He PSlibrary 1.0.0 PAW files are now in the Frontier cache, checksummed, with headers inspected (see `SOURCE.txt` there). Cutoffs are not convergence-tested (the Os suggestion of 53.8 Ry exceeds the campaign `ecutwfc`). No QE run with them yet. Own dispatch, not `slurm_jobs`. |
| 10 | External SPECTRA-PKA adapter | `integrations/external/spectra_pka/` | `EXTERNAL_SMOKE_RUN` | Built on Frontier; PKA data archive (`pkas.zip`) cached, extracted per campaign. One real run (W-182, upstream PWR test spectrum) gave `STRUCTURALLY_VALID`, after the `columns=` line and parser layout were fixed. Not validated against a reference. |
| 11 | PKA-informed ranking | `campaign/pka_acquisition.py` | `IMPLEMENTED_TESTED` | Policy only; no per-PKA jobs. |
| 12 | Surviving-defect production | `irradiation/contracts.py` | `NOT_ESTABLISHED` | Default `NoCascadeModel`; a PKA is not a surviving defect. |

## Compute state

* Job 5986052 (128-atom k-point test, meshes 4/6/8, 4 nodes) reported Slurm `COMPLETED`, but all
  three `pw.x` runs ended with exit 139 (segmentation fault) after printing the FFT grid and
  before the first SCF iteration. No energies exist. Each run left a roughly 7 GB core file
  under `runs/qe-kconv-128atom/`. All shared libraries of `pw.x` resolve under the job's module
  stack (`ldd` shows none missing). A `gdb` backtrace of the `k4` core shows the fault in
  `buiol_write_record` (QE's in-memory buffer), called from `save_buffer` in `wfcinit`. The root
  cause is not established; no fix has been tried. The same binary and inputs need a one-node
  reproduction before any change to the sbatch scripts or the QE input.
* Reproduction on one node (128 atoms, 8 ranks, 36 k-points, `disk_io` default): job 5986978
  also exited 139 in `buiol_write_record` under `save_buffer` in `wfcinit`. The same input with
  `disk_io='medium'` (job 5987031) did not crash but made no progress: it printed the first
  SCF iteration header and was cancelled by the 20 minute limit (Slurm `TIMEOUT`), with no
  converged SCF. So `disk_io='medium'` avoids the crash in this test but is not shown to
  work; it is one variable changed, with an unproven outcome.
* Root cause and fix of the `pw.x` segfault (software-verified on one 128-atom debug job; no
  energies, nothing numerically validated). QE was rebuilt from scratch with the pinned
  toolchain `cpe/24.11`, `cce/18.0.1` (Frontier's default cpe/26.03 / cce 21.0.4 does not
  compile QE). A standalone reproducer showed that Cray `ftn` (cce 18, 19, 20) segfaults at run
  time on an array assignment inside an OpenMP `WORKSHARE` (orphaned; a rebuilt `pw.x` with
  `PARALLEL WORKSHARE` still crashed in the outlined region, found with a core dump taken in
  job 5988713). `PW/src/buffers.f90` uses that construct in `buiol_write_record` and
  `buiol_read_record`. Replacing both with an explicit-shape `PARALLEL DO` copy
  (`buiol_copy_z`; patch in `deployments/frontier/setup/patches/qe-cce-orphaned-workshare/`,
  applied by `build-qe-gpu-frontier.sh`) removes the crash: job 5988725 (same input, default
  `disk_io`) passed `wfcinit` and printed SCF iteration 1, then reached the 15 minute debug
  limit (`TIMEOUT`) with no converged SCF. No convergence or energy exists yet for the
  128-atom case; the wavefunction start-up alone takes several minutes.
* Slurm `COMPLETED` is not QE success. Launchers must propagate the `srun` exit code and
  check for `convergence has been achieved` and `JOB DONE`.
* See [multiproperty-selection.md](multiproperty-selection.md) for the property, selection,
  and acquisition workflow. All of it is software-verified only.
* The 2x2x2 k-mesh and the 51/650 Ry cutoffs used for the 1,024-atom input are unconverged.
* Debug QoS is not sufficient for production 1,024-atom QE.
* `pristine_ferromagnetic` stays `INPUT_BLOCKED`; gas examples are nonmagnetic (`nspin: 1`).

## Tests

`tests/test_radiation_e2e_synthetic.py` chains alloy, transmutation, gas complex, mock
SPECTRA-PKA, PKA sampling and ranking, and checks composition, determinism and labels only.
Mock-executable tests are not nuclear-physics verification.
