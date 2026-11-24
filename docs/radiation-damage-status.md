# Radiation-damage extension: implementation status

This page separates what is implemented and unit-tested from what is scientifically established.
Nothing here is numerically validated: no production DFT or MLIP result, no real SPECTRA-PKA run,
and no kinetic prediction exists. Unit tests establish software behaviour only.

## Qualification levels

| Level | Meaning |
| --- | --- |
| `IMPLEMENTED_TESTED` | Code exists and passes synthetic unit tests. |
| `INPUT_BLOCKED` | Code exists but required external inputs are missing. |
| `EXTERNAL_UNVERIFIED` | Interface to external software; no real run made. |
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
| 9 | 1,024-site QE campaign | `workflows/gas_qe_campaign.py` | `INPUT_BLOCKED` | Re/Os/H/He pseudopotentials absent from the Frontier cache; own dispatch, not `slurm_jobs`. |
| 10 | External SPECTRA-PKA adapter | `integrations/external/spectra_pka/` | `EXTERNAL_UNVERIFIED` | Not installed here; layout and `columns=` block unverified; index file name unresolved. |
| 11 | PKA-informed ranking | `campaign/pka_acquisition.py` | `IMPLEMENTED_TESTED` | Policy only; no per-PKA jobs. |
| 12 | Surviving-defect production | `irradiation/contracts.py` | `NOT_ESTABLISHED` | Default `NoCascadeModel`; a PKA is not a surviving defect. |

## Compute state

* Job 5986052 (128-atom k-point test, meshes 4/6/8, 4 nodes) reported Slurm `COMPLETED`, but all
  three `pw.x` runs ended with exit 139 (segmentation fault) after printing the FFT grid and
  before the first SCF iteration. No energies exist. The cause is undiagnosed. Each run left a
  roughly 7 GB core file under `runs/qe-kconv-128atom/`.
* The 2x2x2 k-mesh and the 51/650 Ry cutoffs used for the 1,024-atom input are unconverged.
* Debug QoS is not sufficient for production 1,024-atom QE.
* `pristine_ferromagnetic` stays `INPUT_BLOCKED`; gas examples are nonmagnetic (`nspin: 1`).

## Tests

`tests/test_radiation_e2e_synthetic.py` chains alloy, transmutation, gas complex, mock
SPECTRA-PKA, PKA sampling and ranking, and checks composition, determinism and labels only.
Mock-executable tests are not nuclear-physics verification.
