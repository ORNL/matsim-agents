# Radiation-damage implementation status

Current-state record of the radiation-damage workflow code. Design is in
[radiation-damage-phase-b-spec.md](radiation-damage-phase-b-spec.md) and
[radiation-damage-phases-c-e-spec.md](radiation-damage-phases-c-e-spec.md). Magnetic ordering is tracked in
[magnetic-ordering-implementation-spec.md](magnetic-ordering-implementation-spec.md) and
[magnetic-scalable-exploration.md](magnetic-scalable-exploration.md).

## Status

- Software: the modules below exist, import, and have unit tests.
- Scientific qualification: none. `workflows/radiation/capabilities.py` reports every capability as
  `NOT_QUALIFIED`, and only the qualification matrix (`workflows/qualification_matrix.py`) can change that, from
  measured, hash-checked evidence.
- Test data is synthetic. The tests do not run a real `neb.x`, DFT, MLIP or HPC job.
- The `radiation-damage` CLI command was removed. The workflow is driven from Python or by `matsim-agents run`.
- Test counts are not recorded here. The radiation-related test files were started in this review but the run was
  stopped before it finished, so no pass count is claimed.

## Capability areas

Paths are under `src/matsim_agents/`.

| Area | Modules |
|---|---|
| Alloy generation, short-range order | `science/structures/alloys.py`, `science/structures/alloy_ensemble.py` |
| Defect generation and analysis | `science/defects/` (`point_defects`, `classification`, `mapping`, `environments`, `energies`, `wigner_seitz`, `tracking`, `survival`, `gas`, `cascade_injection`) |
| NEB contracts, endpoints, paths, images | `science/neb/` (`contracts`, `endpoints`, `paths`, `images`, `validation`, `serialization`), `workflows/neb/` (`workflow`, `mapping`, `convergence`) |
| Quantum ESPRESSO NEB and magnetic input | `backends/dft/qe_neb*.py`, `backends/dft/qe_magnetic.py` |
| Kinetics | `science/kinetics/` (`graph`, `transitions`, `rates`, `harmonic_tst`, `kmc`, `ensemble`, `uncertainty`, `benchmarks`, `event_library`, `rate_construction`, `gas_mechanisms`, `cluster_dynamics`) |
| Cascade | `science/cascade/` (`zbl`, `recoil`, `dynamics`, `engines`, `events`, `protocol`, `boundaries`, `damage_functions`, `qualification`) |
| Radiation workflow | `workflows/radiation/` (`capabilities`, `damage`, `hypothesis`, `properties`, `materials`, `report`, `final_report`, `cascade_*`, `gas_*`, `transmutation_*`) |
| Qualification, validation, scalability | `workflows/qualification_matrix.py`, `workflows/mlip_dft_validation.py`, `workflows/scalability.py` |
| Slurm jobs | `execution/slurm_jobs.py` |

## Examples

- `examples/radiation_damage/neb_vacancy_hop.example.yaml`
- `examples/fusion_damage/` (W reference alloy)
- `examples/fusion_pfc/`

## Open items

- Record measured evidence in the qualification matrix. Until then every row stays at `not_assessed`.
- Run NEB end to end with real `neb.x` and an MLIP pre-screen on HPC. Only parsers and input writers are tested.
- Scalability numbers cover a synthetic KMC chain, not DFT, MLIP or large-cell cost. No lock file pins the environment.
- Run the full test suite and CI for the current head. This file's status comes from reading the code, not from a
  fresh full run.
