# Architectural audit (Phase 1)

Baseline: `main` at `fc70ca9` (after #79). Branch: `refactor/framework-architecture`.

This audit is the Phase 1 deliverable of the repository-wide restructuring. It
records what exists today, what is application-specific, what is duplicated,
and what must be removed, so that implementation work can proceed against an
explicit inventory. The target design is in
[target-architecture.md](target-architecture.md).

Backward compatibility is not a goal. Deprecated aliases, compatibility
wrappers, and campaign-named entry points are removal targets, not
preservation targets.

## 1. Size and shape

`src/matsim_agents` contains 282 Python files and about 80.5k lines.

| Package | Files | Lines | Character today |
| --- | ---: | ---: | --- |
| `campaign/` | 35 | 17.0k | Mix of generic acquisition/UQ/thermodynamics and concrete HEA, gas and radiation campaigns |
| `discovery/` | 47 | 12.1k | Catch-all: composition, alloys, defects, NEB, magnetism, exploration, transmutation |
| `backends/` | 44 | 10.5k | LLM, DFT (QE/VASP), MLIP, MCP adapters; mostly clean |
| `workflows/` | 27 | 8.6k | Generic workflows mixed with radiation/E1/warm-start campaign modules |
| `active_learning/` | 15 | 7.2k | Loop, UQ, DFT protocol, trainer, campaign scaffold |
| `execution/` | 23 | 5.2k | Contracts, run store, Slurm, IRI, facility presets, property families |
| `cli/` | 28 | 4.3k | Typer + argparse, many campaign-named commands |
| `training/` | 8 | 3.4k | MLIP fine-tuning dispatchers |
| `fusion/` | 10 | 3.2k | A single application (fusion PFC) living in production code |
| `kinetics/` | 13 | 2.5k | Mostly generic; H/He/V-specific gas mechanisms |
| `orchestration/` | 10 | 2.1k | Several competing LLM orchestrators |
| `integrations/` | 11 | 2.0k | External codes (SPECTRA-PKA, etc.) |
| `datasets/` | 6 | 1.9k | Training data assembly |
| `irradiation/` | 3 | 0.4k | Damage descriptors |
| `seeding.py` | 1 | – | RNG seeding |

There are no two-way package import cycles today, and
`tests/test_package_boundaries.py` enforces a linear layer order. That order,
however, encodes applications as layers (`campaign` → `fusion` → `cli`), so
the enforced architecture itself is campaign-shaped.

## 2. Application-specific code in production packages

### 2.1 Whole modules that are applications

| Module | Application content | Destination |
| --- | --- | --- |
| `fusion/*` (10 files) | Fusion PFC scoring, acquisition, revision, transmutation, nuclear evidence | `examples/fusion_pfc/` (application config + thin glue); generic parts merged into `science/` |
| `campaign/hierarchy_campaign.py`, `campaign/hierarchy_config.py` | `TARGET_SITES=1024`, composition W640Ta128Cr64V96Hf32Mo64, supercell (8,8,8), `account="LRN110"`, `qe_1024_job.sbatch` | `examples/fusion_bcc_1024/` config; engine becomes generic `workflows/hierarchy` |
| `campaign/gas_qe_campaign.py`, `campaign/gas_transfer.py` | He/H/Re/Os configurations | `examples/gas_transmutation/` |
| `campaign/radiation_properties.py` | `REFERENCE_*` constants, `/lustre/orion/lrn110` path | `examples/radiation_damage/` |
| `campaign/stages.py`, `strategies.py`, `synthetic.py` | Hierarchy-specific stage names and synthetic fixtures | examples / test fixtures |
| `workflows/radiation_*` (6), `workflows/full_supercell_campaign.py` | W interstitial, QE settings, `account="LRN110"` | `examples/radiation_damage/` |
| `workflows/e1.py` | Paper experiment E1 | `research/` |
| `workflows/scalability.py`, `workflows/warmstart_qe.py`, `workflows/warmstart_vasp.py` | Benchmarks | `benchmarks/` (warm-start merged into one) |
| `discovery/exploration/demo.py` | Demo | `examples/` |
| `campaign/agent_eval.py`, `campaign/known_answers.py`, `campaign/_replay_pw.py` | Evaluation harness | `benchmarks/` |

### 2.2 Hard-coded scientific constants in generic code

| Location | Constant | Resolution |
| --- | --- | --- |
| `campaign/registry.py:21` | `TARGET_ELEMENTS=("Cr","Hf","Mo","Ta","V","W")` used as default | Required parameter; value lives in application config |
| `discovery/alloys.py:44-45` | `DEFAULT_REPEATS=(8,8,8)`, `a=3.16` (W) | Required parameters |
| `discovery/transmutation.py:35-39`, `fusion/transmutation.py` | W→Re/Os, 0.01 cap (duplicated) | One parameterized `science/irradiation/transmutation` |
| `discovery/transmutation_source.py` | Fixed `re/os/h/he` columns | Species list from input |
| `discovery/gas_defects.py` | `GAS_SPECIES = H/He` | Parameter |
| `discovery/species_support.py` | Re/Os/H/He table | Data file / config |
| `discovery/magnetism/policies.py:40-90` | Material families | Data-driven policy table |
| `kinetics/gas_mechanisms.py`, `kinetics/cluster_dynamics.py` | H/He/V mechanisms | Mechanism set passed in |
| `execution/properties.py` | Fusion property families | `science/properties` registry, families from config |
| `campaign/budget.py` | `CATEGORIES` = hierarchy stages | Categories from workflow definition |

### 2.3 Site, account and path constants in production code

| Location | Constant | Destination |
| --- | --- | --- |
| `campaign/hierarchy_config.py:169`, `workflows/full_supercell_campaign.py:116` | `account="LRN110"` | Deployment profile |
| `discovery/exploration/dispatch.py:38-52` | `FACILITIES` (frontier/LRN110, aurora, perlmutter) | `deployments/` profiles |
| `execution/iri.py` | `/lustre/orion/{account}`, `/gpfs/wolf2` | `deployments/` profiles |
| `execution/presets.py` | Facility node geometries | `deployments/` profiles |
| `backends/mlip/hydragnn_newhead.py:9` | `/global/cfs/projectdirs/m5216/...` | Required config value |
| `integrations/spectra_pka/config.py:13-14` | `/lustre` docstring path | Remove |

## 3. Duplicated concepts

| Concept | Copies | Canonical home |
| --- | --- | --- |
| Composition parsing/normalization | `discovery/composition` + several private `_composition` helpers | `core/composition` |
| Structure fingerprint / identity | ~8 | `core/structure_identity` |
| Candidate deduplication | 3 | `core/structure_identity` |
| Candidate records | ~6 dataclasses | `core/candidates` |
| `PhaseDiagram` construction | 5 | `science/thermodynamics/hull` |
| `sha256_file` | ~5 | `core/hashing` |
| Canonical JSON | 3 | `core/hashing` |
| Budget / accounting | 6+ | `execution/budget` |
| Campaign/workflow state | 4 | `workflows/state` |
| `sbatch` script construction/submission | `execution/slurm_jobs`, `gas_qe_campaign:379`, `hierarchy_campaign:674`, SPECTRA-PKA runner | `execution/schedulers/slurm` |
| Process-group kill | `backends/dft/process`, former `workflow_runner` | `execution/process` |
| Dispatch pools | `backends/dft/runner`, `backends/mlip/pool`, `execution/deadline_worker` | `execution/pool` |
| Atomic-mass tables | 4 copies in QE modules | `ase.data` via `core/units` |
| eV/Å → Ry/bohr | literal in `relaxation.py:362` vs `ase.units` elsewhere | `core/units` (re-record known answers) |
| `wilson_interval` / `Z95` | 2 | `core/stats` |
| `k_B`, `arrhenius_rate` | `kinetics` + `magnetism/consistency` | `core/units` + `science/kinetics` |
| `CELL_TOLERANCE` (NEB) | 2 | `science/neb` |
| UQ hand-off gate (0.6 / 0.25 / 3) | 3 implementations | `science/uq/gate` |
| Selection scoring | `campaign/selection`, `fusion/scoring` | `science/acquisition/scoring` |
| Property acquisition | `campaign/property_acquisition`, `fusion/acquisition` | `science/acquisition` |
| Selection revision | `campaign/selection_revision`, `fusion/revision` | `agents/revision` |
| Training dispatchers | 3 (`active_learning/trainer`, `training/*`, `hydragnn_newhead`) | `backends/mlip/training` |

## 4. Competing implementations

* **Orchestrators (5):** `orchestration/objective_graph`,
  `orchestration/composition_graph`, `orchestration/interactive_chat`,
  `campaign/orchestrator`, `workflows/investigation`. Default LLM model also
  disagrees (`qwen2.5:14b` vs `llama3.1:8b`). Target: one agent graph in
  `agents/` (based on `objective_graph`); `composition_graph` and the empty
  `orchestration/policies/` are removed.
* **Exploration engines (2):** `workflows/exploration*` and
  `phase_exploration` + `discovery/wrapper`. Target: one engine in
  `workflows/exploration`.
* **NEB (14 files)** spread across `discovery/neb_*`, `backends/dft/qe_neb*`,
  `workflows/neb_convergence`, `workflows/neb_mapping`. Target:
  `science/neb` (method) + `backends/dft/qe/neb` (code adapter).
* **Defects** spread across `discovery/*defect*`, `point_defects`,
  `gas_defects`. Target: `science/defects/{contracts,generators,analysis,formation}`.

## 5. Dependency-direction problems

| Importer | Imports | Problem |
| --- | --- | --- |
| `campaign/selection_revision.py:49`, `fusion/revision.py:58` | `workflows.radiation_hypothesis` | Generic decision logic depends on an application workflow |
| `campaign/gas_qe_campaign.py:48`, `campaign/radiation_properties.py:44` | `workflows.full_supercell_campaign` | Campaigns depend on another campaign |
| `fusion/transmutation.py:19` | `campaign.radiation_properties` | Application depends on application constants |
| `backends/mlip/hydragnn_energy.py:14`, `training/finetune_hydragnn*.py` | `datasets.formation_training` | Backend depends on data assembly |
| `discovery/point_defects` ↔ `discovery/alloy_ensemble` | occupation fingerprint | Must be cut before splitting `discovery` |
| `cli/workflows.py` | `benchmarks.e3.production`, `deployments/aurora/jobs/e4-first-batch/submit_batch.py` | Installed package executes repository-only scripts |
| `cli/config_tools.py` | 7 `examples/*.yaml` paths | Breaks in a wheel install |

## 6. Dead code and legacy aliases

| Module | Status | Action |
| --- | --- | --- |
| `discovery/seeds.py` (13 lines) | Deprecated alias; only `tests/test_renamed_module_aliases.py` and `workflows/exploration_checkpoint.py` use it | Remove; update importer |
| `discovery/selection.py` (14 lines) | Deprecated alias; also imported by `workflows/exploration_checkpoint.py:121` | Remove; update importer |
| `active_learning/candidates.py` (14 lines) | Deprecated alias | Remove |
| `tests/test_renamed_module_aliases.py` | Tests aliases | Remove |
| `execution/launchers.py` (22 lines) | No importers | Remove |
| `execution/deadline_worker.py` | Only used by `benchmarks/e3` and its tests | Move to `benchmarks/e3` or fold into `execution/pool` |
| `orchestration/policies/` (6 lines) | Empty | Remove |

## 7. CLI

* Campaign-named commands: `campaign`, `materials`, `radiation-damage`,
  `e1-development`, `benchmark e1`, `workflow init e3-*`,
  `tools warmstart-*`.
* Two CLI frameworks: Typer (main app) and argparse (14 module-level
  `main()` functions).
* Target: a small generic CLI (`run <config>`, `validate <config>`,
  `status`, `resume`, `inspect`, `config`, `backends`, `deployments`, `mcp`,
  `docs`/`version`) built only on Typer. Applications are selected by config
  file, not by subcommand.

## 8. Configuration

* Pydantic models in 115 files, dataclasses in 70, raw YAML parsing in 25.
* Target: Pydantic for every user-facing configuration (application,
  workflow, backend, deployment); dataclasses only for internal immutable
  value objects. One loader in `core/config` that validates and records the
  resolved config into provenance.
* Config templates shipped as package data, not read from `examples/`.

## 9. Repository hygiene

| Item | Observation | Action |
| --- | --- | --- |
| `research/` | 2,196 tracked files | Keep as research record; excluded from package, tests, and lint scope |
| `runs/` | 44 tracked files | Untrack (keep `.gitignore` rule), move needed fixtures to `tests/data` |
| `docs/project/` | 18 progress notes | Move to `research/notes/` |
| `docs/paper/new-manuscript.zip` | 7.5 MB binary | Remove from tree (belongs with the manuscript repository) |
| `docs/architecture.md` vs `docs/guide/architecture*` | Duplicate | Single architecture doc |
| `docs/*-aurora.md`, `*-frontier.md`, `*-perlmutter.md` | Site docs in general docs | Move under `deployments/<site>/` |
| `examples/paper_cases/` | Per-facility copies | One example per case, deployment selected by profile |
| `scripts/nbtao5_*`, `scripts/e1`, `scripts/e2`, `six_model*`, paper diagnostics | Campaign scripts | `research/` |
| Root `*.err`, `input_tmp.in`, `matsim-agents-qr.png` | Stray artifacts | Remove / ignore |
| `pyproject.toml` | `pyxtal`, `dscribe` in core dependencies; no `all` extra | Move to extras |

## 10. Risks

* **Known-answer drift:** unifying the eV/Å→Ry/bohr constant changes QE
  force-threshold inputs at the 1e-6 relative level; known-answer fixtures
  must be re-recorded in the same commit.
* **Test volume:** many tests import moved modules directly. Every phase must
  land with its test updates; no shims are added to bridge them.
* **Fusion 1024-site requirements:** exact composition, site count and
  supercell must survive the move. They become validated fields in
  `examples/fusion_bcc_1024/application.yaml` with a test that pins them.
* **Environment:** `~/.local` ase 3.28 shadows the venv's ase on this login
  node; tests are run with `PYTHONPATH=src:.` from the venv interpreter.
