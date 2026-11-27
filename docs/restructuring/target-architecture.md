# Target architecture

This document defines the package responsibilities, dependency rules and
acceptance criteria for the restructured `matsim_agents`. It is derived from
the findings in [audit.md](audit.md).

## Principles

1. **The framework is general-purpose.** No element symbol, composition, site
   count, supercell size, account, queue, filesystem path or campaign name
   appears as a default in `src/matsim_agents`. Such values arrive through
   validated application or deployment configuration.
2. **Applications are configuration plus thin glue** and live in
   `examples/`. A campaign is never a package, a layer or a CLI subcommand.
3. **One implementation per concept.** Duplicates listed in the audit are
   merged; no aliases or compatibility wrappers are kept.
4. **Dependencies point downward** through the layers below and are enforced
   by `tests/test_package_boundaries.py`.
5. **Backends are adapters.** Scientific methods do not import a specific DFT,
   MLIP or LLM code; they depend on the protocols owned by the layer that
   defines them (see Core interfaces below).

## Package map

```text
src/matsim_agents/
  core/          contracts, application schema, composition, energy
                 conventions, relaxation results, hashing, atomic file writes
                 (`fileio`), YAML mapping loader (`yamlio`), restart records
                 (`restart`), stats, seeding, stress
  science/       backend-agnostic scientific methods
    structures/      alloy/SQS generation, prototypes, seed structures
    thermodynamics/  references, formation energy, convex hull, surrogate hull
    defects/         contracts, generators, analysis, formation energies
    neb/             path construction, convergence, analysis
    kinetics/        rates, Arrhenius, KMC, cluster dynamics (mechanisms as input)
    irradiation/     damage descriptors, PKA, transmutation (species as input)
    magnetism/       collinear/noncollinear policies (data-driven)
    properties/      elasticity, property recipes, property registry
    uq/              ensembles, uncertainty metrics, transfer, hand-off gate
    acquisition/     acquisition functions, scoring, coverage, selection
    characterization/ structure/property characterization
  backends/      adapters to external codes
    llm/  dft/{qe,vasp}  mlip/{mace,hydragnn,...,training}  mcp/  external/
  execution/     run store, artifacts, restarts, schedulers (slurm, local),
                 process control, worker pools, budget, resource requests
  data/          dataset assembly, formats, training-set construction
  workflows/     generic multi-step workflows: relaxation, hierarchy screening,
                 exploration, active learning, characterization, state
  agents/        LLM agent graph, tools, review, selection revision, debate
  cli/           generic Typer CLI: run, validate, status, resume, inspect,
                 config, backends, deployments, mcp, version
```

Outside the package:

| Directory | Contents |
| --- | --- |
| `examples/` | Applications: config + small glue modules + README. At minimum `fusion_bcc_1024/`, one non-BCC application, one different-paradigm application, plus `fusion_pfc/`, `gas_transmutation/`, `radiation_damage/` |
| `deployments/` | Facility profiles (accounts, queues, node geometry, filesystems, IRI endpoints), job templates, site docs |
| `benchmarks/` | Evaluation harnesses, known answers, scalability, warm-start, E3/E4 |
| `research/` | Paper experiments, campaign scripts, progress notes, results |
| `tests/` | `unit/`, `integration/`, `architecture/`, `applications/`, `data/` |
| `docs/` | User and developer documentation |
| `scripts/` | Repository maintenance only (`check.sh`, setup, doc validation) |

## Layers

Lowest first. A module may import only from its own layer or lower layers.
Sublayers inside a row may not import each other unless listed.

| Layer | Packages |
| --- | --- |
| 0 | `core` |
| 1 | `execution`, `data` |
| 2 | `backends.llm`, `backends.dft`, `backends.mlip`, `backends.mcp` |
| 3 | `science.*` (`structures` < `thermodynamics` < `defects`, `neb`, `magnetism`, `properties` < `kinetics`, `irradiation` < `uq` < `acquisition`, `characterization`) |
| 3b | `backends.external` (adapters that emit `science` contracts) |
| 4 | `workflows` |
| 5 | `agents` |
| 6 | `cli` |

Additional rules enforced by architecture tests:

* `science.*` must not import `backends.*`. It takes calculators and
  results as arguments, so any backend can be plugged in.
* `backends.mlip` must not import `data`. Training-set construction is
  passed in by `workflows`.
* Nothing under `src/` imports `examples`, `benchmarks`, `research`,
  `deployments` or `scripts`.
* `src/` contains no literal facility accounts, `/lustre`, `/gpfs`,
  `/global/cfs` paths, or campaign identifiers (`e1`, `e3`, `e4`, `nbtao5`,
  `fusion`, `pfc`) in module names or CLI command names.
* Removed modules stay removed: a list of old import paths must fail to
  import.

## Protocols

Protocols sit in the layer that owns the capability, so `core` stays free of
backend and execution concepts:

* `DFTBackend` in `backends/dft/base.py` and `MLIPBackend` in
  `backends/mlip/__init__.py`
* `ExecutionPlatform` in `execution/__init__.py` (scheduler + filesystem
  abstraction)
* `RunStore` in `execution/provenance.py` (artifacts, restarts, provenance)

## Configuration model

* Pydantic models for all user-facing configuration:
  `ApplicationConfig` (chemistry, structure generation, properties,
  objectives, budgets), `WorkflowConfig`, `BackendConfig`,
  `DeploymentConfig`.
* YAML configs are read with `core.yamlio`; the resolved config is written into
  the run provenance.
* Application requirements become validation rules. For the fusion
  1024-site study these are `sites == 1024`, the exact composition, the
  (8,8,8) BCC supercell, and the element set.

## Removal inventory

Removed outright, with no replacement alias:

* `discovery/seeds.py`, `discovery/selection.py`,
  `active_learning/candidates.py`, `tests/test_renamed_module_aliases.py`
* `execution/launchers.py`, `orchestration/policies/`,
  `orchestration/composition_graph.py`
* Top-level packages `campaign/`, `fusion/`, `discovery/`,
  `orchestration/`, `active_learning/`, `training/`, `datasets/`,
  `integrations/`, `irradiation/`, `kinetics/` and the module `seeding.py`.
  Their contents are redistributed according to the package map.
* CLI commands `campaign`, `materials`, `radiation-damage`,
  `e1-development`, `benchmark e1`, `workflow init e3-*`,
  `tools warmstart-*`.

## Acceptance criteria

1. Architecture tests pass: layers, `science`-independent-of-`backends`, no
   imports of repository-only directories, no site or campaign literals, and
   removed modules fail to import.
2. `src/matsim_agents` contains no `campaign/`, `fusion/` or `discovery/`
   package.
3. The CLI exposes only generic commands. Every application runs through
   `matsim-agents run examples/<app>/application.yaml` (dry-run in CI).
4. Three applications validate and dry-run in CI:
   * **A:** fusion BCC refractory alloy, 1024 sites (W640Ta128Cr64V96Hf32Mo64,
     8×8×8). Exact requirements are pinned by a test.
   * **B:** a non-BCC crystal (for example an FCC or perovskite system).
   * **C:** a different paradigm (for example a magnetic or 2D material).
5. One implementation each for: composition parsing, structure identity,
   candidate record, hull construction, hashing/canonical JSON, Slurm
   submission, worker pool, UQ gate, orchestrator, exploration engine.
6. The full test suite, `scripts/check.sh lint` and the documentation
   validator pass.
7. Docs describe the new architecture only. Site docs live in
   `deployments/`.

## Status and blockers

Implemented on branch `refactor/framework-architecture` (PR #80):

- Packages `campaign/`, `fusion/` and `discovery/` are gone. Layers are
  `core` → `execution`, `data` → `backends.{llm,dft,mlip,mcp}` → `science` →
  `backends.external` → `workflows` → `agents` → `cli`.
- `science` imports no backend; `RelaxationResult` and the numerical stress
  calculator live in `core`.
- Campaign CLI commands (`campaign`, `materials`, `radiation-damage`,
  `supervisor-run`, `explore-and-learn`, `e1-development`, E3 presets) are
  removed. E1 and E3 run from `benchmarks/` (`python -m benchmarks.e1`,
  `python -m benchmarks.e3.production`).
- Applications: `examples/fusion_bcc_1024` (pinned 1,024-site
  W640Ta128Cr64V96Hf32Mo64, 8×8×8 BCC), `examples/cantor_fcc_256` (FCC,
  hierarchy workflow), `examples/perovskite_oxides` (prototype search).
  `matsim-agents validate FILE` dry-runs each; CI runs all three.
- Every maintained example goes through `matsim-agents validate`
  (`tests/test_runner.py`). Unset `${VAR}` inputs, a missing stability
  `reference_energies` file, and server-backed LLM checks are reported as
  blockers (exit 3) rather than invalid configs; investigation configs have a
  runner handler.
- Transmutation host/products, radiation species and HydraGNN paths come
  from configuration.
- Enforcement: `tests/test_package_boundaries.py`, `tests/test_applications.py`,
  the `architecture` job in `.github/workflows/ci.yml`.

Known blockers and deviations:

- Alloy generation (`AlloySpec`, `build_supercell`) handles sc, bcc, fcc,
  hcp, diamond and custom unit cells, conventional or primitive, with diagonal
  or full integer supercell matrices. `validate` builds every target
  parent, including the Cantor 4x4x4 FCC cell (256 sites). The
  full-supercell defect workflow and `find_cross_cell_equivalents` are still
  bcc-only, because they use bcc interstitial and cubic-axis geometry.
- The perovskite example is mode-driven (`validate`, `generate`, `mlip_relax`,
  `dft_relax`) and runs through `matsim-agents run`; its `run.py` wrapper is
  removed. `*_relax` modes report an MLIP that does not cover every element as
  a blocker. Relaxation and hull ranking need an MLIP or DFT and are not run
  in CI.
- Acceptance criterion 3 is met: `matsim-agents run FILE` executes any
  config; the free-form objective path is `matsim-agents objective "TEXT"`.
- Criterion 5 (single implementations) was audited; see
  [duplication-audit.md](duplication-audit.md) for merges and the duplicates
  kept with a reason.
- `research/paper/e1/d-questions.v1.json` is the hash-pinned historical record
  and still names `matsim_agents.workflows.e1`; `d-questions.v2.json` carries
  the same 18 questions and names the current runner schema. `research/` is
  otherwise frozen.
- `execution/iri.py` keeps facility working-directory templates; it is the
  allowlisted exception to the facility-literal test.
- The full suite was not rerun after every commit; targeted suites were.

## Phase plan and PR scoping

| PR | Phase | Scope |
| --- | --- | --- |
| 1 | 1 | This audit and target architecture |
| 2 | 2 | `core/` (contracts, interfaces, composition, identity, candidates, units, hashing, stats, seeding, config); remove aliases and dead modules; new layer table |
| 3 | 3a | `science/structures`, `thermodynamics`, `defects`, `neb` from `discovery`/`campaign`/`workflows`; parameterize constants |
| 4 | 3b | `science/kinetics`, `irradiation`, `magnetism`, `properties`, `uq`, `acquisition`, `characterization`; merge `fusion` generic parts |
| 5 | 4 | `agents/` (single orchestrator), one exploration engine, `workflows/active_learning`, `backends/mlip/training`, `data/` |
| 6 | 5 | `examples/` applications, `deployments/` profiles, generic CLI, packaged templates, pyproject extras |
| 7 | 6 | Tests reorganized, three-application demonstrations, docs rewrite, CI update, hygiene |

Each PR lands with green tests and lint. No PR introduces new scientific
capability.
