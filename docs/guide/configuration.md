# Configuration reference

## MLIP selection

Every command that runs an MLIP reads one flat `mlip:` block, usually from
`--mlip-config FILE`; an active-learning configuration works too. The
`backend` key selects the model, and each backend accepts only its own fields.
`${VAR}` references are expanded from the environment and the file's `vars:`
mapping. Relative paths resolve differently by backend:

- `hydragnn`: `logdir`, `hydragnn_root`, `mlp_checkpoint`, `newhead_ft_config`,
  and `ft_repo` resolve against the YAML file; `checkpoint` resolves against
  `logdir`.
- `uma`: `model_name` is a registry name unless it names an existing file
  relative to the YAML file's directory, in which case that file is used.
- `mace`: with `family: checkpoint`, `model` resolves against the YAML file;
  otherwise `model` names a foundation model.

Ready-made files are in [examples/mlip/](../../examples/mlip/).

| Backend | Fields |
|---|---|
| `hydragnn` | `logdir` (with `config.json` and checkpoint), `checkpoint`, `mlp_checkpoint` (BranchWeightMLP `.pt`), `inference_head`, `precision`, `mlp_precision`, `hydragnn_root`, `newhead_ft_config`, `ft_repo`, `radius`, `max_neighbours`, `charge`, `spin` |
| `uma` | `model_name`, `task_name` (`omat`, `omol`, `oc20`, `odac`, `omc`), `precision`, `charge`, `spin`, `dropout` |
| `mace` | `family`, `model`, `head`, `precision`, `dispersion`, `dropout` |
| all | `device` (`cuda`, `cpu`, `xpu`), `ensemble_models`, `fine_tuner` |

HydraGNN uses fused 16-head inference by default, which requires
`mlp_checkpoint`. To pin one head for MLIP inference, set `inference_head` to a
case-insensitive dataset name or its index, for example
`inference_head: OMat24`. Pinned mode invokes only that decoder, does not require
the branch MLP, and does not report branch-weight uncertainty. It preserves
the loader's precision/autocast context, including `bf16`, during energy and
force inference.

| Index | Dataset | Index | Dataset |
|---:|---|---:|---|
| 0 | Alexandria | 8 | OMol25 |
| 1 | ANI1x | 9 | OMol25-neutral |
| 2 | MPTrj | 10 | OMol25-non-neutral |
| 3 | OC2020 | 11 | OPoly2026 |
| 4 | OC2022 | 12 | Nabla2DFT |
| 5 | OC25 | 13 | QCML |
| 6 | ODAC23 | 14 | QM7X |
| 7 | OMat24 | 15 | transition1x |

## Agent graph state

### `MatSimState`

| Field | Type | Purpose |
|---|---|---|
| `objective` | `str` | Free-form research goal. |
| `plan` | `list[TaskSpec]` | Tasks emitted by the planner. |
| `pending_tasks` | `list[TaskSpec]` | Queue consumed by the executor. |
| `results` | `list[RelaxationResult]` | Accumulated tool outputs. |
| `mcp_results` | `list[MCPCallRecord]` | Accumulated external MCP call records, separate from scientific results. |
| `handoff_events` | `list[str]` | UQ-driven active-learning handoff events. |
| `analysis` | `str \| None` | Final analyst summary. |
| `iteration` / `max_iterations` | `int` | Executor loop guard. |
| `llm_provider` / `llm_model` / `llm_base_url` | `str \| None` | LLM selection. |

### `TaskSpec`

```python
TaskSpec(
    kind="relax",                  # relax | analyze | report | mcp_tool | mcp_resource
    structure_path="foo.vasp",
    optimizer="FIRE",              # FIRE | BFGS | BFGSLineSearch
    maxiter=200,
    maxstep=1e-2,
    charge=None,                   # None uses the selected MLIP's configuration
    spin=None,
    random_displacement=False,
    displacement_seed=None,        # required when random_displacement is enabled
)
```

MCP tool tasks require `mcp_server`, `mcp_tool`, and optionally
`mcp_arguments`; resource tasks require `mcp_server` and `mcp_resource_uri`.
They need no structure path or MLIP selection. See the [MCP guide](../mcp.md)
for configuration, authorization, and result handling. The planner also
accepts `analyze` and `report`, but the executor currently requires an MLIP
selection and then skips those kinds.

## `RelaxStructureInput` / `RelaxationResult`

See [`src/matsim_agents/backends/mlip/relaxation.py`](../../src/matsim_agents/backends/mlip/relaxation.py).
`RelaxStructureInput` takes the structure path, the `mlip` selection, and the
optimizer settings; the same settings apply to every MLIP backend.

## Standalone DFT relaxers (outside the AL loop)

For cases where the user wants a *real* DFT relaxation rather than the
cheaper MLIP one (e.g. validating a discovered structure, refining a
final candidate), two drivers ship under `src/matsim_agents/backends/dft/`
with matching APIs:

| Module | Backend | Calculation modes | Composition-aware defaults |
|---|---|---|---|
| [`backends/dft/qe_relax.py`](../../src/matsim_agents/backends/dft/qe_relax.py) | Quantum ESPRESSO `pw.x` | `scf`, `relax`, `vc-relax` | `ecutwfc` (SSSP-PBE-eff-1.3 table), smearing, k-mesh |
| [`backends/dft/vasp_relax.py`](../../src/matsim_agents/backends/dft/vasp_relax.py) | VASP `vasp_std` | `scf`, `relax`, `vc-relax`, `vc-relax-shape` | `ENCUT` = 1.3 × max(ENMAX) from POTCARs (else 520 eV); `ISMEAR/SIGMA/KSPACING` flip metallic vs insulator |

Both follow the same workflow:

```python
from ase.build import bulk
from matsim_agents.backends.dft.vasp_relax import (
    recommend_settings, prepare_relax_workdir, run_vasp,
)

atoms    = bulk("Si")
settings = recommend_settings(atoms, potcar_dir="/path/to/potcars",
                              calculation="vc-relax")
workdir  = prepare_relax_workdir(atoms, "./Si_vcrelax", settings,
                                 potcar_dir="/path/to/potcars")
result   = run_vasp(workdir, launcher_cmd=["bash", "run-vasp-frontier.sh"])
print(result.final_energy_eV, result.n_ionic_steps, result.converged)
```

`qe_relax` has the same shape; both honour an env-overridable launcher
(`MATSIM_QE_LAUNCHER` / `MATSIM_VASP_LAUNCHER`) and parse the per-ionic-step
trajectory + walltime + convergence flag from the native output files
(`pw.out` for QE, `vasprun.xml` + `OUTCAR` for VASP).

> **Note:** the active-learning loop itself never calls these relaxers —
> AL labelling always uses the SCF-only backends under
> [`src/matsim_agents/backends/dft/`](../../src/matsim_agents/backends/dft/).
> A relaxation per AL candidate would defeat the point of uncertainty-driven
> sampling. The standalone relaxers are intended for one-off DFT validation
> work outside the AL pipeline.

## MLIP warm-start benchmarks

A second pair of sibling drivers wraps the standalone relaxers in a
"cold start vs MLIP-warm start" experiment and emits a JSON summary
that the integration tests consume:

| Module | Backend | CLI |
|---|---|---|
| [`warmstart_benchmark_qe.py`](../../src/matsim_agents/workflows/warmstart_benchmark_qe.py)     | Quantum ESPRESSO `pw.x` | `python -m matsim_agents.workflows.warmstart_benchmark_qe …`   |
| [`warmstart_benchmark_vasp.py`](../../src/matsim_agents/workflows/warmstart_benchmark_vasp.py) | VASP `vasp_std`         | `python -m matsim_agents.workflows.warmstart_benchmark_vasp …` |

Each driver runs (1) an MLIP ASE relaxation with the backend chosen by
`--mlip-config FILE` (an `mlip:` block selecting `hydragnn`, `uma`, or `mace`),
(2) DFT relaxation from the *original* coordinates (cold), (3) DFT
relaxation from the MLIP-relaxed coordinates (warm), then reports
`Δ ionic-steps`, `Δ total-SCF-iterations`, `Δ energy`, and a `warm_helped`
boolean. The MLIP result is stored in the summary's `mlip` block. If the
MLIP step fails (or `--skip-mlip` is passed) only the cold DFT run is
executed and the `warm` block is left `None`.

---
