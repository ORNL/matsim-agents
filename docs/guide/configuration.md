# Configuration reference

## Configuration reference

### `MatSimState`

| Field | Type | Purpose |
|---|---|---|
| `objective` | `str` | Free-form research goal. |
| `plan` | `list[TaskSpec]` | Tasks emitted by the planner. |
| `pending_tasks` | `list[TaskSpec]` | Queue consumed by the executor. |
| `results` | `list[RelaxationResult]` | Accumulated tool outputs. |
| `analysis` | `str \| None` | Final analyst summary. |
| `iteration` / `max_iterations` | `int` | Executor loop guard. |
| `llm_provider` / `llm_model` / `llm_base_url` | `str \| None` | LLM selection. |

### `TaskSpec`

```python
TaskSpec(
    kind="relax",                  # relax | analyze | report
    structure_path="foo.vasp",
    optimizer="FIRE",              # FIRE | BFGS | BFGSLineSearch
    maxiter=200,
    maxstep=1e-2,
    charge=0.0,
    spin=0.0,
    random_displacement=False,
)
```

### `RelaxStructureInput` / `RelaxationResult`

See [`src/matsim_agents/backends/mlip/relaxation.py`](../../src/matsim_agents/backends/mlip/relaxation.py) — fields mirror the
options of the upstream HydraGNN ASE script
(`structure_optimization_ASE.py`).

### Standalone DFT relaxers (outside the AL loop)

For cases where the user wants a *real* DFT relaxation rather than the
cheap HydraGNN one (e.g. validating a discovered structure, refining a
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

### MLIP warm-start benchmarks

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
