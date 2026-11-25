# Active learning (MLIP and DFT)

## Active-learning loop (MLIP ↔ DFT)

The `matsim-agents al` subcommand runs an end-to-end active-learning loop
that grows a training set from DFT labels of structures the current MLIP
(HydraGNN, UMA, or MACE, selected by the config's `mlip:` block) is most
uncertain about. Both **VASP 6.6** and **Quantum
ESPRESSO `pw.x`** are supported as the labeller — the choice is a single
YAML field.

```
  MLIP ──────── MD ──► candidates ────────────────────────────────────┐
        ▲                       │                                            │
        │                       ▼                                            │
        │             ensemble / MC-dropout                                   │
        │             uncertainty + diversity                                 │
        │                       │                                            │
        │                       ▼                                            │
        │             top-K most informative                                  │
        │                       │                                            │
        │                       ▼                                            │
        │             DFT backend (parallel, in-allocation)                   │
        │             vasp_std  │  pw.x  (one toggle)                         │
        │                       │                                            │
        │                       ▼                                            │
        │             dataset.extxyz / dataset.db  (tagged with backend)      │
        │                       │                                            │
        │                       ├── default: retain validated labels          │
        │                       ▼                                            │
        └─ optional train + approved promotion ── next iteration ────────────┘
```

### Your campaign owns its config

The YAML files under `examples/` are **templates and illustrations**. They were
written for particular chemical systems, elemental references, cutoffs and
budgets. They are not a configuration for your campaign. Each campaign needs its
own `al.yaml` and its own elemental reference plan, covering every element that
can appear in its structures.

`matsim-agents al run` refuses a config that lives under `examples/`. To run a
template on purpose (for example a deployment smoke test), pass
`--allow-template` or set `MATSIM_ALLOW_TEMPLATE_CONFIG=1`. The checked-in
deployment jobs set this variable.

Before the first iteration, `al run` and `al validate-config` also check that the
training references cover every element in the seed structures. A missing
element stops the run.

### Quick start

```bash
# 1. Generate a campaign directory from your own seeds or compositions.
#    Nothing in it is approved yet.
matsim-agents al init-campaign campaigns/wtacrvhfmo \
  --mlip-config examples/mlip/hydragnn-gfm.yaml \
  --dft-backend qe \
  --composition W --composition Ta --composition Cr \
  --composition V --composition Hf --composition Mo \
  --preset frontier
#   -> campaigns/wtacrvhfmo/al.yaml
#      campaigns/wtacrvhfmo/elemental-phases.yaml   (proposed reference phases)
#      campaigns/wtacrvhfmo/references/*.extxyz     (starting structures)

# 2. Review elemental-phases.yaml: phases, magnetic states, k-points and the
#    review notes (for example: Cr is antiferromagnetic but is declared
#    nonmagnetic). Elements that cannot be proposed safely are listed as
#    unresolved and must be added by hand. Then set phases_approved: true
#    in al.yaml.

# 3. Validate the resolved config (no run). --mlip-config swaps in any model.
export PW_BIN=... PSEUDO_DIR=...   # placeholders listed in al.yaml
matsim-agents al validate-config campaigns/wtacrvhfmo/al.yaml

# 4. Submit on Frontier
sbatch --export=ALL,AL_CONFIG=$PWD/campaigns/wtacrvhfmo/al.yaml \
    -N 64 -t 12:00:00 \
    deployments/frontier/launchers/run-active-learning-frontier.sh
```

`al init-campaign` never overwrites an existing campaign directory. Its
proposals are starting points, and the reference DFT still needs your approval
(`dft_approved`) before the elemental references are computed.

### Backend toggle

The example YAML carries both backend sub-blocks; flip `dft.backend:` to
select one. The unused sub-block is ignored.

```yaml
dft:
  backend: ${DFT_BACKEND:-vasp}    # vasp | qe
  vasp:
    vasp_bin: ${VASP_BIN}
    potcar_dir: ${POTCAR_DIR}
    incar_template: ${PROJ_ROOT}/examples/active_learning/INCAR.template
  qe:
    pw_bin: ${PW_BIN}
    pseudo_dir: ${PSEUDO_DIR}
    pw_template: ${PROJ_ROOT}/examples/active_learning/pw.template
```

### Variable substitution in YAMLs

All AL example configs use shell-style placeholders that are expanded
at load time by `ALConfig.from_yaml`:

| Syntax                  | Meaning                                          |
| ----------------------- | ------------------------------------------------ |
| `${VAR}`                | required; raises if unset                        |
| `${VAR:-default}`       | falls back to `default` if unset                 |
| `${VAR:?error message}` | aborts with `error message`                      |

Resolution order: (1) `os.environ`, (2) optional top-level `vars:`
block in the YAML itself. Nested references inside `vars:` resolve
iteratively, so `VASP_BIN: ${PROJ_ROOT}/external/.../vasp_std` just
works. The `vars:` block is consumed before pydantic validation and
never appears in the parsed `ALConfig`.

### Seed sources for MD

`md.seed_source.kind` selects how initial MD structures are obtained:

- `paths` — a curated list of POSCAR / CIF / XYZ files on disk.
- `compositions` — an explicit list of formulas expanded into prototype
  seeds, with optional supplementary pyXtal seeds.
- `prompt` — the LLM proposes plausible compositions for a target
  objective (e.g. *“Pb-free halide perovskites for PV”*) and the loop
  materialises seed structures by running the same crystal-prototype
  enumerator used by the discovery wrapper. No curated structure
  collection is required.

For `compositions` and `prompt`, `max_phases_per_composition` caps
prototype-derived seeds and `n_random` requests additional pyXtal seeds
(default `0`); the generated count can be lower.

### Random seed

The required top-level `random_seed` is the root of every random draw in the
run: pyXtal seed structures, MD velocities and thermostat noise, acquisition
(random scores and MC-dropout masks), the validation split, and each
iteration's fine-tuning. Each draw uses its own named stream of the seed; see
[Reproducibility and random seeds](../reproducibility.md).

### Energy-reference warning

VASP PAW totals and QE pseudopotential totals are **not** directly
comparable. Every frame written to the dataset is tagged with
`info["dft_backend"]`; never train one MLIP on a mixed
VASP+QE dataset without an explicit per-backend energy offset. See
[MLIP and DFT energy conventions](../mlip-energy-conventions.md).

### Energy comparison and promotion

For each method `m`, the comparison uses:

```text
formation_energy_m = (compound_total_m - sum_i(n_i * elemental_energy_per_atom_m[i])) / N
```

Each MLIP first evaluates the same fixed elemental geometries supplied with
the test. Its compound totals subtract **its own predictions**, while DFT
totals subtract **DFT elemental energies**. Do not subtract DFT references
from raw MLIP totals. O2 and other molecular references are divided by their
actual atom count. References are not independently relaxed by each model.

Use this for held-out evaluation of incumbent/candidate models and promotion:

```yaml
trainer:
  # Add these to an otherwise valid training configuration.
  compare_after_training: true
  validation_set: /path/to/held-out.extxyz
  validation_reference_set: /path/to/elemental_references.json
  promotion_max_energy_mae_eV_per_atom: 0.1
```

Promotion still requires its separate approval flags and force/regression
checks. `promotion_max_energy_mae_eV_per_atom` now limits
`formation_energy_mae_eV_per_atom`, not compound-fitted energy offsets.
Recalibrate historical thresholds on real held-out data; the value above is
an example, not a scientifically qualified recommendation.

For standalone energy evaluation:

```bash
python -m matsim_agents.active_learning.evaluate \
  --al-config /path/to/al-config.yaml \
  --test-set /path/to/held-out.extxyz \
  --elemental-reference-manifest /path/to/elemental_references.json \
  --out-json /path/to/metrics.json
```

Raw energy errors and elemental-baseline errors remain diagnostic outputs,
alongside force errors. Force-only evaluation needs no elemental manifest.
Training without model comparison, MD acquisition, within-method
fixed-composition ranking, and method-specific hull construction are unchanged.
Formation subtraction does not authorize pooling incompatible DFT methods
into a training dataset or hull.

The caller must supply converged pure-element DFT calculations; these
evaluators do not launch them. Campaign comparisons check coverage and the
DFT method signature before candidate training. Separate DFT protocols require
separate matching manifests, even when the comparison geometries are identical.
File-based energy comparisons and promotion require a matching compound
`*.extxyz.manifest.json` sidecar containing the data hash, DFT backend, and
method signature; missing metadata is an error. Fine-tune/evaluation propagates
verified method metadata into newly generated train/test split sidecars.
Campaign AL training accumulates within each formula's dataset across iterations
and restarts. Existing train/held-out assignments never change. Small batches
are retained, with training and promotion explicitly deferred until the
cumulative training and validation partitions meet their minimum sizes;
there is no automatic pooling across formulas or increase in DFT selection.
See [the manifest schema and compatibility checks](../mlip-training-and-promotion.md).

Full walkthrough — including templated INCAR / `pw.in` files, in-allocation
launcher details, and per-backend ROCm/MPI gotchas — lives in
[`examples/active_learning/README.md`](../../examples/active_learning/README.md).

---
