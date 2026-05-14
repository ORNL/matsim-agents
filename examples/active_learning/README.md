# Active-learning workflow: HydraGNN ↔ DFT (VASP or Quantum ESPRESSO)

End-to-end loop that uses HydraGNN as a fast surrogate force field and either
**VASP 6.x** or **Quantum ESPRESSO `pw.x`** as the ground-truth labeller for
high-uncertainty structures discovered by HydraGNN-driven MD. The choice is
a single YAML field (`dft.backend: vasp | qe`).

## Files

- `al_config.example.yaml`         — Unified config: contains both `dft.vasp:`
  and `dft.qe:` sub-blocks; flip `dft.backend:` to choose the labeller.
- `al_config.prompt.example.yaml`  — LLM-generated seeds (works with either backend).
- `INCAR.template`                 — VASP single-point INCAR (MI250X-tuned).
- `seeds/`                          — Drop seed POSCAR/CIF/XYZ files here
  (only used when `md.seed_source.kind: paths`).

> **Energy-reference warning.** VASP PAW totals and QE pseudopotential totals
> are NOT directly comparable. The dataset writer tags every frame with
> `info["dft_backend"]`; never train one HydraGNN model on a mixed VASP+QE
> dataset without an explicit per-backend energy offset.

## Quick start on Frontier

1. Build VASP (one-time):
   ```bash
   nohup bash scripts/setup/frontier/build-vasp-gpu-frontier.sh \
       > runs/build-vasp-gpu-login/build.log 2>&1 &
   ```

2. Edit `al_config.example.yaml`:
   - Point `hydragnn.logdir` at a trained HydraGNN logdir.
   - Choose a seed source under `md.seed_source` — see *Seed sources* below.
   - Choose `dft.backend: vasp` (and fill `dft.vasp.*`) **or** `dft.backend: qe`
     (and fill `dft.qe.*`).
   - For VASP: point `dft.vasp.potcar_dir` at your POTCAR collection (one
     POTCAR per element). For QE: point `dft.qe.pseudo_dir` at a UPF
     pseudopotential library (e.g. SSSP-PBE-efficiency).

3. Validate:
   ```bash
   matsim-agents al validate-config examples/active_learning/al_config.example.yaml
   ```

4. Submit the SLURM job:
   ```bash
   sbatch --export=ALL,AL_CONFIG=$PWD/examples/active_learning/al_config.example.yaml \
       -N 64 -t 12:00:00 \
       scripts/launchers/frontier/run-active-learning-frontier.sh
   ```

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│  SLURM allocation (64 nodes, 12 h)                                   │
│  Parent shell: PrgEnv-gnu + rocm/7.2.0 + HydraGNN venv               │
│                                                                      │
│  python -m matsim_agents.cli al run al.yaml                          │
│    │                                                                 │
│    ├── for iter in 0..N:                                             │
│    │     ├── load HydraGNN calculator(s)                             │
│    │     ├── sample MD candidates (in-process)                       │
│    │     ├── score: ensemble / MC-dropout / random                   │
│    │     ├── select top-K with diversity filter                      │
│    │     ├── ── ThreadPoolExecutor(K parallel) ──┐                   │
│    │     │     bash _vasp-step-frontier.sh       │ inner srun step   │
│    │     │       module reset → PrgEnv-cray      │ rocm/7.1.1        │
│    │     │       srun -N1 -n8 vasp_std           │                   │
│    │     │     parse OUTCAR/vasprun → labels  ←──┘                   │
│    │     ├── append to dataset.extxyz                                │
│    │     └── bash _hydragnn-train-step-frontier.sh                   │
│    │           srun -N8 -n64 python train.py --dataset ... --resume  │
│    └──                                                                │
└──────────────────────────────────────────────────────────────────────┘
```

Why two module stacks: the DFT labeller (VASP or QE) needs Cray Fortran
(`PrgEnv-cray + cce`) for OpenMP target offload. The HydraGNN/PyTorch venv
was built against `PrgEnv-gnu + rocm/7.2.0`. They cannot coexist in one
shell, but they do coexist in one SLURM job — every DFT step does
`module reset` in a fresh bash subshell
(`_vasp-step-frontier.sh` or `_qe-step-frontier.sh`),
then `srun` against the allocation.

## Resume

Each iteration writes `out_dir/iteration_NNNN/state.json` with `status:
"complete"` only on success. On restart the loop scans for the highest
completed iteration, deletes any partial directory, and resumes from there
without double-counting DFT results.

## Seed sources

The MD sampler’s starting structures come from `md.seed_source` — pick **one** of:

```yaml
md:
  seed_source:
    kind: paths            # explicit list of files
    paths: [seeds/Si.vasp, seeds/SiO2.vasp]
```

```yaml
md:
  seed_source:
    kind: compositions     # formulas → prototype seeds via discovery enumerator
    compositions: [LiCoO2, LiFePO4, Cs2AgBiBr6]
    max_phases_per_composition: 2
    min_atoms: 32
```

```yaml
md:
  seed_source:
    kind: prompt           # LLM expands a free-text target into formulas
    prompt: "Pb-free halide double perovskites for photovoltaics"
    llm:
      provider: vllm       # ollama | vllm | openai | anthropic | huggingface
      model: Qwen/Qwen2.5-72B-Instruct
      base_url: http://localhost:8000/v1
    max_compositions: 6
    max_phases_per_composition: 2
```

For `kind: prompt` the LLM proposal is persisted as
`<out_dir>/seeds/llm_proposed_compositions.json` so the run is reproducible.
For `kind: compositions` and `kind: prompt`, formulas are expanded into
prototype crystal seeds via
[`matsim_agents.discovery.enumerate_phases`](../../src/matsim_agents/discovery/phase_explorer.py)
(fcc/bcc/hcp/diamond, rocksalt/CsCl/zincblende/wurtzite/rutile, perovskite,
spinel, double perovskite, optional 2-D graphene/h-BN/MX2 …). The seeds are
intentionally **not** pre-relaxed — the MD heat-up immediately drives them
off their idealised positions, which is exactly the regime AL needs.
