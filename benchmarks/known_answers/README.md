# Si/SiC known-answer suite

Five characterization questions with established PBE answers. Each case runs
its contract through the real characterization pipeline. A deterministic
grader (the benchmark's NO_LLM arm) checks the report's status, ground state,
whether the conclusion places it on or above the hull, observables and missing
evidence against [cases.yaml](cases.yaml). Each
verdict is written to the evidence ledger as a `known_answer_pass` or
`known_answer_fail` decision that cites the run's evidence.

| Case | Question | Expected answer |
|---|---|---|
| `si-lattice-constant` | PBE lattice constant of diamond Si | 5.44–5.50 Å, residual force ≤ 0.02 eV/Å |
| `si-diamond-vs-beta-tin` | Diamond vs β-tin ranking | diamond lowest; β-tin 0.15–0.40 eV/atom above |
| `sic-3c-on-hull` | Is 3C-SiC on the Si–C hull? | E_hull ≤ 5 meV/atom; formation energy −0.40 to −0.15 eV/atom |
| `sic-rocksalt-above-hull` | Is rocksalt SiC on the hull? | E_hull ≥ 0.3 eV/atom |
| `si-step-limited` | One-step relaxation | `inconclusive`, missing evidence says "not converged" |

All contracts share one DFT method: Quantum ESPRESSO, PBE pslibrary
pseudopotentials, 45/360 Ry, smearing (`qe-pbe-pslibrary-ecut45-smearing`).
The hull cases use [references.json](references.json): diamond Si, diamond C
and 2H-SiC relaxed with that method.

## Tiers

```bash
# Offline (CI): recorded pw.x runs, no DFT executed
matsim-agents benchmark known-answers --tier replay

# Live: runs pw.x for every case (pw.x on PATH, pseudopotentials in pseudo_dir)
matsim-agents benchmark known-answers --tier live -o runs/known-answers-live
```

The command exits 1 if any case fails. `--case ID` selects cases. Each
invocation runs in a new `<output>/<tier>-*` directory, so no step is reused
from an earlier run. `--ledger` appends to an existing ledger. The default
ledger is `known-answer-ledger.jsonl` in that run directory.

On the replay tier the launcher is replaced by
`matsim_agents/campaign/_replay_pw.py`. It returns the recorded `pw.out` only
if the `pw.in` the pipeline wrote matches the recorded input exactly. If input
generation changes, the replay tier fails instead of grading stale output.

## Re-recording

Re-record after changing a contract, structure, or the QE input writer:

```bash
export MPICH_GPU_SUPPORT_ENABLED=0 OMP_NUM_THREADS=8
python benchmarks/known_answers/record.py --pw-x /path/to/pw.x
```

Each invocation works in a new directory under `runs/known-answer-record`
(git-ignored). It relaxes the references and runs the live tier with `--pw-x`
on a copy of the suite. Only if every case passes does it replace
`references.json` and copy each relaxation's `pw.in`/`pw.out` to
`replay/<case_id>/<step_id>/`; otherwise it exits 1 and leaves the suite
unchanged. The runs are serial and
take a few minutes in total. On another machine, change `pseudo_dir` in the
contracts first; the recorded inputs contain it.
