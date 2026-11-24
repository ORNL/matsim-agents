# Magnetic ordering and scalable exploration: capability and qualification

Branch `feat/radiation-damage-phase-a`. Four evidence levels are kept apart:
implemented, tested, numerically validated, scientifically qualified.

## Qualification matrix

| Capability | Implemented | Tested (software) | Numerically validated | Scientifically qualified |
|---|---|---|---|---|
| Magnetic configuration contract and generators (M1, M2) | yes | yes | no | no |
| Quantum ESPRESSO magnetic input (M3, M4) | yes | yes (input text only) | no (no QE run compared) | no |
| Result parsing, competition, defect magnetism, consistency (M5-M8) | yes | yes (synthetic data) | no | no |
| Material policies (M9) | yes | yes | no | no, `policy_readiness()` always reports unqualified |
| Candidate manifest, reduction, sampling (M12-M14) | yes | yes | no | no |
| Multifidelity promotion (M15) | yes | yes | no | no; F1 never supports magnetic claims |
| Batch dispatch (M16) | yes | yes, fake scheduler only | no real Slurm job submitted | no |
| Surrogate interface (M17) | yes | yes | no, no real labelled data | no |
| Coverage and convergence (M18) | yes | yes | no | no; statistical uncertainty only |
| Scale and adaptive demonstrations (M19) | yes | yes | synthetic landscape only | no |

Synthetic evidence is never "qualified". Real DFT comparison campaigns are pending.

## Measured synthetic results (M19)

- 5000 candidates ingested, killed after 50 percent, reopened, recovered and finished:
  5000 done, 0 duplicates, each evaluated exactly once, 24.1 s wall, 25.3 MiB Python heap peak
  (tracemalloc; excludes numpy and ASE native memory). Login node, single process.
- Adaptive versus random selection on a synthetic heteroscedastic landscape (budget 80, 20 seeds):
  mean absolute error of stratum means 0.164 (sd 0.055) adaptive versus 0.194 (sd 0.082) random;
  adaptive was better in 13 of 20 seeds. This is a modest, not decisive, difference on a landscape
  built to favour uncertainty-driven sampling. It says nothing about real magnetic energies.

## Limitations

- No GPU smoke test was run and no Slurm job was submitted in this work.
- Aurora and Perlmutter dispatch profiles are unverified and refuse to dispatch.
- The fallback surrogate is a per-stratum mean; no learned model is integrated.
- Physical and numerical uncertainty are reported as "not assessed".
