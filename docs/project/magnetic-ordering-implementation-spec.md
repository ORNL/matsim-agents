# Magnetic ordering and scalable exploration: implementation status

Scope: collinear NM/FM/AFM/ferrimagnetic/custom magnetism (M1-M11) and scalable
chemical-magnetic-defect exploration (M12-M19). Noncollinear magnetism, spin-orbit coupling,
spin spirals and DLM models are out of scope. Synthetic evidence is never numerically qualified.

Status is append-only; the latest line per task wins.

| Task | Status | Tests | Commit |
|---|---|---|---|
| M1 magnetic configuration contract | software tested | tests/test_magnetism_contract.py (22 passed) | see log |
| M1 (final) | software tested | 22 | 1b3b0ce |
| M2 generators | software tested | 11 | f5c0df4 |
| M3 QE type builder | software tested | 11 | a61067a |
| M4 QE input integration | software tested | in test_qe_magnetic_input.py | 3cab2ef |
| M5 result parsing | software tested | 13 | 6b7350b |
| M6 competition ranking | software tested | 9 | 65c090c |
| M7 defect magnetism | software tested | 9 | 1597c4a |
| M8 consistency checks | software tested | 7 | 184791c |
| M9 material policies (not scientifically qualified) | software tested | 8 | 5a4d181 |
| M10 end-to-end (synthetic) | software tested | 4 | f57654c |
| M11 CLI | software tested | 4 | 8212550 |
| M12 candidate identity and manifest | software tested | 6 | c1dc9f9 |
| M13 symmetry reduction | software tested | 6 | 32b90af |
| M14 sampling and stop rules | software tested | 7 | e2c344d |
| M15 multifidelity promotion | software tested | 7 | 5d147e3 |
| M16 non-blocking dispatch (fake scheduler; no real job) | software tested | 9 | 7c8b10c |
| M17 surrogate interface | software tested | 6 | d54d9f7 |
| M18 coverage and convergence | software tested | 5 | fbdeab2 |
| M19 synthetic demonstrations | software tested | 7 | 38d937a |
