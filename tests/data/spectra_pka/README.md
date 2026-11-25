# Real SPECTRA-PKA output excerpt

`real_w182_excerpt.out` and `real_w182_excerpt.indexes` are cut verbatim from one real run.
Only whole blocks were kept, and the data rows of each kept block were truncated. Nothing was
edited or invented.

- Code: SPECTRA-PKA, upstream commit `fff9e3f`, built with serial `make`.
- Matrix: TENDL-2017 neutron `W182s.asc`, from `pkas.zip` (FISPACT nuclear-data downloads page;
  see `docs/external/spectra-pka.md`).
- Flux: the upstream `manual/test/fluxes_specter.dat` PWR test spectrum (709 groups), rescaled to
  3.25e14. This is a build check, not a physical scenario.
- Target: pure W-182, fraction 1.0.

The excerpt is a parser layout check only. It is not nuclear-physics verification, and because
rows were truncated, the `norm_sum` and cumulative columns do not add up within the excerpt.
