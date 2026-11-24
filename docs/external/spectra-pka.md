# SPECTRA-PKA (external)

**MATSIM-AGENTS does not implement SPECTRA-PKA's nuclear physics. It provides an optional
interface to externally installed SPECTRA-PKA software.**

SPECTRA-PKA is a Fortran program by M. R. Gilbert et al. (GPL-3.0,
<https://github.com/fispact/SPECTRA-PKA>). It computes primary-knock-on-atom (PKA) recoil
spectra from a neutron spectrum and NJOY/GROUPR PKA matrices. Nothing from it is vendored in
this repository: no source, no binaries, no nuclear data.

## 1. What this interface is

A thin adapter in `matsim_agents.integrations.external.spectra_pka`:

| Module | Role |
| --- | --- |
| `config` | strict YAML schema (`extra="forbid"`, disabled by default) |
| `availability` | separate readiness checks and a status |
| `input_writer` | native `.in` file from a validated campaign; user data linked, never modified |
| `runner` | `subprocess.run([exe, input])` with `cwd` and `timeout` (no shell), or one `slurm_jobs` submission |
| `parser`, `validation` | read `<stub>.out`/`.indexes`, reject unknown layouts, check structure |
| `adapter` | validate -> stage -> run -> collect -> parse -> validate |

The application-independent contracts live in `matsim_agents.irradiation` and import nothing
from the adapter, so another PKA source can replace it.

## 2. Installing SPECTRA-PKA yourself

Follow the upstream README (a `make` build of the Fortran source) and obtain PKA matrices from
your FISPACT-II / NJOY nuclear-data distribution. MATSIM-AGENTS never downloads, compiles or
installs either. Point `external_tools.spectra_pka.executable` and `nuclear_data_root` at what
you installed.

## 3. Configuration

See `examples/external/spectra-pka/campaign.yaml`. All paths there are placeholders. Relative
paths resolve against the YAML file's directory; `${VAR}` is expanded.

## 4. Commands

```bash
matsim-agents external spectra-pka check   -c campaign.yaml
matsim-agents external spectra-pka prepare -c campaign.yaml
matsim-agents external spectra-pka run     -c campaign.yaml            # dry run
matsim-agents external spectra-pka run     -c campaign.yaml --execute  # real run
matsim-agents external spectra-pka status  -c campaign.yaml
matsim-agents external spectra-pka import  -c campaign.yaml            # parse existing output
```

`run` stages inputs and stops unless `--execute` is given. With `execution.mode: slurm` the
job is submitted once through `execution/slurm_jobs.py` (1 node, no GPU, idempotent by spec
hash, never polled).

## 5. Statuses

Readiness only, not scientific qualification:

| Status | Meaning |
| --- | --- |
| `NOT_INSTALLED` | executable not configured, missing, or not executable |
| `INSTALLED` | the executable starts and behaves like SPECTRA-PKA; no campaign was assessed |
| `INPUTS_MISSING` | executable present but nuclear data, a PKA file or the spectrum is missing |
| `READY` | executable, data, flux file and work directory are all in place |
| `EXECUTION_FAILED` | the invocation probe or a smoke record failed |

Each check (executable, permission, invocation, data root, per-isotope coverage, spectrum,
work directory, smoke test) is reported separately by `check`. The identity of the executable is
its SHA-256 and install-root metadata; no `--version` flag is invoked because none is verified.
The invocation probe passes a deliberately missing input file and looks for the upstream
message `error opening parameter input file`.

## 6. Native input

Only labels documented upstream are written (an unrecognised label makes SPECTRA-PKA stop):
`flux_filename`, `results_stub`, `num_columns`, `columns=`, `number_pka_files`,
`flux_norm_type`, `pka_filetype=2`, `do_mtd_sums`, `do_global_sums`,
`do_exclude_light_from_total`, `energies_once_perfile`, `max_global_recoils`,
`flux_rescale_value`. SPECTRA-PKA silently renormalises `pka_ratios`, so the adapter requires
isotope fractions to agree with the stated composition up front.

The column block written has four columns. The upstream examples that also give the two
`ngamma_*_mass` columns do so for `do_ngamma_estimate`, which is not enabled here. Whether a
four-column block is accepted has not been checked against a real build.

## 7. Output handling

The layout parsed was read from the upstream output routines, not from a real run. The parser
accepts only blocks it recognises and column counts it expects, and raises `ParseError`
otherwise. Zero-PKA bins stay absent, `unknown` daughters are kept, energies are MeV, rates
are per target atom per second for the global matrices. Raw files and their SHA-256 hashes are
preserved; parsing never rewrites them. Collection is restartable from the raw files
(`import`).

## 8. Validation

`STRUCTURALLY_VALID` means the file is complete and self-consistent (end marker, ascending
non-negative bins, finite values, input-spectrum total). It is not a physics validation.
Every spectrum carries `numerical_validation = NOT_VALIDATED`.

## 9. Species, sites and sampling

`PKASpectrum` holds species-resolved rates, with isotope-to-element aggregation. 
`irradiation.pka_sampling.sample_pka_events` draws reproducible events (seeded through
`seeding.numpy_rng`), places each on a lattice site of the matching element, optionally
records the neighbouring chemistry, and reports the share of the spectrum that could not be
placed (for example H and He recoils on a metal lattice) instead of hiding it. Event times
exist only when the rates are stated per target atom per second. Within-bin energies are drawn
uniformly (or log-uniformly): a sampling convention, not data. Daughter-nuclide parentage of
capture products is not yet used for site assignment; sites are matched by the recoiling
element.

## 10. PKA is not surviving defects

A PKA spectrum says how many recoils of which species and energy occur. How many defects
survive needs a cascade model (MD, BCA, a surrogate). The default `NoCascadeModel` returns
`NOT_ESTABLISHED`. Nothing here converts PKAs to defects.

## 11. Provenance

Each spectrum records the executable path and SHA-256, the nuclear-data identity, SHA-256 of
the input file and of every staged data file, SHA-256 of the raw outputs, the run record
(command, cwd, exit code, wall time, marker) and the raw-output directory.

## 12. What is tested

`tests/test_spectra_pka_adapter.py` and `tests/test_spectra_pka_cli.py` use a mock executable
and invented fixtures. They test file handling, status logic, parsing of the assumed layout,
contracts, sampling and restartability. **They are not nuclear-physics verification.** No real
SPECTRA-PKA run has been made by this repository's tests.

## 13. Limitations and open steps

* The output layout and the four-column `columns=` block are unverified against a real build.
* Real validation (a published or independently computed reference case) is pending.
* Time-dependent inventories and capture-product parentage are not modelled here.
* Citations to use when publishing results: J. Nucl. Mater. 467 (2015) 121-134;
  Nucl. Mater. Energy 9 (2016) 576-580; J. Nucl. Mater. 504 (2018) 101-108;
  Nucl. Sci. Eng. (2024) doi:10.1080/00295639.2024.2342506.
* The index file name is unresolved: the upstream manual names `index_summary.dat`, while
  reading the source suggested `<stub>.indexes`. The parser does not depend on this file for
  the spectrum. Confirm against a real build before relying on either name.
* SPECTRA-PKA was not installed on the development machine when this was written
  (`spectra-pka` not on `PATH`), so no real run exists.

## 14. PKA-informed ranking and the synthetic chain

`matsim_agents.campaign.pka_acquisition.select_pka_informed` ranks candidate calculations by
species-resolved PKA production, a diminishing return for repeating a species, user-supplied
kinetic importance and uncertainty, and cost in node-hours. It is a policy, not a fitted
model, and it has not been shown to beat random selection. It launches no jobs, and
surviving-defect production stays `NOT_ESTABLISHED`.

`tests/test_radiation_e2e_synthetic.py` chains alloy generation, transmutation, a He/H
vacancy complex, a mock SPECTRA-PKA run, PKA sampling and the ranking. It checks composition,
determinism and labels only.
