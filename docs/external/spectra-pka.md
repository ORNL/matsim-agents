# SPECTRA-PKA (external)

**MATSIM-AGENTS does not implement SPECTRA-PKA's nuclear physics. It provides an optional
interface to externally installed SPECTRA-PKA software.**

SPECTRA-PKA is a Fortran program by M. R. Gilbert et al. (GPL-3.0,
<https://github.com/fispact/SPECTRA-PKA>). It computes primary-knock-on-atom (PKA) recoil
spectra from a neutron spectrum and NJOY/GROUPR PKA matrices. Nothing from it is vendored in
this repository: no source, no binaries, no nuclear data.

## 1. What this interface is

A thin adapter in `matsim_agents.backends.external.spectra_pka`:

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

MATSIM-AGENTS never downloads, compiles or installs SPECTRA-PKA or its data from Python. Build
it outside the tracked tree (`external/` is gitignored). The serial gfortran build is
facility-neutral: `deployments/common/setup/build-spectra-pka.sh` (clone, pinned commit, serial
build, smoke test, build record). It was validated on Frontier only; on Perlmutter or Aurora
load a GNU Fortran compiler first (for example the `PrgEnv-gnu` stack) and run it on a login
node. Data retrieval is described in [spectra-pka-data.md](spectra-pka-data.md). Manual
equivalent:

```bash
cd external
git clone https://github.com/fispact/SPECTRA-PKA.git      # GPL-3.0, upstream `master`
cd SPECTRA-PKA
make                                                       # serial; produces ./spectra-pka
```

* Requirements: GNU make and a Fortran compiler. The Makefile uses `gfortran` with
  `-ffree-line-length-none`. It built with `gfortran` 7.5 on Frontier (upstream commit
  `fff9e3f`).
* **Do not use `make -j`.** The Makefile has no module dependency ordering, so a parallel
  build fails with `Can't open module file 'globals.mod'`. Run `make clean && make` to retry.
* Smoke test (bundled data, no download): run `../../spectra-pka ZR.in` in a scratch copy of
  `manual/test`. The last line of output must be `END as expected`, and `ZR.indexes` must match
  `example_results/ZR.indexes`. This checks the build only, not any campaign.
* Observed on Frontier (`fff9e3f`, gfortran 7.5): `ZR.indexes` matches upstream byte for byte.
  `ZR.out` is not byte-identical to upstream's shipped `example_results/ZR.out`, because that
  file predates the current output format: it prints energies with 4 digits (`0.3020E-03`, the
  binary prints `3.019952E-04`) and lacks the `disp_energy (eV/s)` and `NRT_dpa/s` columns.
  A numeric comparison of the data rows (ignoring `#` lines) found 43117 rows in each file, 657
  rows differing only in those "estimated (n,g) recoil matrix" blocks, and a maximum relative
  difference of 0 at the printed precision. Two Frontier runs were byte-identical, so the build
  is deterministic. The build script therefore requires `ZR.indexes` to match and reports the
  `ZR.out` comparison as informational. This is a build check, not a physics validation.
* The program reads `specter.input` by default, or the input file named on the command line.

### PKA matrices (nuclear data)

`manual/test/example_data` holds only a Zr sample. The production matrices (NJOY/GROUPR recoil
matrices, `*.asc`) come from the "Data download for SPECTRA-PKA" table on
<https://fispact.github.io/resources/nuclear_data_downloads/>. Every row links the same Dropbox
folder. Change `dl=0` to `dl=1` to download the whole folder as one zip (`pkas.zip`, about
7.9 GB):

```bash
curl -sSL -A "Mozilla/5.0" --retry 5 -C - -o pkas.zip \
  'https://www.dropbox.com/scl/fo/jqgdshufqxvwea8xzecwf/ACablw3sL4Ru0rgiogr17Xs?rlkey=ryqsvfgjm4u6wzoxaqcmkteip&dl=1'
unzip -l pkas.zip     # TENDL2023n/p, TENDL2019-n, TENDL2015data-n, TENDL2017data-n/p/a tarballs
unzip -o pkas.zip TENDL2017data-n-pka.tar.bz2 && tar -xjf TENDL2017data-n-pka.tar.bz2
# -> njoy_results-n-tendl2017-jc/pka/<El><AAA>s.asc   (e.g. W182s.asc, Ta181s.asc)
```

The licence and terms of use are those set by FISPACT/TENDL. Do not commit these files.

**On Frontier (lrn110) the download is done.** The cache
`/lustre/orion/lrn110/proj-shared/cache/nuclear-data/spectra-pka/` holds only `pkas.zip`, its
`SHA256SUMS`, and `SOURCE.txt` (URL, date, fetch/extract/verify commands). Nothing is kept
unpacked. Extract only the matrices a campaign needs into its own work area, for example:

```bash
C=/lustre/orion/lrn110/proj-shared/cache/nuclear-data/spectra-pka
(cd "$C" && sha256sum -c --quiet SHA256SUMS)
mkdir -p data && cd data
unzip -p "$C/pkas.zip" TENDL2017data-n-pka.tar.bz2 \
  | tar -xj --wildcards 'njoy_results-n-tendl2017-jc/pka/W18*s.asc'
export PKA_DATA_DIRECTORY=$PWD/njoy_results-n-tendl2017-jc/pka
```

The same extraction is available as a command. `--library` must name one member exactly (with or
without `-pka`, case-insensitive), for example `TENDL2023n`, `TENDL2019-n` or `TENDL2017data-n`.
Extraction is verified against the manifest, so a modified or missing file is extracted again:

```bash
matsim-agents external spectra-pka fetch-data --dest data --library TENDL2023n \
  --elements W,Ta --archive "$C/pkas.zip"
```

The neutron spectrum is always supplied by the user. The upstream `manual/test/fluxes_specter.dat`
(709-group PWR spectrum) is suitable for build checks only.

Then point the campaign YAML at the build:

```bash
export SPECTRA_PKA_EXECUTABLE_PATH=$PWD/spectra-pka
export PKA_DATA_DIRECTORY=/path/to/pka-matrices   # see above for the Frontier path
matsim-agents external spectra-pka check -c campaign.yaml
```

`INSTALLED` from `check` means the executable starts, not that results are qualified.

## 3. Configuration

See `examples/external/spectra-pka/campaign.yaml`. All paths there are placeholders. Relative
paths resolve against the YAML file's directory; `${VAR}` is expanded.

The neutron flux is given by exactly one of two keys under `flux`:

- `file`: a native SPECTRA-PKA flux file, linked into the run directory unchanged.
- `spectrum_file`: a CSV `e_low,e_high,flux`, written as a native itype-2 file (MeV, `ksail -2`).
  Declare `energy_units` (eV, keV, MeV), `flux_units`, `value_kind` (integrated, per_energy,
  per_lethargy) and `absolute`. A line spectrum, `norm_type: 3` (upstream skips the cm2-to-barn
  factor there), or a shape-only spectrum without `rescale_value` is rejected. SPECTRA-PKA
  collapses the flux onto the library grid itself.

The run record keeps the SHA-256 of the source spectrum and of the generated `flux.dat`.

## 4. Commands

```bash
matsim-agents external spectra-pka check   -c campaign.yaml
matsim-agents external spectra-pka prepare -c campaign.yaml
matsim-agents external spectra-pka run     -c campaign.yaml            # dry run
matsim-agents external spectra-pka run     -c campaign.yaml --execute  # real run
matsim-agents external spectra-pka status  -c campaign.yaml
matsim-agents external spectra-pka import  -c campaign.yaml            # parse existing output
```

`run --execute` is cached for local runs. The key is the SHA-256 of the executable, the input
text, every staged data file and the output stub. A matching, structurally valid previous run is
parsed again and not re-executed (the message says `cache hit` and `cached` is true). The
cache record stores the original execution outcome, so a hit keeps the exit code, end marker
and wall time of the process that wrote the bytes. A failed run never leaves a cache record.

Raw outputs are never deleted. Before a fresh execution, the previous `.out`, `.indexes`,
`.log`, input snapshot, stdout/stderr, run record and cache record are moved to
`archive/<utc stamp>/` in the scenario directory. A campaign that still has
`derive_channels` set (not loaded through `load_config_file`) is refused, because it has no
channels to check or stage.

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

The PKA-file table has four columns (`pka_filename pka_ratios parent_ele parent_num`). The upstream examples that also give the two
`ngamma_*_mass` columns do so for `do_ngamma_estimate`, which is not enabled here. The column
headers must sit on the same line as the label (`columns= ...`), because `read_input.f90` reads
them from the label's value. An earlier two-line form made the real binary stop with
`variable columns not recognised`. The one-line, four-column form ran successfully with the real
build.

## 7. Output handling

The layout was first read from the upstream output routines and then corrected against a real
run (see section 13). In the real output:

* per-file blocks are titled `(n,np) proton matrix [ H1 ] from [ W182 ]`, where the bracketed
  daughter is the recoil;
* global blocks are titled `recoil matrix of W182`, `elemental recoil matrix of W` and
  `total recoil matrix (He+H+unknowns excluded)`;
* `<stub>.indexes` lists `<n>  <title>pka_<file>.asc`, with the file name glued to `]`.

The parser accepts only blocks it recognises and column counts it expects, and raises `ParseError`
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
and invented fixtures. They test file handling, status logic, parsing, contracts, sampling and
restartability. `test_parse_real_output_excerpt` parses a verbatim excerpt of the real run in
`tests/data/spectra_pka/` (provenance in its README). **None of this is nuclear-physics
verification.** The tests do not run SPECTRA-PKA.

## 13. Limitations and open steps

* One real software smoke run exists (Frontier, upstream `fff9e3f`). It used pure W-182, the
  TENDL-2017 `W182s.asc` matrix and the upstream PWR test spectrum rescaled to 3.25e14. The
  adapter reported `STRUCTURALLY_VALID` and `NOT_VALIDATED`. This is a software check, not a
  physical scenario.
* Real validation (a published or independently computed reference case) is pending.
* Time-dependent inventories and capture-product parentage are not modelled here.
* Citations to use when publishing results: J. Nucl. Mater. 467 (2015) 121-134;
  Nucl. Mater. Energy 9 (2016) 576-580; J. Nucl. Mater. 504 (2018) 101-108;
  Nucl. Sci. Eng. (2024) doi:10.1080/00295639.2024.2342506.
* The real build writes the index file as `<stub>.indexes`.
* SPECTRA-PKA is built on Frontier at `external/SPECTRA-PKA` (gitignored). The PKA data archive is
  in the shared cache; matrices are extracted per campaign (section 2).

## 14. PKA-informed ranking and the synthetic chain

`matsim_agents.science.acquisition.pka_acquisition.select_pka_informed` ranks candidate calculations by
species-resolved PKA production, a diminishing return for repeating a species, user-supplied
kinetic importance and uncertainty, and cost in node-hours. It is a policy, not a fitted
model, and it has not been shown to beat random selection. It launches no jobs, and
surviving-defect production stays `NOT_ESTABLISHED`.

`tests/test_radiation_e2e_synthetic.py` chains alloy generation, transmutation, a He/H
vacancy complex, a mock SPECTRA-PKA run, PKA sampling and the ranking. It checks composition,
determinism and labels only.
