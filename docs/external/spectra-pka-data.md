# SPECTRA-PKA nuclear data: the 7.9 GB `pkas.zip`

## What it is

The official PKA recoil matrices (NJOY-generated, 294 K) are published as one Dropbox shared
folder, linked from the "Data download for SPECTRA-PKA" table at
<https://fispact.github.io/resources/nuclear_data_downloads/>. Downloading it gives a single
`pkas.zip` of 7,946,418,171 bytes (observed 2026-11-25). The folder lists these libraries:

| Library | Incident particle |
|---|---|
| TENDL-2015, 2017, 2019, 2023 | neutron |
| TENDL-2017 | alpha, proton |
| TENDL-2023 | proton |

## Layout (observed)

- `pkas.zip` stores (method 0, no ZIP compression) one tarball per library, for example
  `TENDL2023n-pka.tar.bz2`. Despite the name it is **gzip**, not bzip2.
- Each tarball holds `pka/<El><AAA>s.asc`, one file per target isotope (`W184s.asc`,
  `Fe056s.asc`, `Cu063s.asc`). Mass numbers are zero-padded to three digits. Files are
  roughly 20-90 MB each, so a full library is several GB.
- Each `.asc` file starts with a header line such as `(n,x) proton matrix 5 0.30000E+08 1026
  1025 1`, then the 1026-point incident energy grid (eV, from 1e-11), then the matrices.
- Only the isotopes with evaluated data exist. A missing file means the library has no data
  for that isotope; it must not be substituted from another library.
- The publisher gives no checksums. Local SHA-256 values are recorded and never reported as
  verified.

The server ignores HTTP range requests (a ranged request returns the whole 7.9 GB), so the
archive cannot be sampled. It must be streamed once from the start.

## Extracting only what a material needs

`scripts/spectra_pka_fetch_data.py` (library code in
`matsim_agents.backends.external.spectra_pka.data_fetch`) streams the archive, skips every
other library, and writes only the requested files. It works for any material: the isotope
list comes from the elements you name (all natural isotopes, NIST/IUPAC abundances through
`mendeleev`), from a composition, or from explicit nuclides. Nothing is specific to a campaign.

Run it on a login node (compute nodes have no internet). The stream is 7.9 GB, so expect a
long transfer even when you keep only a few files, because the library you want can sit
anywhere in the archive. Extraction stops once every requested file is found.

```bash
pip install 'matsim-agents[spectra-pka]'        # provides natural abundances
DATA=/path/to/shared/spectra-pka-data           # shared cache, never inside Git

# what would be needed (no download)
python scripts/spectra_pka_fetch_data.py --dest $DATA --elements W,Fe,Cu --dry-run

# elements -> all natural isotopes; composition and extra nuclides are also accepted
python scripts/spectra_pka_fetch_data.py --dest $DATA --library TENDL2023n \
    --composition W:0.4,Ta:0.3,Cr:0.3 --nuclides Re187

# a pkas.zip you already have, no network
python scripts/spectra_pka_fetch_data.py --dest $DATA --archive /path/pkas.zip --elements W
```

Options: `--library` is a substring of the archive member name (`TENDL2023n`, `TENDL2019n`,
`TENDL2017p`, ...), default `TENDL2023n`. `--force` re-extracts. `--allow-missing` returns 0
when isotopes are absent. Exit code 2 means some isotopes are not in that library.

## Result on disk

```text
$DATA/TENDL2023n-pka/W180s.asc ... W186s.asc
$DATA/TENDL2023n-pka/manifest.json
```

`manifest.json` records per file: library, file name, target isotope, source URL (without the
share key), retrieval time (UTC), byte size, SHA-256, the header line, and
`"checksum_verified": false`. Files already present are reused, writes are atomic
(temporary file then rename), and the manifest is updated under a lock, so several
users sharing one cache directory do not corrupt it.

Point `external_tools.spectra_pka.nuclear_data_root` at `$DATA/TENDL2023n-pka` (the runtime
`PKA_DATA_DIRECTORY`).

## Rules

- Keep one library per run. Do not mix files from different TENDL releases for one material.
- Never commit the data to Git; it is licensed by its publishers.
- Choose the library (release and particle) deliberately and record it with results.
- Isotopes absent from the library are a stop condition for that material, not a prompt to
  fall back to another release.
