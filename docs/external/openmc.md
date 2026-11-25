# OpenMC activation and transmutation (open alternative to FISPACT-II)

OpenMC (MIT licence) solves the transmutation and decay equations with `openmc.deplete`. Unlike
FISPACT-II it is open source, so it can be cloned, built and run here. It is not a drop-in
replacement: it reports atoms per nuclide, and the activity and decay heat follow from the
depletion chain's decay data. It reports **no uncertainties and no shutdown dose rate**, so the
evidence it produces omits both.

## Status

| Part | State |
|---|---|
| Inventory model and evidence converters (`backends/external/openmc_activation/inventory.py`) | written; `tests/test_openmc_inventory.py` covers it with a synthetic inventory (10 tests, pass) |
| `deployments/common/setup/build-openmc.sh` and `openmc-smoke.py` | written; `bash -n` and `ruff` pass; **the script has not been run end to end** |
| Manual build of v0.15.3 on Frontier (serial, GCC 12.3, Cray HDF5 1.14.3.9) | done; `import openmc.deplete` works with the library path below |
| Co-59 activation against the analytic Co-60 inventory | run once on a Frontier login node (v0.15.3, ENDF/B-VIII.0): 1.522e12 atoms against 1.524e12 analytic, 0.15 % apart, PASS. This checks the solver bookkeeping only (see section 3) |
| A runner that writes an inventory from a material and a flux | **not written**; `OpenMCInventory` JSON is the contract it will fill |

## 1. Clone, install and build

Run on a login node (compute nodes have no internet). No allocation is needed.

```bash
source /usr/share/lmod/lmod/init/bash          # non-login shells only
module load cray-hdf5                          # exports HDF5_ROOT; or set OPENMC_HDF5_ROOT
source .venv/bin/activate                      # Python >= 3.11
bash deployments/common/setup/build-openmc.sh
```

The script has passed `bash -n` but has not been run start to finish; the same steps were done
by hand to build the existing install. It is meant to be idempotent. It:

1. clones `openmc-dev/openmc` at `v0.15.3` with its vendored submodules into `external/openmc`
   (Git-ignored), and refuses a checkout at another tag;
2. installs the OpenMC dependencies and the Python package (`pip install --no-deps`);
3. builds `libopenmc.so` serially (no MPI, no OpenMP, no DAGMC or libMesh) and copies it to
   `openmc/lib/` in the active environment, because the Python package ships without it;
4. downloads the ENDF/B-VIII.0 HDF5 library (3.4 GB) and the thermal-spectrum depletion chain
   (27.5 MB) to `.hpc-build/openmc-data`, checking sizes;
5. writes `openmc-build-record.txt` (tag, commit, compiler, SHA-256 of the library, chain and
   `cross_sections.xml`);
6. runs `deployments/common/setup/openmc-smoke.py` (see section 3).

v0.15.3 is used because v0.16.0 needs Python >= 3.12. Set `OPENMC_TAG` to change it.

| Variable | Default |
|---|---|
| `OPENMC_DIR` | `external/openmc` |
| `OPENMC_DATA_DIR` | `.hpc-build/openmc-data`; use a shared cache outside Git for several users |
| `OPENMC_HDF5_ROOT` | `$HDF5_ROOT`, else `$HDF5_DIR` |
| `OPENMC_CXX`, `OPENMC_CC` | `g++`, `gcc` from `PATH` |
| `OPENMC_JOBS` | 8 |
| `OPENMC_SKIP_DATA=1`, `OPENMC_SKIP_SMOKE=1` | off |
| `OPENMC_RUNTIME_LIBS` | extra `LD_LIBRARY_PATH` entries for the smoke test |

The data come from the OpenMC team's downloads (<https://openmc.org/data/>), hosted on
`anl.box.com`. The publisher gives no checksums, so the recorded hashes are local and are not a
verification.

## 2. Environment

```bash
export OPENMC_CROSS_SECTIONS=$PWD/.hpc-build/openmc-data/endfb-viii.0-hdf5/cross_sections.xml
export OPENMC_CHAIN_FILE=$PWD/.hpc-build/openmc-data/chain_endfb80_pwr.xml
```

On Frontier, importing `openmc.deplete` also needs these libraries on `LD_LIBRARY_PATH`
(otherwise `libhdf5` or `mpi4py` fails to load):

```bash
export LD_LIBRARY_PATH=/opt/cray/pe/hdf5/1.14.3.9/gnu/12.3/lib:/opt/cray/pe/gcc-native/12/snos/lib64:/opt/cray/pe/mpich/9.1.0/ofi/crayclang/20.0/lib:$LD_LIBRARY_PATH
export XALT_EXECUTABLE_TRACKING=no
```

Check it:

```bash
python -c "import openmc, openmc.deplete as d; print(openmc.__version__, hasattr(d, 'IndependentOperator'))"
```

The expected output is `0.15.3 True`.

## 3. Run

The smoke test depletes cobalt-59 with `IndependentOperator` (one-group cross sections from
`MicroXS.from_multigroup_flux`, `source-rate` normalisation) at a flux low enough that burn-up
is negligible. It compares the Co-60 plus Co-60m atom count with $N_{59}\,\sigma\,\phi\,t$ and
fails if they differ by more than 1 %.

```bash
python deployments/common/setup/openmc-smoke.py
```

It exits 0 on `PASS`, 1 on a mismatch and 2 when the data variables are unset. It runs on one
core on a login node.

What the test does not show: the single group is flat in energy over 1e-5 eV to 20 MeV, so the
collapsed Co-59 (n,gamma) cross section is 0.00194 b, far from the thermal value (about 37 b).
The analytic value uses that same cross section, so the test validates depletion, decay and
unit handling, not the cross-section data or a realistic spectrum. The 0.15 % residual is
unexplained; it is well inside the 1 % tolerance but has not been traced. Use a spectrum that
matches your irradiation field for real work.

## 4. Use in matsim-agents

`backends/external/openmc_activation/inventory.py` turns a stored inventory into the same
`ExternalNuclearEvidence` the FISPACT-II readers produce, so downstream code is unchanged:

```python
from matsim_agents.backends.external.openmc_activation.inventory import (
    read_inventory, to_activation_evidence, to_transmutation_evidence,
)

inv = read_inventory("co59-inventory.json")
activation = to_activation_evidence(
    inv, cooling_time_s=86400.0, material_id="Co", irradiation_scenario_id="flux-1e11"
)
helium = to_transmutation_evidence(
    inv, cooling_time_s=0.0, material_id="Co", irradiation_scenario_id="flux-1e11"
)
```

- Activation evidence has `specific_activity` (Bq/kg) and `decay_heat` (W/kg). The shutdown dose
  rate is omitted, never set to zero.
- Transmutation evidence has `concentration_appm` for the chosen elements (default H and He),
  per atom of the initial material. Metastable states fold into their ground state.
- The inventory carries the SHA-256 of its inputs and outputs, the OpenMC version, the library
  name and the one-group flux, so the evidence is traceable.

## Limits that apply to every result

- One-group cross sections collapse the whole spectrum at one temperature. They are as good as
  the spectrum you supply; a thermal-spectrum chain (the one downloaded) has capture branching
  ratios averaged for a thermal reactor, so it is not suited to a fusion or fast spectrum.
  Fast-spectrum chains are listed on the OpenMC data page.
- Cross sections are fixed for the whole run, so self-shielding and burn-up of the
  cross-section spectrum are not modelled.
- Gas production relies on the chain's reaction list. Check that the (n,p) and (n,alpha)
  reactions you care about are present before trusting a helium or hydrogen number.
- There are no uncertainties. Do not quote an error bar for an OpenMC activation result.
