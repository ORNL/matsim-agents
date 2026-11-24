"""Synthetic SPECTRA-PKA fixtures for tests.

Everything here is invented to exercise the adapter's file handling. None of it is real
SPECTRA-PKA output or nuclear data, and tests using it are not nuclear-physics verification.
"""

from __future__ import annotations

import stat
import sys
from pathlib import Path

from matsim_agents.integrations.external.spectra_pka.config import (
    CampaignSpec,
    ChannelSpec,
    FluxSpec,
    SpectraPKAConfig,
)

SYNTHETIC_OUT = """\
### index 0 #####
#INPUT SPECTRUM - E(MEV) lower+upper bound  VS.  FLUX
 1.0E-01 1.0E+00 4.0E+09
 1.0E+00 1.0E+01 6.0E+09
# TOTAL FLUX = 1.0E+10  FLUENCE = 1.0E+10


### index 1 #####
#  W [ W183 ] from [ W182 ]
#PKA RECOIL DISTRIBUTIONS
#RECOIL energy (MeV low & high)  PKAs  norm_sum
 5.0E-07 1.0E-06 2.0E-03 5.0E-01
 1.0E-06 2.0E-06 2.0E-03 1.0E+00
# AVERAGE PKA ENERGY = 1.2E+00 (eV)


### index 2 #####
#  Ta [ unknown ] from [ Ta181 ]
#PKA RECOIL DISTRIBUTIONS
#RECOIL energy (MeV low & high)  PKAs  norm_sum
 5.0E-07 1.0E-06 1.0E-03 1.0E+00
# AVERAGE PKA ENERGY = 7.0E-01 (eV)


### index 3 #####
#  recoil matrix of W182
#PKA RECOIL DISTRIBUTIONS - per target atom
#RECOIL energy (MeV low & high)  PKAs/s  norm_sum  cumulative_sum
 5.0E-07 1.0E-06 3.0E-03 5.0E-01 5.0E-01
 1.0E-06 2.0E-06 3.0E-03 1.0E+00 1.0E+00


### index 4 #####
#  recoil matrix of Ta181
#PKA RECOIL DISTRIBUTIONS - per target atom
#RECOIL energy (MeV low & high)  PKAs/s  norm_sum  cumulative_sum
 5.0E-07 1.0E-06 2.0E-03 1.0E+00 1.0E+00
"""

SYNTHETIC_INDEXES = (
    "0 input spectrum\n"
    "1 W [W183] from [W182] pka_W182.asc\n"
    "2 Ta [unknown] from [Ta181] pka_Ta181.asc\n"
)

MOCK_EXECUTABLE = """\
#!{python}
import re, sys
from pathlib import Path

if len(sys.argv) < 2 or not Path(sys.argv[1]).is_file():
    print(" error opening parameter input file")
    sys.exit(1)
text = Path(sys.argv[1]).read_text()
stub = re.search(r'results_stub="([^"]+)"', text).group(1)
Path(stub + ".out").write_text({out!r})
Path(stub + ".indexes").write_text({idx!r})
Path(stub + ".log").write_text("END as expected\\n")
print("END as expected")
"""


def write_mock_executable(path: Path, *, out: str = SYNTHETIC_OUT) -> Path:
    path.write_text(MOCK_EXECUTABLE.format(python=sys.executable, out=out, idx=SYNTHETIC_INDEXES))
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def make_campaign(root: Path) -> CampaignSpec:
    (root / "data").mkdir(exist_ok=True)
    w = root / "data" / "W182.asc"
    ta = root / "data" / "Ta181.asc"
    flux = root / "data" / "flux.dat"
    for p in (w, ta, flux):
        p.write_text(f"synthetic placeholder for {p.name}\n")
    return CampaignSpec(
        scenario_id="synthetic-w-ta",
        parent_alloy_id="W0.9Ta0.1",
        composition={"W": 0.9, "Ta": 0.1},
        nuclear_data_identity="synthetic-test-data",
        channels=(
            ChannelSpec(pka_file=w, parent_ele="W", parent_num=182, atom_fraction=0.9),
            ChannelSpec(pka_file=ta, parent_ele="Ta", parent_num=181, atom_fraction=0.1),
        ),
        flux=FluxSpec(
            file=flux,
            identity="synthetic-flat",
            normalization="synthetic; n/cm^2/s",
            norm_type=2,
        ),
        results_stub="spka",
    )


def make_config(root: Path, executable: Path | None) -> SpectraPKAConfig:
    return SpectraPKAConfig(
        enabled=True,
        executable=executable,
        nuclear_data_root=root / "data",
        workdir=root / "work",
    )
