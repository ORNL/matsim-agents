#!/usr/bin/env python3
"""Extract the PKA recoil matrices a material needs from the official pkas.zip.

Run on a login node (compute nodes have no internet). Example:
    PYTHONPATH=src python scripts/spectra_pka_fetch_data.py --dest $DATA --elements W,Fe,Cu
See docs/external/spectra-pka-data.md.
"""

from matsim_agents.backends.external.spectra_pka.data_fetch import main

if __name__ == "__main__":
    raise SystemExit(main())
