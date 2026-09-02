from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

INSTALLERS = {
    "frontier": "scripts/hpc/olcf/frontier/installation/install-rocm72.sh",
    "aurora": "scripts/hpc/alcf/aurora/installation/install.sh",
    "perlmutter": "scripts/hpc/nersc/perlmutter/installation/install.sh",
}


def test_facilities_have_one_canonical_current_hydragnn_installer() -> None:
    for facility, hydragnn_path in INSTALLERS.items():
        installer = ROOT / "deployments" / facility / "setup" / "install.sh"
        text = installer.read_text(encoding="utf-8")
        assert installer.stat().st_mode & 0o111
        assert hydragnn_path in text
        assert "https://github.com/ORNL/HydraGNN.git" in text
        assert "hf_transfer" in text
        assert 'INSTALL_UMA="${INSTALL_UMA:-0}"' in text
        assert 'MATSIM_EXTRAS="${MATSIM_EXTRAS},uma"' in text
        assert "from fairchem.core import FAIRChemCalculator, pretrained_mlip" in text
        assert "pip check" in text
        assert "installation_DOE_supercomputers/hydragnn_installation" not in text
        subprocess.run(["bash", "-n", str(installer)], check=True)
