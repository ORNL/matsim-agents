"""Synthetic UPF headers for input-writer tests (software verification only, never physics)."""

from __future__ import annotations

from pathlib import Path

_TEMPLATE = (
    '<UPF version="2.0.1">\n<PP_INFO/>\n<PP_HEADER element=" {el}" pseudo_type="PAW" '
    'is_paw="true" has_so="false" core_correction="true" functional=" SLA  PW   PBX  PBC" '
    'z_valence="{z}" wfc_cutoff="{wfc}" rho_cutoff="{rho}"/>\n</UPF>\n'
)


def write_synthetic_upf(
    directory: str | Path,
    filename: str,
    element: str,
    *,
    z: float = 4.0,
    wfc: float = 30.0,
    rho: float = 240.0,
) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    path.write_text(_TEMPLATE.format(el=element, z=z, wfc=wfc, rho=rho))
    return path
