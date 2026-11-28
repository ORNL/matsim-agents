"""UPF header parsing; identity comes from the header, not the file name."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path


class UPFError(ValueError):
    """Raised for an unreadable or incomplete UPF header."""


_ATTR = re.compile(r'(\w+)\s*=\s*"([^"]*)"')

# QE spellings of PBE: short name, explicit components, and legacy "PBE" in the gradient slots.
_PBE_SPELLINGS = {"PBE", "SLA PW PBX PBC", "SLA PW PBE PBE", "SLA PW PBX PBE", "SLA PW PBE PBC"}


def canonical_functional(functional: str) -> str:
    """Map equivalent QE spellings of PBE to ``PBE``; other functionals are returned unchanged."""
    name = " ".join(functional.upper().split())
    return "PBE" if name in _PBE_SPELLINGS else name


@dataclass(frozen=True)
class UPFHeader:
    element: str
    z_valence: float
    functional: str
    pseudo_type: str
    is_paw: bool
    has_so: bool
    core_correction: bool
    wfc_cutoff_ry: float | None
    rho_cutoff_ry: float | None
    sha256: str
    relativistic: str = ""
    upf_version: int = 2


def _parse_upf_v1(path: Path, text: bytes) -> UPFHeader:
    """UPF v1: positional ``<PP_HEADER>`` lines; relativistic treatment comes from ``PP_INFO``."""
    block = re.search(rb"<PP_HEADER>\s*\n(.*?)</PP_HEADER>", text, re.S)
    if block is None:
        raise UPFError(f"{path.name}: no UPF PP_HEADER found")
    lines = block.group(1).decode("latin-1").splitlines()
    try:
        element = lines[1].split()[0]
        pseudo_type = lines[2].split()[0]
        core_correction = lines[3].split()[0].upper().startswith("T")
        functional = " ".join(lines[4].split()[:4])
        z_valence = float(lines[5].split()[0])
        wfc, rho = (float(x) for x in lines[7].split()[:2])
    except (IndexError, ValueError) as exc:
        raise UPFError(f"{path.name}: unreadable UPF v1 PP_HEADER") from exc
    info = text.split(b"<PP_HEADER>", 1)[0].decode("latin-1").lower()
    if "scalar-relativistic" in info or "scalar relativistic" in info:
        relativistic = "scalar"
    elif re.search(r"full\w*[- ]relativistic", info):
        relativistic = "full"
    elif "non-relativistic" in info or "nonrelativistic" in info:
        relativistic = "no"
    else:
        relativistic = ""
    return UPFHeader(
        element=element,
        z_valence=z_valence,
        functional=functional,
        pseudo_type=pseudo_type,
        is_paw=pseudo_type == "PAW",
        has_so=relativistic == "full",
        core_correction=core_correction,
        wfc_cutoff_ry=wfc or None,
        rho_cutoff_ry=rho or None,
        sha256=hashlib.sha256(text).hexdigest(),
        relativistic=relativistic,
        upf_version=1,
    )


def parse_upf_header(path: str | Path) -> UPFHeader:
    """Read identity from the UPF ``PP_HEADER`` (v2 attributes or v1 lines), not the file name."""
    path = Path(path)
    text = path.read_bytes()
    match = re.search(rb"<PP_HEADER(\s[^>]*?)/?>", text, re.S)
    if match is None:
        return _parse_upf_v1(path, text)
    attrs = {k: v for k, v in _ATTR.findall(match.group(1).decode("latin-1"))}
    try:
        z_valence = float(attrs["z_valence"])
        element = attrs["element"].strip()
    except (KeyError, ValueError) as exc:
        raise UPFError(f"{path.name}: header lacks element or z_valence") from exc

    def number(key: str) -> float | None:
        return float(attrs[key]) if key in attrs else None

    # ONCVPSP writes rho_cutoff as a radius (bohr) and no wfc_cutoff; only a pair is a Ry hint.
    wfc_cutoff = number("wfc_cutoff")
    rho_cutoff = number("rho_cutoff") if wfc_cutoff is not None else None

    def flag(key: str) -> bool:
        return attrs.get(key, "false").strip().lower() in ("true", ".true.", "t")

    return UPFHeader(
        element=element,
        z_valence=z_valence,
        functional=" ".join(attrs.get("functional", "").split()),
        pseudo_type=attrs.get("pseudo_type", "").strip(),
        is_paw=flag("is_paw"),
        has_so=flag("has_so"),
        core_correction=flag("core_correction"),
        wfc_cutoff_ry=wfc_cutoff,
        rho_cutoff_ry=rho_cutoff,
        sha256=hashlib.sha256(text).hexdigest(),
        relativistic=attrs.get("relativistic", "").strip(),
    )
