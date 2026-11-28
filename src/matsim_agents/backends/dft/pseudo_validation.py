"""Pseudopotential qualification for every Quantum ESPRESSO input the package writes.

One function, :func:`validate_pseudopotentials`, is called by the ``pw.x`` and ``neb.x`` input
writers. The minimum cutoffs are the maximum over the selected pseudopotentials; they are
read from the UPF header (or the library manifest) and never guessed. A missing recommendation
is reported as ``unavailable`` and is not replaced by a default.

A configuration below a recommendation is rejected. ``override_reason`` allows it for a
documented diagnostic run; the result is then ``diagnostic_only`` and can never qualify for
production.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from matsim_agents.core.upf import UPFError, UPFHeader, canonical_functional, parse_upf_header
from matsim_agents.core.yamlio import load_yaml_mapping


class PseudopotentialError(ValueError):
    """The pseudopotential set cannot be used for the requested calculation."""


@dataclass(frozen=True)
class PseudoCutoffs:
    element: str
    filename: str
    ecutwfc_ry: float | None
    ecutrho_ry: float | None
    source: str  # "upf_header" | "manifest" | "unavailable"


@dataclass
class PseudoValidation:
    minimum_ecutwfc_ry: float | None
    minimum_ecutrho_ry: float | None
    configured_ecutwfc_ry: float
    configured_ecutrho_ry: float
    per_element: dict[str, PseudoCutoffs]
    problems: list[str] = field(default_factory=list)
    unavailable: list[str] = field(default_factory=list)
    override_reason: str | None = None

    @property
    def below_recommendation(self) -> bool:
        return any("below the pseudopotential recommendation" in p for p in self.problems)

    @property
    def diagnostic_only(self) -> bool:
        return self.override_reason is not None and bool(self.problems)

    @property
    def qualified(self) -> bool:
        """True only when nothing was overridden, nothing is missing, and every cutoff is known."""
        return not self.problems and not self.unavailable and self.override_reason is None

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["qualified"] = self.qualified
        out["diagnostic_only"] = self.diagnostic_only
        return out


def load_manifest(path: str | Path) -> dict[str, Any]:
    data = load_yaml_mapping(path)
    if not isinstance(data, dict) or not isinstance(data.get("pseudopotentials"), dict):
        raise PseudopotentialError(f"{path}: not a pseudopotential manifest")
    return data


def _manifest_cutoffs(entry: Mapping[str, Any]) -> tuple[float | None, float | None]:
    wfc = entry.get("recommended_ecutwfc_ry", entry.get("sssp_cutoff_wfc_ry"))
    rho = entry.get("recommended_ecutrho_ry", entry.get("sssp_cutoff_rho_ry"))
    return (
        _positive_cutoff(wfc),
        _positive_cutoff(rho),
    )


def _positive_cutoff(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) and value > 0 else None


def validate_pseudopotentials(
    pseudopotentials: Mapping[str, str],
    pseudo_dir: str | Path,
    *,
    ecutwfc_ry: float,
    ecutrho_ry: float,
    manifest: Mapping[str, Any] | None = None,
    functional: str = "PBE",
    override_reason: str | None = None,
) -> PseudoValidation:
    """Check files, identity, checksums, functional and cutoffs for ``{element: filename}``."""
    problems: list[str] = []
    unavailable: list[str] = []
    per_element: dict[str, PseudoCutoffs] = {}
    entries = (manifest or {}).get("pseudopotentials", {})
    for element, filename in sorted(pseudopotentials.items()):
        path = Path(pseudo_dir) / filename
        if not path.is_file():
            problems.append(f"{element}: pseudopotential file not found: {path}")
            continue
        try:
            header: UPFHeader = parse_upf_header(path)
        except UPFError as exc:
            problems.append(f"{element}: {exc}")
            continue
        if header.element.strip().capitalize() != element.capitalize():
            problems.append(f"{element}: file {filename} is for element {header.element!r}")
        if canonical_functional(header.functional) != canonical_functional(functional):
            problems.append(f"{element}: functional {header.functional!r} is not {functional}")
        wfc = _positive_cutoff(header.wfc_cutoff_ry)
        rho = _positive_cutoff(header.rho_cutoff_ry)
        source = "upf_header"
        entry = entries.get(element)
        if manifest is not None:
            if entry is None:
                problems.append(f"{element}: not listed in the pseudopotential manifest")
            else:
                if entry.get("filename") != filename:
                    problems.append(f"{element}: manifest lists {entry.get('filename')!r}")
                if entry.get("sha256") != header.sha256:
                    problems.append(f"{element}: checksum differs from the manifest")
                m_wfc, m_rho = _manifest_cutoffs(entry)
                if wfc is None and m_wfc is not None:
                    wfc, source = m_wfc, "manifest"
                if rho is None and m_rho is not None:
                    rho, source = m_rho, "manifest"
        if wfc is None or rho is None:
            unavailable.append(element)
            source = "unavailable"
        per_element[element] = PseudoCutoffs(element, filename, wfc, rho, source)
    known_wfc = [c.ecutwfc_ry for c in per_element.values() if c.ecutwfc_ry is not None]
    known_rho = [c.ecutrho_ry for c in per_element.values() if c.ecutrho_ry is not None]
    min_wfc = max(known_wfc) if known_wfc else None
    min_rho = max(known_rho) if known_rho else None
    for label, configured, minimum in (
        ("ecutwfc", ecutwfc_ry, min_wfc),
        ("ecutrho", ecutrho_ry, min_rho),
    ):
        if _positive_cutoff(configured) is None:
            problems.append(f"{label} must be a positive finite cutoff")
        elif minimum is not None and configured < minimum - 1e-9:
            problems.append(
                f"{label} {configured:g} Ry is below the pseudopotential recommendation "
                f"{minimum:g} Ry"
            )
    return PseudoValidation(
        minimum_ecutwfc_ry=min_wfc,
        minimum_ecutrho_ry=min_rho,
        configured_ecutwfc_ry=ecutwfc_ry,
        configured_ecutrho_ry=ecutrho_ry,
        per_element=per_element,
        problems=problems,
        unavailable=unavailable,
        override_reason=override_reason,
    )


def require_valid_pseudopotentials(
    pseudopotentials: Mapping[str, str],
    pseudo_dir: str | Path,
    **kwargs: Any,
) -> PseudoValidation:
    """Raise unless valid; with ``override_reason`` set, only cutoff shortfalls are tolerated."""
    result = validate_pseudopotentials(pseudopotentials, pseudo_dir, **kwargs)
    tolerated = bool(result.override_reason and result.override_reason.strip())
    blocking = [
        p
        for p in result.problems
        if not (tolerated and "below the pseudopotential recommendation" in p)
    ]
    if result.unavailable:
        blocking.append(f"cutoff recommendations unavailable for: {', '.join(result.unavailable)}")
    if blocking:
        raise PseudopotentialError(
            "pseudopotential validation failed: "
            + "; ".join(blocking)
            + ". A documented diagnostic override covers cutoff shortfalls only."
        )
    return result


def main(argv: list[str] | None = None) -> int:
    """``python -m matsim_agents.backends.dft.pseudo_validation``: validate one cutoff pair."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("pseudo_dir")
    parser.add_argument("--map", action="append", required=True, metavar="EL=FILE")
    parser.add_argument("--ecutwfc", type=float, required=True)
    parser.add_argument("--ecutrho", type=float, required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--functional", default="PBE")
    parser.add_argument("--override-reason")
    args = parser.parse_args(argv)
    mapping = dict(item.split("=", 1) for item in args.map)
    result = validate_pseudopotentials(
        mapping,
        os.path.expanduser(args.pseudo_dir),
        ecutwfc_ry=args.ecutwfc,
        ecutrho_ry=args.ecutrho,
        manifest=load_manifest(args.manifest) if args.manifest else None,
        functional=args.functional,
        override_reason=args.override_reason,
    )
    print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
    return 0 if result.qualified else 1


if __name__ == "__main__":
    raise SystemExit(main())
