"""Build a pseudopotential manifest from UPF headers (no external metadata file needed).

Every field comes from the UPF file itself or from an explicit argument. A value the header
does not state is written as ``unavailable``; nothing is inferred.

    python scripts/utils/build_pslibrary_manifest.py DIR --library-id ID --library "PSlibrary" \
        --version 1.0.0 --source-url-template 'https://host/{filename}' \
        --element W --element Ta --out manifest.yaml
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
from pathlib import Path

import yaml

from matsim_agents.core.fileio import atomic_write_text
from matsim_agents.core.upf import canonical_functional, parse_upf_header

UNAVAILABLE = "unavailable"
_CONFIG = re.compile(rb"config\s*=\s*'([^']*)'")
_GENERATED = re.compile(rb"Generated using \"?([^\n\"]*)")


def entry(path: Path, element: str, url_template: str) -> dict:
    header = parse_upf_header(path)
    if header.element.strip().capitalize() != element:
        raise ValueError(f"{path.name}: header element {header.element!r}, expected {element}")
    raw = path.read_bytes()
    config = _CONFIG.search(raw)
    return {
        "filename": path.name,
        "sha256": header.sha256,
        "source_url": url_template.format(filename=path.name),
        "xc_functional": canonical_functional(header.functional),
        "pseudo_type": header.pseudo_type or UNAVAILABLE,
        "is_paw": header.is_paw,
        "relativistic": header.relativistic or UNAVAILABLE,
        "has_spin_orbit": header.has_so,
        "z_valence": header.z_valence,
        "valence_configuration": config.group(1).decode("latin-1") if config else UNAVAILABLE,
        "nonlinear_core_correction": header.core_correction,
        "recommended_ecutwfc_ry": header.wfc_cutoff_ry
        if header.wfc_cutoff_ry is not None
        else UNAVAILABLE,
        "recommended_ecutrho_ry": header.rho_cutoff_ry
        if header.rho_cutoff_ry is not None
        else UNAVAILABLE,
        "upf_version": header.upf_version,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("library_dir")
    parser.add_argument("--library-id", required=True)
    parser.add_argument("--library", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--source-url-template", required=True)
    parser.add_argument("--library-dir-var", required=True)
    parser.add_argument("--element", action="append", required=True)
    parser.add_argument("--note", default="")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    directory = Path(args.library_dir)
    pseudos = {}
    for element in args.element:
        matches = sorted(directory.glob(f"{element}.*UPF")) + sorted(directory.glob(f"{element}_*"))
        if len(matches) != 1:
            print(f"{element}: expected one UPF file in {directory}, found {len(matches)}")
            return 2
        pseudos[element] = entry(matches[0], element, args.source_url_template)
    manifest = {
        "library_id": args.library_id,
        "library_dir": "${" + args.library_dir_var + "}",
        "provenance": {
            "library": args.library,
            "version": args.version,
            "note": args.note or UNAVAILABLE,
            "manifest_generated_utc": dt.datetime.now(dt.timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
        },
        "campaign_minimum_cutoffs_ry": {
            "ecutwfc": max(
                p["recommended_ecutwfc_ry"]
                for p in pseudos.values()
                if p["recommended_ecutwfc_ry"] != UNAVAILABLE
            ),
            "ecutrho": max(
                p["recommended_ecutrho_ry"]
                for p in pseudos.values()
                if p["recommended_ecutrho_ry"] != UNAVAILABLE
            ),
            "note": "max over listed elements of the UPF-header recommendation; not a "
            "convergence result",
        },
        "pseudopotentials": pseudos,
    }
    atomic_write_text(Path(args.out), yaml.safe_dump(manifest, sort_keys=False))
    print(f"wrote {args.out} ({len(pseudos)} elements)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
