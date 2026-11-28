"""Verify an installed SSSP library against its official JSON and write a versioned manifest.

Usage:
  python scripts/utils/build_pseudo_manifest.py LIBRARY_DIR --sssp-json SSSP.json \
      --library-id sssp-1.3.0-pbe-precision --source-url URL --archive-md5 MD5 \
      --element W --element Ta ... --out manifest.yaml

Every UPF is checked against the official MD5; identity (element, functional, type, valence,
relativistic treatment) is read from the UPF header, never from the file name. Any mismatch or
missing element exits non-zero and no manifest is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml

from matsim_agents.core.upf import canonical_functional, parse_upf_header


def build(args: argparse.Namespace) -> tuple[dict, list[str]]:
    library = Path(args.library_dir)
    official = json.loads(Path(args.sssp_json).read_text())
    errors: list[str] = []
    entries: dict[str, dict] = {}
    for element in args.element:
        meta = official.get(element)
        if meta is None:
            errors.append(f"{element}: not in {Path(args.sssp_json).name}")
            continue
        path = library / meta["filename"]
        if not path.is_file():
            errors.append(f"{element}: {meta['filename']} missing from {library}")
            continue
        data = path.read_bytes()
        md5 = hashlib.md5(data).hexdigest()  # noqa: S324 - matches the published checksum
        if md5 != meta["md5"]:
            errors.append(f"{element}: md5 {md5} != official {meta['md5']}")
        header = parse_upf_header(path)
        if header.element.strip().lower() != element.lower():
            errors.append(f"{element}: UPF header says element {header.element!r}")
        if canonical_functional(header.functional) != "PBE":
            errors.append(f"{element}: UPF functional {header.functional!r} is not PBE")
        entries[element] = {
            "filename": meta["filename"],
            "sha256": header.sha256,
            "md5": md5,
            "md5_official": meta["md5"],
            "source_family": meta["pseudopotential"],
            "sssp_cutoff_wfc_ry": meta["cutoff_wfc"],
            "sssp_cutoff_rho_ry": meta["cutoff_rho"],
            "upf_version": header.upf_version,
            "upf_element": header.element.strip(),
            "upf_pseudo_type": header.pseudo_type,
            "upf_is_paw": header.is_paw,
            "upf_functional": header.functional,
            "upf_z_valence": header.z_valence,
            "upf_relativistic": header.relativistic or "unstated",
            "upf_relativistic_source": (
                "PP_HEADER relativistic attribute"
                if header.upf_version == 2
                else "PP_INFO generation text (UPF v1 has no field; QE upfconv writes 'no')"
            ),
            "upf_has_so": header.has_so,
            "upf_core_correction": header.core_correction,
        }
        if header.wfc_cutoff_ry is not None and header.rho_cutoff_ry is not None:
            entries[element]["upf_suggested_wfc_ry"] = header.wfc_cutoff_ry
            entries[element]["upf_suggested_rho_ry"] = header.rho_cutoff_ry
    wfc = max((e["sssp_cutoff_wfc_ry"] for e in entries.values()), default=None)
    rho = max((e["sssp_cutoff_rho_ry"] for e in entries.values()), default=None)
    manifest = {
        "library_id": args.library_id,
        "library_dir": str(library.resolve()),
        "provenance": {
            "source_url": args.source_url,
            "doi": args.doi,
            "archive": args.archive_name,
            "archive_md5_official": args.archive_md5,
            "metadata_json": Path(args.sssp_json).name,
            "metadata_json_sha256": hashlib.sha256(Path(args.sssp_json).read_bytes()).hexdigest(),
            "license": args.license,
            "manifest_generated_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "pseudo_types": sorted({e["upf_pseudo_type"] for e in entries.values()}),
        "campaign_minimum_cutoffs_ry": {
            "ecutwfc": wfc,
            "ecutrho": rho,
            "note": "max over listed elements of SSSP-recommended cutoffs; not a convergence test",
        },
        "pseudopotentials": entries,
    }
    return manifest, errors


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("library_dir")
    p.add_argument("--sssp-json", required=True)
    p.add_argument("--library-id", required=True)
    p.add_argument("--source-url", required=True)
    p.add_argument("--doi", default="")
    p.add_argument("--archive-name", default="")
    p.add_argument("--archive-md5", default="")
    p.add_argument("--license", default="")
    p.add_argument("--element", action="append", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--sha256sums", help="also write a sha256sum -c compatible file here")
    args = p.parse_args(argv)
    manifest, errors = build(args)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    Path(args.out).write_text(yaml.safe_dump(manifest, sort_keys=False))
    if args.sha256sums:
        Path(args.sha256sums).write_text(
            "".join(
                f"{e['sha256']}  {e['filename']}\n" for e in manifest["pseudopotentials"].values()
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
