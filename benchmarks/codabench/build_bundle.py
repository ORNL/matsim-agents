#!/usr/bin/env python3
"""Build the exact self-contained Codabench upload archive."""

from __future__ import annotations

import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path

from validate_bundle import validate_bundle

HERE = Path(__file__).resolve().parent
STATIC_ENTRIES = (
    "competition.yaml",
    "logo.png",
    "pages",
    "scoring_program",
    "starting_kit",
)


def _copy(source: Path, target: Path) -> None:
    if source.is_dir():
        shutil.copytree(source, target, dirs_exist_ok=True)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def build_bundle(output: Path, public_data: Path, reference_data: Path) -> Path:
    """Assemble and validate an upload without modifying the source tree."""
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="matsim-codabench-") as temp:
        stage = Path(temp) / "competition"
        stage.mkdir()
        for rel in STATIC_ENTRIES:
            _copy(HERE / rel, stage / rel)
        _copy(public_data.resolve(), stage / "public_data")
        _copy(reference_data.resolve(), stage / "reference_data")

        # Participant-facing utilities are deliberately copied into the
        # starting kit because Codabench does not include sibling directories.
        for rel in (
            "baselines",
            "run_baselines.py",
            "package_submission.py",
            "requirements.txt",
            "requirements-mace.txt",
            "requirements-fairchem.txt",
        ):
            _copy(HERE / rel, stage / "starting_kit" / rel)

        errors = validate_bundle(stage, release=True)
        if errors:
            raise ValueError("Invalid Codabench release bundle:\n  - " + "\n  - ".join(errors))

        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(stage.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(stage).as_posix())
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--public-data", required=True, type=Path)
    parser.add_argument("--reference-data", required=True, type=Path)
    args = parser.parse_args()
    print(build_bundle(args.output, args.public_data, args.reference_data))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
