#!/usr/bin/env python3
"""Stage selected HydraGNN code with current upstream dependency metadata."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version


def prepare_source(code: Path, dependencies: Path, output_root: Path, torch_version: str) -> Path:
    if Version(torch_version).release[:2] not in {(2, 13), (2, 14)}:
        raise ValueError(f"Unsupported Aurora Torch series: {torch_version}")
    code, dependencies, output_root = code.resolve(), dependencies.resolve(), output_root.resolve()
    if output_root == code or code in output_root.parents:
        raise ValueError("Staged source must be outside the selected HydraGNN checkout")
    for name in ("requirements-base.txt", "requirements-torch.txt", "requirements-pyg.txt"):
        if not (dependencies / name).is_file():
            raise ValueError(f"Upstream dependency profile is missing {name}")
    files = subprocess.check_output(["git", "-C", str(code), "ls-files", "-z"]).split(b"\0")
    output_root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="hydragnn-source-", dir=output_root))
    try:
        for entry in files:
            if not entry:
                continue
            relative = Path(entry.decode())
            source = code / relative
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe tracked source path: {relative}")
            if source.is_file():
                target = stage / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
        package_root = stage / "hydragnn"
        for source in package_root.rglob("*.py"):
            directory = source.parent
            while directory == package_root or package_root in directory.parents:
                (directory / "__init__.py").touch(exist_ok=True)
                directory = directory.parent
        for source in dependencies.glob("requirements*.txt"):
            shutil.copy2(source, stage / source.name)
        installer_path = Path("scripts/hpc/alcf/aurora/installation/install.sh")
        installer = dependencies / installer_path
        if installer.is_file():
            text, replacements = re.subn(
                r"(?m)^([ \t]*)module load frameworks[ \t]*$",
                r'\1module load "${FRAMEWORKS_MODULE:-frameworks}"',
                installer.read_text(),
            )
            if replacements != 1 and 'module load "${FRAMEWORKS_MODULE:-frameworks}"' not in text:
                raise ValueError("Unknown upstream Aurora framework module selection")
            target = stage / installer_path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(installer, target)
            target.write_text(text)
        requirement_path = stage / "requirements-torch.txt"
        requirements = []
        for line in requirement_path.read_text().splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith(("#", "-")):
                requirement = Requirement(stripped)
                if requirement.name.lower().replace("_", "-") == "torch":
                    line = f"torch=={torch_version}"
                elif requirement.name.lower().replace("_", "-") == "torchvision":
                    line = "torchvision>=0.28.0.dev0,<0.30"
            requirements.append(line)
        requirement_path.write_text("\n".join(requirements) + "\n")
        return stage
    except Exception:
        shutil.rmtree(stage)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code", required=True, type=Path)
    parser.add_argument("--dependencies", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--torch-version", required=True)
    args = parser.parse_args()
    print(prepare_source(args.code, args.dependencies, args.output_root, args.torch_version))


if __name__ == "__main__":
    main()
