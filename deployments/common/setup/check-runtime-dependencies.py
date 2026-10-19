#!/usr/bin/env python3
"""Check dependency metadata reachable from selected runtime requirements."""

from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, distribution

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def check_dependencies(roots: list[str]) -> list[str]:
    pending = [Requirement(root) for root in roots]
    visited = set()
    errors = set()
    while pending:
        requirement = pending.pop()
        try:
            installed = distribution(requirement.name)
        except PackageNotFoundError:
            errors.add(f"Missing runtime dependency: {requirement}")
            continue
        if not requirement.specifier.contains(installed.version, prereleases=True):
            errors.add(f"{requirement} is required, installed {installed.version}")
        for extra in requirement.extras | {""}:
            key = (canonicalize_name(requirement.name), extra)
            if key in visited:
                continue
            visited.add(key)
            for value in installed.requires or []:
                dependency = Requirement(value)
                if dependency.marker is None or dependency.marker.evaluate({"extra": extra}):
                    pending.append(dependency)
    return sorted(errors)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("requirements", nargs="+")
    errors = check_dependencies(parser.parse_args().requirements)
    if errors:
        raise SystemExit("\n".join(errors))
    print("Selected runtime dependency metadata verified")


if __name__ == "__main__":
    main()
