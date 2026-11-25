"""Generate the 1,024-atom tungsten reference alloy from its YAML configuration.

Usage (from the repository root):
    PYTHONPATH=src .venv/bin/python examples/fusion_damage/generate_reference_alloy.py \
        --output /tmp/w_reference.extxyz
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import yaml
from ase.io import write

from matsim_agents.discovery.alloys import AlloySpec, generate_alloy

CONFIG = Path(__file__).with_name("tungsten_reference_alloy.yaml")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    spec = AlloySpec.model_validate(yaml.safe_load(args.config.read_text()))
    atoms = generate_alloy(spec)
    if Counter(atoms.get_chemical_symbols()) != Counter(spec.counts):
        raise SystemExit("generated species counts do not match the configuration")

    write(args.output, atoms, format="extxyz")
    print(f"wrote {len(atoms)} atoms to {args.output}")
    print("counts:", dict(Counter(atoms.get_chemical_symbols())))


if __name__ == "__main__":
    main()
