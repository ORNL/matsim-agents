#!/usr/bin/env python3
"""Create an explicitly experimental Aurora-only FairChem compatibility wheel."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from email import policy
from email.parser import BytesParser
from pathlib import Path

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version
from wheel.wheelfile import WheelFile


def adapt_xpu_source(source: str) -> str:
    tree = ast.parse(source)
    predictor = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MLIPPredictUnit"
    )
    methods = {node.name: node for node in predictor.body if isinstance(node, ast.FunctionDef)}
    setup = methods["_setup_device"]
    assertion = next(node for node in setup.body if isinstance(node, ast.Assert))
    load = next(
        node
        for node in ast.walk(methods["__init__"])
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "_load_model"
    )
    settings = (
        '        if torch.device(device).type == "xpu":\n'
        "            if not torch.xpu.is_available():\n"
        '                raise RuntimeError("XPU unavailable; CPU fallback is forbidden")\n'
        "            self.inference_settings.compile = False\n"
        "            self.inference_settings.tf32 = False\n"
        "            self.inference_settings.merge_mole = False\n"
        "            self.inference_settings.execution_mode = ExecutionMode.GENERAL\n"
    )
    device = (
        '        if torch.device(device).type == "xpu":\n'
        "            self.device = torch.device(device)\n"
        "            return\n"
    )
    lines = source.splitlines(keepends=True)
    for position, addition in sorted(
        ((load.lineno, settings), (assertion.lineno, device)), reverse=True
    ):
        lines.insert(position - 1, addition)
    updated = "".join(lines)
    compile(updated, "fairchem-xpu-predict", "exec")
    return updated


def prepare_wheel(source: Path, output: Path, torch_version: str, enable_xpu: bool = False) -> Path:
    name, version, build, _ = parse_wheel_filename(source.name)
    if canonicalize_name(name) != "fairchem-core" or version not in SpecifierSet(">=2.22,<2.24"):
        raise ValueError("Only FairChem 2.22/2.23 Torch-2.13 profiles are supported")
    if build or Version(torch_version).release[:2] != (2, 13):
        raise ValueError("Expected an upstream wheel and Aurora Torch 2.13")
    parts = source.name.split("-")
    parts.insert(2, "2auroraxpu" if enable_xpu else "1aurora")
    output.mkdir(parents=True, exist_ok=True)
    target = output / "-".join(parts)
    if target.resolve() == source.resolve():
        raise ValueError("The upstream wheel must not be overwritten")
    with WheelFile(source) as upstream:
        metadata_paths = [
            name for name in upstream.namelist() if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_paths) != 1:
            raise ValueError("Expected exactly one wheel METADATA file")
        metadata_path = metadata_paths[0]
        metadata = BytesParser(policy=policy.compat32).parsebytes(upstream.read(metadata_path))
        requirements = metadata.get_all("Requires-Dist", [])
        replacements = []
        original_torch = []
        for value in requirements:
            requirement = Requirement(value)
            if canonicalize_name(requirement.name) == "torch":
                if not requirement.specifier.contains("2.13.0"):
                    raise ValueError(f"Unexpected upstream Torch requirement: {requirement}")
                original_torch.append(value)
                requirement.specifier = SpecifierSet(f"=={torch_version}")
                value = str(requirement)
            replacements.append(value)
        if len(original_torch) != 1:
            raise ValueError("Expected exactly one upstream Torch requirement")
        if any(name.endswith(("/RECORD.jws", "/RECORD.p7s")) for name in upstream.namelist()):
            raise ValueError("Refusing to rewrite a signed wheel")
        del metadata["Requires-Dist"]
        for value in replacements:
            metadata["Requires-Dist"] = value
        predictor_path = "fairchem/core/units/mlip_unit/predict.py"
        if enable_xpu and version != Version("2.23.0"):
            raise ValueError("XPU source adaptation requires the FairChem 2.23.0 layout")
        if enable_xpu and predictor_path not in upstream.namelist():
            raise ValueError("FairChem wheel is missing its inference device controller")
        with WheelFile(target, "w") as patched:
            for entry in upstream.infolist():
                if entry.filename.endswith(".dist-info/RECORD"):
                    continue
                data = (
                    metadata.as_bytes(policy=policy.compat32.clone(max_line_length=0))
                    if entry.filename == metadata_path
                    else upstream.read(entry)
                )
                if enable_xpu and entry.filename == predictor_path:
                    data = adapt_xpu_source(data.decode()).encode()
                patched.writestr(entry, data)
    manifest = {
        "experimental": True,
        "upstream_wheel": source.name,
        "upstream_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "patched_wheel": target.name,
        "patched_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "original_torch_requirement": original_torch[0],
        "aurora_torch_version": torch_version,
        "experimental_xpu_execution": enable_xpu,
    }
    target.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--torch-version", required=True)
    parser.add_argument("--enable-xpu", action="store_true")
    args = parser.parse_args()
    print(prepare_wheel(args.wheel, args.output, args.torch_version, args.enable_xpu))


if __name__ == "__main__":
    main()
