#!/usr/bin/env python3
"""Guard MACE's optional OpenEquivariance import on non-CUDA/HIP Torch builds."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path


def guard_openequivariance(path: Path) -> bool:
    source = path.read_text()
    imports = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Import)
        and any(alias.name == "openequivariance" for alias in node.names)
    ]
    if len(imports) != 1:
        raise ValueError("Expected exactly one MACE OpenEquivariance import")
    node = imports[0]
    lines = source.splitlines(keepends=True)
    line = lines[node.lineno - 1]
    if (
        node.lineno != node.end_lineno
        or len(node.names) != 1
        or node.names[0].asname not in {"oeq", "_oeq"}
    ):
        raise ValueError("Unsupported MACE OpenEquivariance import layout")
    indentation = line[: len(line) - len(line.lstrip())]
    if indentation.endswith("    "):
        parent_indent = indentation[:-4]
        guard_line = f"{parent_indent}if torch.version.cuda or torch.version.hip:\n"
        if node.lineno > 1 and lines[node.lineno - 2] == guard_line:
            return False
    replacement = (
        f"{indentation}if torch.version.cuda or torch.version.hip:\n"
        f"{indentation}    {line.strip()}\n"
        f"{indentation}else:\n"
        f'{indentation}    raise ImportError("OpenEquivariance requires CUDA or HIP Torch")\n'
    )
    lines[node.lineno - 1] = replacement
    updated = "".join(lines)
    compile(updated, str(path), "exec")
    path.write_text(updated)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, required=True)
    args = parser.parse_args()
    site = args.site.resolve()
    for relative in ("mace/modules/wrapper_ops.py", "mace/cli/convert_e3nn_hybrid.py"):
        path = (site / relative).resolve()
        if site not in path.parents:
            raise ValueError("Refusing to patch MACE outside its compatibility environment")
        changed = guard_openequivariance(path)
        status = "installed" if changed else "already installed"
        print(f"MACE optional backend guard {status}: {relative}")


if __name__ == "__main__":
    main()
