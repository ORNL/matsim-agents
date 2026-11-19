"""Repository-wide CLI diagnostics."""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from typing import Any

import typer

from matsim_agents.cli._app import app, console, package_version
from matsim_agents.execution.capabilities import check_all


@app.command("doctor")
def doctor(
    as_json: bool = typer.Option(False, "--json", help="Print the diagnostics as JSON."),
    require: list[str] = typer.Option(
        None, "--require", help="Exit non-zero unless this backend is available (repeatable)."
    ),
) -> None:
    """Inspect Python, package, and backend readiness without loading a model."""
    results = check_all()
    by_name = {result.name: result for result in results}
    unknown = sorted(set(require or ()) - by_name.keys())
    if unknown:
        console.print(
            f"Unknown required backend(s): {', '.join(unknown)}", style="red", markup=False
        )
        raise typer.Exit(2)

    counts = {
        kind: {
            "available": sum(item.kind == kind and item.available for item in results),
            "configured": sum(item.kind == kind for item in results),
        }
        for kind in ("dft", "mlip", "llm")
    }
    required = {
        name: {
            "available": by_name[name].available,
            "missing": list(by_name[name].missing),
        }
        for name in require or ()
    }
    report: dict[str, Any] = {
        "package": "matsim-agents",
        "version": package_version(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "working_directory": str(Path.cwd()),
        "python_supported": (3, 11) <= sys.version_info[:2] < (3, 15),
        "backends": counts,
        "required": required,
        "capabilities": [result.model_dump(mode="json") for result in results],
    }
    if as_json:
        typer.echo(json.dumps(report, indent=2))
    else:
        console.print(f"matsim-agents {report['version']}", style="bold")
        console.print(f"Python: {report['python']} ({report['platform']})", markup=False)
        console.print(
            "Python support: " + ("passed" if report["python_supported"] else "failed"),
            markup=False,
        )
        console.print(f"Working directory: {report['working_directory']}", markup=False)
        for kind, count in counts.items():
            console.print(
                f"{kind.upper()} backends: {count['available']}/{count['configured']} available",
                markup=False,
            )
        for item in results:
            status = "available" if item.available else "missing"
            detail = f" ({'; '.join(item.missing)})" if item.missing else ""
            console.print(f"  {item.kind:4} {item.name:12} {status}{detail}", markup=False)
        console.print("No model, scientific workflow, or scheduler job was started.")

    if not report["python_supported"] or any(not item["available"] for item in required.values()):
        raise typer.Exit(1)
