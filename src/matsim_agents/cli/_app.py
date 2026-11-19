"""Root Typer application and shared console for the matsim-agents CLI."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import NoReturn

import click
import typer
from pydantic import ValidationError
from rich.console import Console
from typer.core import TyperGroup

console = Console()
error_console = Console(stderr=True)


@dataclass
class CLIState:
    debug: bool = False


def describe_error(error: BaseException) -> str:
    if isinstance(error, ValidationError):
        lines = [f"invalid {error.title} ({error.error_count()} problem(s)):"]
        for item in error.errors():
            location = ".".join(str(part) for part in item["loc"]) or "(top level)"
            lines.append(f"  {location}: {item['msg']}")
        return "\n".join(lines)
    return str(error)


class FriendlyTyperGroup(TyperGroup):
    def invoke(self, ctx: click.Context):
        try:
            return super().invoke(ctx)
        except (click.ClickException, click.Abort, typer.Exit):
            raise
        except (KeyError, OSError, RuntimeError, ValueError, ValidationError) as error:
            state = ctx.find_root().ensure_object(CLIState)
            if state.debug:
                raise
            error_console.print(
                f"Error: {describe_error(error)}", style="red", markup=False, highlight=False
            )
            error_console.print(
                "Hint: rerun with --debug before the command for a traceback.", markup=False
            )
            raise typer.Exit(2) from None


app = typer.Typer(
    add_completion=True,
    cls=FriendlyTyperGroup,
    help=(
        "Multi-agent AI for atomistic materials simulation.\n\n"
        "Start with `matsim-agents workflow list`, `workflow check CONFIG.yaml`, "
        "or `explore check CONFIG.yaml`."
    ),
    no_args_is_help=True,
    pretty_exceptions_enable=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)


def package_version() -> str:
    try:
        return version("matsim-agents")
    except PackageNotFoundError:
        return "development"


def fail(message: str, *, code: int = 1) -> NoReturn:
    error_console.print(f"Error: {message}", style="red", markup=False, highlight=False)
    raise typer.Exit(code)


def version_callback(value: bool) -> bool:
    if value:
        typer.echo(package_version())
        raise typer.Exit()
    return value


@app.callback()
def main_options(
    ctx: typer.Context,
    debug: bool = typer.Option(
        False, "--debug", help="Show Python tracebacks for unexpected command failures."
    ),
    show_version: bool = typer.Option(
        False,
        "--version",
        is_eager=True,
        callback=version_callback,
        help="Show the installed version and exit.",
    ),
) -> None:
    ctx.ensure_object(CLIState).debug = debug
