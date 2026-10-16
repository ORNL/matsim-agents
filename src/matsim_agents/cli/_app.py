"""Root Typer application and shared console for the matsim-agents CLI."""

import typer
from rich.console import Console

app = typer.Typer(add_completion=False, help="Multi-agent AI for atomistic materials simulation.")
console = Console()
