"""Resource-dispatch subcommand group: ``matsim-agents resources ...``."""

from __future__ import annotations

from pathlib import Path

import typer

from matsim_agents.cli._app import app, console

resources_app = typer.Typer(
    add_completion=False,
    help="Preview and validate how LLM, MLIP, DFT, and training work is placed on nodes.",
)
app.add_typer(resources_app, name="resources")

PLAN_ARGUMENT = typer.Argument(
    ..., help="Workflow config or plan file with a top-level 'resources:' block."
)
NODES_OPTION = typer.Option(
    None, "--nodes", min=1, help="Preview on N synthetic nodes instead of the current allocation."
)
HOSTS_OPTION = typer.Option(
    None, "--hosts", help="Preview on these comma-separated hosts instead of the allocation."
)


def _allocation(nodes: int | None, hosts: str | None):
    from matsim_agents.execution.allocation import Allocation, discover_allocation

    if nodes is not None and hosts is not None:
        raise typer.BadParameter("use either --nodes or --hosts, not both")
    if hosts is not None:
        names = tuple(host.strip() for host in hosts.split(",") if host.strip())
        if not names:
            raise typer.BadParameter("--hosts needs at least one host name")
        return Allocation(scheduler="preview", nodes=names)
    if nodes is not None:
        return Allocation(
            scheduler="preview", nodes=tuple(f"node{index:03d}" for index in range(nodes))
        )
    return discover_allocation()


def _resolve(config: Path, nodes: int | None, hosts: str | None):
    from matsim_agents.execution.placement import ResourcePlan

    try:
        return ResourcePlan.from_yaml(config).resolve(_allocation(nodes, hosts))
    except ValueError as exc:
        console.print(f"[red]resource plan {config} is invalid:[/red]\n{exc}", markup=True)
        raise typer.Exit(1) from None


def _print_placement(placement) -> None:
    from rich.table import Table

    geometry = placement.geometry
    console.print(
        f"[bold]{placement.preset or 'custom geometry'}[/bold]: {len(placement.nodes)} node(s) "
        f"from {placement.scheduler}, {geometry.devices_per_node} {geometry.vendor} device(s) "
        f"and {geometry.cores_per_node} cores per node, launcher {placement.launcher}"
    )
    table = Table(show_lines=False)
    for column in ("pool", "server", "hosts", "devices", "details"):
        table.add_column(column)
    for pool in placement.pools.values():
        details = []
        if pool.share_with:
            details.append(f"shares with {', '.join(pool.share_with)}")
        if pool.memory_fraction is not None:
            details.append(f"memory_fraction {pool.memory_fraction:g}")
        if pool.max_concurrent is not None:
            details.append(f"{pool.max_concurrent} concurrent")
        if pool.launches:
            launch = pool.launches[0]
            details.append(
                f"{len(pool.launches)} launch(es) x {launch.node_count} node(s), "
                f"{launch.ranks_per_node} ranks x {launch.threads_per_rank} threads per node"
            )
        table.add_row(pool.name, "", _hosts(pool.slices), _devices(pool.slices), "; ".join(details))
        for server in pool.servers:
            if server.kind == "external":
                server_details = f"external {server.base_url}"
            else:
                server_details = f"{server.kind} {server.served_name}"
            if server.tensor_parallel_size is not None:
                server_details += (
                    f", tp {server.tensor_parallel_size} x pp {server.pipeline_parallel_size}"
                )
            if server.max_model_len is not None:
                server_details += f", max_model_len {server.max_model_len}"
            table.add_row(
                "",
                server.name,
                _hosts(server.slices),
                _devices(server.slices),
                server_details,
            )
    console.print(table)
    for warning in placement.warnings:
        console.print(f"[yellow]warning:[/yellow] {warning}")


def _hosts(slices) -> str:
    if not slices:
        return "-"
    if len(slices) <= 3:
        return ",".join(item.host for item in slices)
    return f"{slices[0].host} .. {slices[-1].host} ({len(slices)})"


def _devices(slices) -> str:
    patterns = {item.devices for item in slices}
    if not patterns:
        return "-"
    if len(patterns) == 1:
        devices = patterns.pop()
        return ",".join(str(device) for device in devices) if devices else "node"
    return "varies"


@resources_app.command("plan")
def resources_plan(
    config: Path = PLAN_ARGUMENT,
    nodes: int | None = NODES_OPTION,
    hosts: str | None = HOSTS_OPTION,
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Write the resolved placement JSON to this file or directory."
    ),
    as_json: bool = typer.Option(False, "--json", help="Print the placement as JSON."),
):
    """Resolve a resource plan and print where every pool and LLM server runs."""

    placement = _resolve(config, nodes, hosts)
    if as_json:
        typer.echo(placement.to_json())
    else:
        _print_placement(placement)
    if output is not None:
        written = placement.write(output)
        if not as_json:
            console.print(f"placement written to {written}")


@resources_app.command("validate")
def resources_validate(
    config: Path = PLAN_ARGUMENT,
    nodes: int | None = NODES_OPTION,
    hosts: str | None = HOSTS_OPTION,
):
    """Exit non-zero when a resource plan does not fit the allocation."""

    placement = _resolve(config, nodes, hosts)
    for warning in placement.warnings:
        console.print(f"[yellow]warning:[/yellow] {warning}")
    console.print(f"[green]ok[/green]: {config} fits {len(placement.nodes)} node(s)")


@resources_app.command("serve")
def resources_serve(
    config: Path = PLAN_ARGUMENT,
    command: list[str] = typer.Argument(
        ..., help="Command to run while the servers are up; put it after `--`."
    ),
    output: Path = typer.Option(
        Path("./llm-servers"),
        "--output",
        "-o",
        help="Directory for placement.json, llm-endpoints.json, and server logs.",
    ),
    server: str | None = typer.Option(
        None,
        "--server",
        help="Server exported as MATSIM_LLM_PROVIDER/MODEL/VLLM_BASE_URL (default: the only one).",
    ),
):
    """Start the plan's LLM servers, run a command, and stop the servers afterwards.

    The command receives MATSIM_LLM_ENDPOINTS (a JSON file with every server's
    endpoint) and, for the selected server, MATSIM_LLM_PROVIDER, MATSIM_LLM_MODEL,
    and MATSIM_VLLM_BASE_URL. Its exit code becomes this command's exit code.
    """
    import os
    import signal
    import subprocess

    from matsim_agents.backends.llm.servers import ENDPOINTS_VARIABLE, LLMServices
    from matsim_agents.execution.placement import ResourcePlan

    try:
        placement = ResourcePlan.from_yaml(config).place()
    except ValueError as exc:
        console.print(f"[red]resource plan {config} is invalid:[/red]\n{exc}", markup=True)
        raise typer.Exit(1) from None
    if "llm" not in placement.pools:
        console.print(f"[red]{config} has no llm pool to serve[/red]", markup=True)
        raise typer.Exit(1)
    in_process = [s.name for s in placement.pool("llm").servers if s.kind == "in-process"]
    if in_process:
        console.print(
            f"[red]in-process servers {in_process} load inside one process and cannot be "
            "served to another command[/red]",
            markup=True,
        )
        raise typer.Exit(1)

    def _terminate(signum, _frame):
        raise SystemExit(128 + signum)

    # Slurm and PBS send SIGTERM at the time limit; stop the servers cleanly.
    signal.signal(signal.SIGTERM, _terminate)
    placement.write(output / "placement.json")
    with LLMServices.start(placement, log_dir=output) as services:
        env = dict(os.environ)
        env[ENDPOINTS_VARIABLE] = str(services.write(output / "llm-endpoints.json"))
        if server is not None or len(services.endpoints) == 1:
            selected = services.endpoint(server)
            env["MATSIM_LLM_PROVIDER"] = selected.provider
            env["MATSIM_LLM_MODEL"] = selected.model
            if selected.base_url:
                env["MATSIM_VLLM_BASE_URL"] = selected.base_url
            env.setdefault("MATSIM_VLLM_API_KEY", "EMPTY")
        console.print(f"LLM servers ready: {', '.join(services.endpoints)}")
        returncode = subprocess.run(command, env=env, check=False).returncode
    raise typer.Exit(returncode)
