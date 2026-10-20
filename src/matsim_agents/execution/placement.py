"""Declarative placement of LLM, MLIP, DFT, and training work on an allocation.

A :class:`ResourcePlan` is the validated ``resources:`` block of a workflow
config or standalone plan file. :meth:`ResourcePlan.resolve` maps it onto a
discovered :class:`~matsim_agents.execution.allocation.Allocation` and returns
a :class:`Placement` with concrete hosts, device indices, device-mask
environment, and derived launch values. See ``docs/resource-dispatch.md``.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from matsim_agents.execution.allocation import Allocation
from matsim_agents.execution.presets import (
    LauncherFlavor,
    NodeGeometry,
    Vendor,
    get_preset,
)
from matsim_agents.execution.variables import (
    VARIABLE_PATTERN,
    expand_variables,
    variable_defaults,
)

PoolName = Literal["llm", "mlip", "dft", "training"]
POOL_NAMES: tuple[str, ...] = ("llm", "mlip", "dft", "training")
SHARED_MEMORY_LIMIT = 0.95

PositiveInt = Annotated[int, Field(ge=1)]
DeviceIndex = Annotated[int, Field(ge=0)]
NodesSpec = PositiveInt | Literal["all", "rest"] | Annotated[list[str], Field(min_length=1)]
ServerNodesSpec = PositiveInt | Literal["all"] | Annotated[list[str], Field(min_length=1)]
DevicesSpec = PositiveInt | Literal["all"] | Annotated[list[DeviceIndex], Field(min_length=1)]

# A claimable unit: one device on one host, or the whole host (device ``None``)
# when the geometry has no accelerators.
Unit = tuple[str, int | None]
# A host choice: the selected devices, or a reason the host does not qualify.
Pick = tuple[int, ...] | str


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GeometryOverride(_StrictModel):
    vendor: Vendor | None = None
    devices_per_node: int | None = Field(default=None, ge=0)
    cores_per_node: int | None = Field(default=None, ge=1)
    device_memory_gb: float | None = Field(default=None, gt=0)


class LLMServerSpec(_StrictModel):
    name: str = Field(min_length=1)
    kind: Literal["managed-vllm", "external", "in-process"]
    model: str | None = None
    served_name: str | None = None
    base_url: str | None = None
    nodes: ServerNodesSpec | None = None
    devices: DevicesSpec | None = None
    tensor_parallel_size: PositiveInt | None = None
    max_model_len: PositiveInt | None = None
    port: int | None = Field(default=None, ge=1024, le=65535)
    launcher: list[str] | None = Field(
        default=None,
        min_length=1,
        description=(
            "Command that runs `vllm serve`; the server's arguments are appended. "
            "Defaults to `vllm serve` on PATH."
        ),
    )
    extra_args: list[str] = Field(default_factory=list)
    startup_timeout_s: float = Field(default=2400.0, gt=0)
    api_key_env: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z_][A-Za-z0-9_]*$",
        description=(
            "Environment variable holding the server's API key. Clients send it; a "
            "managed vLLM server receives it as VLLM_API_KEY. The key itself is never "
            "stored in the plan, placement.json, or llm-endpoints.json."
        ),
    )

    @model_validator(mode="after")
    def _check_kind_fields(self) -> LLMServerSpec:
        where = f"LLM server {self.name!r} ({self.kind})"
        if self.kind == "external":
            if not self.base_url:
                raise ValueError(f"{where} requires base_url")
            placed = (
                "nodes",
                "devices",
                "tensor_parallel_size",
                "max_model_len",
                "port",
                "launcher",
            )
            if set_fields := [field for field in placed if getattr(self, field) is not None]:
                raise ValueError(f"{where} runs elsewhere and takes no {', '.join(set_fields)}")
            return self
        if not self.model:
            raise ValueError(f"{where} requires model")
        if self.base_url:
            raise ValueError(f"{where} gets its endpoint from the placement; remove base_url")
        if self.kind == "in-process":
            vllm_only = ("tensor_parallel_size", "max_model_len", "port", "launcher", "api_key_env")
            for field in vllm_only:
                if getattr(self, field) is not None:
                    raise ValueError(f"{where} does not accept {field}; it applies to vLLM only")
            if self.extra_args:
                raise ValueError(f"{where} does not accept extra_args; it applies to vLLM only")
        return self


class _PoolSpec(_StrictModel):
    nodes: NodesSpec = 1
    devices: DevicesSpec = "all"
    share_with: list[PoolName] = Field(default_factory=list)
    memory_fraction: float | None = Field(default=None, gt=0, le=1)


class LLMPoolSpec(_PoolSpec):
    servers: Annotated[list[LLMServerSpec], Field(min_length=1)]

    @model_validator(mode="after")
    def _unique_server_names(self) -> LLMPoolSpec:
        names = [server.name for server in self.servers]
        if duplicates := sorted({name for name in names if names.count(name) > 1}):
            raise ValueError(f"pool 'llm' has duplicate server names: {', '.join(duplicates)}")
        return self

    @property
    def placed_servers(self) -> list[LLMServerSpec]:
        return [server for server in self.servers if server.kind != "external"]


class MLIPPoolSpec(_PoolSpec):
    max_concurrent: PositiveInt | None = None


class DFTPoolSpec(_PoolSpec):
    nodes_per_job: PositiveInt = 1
    ranks_per_device: PositiveInt = 1
    threads_per_rank: PositiveInt | None = None
    max_concurrent: PositiveInt | None = None


class TrainingPoolSpec(_PoolSpec):
    devices: DevicesSpec = 1


PoolSpec = LLMPoolSpec | MLIPPoolSpec | DFTPoolSpec | TrainingPoolSpec
POOL_TYPES: dict[str, type[_PoolSpec]] = {
    "llm": LLMPoolSpec,
    "mlip": MLIPPoolSpec,
    "dft": DFTPoolSpec,
    "training": TrainingPoolSpec,
}


class NodeSlice(_StrictModel):
    host: str
    devices: tuple[int, ...]
    env: dict[str, str] = Field(default_factory=dict)
    shared_devices: tuple[int, ...] = ()


class ProcessLaunch(_StrictModel):
    """One multi-process step: its hosts, devices, rank layout, and environment.

    Launch wrappers keep their positional ``<nodes> <ranks_per_node>
    <threads_per_rank>`` contract (:meth:`wrapper_args`) and read the host list
    and per-node device count from :meth:`environment`.
    """

    hosts: tuple[str, ...]
    devices: tuple[int, ...]
    ranks_per_node: int
    threads_per_rank: int
    env: dict[str, str] = Field(default_factory=dict)

    @property
    def node_count(self) -> int:
        return len(self.hosts)

    @property
    def total_ranks(self) -> int:
        return self.node_count * self.ranks_per_node

    def wrapper_args(self) -> list[str]:
        return [str(self.node_count), str(self.ranks_per_node), str(self.threads_per_rank)]

    def environment(
        self, base: Mapping[str, str] | None = None, *, local: bool = False
    ) -> dict[str, str]:
        """Return *base* (default ``os.environ``) plus this launch's variables.

        With ``local=True`` the process runs directly on this node, so its device
        mask is translated through the mask this process already has. Launches
        through ``srun``/``mpiexec`` keep node-wide indices for the launcher.
        """

        import os

        from matsim_agents.execution.devices import child_mask_environment

        env = dict(os.environ if base is None else base)
        env.update(child_mask_environment(self.env, owner="this launch") if local else self.env)
        env["MATSIM_ASSIGNED_NODES"] = ",".join(self.hosts)
        env["MATSIM_DEVICES_PER_NODE"] = str(len(self.devices))
        return env


class ServerPlacement(_StrictModel):
    name: str
    kind: str
    model: str | None = None
    served_name: str | None = None
    base_url: str | None = None
    slices: list[NodeSlice] = Field(default_factory=list)
    tensor_parallel_size: int | None = None
    pipeline_parallel_size: int | None = None
    max_model_len: int | None = None
    memory_fraction: float | None = None
    port: int | None = None
    launcher: list[str] | None = None
    extra_args: list[str] = Field(default_factory=list)
    startup_timeout_s: float | None = None
    api_key_env: str | None = None


class PoolPlacement(_StrictModel):
    name: str
    slices: list[NodeSlice] = Field(default_factory=list)
    share_with: list[str] = Field(default_factory=list)
    memory_fraction: float | None = None
    servers: list[ServerPlacement] = Field(default_factory=list)
    max_concurrent: int | None = None
    launches: list[ProcessLaunch] = Field(default_factory=list)

    @property
    def hosts(self) -> list[str]:
        return [item.host for item in self.slices]

    @property
    def device_count(self) -> int:
        return sum(len(item.devices) for item in self.slices)

    def server(self, name: str) -> ServerPlacement:
        for server in self.servers:
            if server.name == name:
                return server
        raise KeyError(f"pool {self.name!r} has no server {name!r}")


class Placement(_StrictModel):
    scheduler: str
    preset: str | None
    launcher: LauncherFlavor
    geometry: NodeGeometry
    nodes: list[str]
    pools: dict[str, PoolPlacement]
    idle: list[NodeSlice] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    def pool(self, name: str) -> PoolPlacement:
        try:
            return self.pools[name]
        except KeyError:
            raise KeyError(f"placement has no pool {name!r}") from None

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)

    def write(self, path: str | Path) -> Path:
        """Write the placement as JSON; a directory receives ``placement.json``."""

        target = Path(path)
        if target.is_dir():
            target = target / "placement.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json() + "\n", encoding="utf-8")
        return target


class ResourcePlan(_StrictModel):
    """Validated ``resources:`` block; see ``docs/resource-dispatch.md``."""

    preset: str | None = None
    launcher: LauncherFlavor | None = None
    geometry: GeometryOverride | None = None
    pools: dict[PoolName, PoolSpec] = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def _type_pools(cls, data: Any) -> Any:
        if not isinstance(data, dict) or not isinstance(data.get("pools"), dict):
            return data
        typed: dict[str, Any] = {}
        for name, spec in data["pools"].items():
            if name not in POOL_TYPES:
                raise ValueError(f"unknown pool {name!r}; pools are named {', '.join(POOL_NAMES)}")
            typed[name] = POOL_TYPES[name].model_validate(spec if spec is not None else {})
        return {**data, "pools": typed}

    @model_validator(mode="after")
    def _check_plan(self) -> ResourcePlan:
        if self.preset is not None:
            get_preset(self.preset)
        for name, spec in self.pools.items():
            for other in spec.share_with:
                if other == name:
                    raise ValueError(f"pool {name!r} cannot list itself in share_with")
                if other not in self.pools:
                    raise ValueError(f"pool {name!r} shares with undeclared pool {other!r}")
                if name not in self.pools[other].share_with:
                    raise ValueError(
                        f"pool {name!r} lists {other!r} in share_with, but {other!r} does "
                        f"not list {name!r}; sharing must be declared on both pools"
                    )
        rest = [name for name, spec in self.pools.items() if spec.nodes == "rest"]
        if len(rest) > 1:
            raise ValueError(f"only one pool may use nodes: rest; found {', '.join(rest)}")
        llm = self.pools.get("llm")
        external_only = isinstance(llm, LLMPoolSpec) and not llm.placed_servers
        if external_only and (fields := sorted({"nodes", "devices"} & llm.model_fields_set)):
            raise ValueError(
                "pool 'llm' has only external servers and takes no "
                f"{' or '.join(fields)}; remove them"
            )
        return self

    @classmethod
    def from_mapping(cls, data: Any, *, source: object = "<resources>") -> ResourcePlan:
        """Build a plan from a parsed document that has a top-level ``resources:`` key."""

        if not isinstance(data, dict) or not isinstance(data.get("resources"), dict):
            raise ValueError(f"{source}: no top-level 'resources:' mapping")
        defaults = variable_defaults(data.get("vars"), source=source)
        return cls.model_validate(_substitute(data["resources"], defaults, source=source))

    @classmethod
    def from_yaml(cls, path: str | Path) -> ResourcePlan:
        """Load the ``resources:`` block of a workflow config or standalone plan file."""

        import yaml

        path = Path(path)
        return cls.from_mapping(yaml.safe_load(path.read_text(encoding="utf-8")), source=path)

    def resolve(self, allocation: Allocation) -> Placement:
        return _Resolver(self, allocation).run()

    def place(self, run_dir: str | Path | None = None) -> Placement:
        """Resolve against the current allocation and record ``placement.json``."""

        import logging

        from matsim_agents.execution.allocation import discover_allocation

        placement = self.resolve(discover_allocation())
        logger = logging.getLogger(__name__)
        for warning in placement.warnings:
            logger.warning("resource placement: %s", warning)
        if run_dir is not None:
            Path(run_dir).mkdir(parents=True, exist_ok=True)
            placement.write(Path(run_dir) / "placement.json")
        return placement


def _substitute(value: Any, defaults: Mapping[str, str], *, source: object) -> Any:
    if isinstance(value, dict):
        return {key: _substitute(item, defaults, source=source) for key, item in value.items()}
    if isinstance(value, list):
        return [_substitute(item, defaults, source=source) for item in value]
    if not isinstance(value, str) or "${" not in value:
        return value
    expanded = expand_variables(value, defaults, source=source)
    if VARIABLE_PATTERN.fullmatch(value) and expanded:
        import yaml

        # A value that is exactly one variable becomes a number or list when it
        # spells one (``nodes: ${LLM_NODES}``, ``devices: ${GPUS}``). Anything else
        # stays the literal string, so names like ``on`` or ``null`` and paths are
        # never reinterpreted as booleans, null, or mappings.
        try:
            typed = yaml.safe_load(expanded)
        except yaml.YAMLError:
            return expanded
        if isinstance(typed, int | float | list) and not isinstance(typed, bool):
            return typed
    return expanded


class _Resolver:
    def __init__(self, plan: ResourcePlan, allocation: Allocation) -> None:
        if not allocation.nodes:
            raise ValueError("allocation has no nodes")
        if len(set(allocation.nodes)) != len(allocation.nodes):
            raise ValueError("allocation lists a node more than once")
        self.plan = plan
        self.allocation = allocation
        self.nodes = list(allocation.nodes)
        self.geometry = _resolve_geometry(plan, allocation)
        self.claims: dict[Unit, list[str]] = {}
        self.warnings: list[str] = []

    # -- generic helpers ------------------------------------------------
    def _units(self, host: str) -> list[Unit]:
        count = self.geometry.devices_per_node
        return [(host, device) for device in range(count)] if count else [(host, None)]

    def _blockers(self, pool: str, unit: Unit) -> list[str]:
        share_with = self.plan.pools[pool].share_with
        return [owner for owner in self.claims.get(unit, []) if owner not in share_with]

    def _slice(self, host: str, devices: tuple[int, ...]) -> NodeSlice:
        env = self.geometry.device_mask_environment(devices) if devices else {}
        shared = tuple(d for d in devices if len(self.claims.get((host, d), [])) > 1)
        return NodeSlice(host=host, devices=devices, env=env, shared_devices=shared)

    # -- pools ----------------------------------------------------------
    def run(self) -> Placement:
        order = [name for name, spec in self.plan.pools.items() if spec.nodes != "rest"]
        order += [name for name, spec in self.plan.pools.items() if spec.nodes == "rest"]
        placements = {name: self._resolve_pool(name) for name in order}
        for placement in placements.values():
            # Sharing is only known once every pool has claimed its devices.
            placement.slices = [self._slice(item.host, item.devices) for item in placement.slices]
        self._check_shared_memory()
        for name in self.plan.pools:
            self._finish_pool(placements[name])
        idle = self._idle()
        if idle:
            described = "; ".join(f"{s.host}: {list(s.devices) or 'node'}" for s in idle)
            self.warnings.append(f"not assigned to any pool: {described}")
        launcher = self.plan.launcher or (
            get_preset(self.plan.preset).launcher if self.plan.preset else "local"
        )
        return Placement(
            scheduler=self.allocation.scheduler,
            preset=self.plan.preset,
            launcher=launcher,
            geometry=self.geometry,
            nodes=self.nodes,
            pools={name: placements[name] for name in self.plan.pools},
            idle=idle,
            warnings=self.warnings,
        )

    def _resolve_pool(self, name: str) -> PoolPlacement:
        spec = self.plan.pools[name]
        placement = PoolPlacement(
            name=name, share_with=list(spec.share_with), memory_fraction=spec.memory_fraction
        )
        if isinstance(spec, LLMPoolSpec) and not spec.placed_servers:
            return placement
        devices_spec = spec.devices
        if not self.geometry.devices_per_node:
            if "devices" in spec.model_fields_set and spec.devices != "all":
                raise ValueError(
                    f"pool {name!r} requests devices {spec.devices!r}, but the geometry "
                    "has no accelerator devices"
                )
            devices_spec = "all"
        selected = _select_hosts(
            owner=f"pool {name!r}",
            nodes_spec=spec.nodes,
            candidates=self.nodes,
            pick=lambda host: self._pick_pool_devices(name, host, devices_spec),
            rest_filter=lambda host: not any(self.claims.get(u) for u in self._units(host)),
        )
        for host, devices in selected:
            for unit in [(host, d) for d in devices] if devices else [(host, None)]:
                self.claims.setdefault(unit, []).append(name)
            placement.slices.append(self._slice(host, devices))
        return placement

    def _pick_pool_devices(self, pool: str, host: str, devices_spec: DevicesSpec) -> Pick:
        if not self.geometry.devices_per_node:
            if blockers := self._blockers(pool, (host, None)):
                return f"{host} is used by pool {blockers[0]!r}"
            return ()

        def explain(device: int) -> str:
            blocker = self._blockers(pool, (host, device))[0]
            return (
                f"device {device} on {host} overlaps pool {blocker!r}; "
                "add share_with to both pools to allow sharing"
            )

        return _pick_devices(
            owner=f"pool {pool!r}",
            host=host,
            devices_spec=devices_spec,
            candidates=list(range(self.geometry.devices_per_node)),
            available=lambda device: not self._blockers(pool, (host, device)),
            explain=explain,
            outside=lambda device: (
                f"pool {pool!r} device {device} does not exist on {host} "
                f"({self.geometry.devices_per_node} devices per node)"
            ),
        )

    def _check_shared_memory(self) -> None:
        for (host, device), owners in self.claims.items():
            if len(owners) < 2:
                continue
            fractions = {
                owner: self.plan.pools[owner].memory_fraction
                for owner in owners
                if self.plan.pools[owner].memory_fraction is not None
            }
            total = sum(fractions.values())
            if total > SHARED_MEMORY_LIMIT + 1e-9:
                parts = ", ".join(f"{owner!r} ({value:g})" for owner, value in fractions.items())
                raise ValueError(
                    f"device {device} on {host} is shared by pools {parts}; memory_fraction "
                    f"values sum to {total:.2f}, above the {SHARED_MEMORY_LIMIT} limit"
                )

    def _shared_units(self, slices: Sequence[NodeSlice]) -> list[tuple[str, int]]:
        return [
            (item.host, device)
            for item in slices
            for device in item.devices
            if len(self.claims.get((item.host, device), [])) > 1
        ]

    def _finish_pool(self, placement: PoolPlacement) -> None:
        spec = self.plan.pools[placement.name]
        if isinstance(spec, LLMPoolSpec):
            self._resolve_servers(spec, placement)
        elif isinstance(spec, MLIPPoolSpec):
            self._finish_mlip(spec, placement)
        elif isinstance(spec, DFTPoolSpec):
            self._finish_dft(spec, placement)
        elif isinstance(spec, TrainingPoolSpec):
            self._finish_training(placement)

    def _finish_mlip(self, spec: MLIPPoolSpec, placement: PoolPlacement) -> None:
        workers = placement.device_count or len(placement.slices)
        if spec.max_concurrent is not None and spec.max_concurrent > workers:
            raise ValueError(
                f"pool 'mlip' max_concurrent={spec.max_concurrent} exceeds its {workers} "
                "device(s); concurrent relaxations run one per device"
            )
        placement.max_concurrent = spec.max_concurrent or workers
        if placement.max_concurrent < workers:
            self.warnings.append(
                f"pool 'mlip' uses {placement.max_concurrent} of its {workers} device(s) "
                "concurrently"
            )

    def _uniform_devices(self, placement: PoolPlacement) -> tuple[int, ...]:
        patterns = {item.devices for item in placement.slices}
        if len(patterns) != 1:
            described = "; ".join(f"{item.host}: {list(item.devices)}" for item in placement.slices)
            raise ValueError(
                f"pool {placement.name!r} must use the same devices on every node so one "
                f"launch can span them; it resolved to {described}"
            )
        return patterns.pop()

    def _launch(
        self, hosts: Sequence[str], devices: tuple[int, ...], ranks: int, threads: int
    ) -> ProcessLaunch:
        env = self.geometry.device_mask_environment(devices) if devices else {}
        return ProcessLaunch(
            hosts=tuple(hosts),
            devices=devices,
            ranks_per_node=ranks,
            threads_per_rank=threads,
            env=env,
        )

    def _finish_dft(self, spec: DFTPoolSpec, placement: PoolPlacement) -> None:
        node_count = len(placement.slices)
        if node_count % spec.nodes_per_job:
            raise ValueError(
                f"pool 'dft' nodes_per_job={spec.nodes_per_job} does not divide its "
                f"{node_count} node(s)"
            )
        devices = self._uniform_devices(placement)
        cores = self.geometry.cores_per_node
        if devices:
            ranks = len(devices) * spec.ranks_per_device
        else:
            ranks = cores // (spec.threads_per_rank or 1)
        if ranks < 1:
            raise ValueError(
                f"pool 'dft' threads_per_rank={spec.threads_per_rank} exceeds the {cores} "
                "cores per node"
            )
        threads = spec.threads_per_rank or cores // ranks
        if threads < 1 or ranks * threads > cores:
            raise ValueError(
                f"pool 'dft' needs {ranks} rank(s) x {max(threads, 1)} thread(s) per node, "
                f"but nodes have {cores} cores"
            )
        hosts = placement.hosts
        placement.launches = [
            self._launch(hosts[start : start + spec.nodes_per_job], devices, ranks, threads)
            for start in range(0, node_count, spec.nodes_per_job)
        ]
        groups = len(placement.launches)
        if spec.max_concurrent is not None and spec.max_concurrent > groups:
            raise ValueError(
                f"pool 'dft' max_concurrent={spec.max_concurrent} exceeds its {groups} "
                f"job group(s) of {spec.nodes_per_job} node(s)"
            )
        placement.max_concurrent = spec.max_concurrent or groups

    def _finish_training(self, placement: PoolPlacement) -> None:
        devices = self._uniform_devices(placement)
        ranks = len(devices) or 1
        threads = max(1, self.geometry.cores_per_node // ranks)
        placement.launches = [self._launch(placement.hosts, devices, ranks, threads)]

    # -- LLM servers ----------------------------------------------------
    def _resolve_servers(self, spec: LLMPoolSpec, placement: PoolPlacement) -> None:
        pool_devices = {item.host: item.devices for item in placement.slices}
        used: dict[tuple[str, int], str] = {}
        single = len(spec.placed_servers) == 1
        managed = [server.name for server in spec.servers if server.kind == "managed-vllm"]
        ports: dict[tuple[str, int], str] = {}
        for server in spec.servers:
            result = ServerPlacement(
                name=server.name,
                kind=server.kind,
                model=server.model,
                served_name=server.served_name
                or (Path(server.model).name if server.model else None),
                base_url=server.base_url,
                max_model_len=server.max_model_len,
                memory_fraction=spec.memory_fraction,
                api_key_env=server.api_key_env,
            )
            if server.kind != "external":
                result.slices = self._place_server(server, pool_devices, used, single)
                self._check_server(server, result, spec)
            if server.kind == "managed-vllm":
                port = server.port or 8000 + managed.index(server.name)
                for item in result.slices:
                    if (item.host, port) in ports:
                        raise ValueError(
                            f"LLM servers {ports[(item.host, port)]!r} and {server.name!r} both "
                            f"use port {port} on {item.host}; set port on one of them"
                        )
                    ports[(item.host, port)] = server.name
                result.port = port
                result.launcher = server.launcher
                result.extra_args = list(server.extra_args)
                result.startup_timeout_s = server.startup_timeout_s
            placement.servers.append(result)
        unused = [
            f"{host}: {[d for d in devices if (host, d) not in used]}"
            for host, devices in pool_devices.items()
            if any((host, d) not in used for d in devices)
        ]
        if unused:
            self.warnings.append(f"pool 'llm' devices not used by any server: {'; '.join(unused)}")

    def _place_server(
        self,
        server: LLMServerSpec,
        pool_devices: dict[str, tuple[int, ...]],
        used: dict[tuple[str, int], str],
        single: bool,
    ) -> list[NodeSlice]:
        owner = f"LLM server {server.name!r}"
        if not self.geometry.devices_per_node:
            raise ValueError(f"{owner} needs accelerator devices, but the geometry has none")
        nodes_spec: ServerNodesSpec | None = server.nodes
        if nodes_spec is None:
            if not single:
                raise ValueError(
                    f"{owner}: set nodes (and optionally devices) when pool 'llm' has "
                    "several servers that run on the allocation"
                )
            nodes_spec = "all"
        devices_spec: DevicesSpec = server.devices if server.devices is not None else "all"

        def pick(host: str) -> Pick:
            return _pick_devices(
                owner=owner,
                host=host,
                devices_spec=devices_spec,
                candidates=list(pool_devices[host]),
                available=lambda device: (host, device) not in used,
                explain=lambda device: (
                    f"device {device} on {host} is already used by LLM server "
                    f"{used[(host, device)]!r}"
                ),
                outside=lambda device: (
                    f"{owner} device {device} on {host} is outside pool 'llm' "
                    f"(pool devices there: {list(pool_devices[host])})"
                ),
            )

        selected = _select_hosts(
            owner=owner,
            nodes_spec=nodes_spec,
            candidates=list(pool_devices),
            pick=pick,
            rest_filter=None,
        )
        slices = []
        for host, devices in selected:
            for device in devices:
                used[(host, device)] = server.name
            slices.append(self._slice(host, devices))
        return slices

    def _check_server(
        self, server: LLMServerSpec, result: ServerPlacement, spec: LLMPoolSpec
    ) -> None:
        owner = f"LLM server {server.name!r}"
        if server.kind == "in-process" and len(result.slices) != 1:
            raise ValueError(f"{owner} runs in-process and must use exactly 1 node")
        shared = self._shared_units(result.slices)
        if server.kind == "managed-vllm" and shared and spec.memory_fraction is None:
            host, device = shared[0]
            others = [owner for owner in self.claims[(host, device)] if owner != "llm"]
            raise ValueError(
                f"{owner} runs vLLM on device {device} of {host}, which is shared with "
                f"{others}; set memory_fraction on pool 'llm' (vLLM otherwise reserves "
                "90% of the device)"
            )
        if server.kind != "managed-vllm":
            return
        per_node = {len(item.devices) for item in result.slices}
        if len(per_node) != 1:
            raise ValueError(f"{owner} must use the same number of devices on every node")
        total = sum(len(item.devices) for item in result.slices)
        tensor = server.tensor_parallel_size or per_node.pop()
        if total % tensor:
            raise ValueError(
                f"{owner} tensor_parallel_size={tensor} does not divide its {total} device(s)"
            )
        heads = _attention_heads(server.model)
        if heads is not None and heads % tensor:
            raise ValueError(
                f"{owner} tensor_parallel_size={tensor} does not divide the model's "
                f"{heads} attention heads"
            )
        result.tensor_parallel_size = tensor
        result.pipeline_parallel_size = total // tensor

    def _idle(self) -> list[NodeSlice]:
        idle = []
        for host in self.nodes:
            free = [unit for unit in self._units(host) if not self.claims.get(unit)]
            if not free:
                continue
            devices = tuple(device for _, device in free if device is not None)
            if devices or not self.geometry.devices_per_node:
                idle.append(NodeSlice(host=host, devices=devices))
        return idle


def _pick_devices(
    *,
    owner: str,
    host: str,
    devices_spec: DevicesSpec,
    candidates: list[int],
    available: Callable[[int], bool],
    explain: Callable[[int], str],
    outside: Callable[[int], str],
) -> Pick:
    """Return the chosen devices on *host*, or a reason the host does not qualify."""

    if isinstance(devices_spec, list):
        if len(set(devices_spec)) != len(devices_spec):
            raise ValueError(f"{owner} lists a device more than once: {devices_spec}")
        # Hosts can differ in which devices are available (e.g. an LLM pool slice),
        # so a missing device rejects this host; callers raise if none qualify.
        for device in devices_spec:
            if device not in candidates:
                return outside(device)
        wanted = list(devices_spec)
    elif devices_spec == "all":
        wanted = candidates
    else:
        if devices_spec > len(candidates):
            return (
                f"{owner} requests {devices_spec} device(s) per node, but only "
                f"{len(candidates)} are available to it on {host}"
            )
        free = [device for device in candidates if available(device)]
        if len(free) < devices_spec:
            blocked = next(device for device in candidates if not available(device))
            return f"{host} has {len(free)} free device(s) ({explain(blocked)})"
        return tuple(free[:devices_spec])
    for device in wanted:
        if not available(device):
            return explain(device)
    return tuple(wanted)


def _select_hosts(
    *,
    owner: str,
    nodes_spec: NodesSpec | ServerNodesSpec,
    candidates: list[str],
    pick: Callable[[str], Pick],
    rest_filter: Callable[[str], bool] | None,
) -> list[tuple[str, tuple[int, ...]]]:
    def require(hosts: list[str], context: str) -> list[tuple[str, tuple[int, ...]]]:
        selected = []
        for host in hosts:
            result = pick(host)
            if isinstance(result, str):
                raise ValueError(f"{owner}{context}: {result}")
            selected.append((host, result))
        return selected

    if isinstance(nodes_spec, list):
        if len(set(nodes_spec)) != len(nodes_spec):
            raise ValueError(f"{owner} lists a host more than once: {nodes_spec}")
        for host in nodes_spec:
            if host not in candidates:
                raise ValueError(f"{owner} host {host!r} is not available to it: {candidates}")
        return require(nodes_spec, "")
    if nodes_spec == "all":
        return require(candidates, " uses all nodes")
    if nodes_spec == "rest":
        assert rest_filter is not None
        hosts = [host for host in candidates if rest_filter(host)]
        if not hosts:
            raise ValueError(f"{owner} uses nodes: rest, but every node is used by another pool")
        return require(hosts, "")
    if nodes_spec > len(candidates):
        raise ValueError(
            f"{owner} requests {nodes_spec} node(s), but only {len(candidates)} are available"
        )
    selected, reasons = [], []
    for host in candidates:
        result = pick(host)
        if isinstance(result, str):
            reasons.append(result)
            continue
        selected.append((host, result))
        if len(selected) == nodes_spec:
            return selected
    raise ValueError(
        f"{owner} needs {nodes_spec} node(s) with the requested devices free, but only "
        f"{len(selected)} qualify: {'; '.join(reasons)}"
    )


def _resolve_geometry(plan: ResourcePlan, allocation: Allocation) -> NodeGeometry:
    values: dict[str, Any] = {}
    if plan.preset is not None:
        values.update(get_preset(plan.preset).geometry.model_dump())
    overrides = plan.geometry.model_dump(exclude_none=True) if plan.geometry else {}
    detected = allocation.gpus_per_node
    if detected is not None:
        explicit = overrides.get("devices_per_node")
        if explicit is not None and explicit != detected:
            raise ValueError(
                f"geometry.devices_per_node={explicit} contradicts the {detected} device(s) "
                f"per node reported by {allocation.scheduler}"
            )
        values["devices_per_node"] = detected
    values.update(overrides)
    if missing := [f for f in ("vendor", "devices_per_node", "cores_per_node") if f not in values]:
        raise ValueError(
            "cannot determine node geometry: set preset or "
            + ", ".join(f"geometry.{field}" for field in missing)
        )
    geometry = NodeGeometry.model_validate(values)
    if geometry.vendor == "cpu" and geometry.devices_per_node:
        raise ValueError("geometry vendor 'cpu' cannot have accelerator devices")
    if geometry.vendor != "cpu" and not geometry.devices_per_node:
        raise ValueError(
            f"geometry vendor {geometry.vendor!r} reports no devices per node; use vendor "
            "'cpu' for CPU-only nodes"
        )
    return geometry


def _attention_heads(model: str | None) -> int | None:
    """Read ``num_attention_heads`` from a local model directory when available."""

    if not model:
        return None
    try:
        config = json.loads((Path(model) / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    for section in (config, config.get("text_config") or {}):
        heads = section.get("num_attention_heads") if isinstance(section, dict) else None
        if isinstance(heads, int) and heads > 0:
            return heads
    return None


__all__ = [
    "POOL_NAMES",
    "SHARED_MEMORY_LIMIT",
    "DFTPoolSpec",
    "GeometryOverride",
    "LLMPoolSpec",
    "LLMServerSpec",
    "MLIPPoolSpec",
    "NodeSlice",
    "Placement",
    "PoolPlacement",
    "ProcessLaunch",
    "ResourcePlan",
    "ServerPlacement",
    "TrainingPoolSpec",
]
