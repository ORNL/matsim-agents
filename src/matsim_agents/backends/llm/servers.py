"""Start, wait for, and stop the LLM servers named in a resource placement.

``managed-vllm`` servers are started with their device mask on the node the
placement assigns them (directly when that is this node, otherwise through the
facility launcher: ``srun`` or ``mpiexec``), and are ready once ``/v1/models``
lists their served name. ``external`` servers are only looked up, and
``in-process`` servers load into this process, restricted to their devices.
Workflows refer to every server by its name in the plan.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from matsim_agents.execution.devices import child_mask_environment, visible_ordinals

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from matsim_agents.execution.placement import Placement, ResourcePlan, ServerPlacement

log = logging.getLogger(__name__)

ENDPOINTS_VARIABLE = "MATSIM_LLM_ENDPOINTS"


@dataclass(frozen=True)
class LLMEndpoint:
    """How a workflow reaches one named server of the plan."""

    name: str
    provider: Literal["vllm", "huggingface"]
    model: str
    base_url: str | None = None
    tensor_parallel_size: int | None = None
    pipeline_parallel_size: int | None = None
    device_count: int | None = None
    max_model_len: int | None = None
    # Device ordinals an in-process model may use (positions among the visible devices).
    devices: tuple[int, ...] | None = None

    def chat_model(self, **kwargs: Any) -> BaseChatModel:
        from matsim_agents.backends.llm.provider import get_chat_model

        return get_chat_model(
            provider=self.provider, model=self.model, base_url=self.base_url, **kwargs
        )


def _short_host(name: str) -> str:
    return name.split(".", 1)[0]


def _tail(path: Path, lines: int = 40) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])
    except OSError:
        return "(no log)"


def served_models(base_url: str, timeout: float = 5.0) -> list[str]:
    """Return the model ids an OpenAI-compatible server lists, or raise ``OSError``."""

    with urllib.request.urlopen(f"{base_url.rstrip('/')}/models", timeout=timeout) as response:
        payload = json.loads(response.read().decode())
    return [str(item.get("id")) for item in payload.get("data", [])]


class _ManagedServer:
    def __init__(
        self,
        server: ServerPlacement,
        placement: Placement,
        *,
        log_dir: Path,
        local_host: str,
    ) -> None:
        if len(server.slices) != 1:
            raise ValueError(
                f"LLM server {server.name!r} spans {len(server.slices)} nodes; managed vLLM "
                "servers run on one node (multi-node serving is phase 5 of "
                "docs/resource-dispatch.md); run it as an external server instead"
            )
        model = Path(str(server.model))
        if model.is_absolute() and not model.exists():
            raise ValueError(f"LLM server {server.name!r}: model directory not found: {model}")
        self.server = server
        self.slice = server.slices[0]
        self.local = placement.scheduler in {"local", "preview"} or (
            _short_host(self.slice.host) == _short_host(local_host)
        )
        bind = "127.0.0.1" if self.local else "0.0.0.0"
        address = "127.0.0.1" if self.local else self.slice.host
        self.base_url = f"http://{address}:{server.port}/v1"
        self.log_path = log_dir / f"{server.name}.log"
        self.command = self._command(placement, bind)
        self.env = self._environment()
        self.process: subprocess.Popen | None = None
        self.ready = False

    def _serve_args(self, bind: str) -> list[str]:
        server = self.server
        args = [
            str(server.model),
            "--served-model-name",
            str(server.served_name),
            "--host",
            bind,
            "--port",
            str(server.port),
            "--tensor-parallel-size",
            str(server.tensor_parallel_size),
        ]
        if (server.pipeline_parallel_size or 1) > 1:
            args += ["--pipeline-parallel-size", str(server.pipeline_parallel_size)]
        if server.max_model_len is not None:
            args += ["--max-model-len", str(server.max_model_len)]
        if self.slice.shared_devices and server.memory_fraction is not None:
            args += ["--gpu-memory-utilization", str(server.memory_fraction)]
        return args + list(server.extra_args)

    def _command(self, placement: Placement, bind: str) -> list[str]:
        command = list(self.server.launcher or ["vllm", "serve"]) + self._serve_args(bind)
        if self.local:
            return command
        mask = [f"{key}={value}" for key, value in self.slice.env.items()]
        host = self.slice.host
        geometry = placement.geometry
        if placement.launcher == "srun":
            cpus = max(
                1, geometry.cores_per_node * len(self.slice.devices) // geometry.devices_per_node
            )
            return [
                "srun",
                "--nodes=1",
                "--ntasks=1",
                f"--nodelist={host}",
                "--overlap",
                f"--gpus-per-node={geometry.devices_per_node}",
                f"--cpus-per-task={cpus}",
                "env",
                *mask,
                *command,
            ]
        if placement.launcher == "mpiexec":
            return ["mpiexec", "-n", "1", "--ppn", "1", "--hosts", host, "env", *mask, *command]
        raise ValueError(
            f"LLM server {self.server.name!r} is placed on {host}, but launcher "
            f"{placement.launcher!r} cannot start processes on other nodes"
        )

    def _environment(self) -> dict[str, str]:
        env = dict(os.environ)
        if self.local:
            env.update(
                child_mask_environment(self.slice.env, owner=f"LLM server {self.server.name!r}")
            )
        return env

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log.info("Starting LLM server %s: %s", self.server.name, " ".join(self.command))
        with self.log_path.open("w") as log_file:
            self.process = subprocess.Popen(
                self.command,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=self.env,
                start_new_session=True,
            )

    def check(self) -> bool:
        """Return True once the server lists its served name; raise if it exited."""
        assert self.process is not None
        if self.process.poll() is not None:
            raise RuntimeError(
                f"LLM server {self.server.name!r} exited with code {self.process.returncode} "
                f"before becoming ready; tail of {self.log_path}:\n{_tail(self.log_path)}"
            )
        try:
            self.ready = str(self.server.served_name) in served_models(self.base_url)
        except (OSError, ValueError):
            self.ready = False
        return self.ready

    def stop(self, grace_s: float = 30.0) -> None:
        process = self.process
        if process is None or process.poll() is not None:
            return
        log.info("Stopping LLM server %s (pid %d)", self.server.name, process.pid)
        for sig, wait_s in ((signal.SIGTERM, grace_s), (signal.SIGKILL, 10.0)):
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                return
            try:
                process.wait(timeout=wait_s)
                return
            except subprocess.TimeoutExpired:
                continue


class LLMServices:
    """Endpoints for every server of the plan's ``llm`` pool, and the processes behind them."""

    def __init__(
        self, endpoints: dict[str, LLMEndpoint], managed: list[_ManagedServer] | None = None
    ):
        self.endpoints = endpoints
        self._managed = managed or []

    @classmethod
    def start(
        cls,
        placement: Placement,
        *,
        log_dir: str | Path,
        local_host: str | None = None,
        poll_interval_s: float = 10.0,
    ) -> LLMServices:
        """Start managed servers, register in-process models, and wait until all are ready."""

        if "llm" not in placement.pools:
            return cls({})
        local_host = local_host or socket.gethostname()
        log_dir = Path(log_dir)
        endpoints: dict[str, LLMEndpoint] = {}
        managed: list[_ManagedServer] = []
        for server in placement.pool("llm").servers:
            if server.kind == "external":
                endpoints[server.name] = LLMEndpoint(
                    server.name, "vllm", str(server.served_name or server.name), server.base_url
                )
            elif server.kind == "in-process":
                endpoints[server.name] = _in_process_endpoint(server, placement, local_host)
            else:
                process = _ManagedServer(server, placement, log_dir=log_dir, local_host=local_host)
                managed.append(process)
                endpoints[server.name] = LLMEndpoint(
                    server.name,
                    "vllm",
                    str(server.served_name),
                    process.base_url,
                    tensor_parallel_size=server.tensor_parallel_size,
                    pipeline_parallel_size=server.pipeline_parallel_size,
                    device_count=sum(len(item.devices) for item in server.slices),
                    max_model_len=server.max_model_len,
                )
        services = cls(endpoints, managed)
        try:
            for process in managed:
                process.start()
            services._wait_ready(poll_interval_s)
        except BaseException:
            services.close()
            raise
        return services

    def _wait_ready(self, poll_interval_s: float) -> None:
        started = time.monotonic()
        pending = list(self._managed)
        while pending:
            pending = [process for process in pending if not process.check()]
            elapsed = time.monotonic() - started
            for process in pending:
                if elapsed > float(process.server.startup_timeout_s or 0):
                    raise TimeoutError(
                        f"LLM server {process.server.name!r} did not list "
                        f"{process.server.served_name!r} at {process.base_url} within "
                        f"{process.server.startup_timeout_s:.0f} s; tail of "
                        f"{process.log_path}:\n{_tail(process.log_path)}"
                    )
            if pending:
                time.sleep(poll_interval_s)
        for process in self._managed:
            log.info("LLM server %s ready at %s", process.server.name, process.base_url)

    def write(self, path: str | Path) -> Path:
        """Publish the endpoints as JSON for other processes (see :func:`llm_endpoints`)."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {name: asdict(endpoint) for name, endpoint in self.endpoints.items()}
        target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return target

    @classmethod
    def from_file(cls, path: str | Path) -> LLMServices:
        """Endpoints published by another process; this object owns no servers."""

        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        endpoints = {}
        for name, values in payload.items():
            if values.get("devices") is not None:
                values = {**values, "devices": tuple(values["devices"])}
            endpoints[name] = LLMEndpoint(**values)
        return cls(endpoints)

    def endpoint(self, name: str | None = None) -> LLMEndpoint:
        """Return the named endpoint; without a name, the only server of the plan."""

        if name is None:
            if len(self.endpoints) != 1:
                raise ValueError(
                    f"the plan's llm pool has servers {sorted(self.endpoints)}; choose one by name"
                )
            return next(iter(self.endpoints.values()))
        try:
            return self.endpoints[name]
        except KeyError:
            raise KeyError(
                f"the plan's llm pool has no server {name!r}; servers: {sorted(self.endpoints)}"
            ) from None

    def close(self) -> None:
        for process in reversed(self._managed):
            process.stop()

    def __enter__(self) -> LLMServices:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def _in_process_endpoint(
    server: ServerPlacement, placement: Placement, local_host: str
) -> LLMEndpoint:
    from matsim_agents.backends.llm.provider import limit_in_process_devices

    owner = f"LLM server {server.name!r}"
    item = server.slices[0]
    if placement.scheduler not in {"local", "preview"} and (
        _short_host(item.host) != _short_host(local_host)
    ):
        raise ValueError(
            f"{owner} runs in-process, so it must be placed on this node ({local_host}), "
            f"not {item.host}"
        )
    ordinals = visible_ordinals(placement.geometry.vendor, item.devices, owner=owner)
    fractions = {
        ordinal: (server.memory_fraction if device in item.shared_devices else None)
        for ordinal, device in zip(ordinals, item.devices, strict=True)
    }
    limit_in_process_devices(str(server.model), fractions)
    return LLMEndpoint(
        server.name,
        "huggingface",
        str(server.model),
        device_count=len(ordinals),
        devices=tuple(ordinals),
    )


@contextmanager
def llm_endpoints(
    resources: ResourcePlan | None, *, output_dir: str | Path
) -> Iterator[LLMServices]:
    """Endpoints for named servers: started from *resources*, or published by a parent.

    With a plan, its servers are started (and stopped on exit) and the placement
    is recorded in *output_dir*. Without one, the endpoints come from the file
    named by ``MATSIM_LLM_ENDPOINTS``, which ``matsim-agents resources serve``
    sets for the command it runs.
    """

    if resources is not None:
        placement = resources.place(output_dir)
        with LLMServices.start(placement, log_dir=Path(output_dir) / "llm-servers") as services:
            yield services
        return
    published = os.environ.get(ENDPOINTS_VARIABLE)
    if not published:
        raise ValueError(
            "LLM servers are named, but there is no resources: block and "
            f"{ENDPOINTS_VARIABLE} is unset; add a plan or run under "
            "`matsim-agents resources serve`"
        )
    yield LLMServices.from_file(published)


__all__ = [
    "ENDPOINTS_VARIABLE",
    "LLMEndpoint",
    "LLMServices",
    "llm_endpoints",
    "served_models",
]
