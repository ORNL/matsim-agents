"""DOE Integrated Research Infrastructure (IRI) Facility API execution platform.

Submits jobs through the facility-neutral IRI REST API
(https://github.com/doe-iri/iri-facility-api-python) instead of a local
scheduler, so one client reaches NERSC, ALCF, and OLCF (Frontier via S3M).
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from matsim_agents.execution.resources import ResourceRequest

TERMINAL_STATES = frozenset({"completed", "failed", "canceled"})

Transport = Callable[[str, str, Mapping[str, str], bytes | None, float], tuple[int, bytes]]


class IRIError(RuntimeError):
    """An IRI Facility API request failed."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class IRIWaitTimeout(IRIError, TimeoutError):
    """A job did not reach a terminal state before the wait deadline."""

    def __init__(self, message: str, *, last_state: str) -> None:
        super().__init__(message)
        self.last_state = last_state


class IRIFacility(BaseModel):
    """Connection settings for one IRI deployment and its target compute resource."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    base_url: str
    compute_resource_id: str = Field(min_length=1)
    token_env: str = Field(min_length=1, description="Environment variable holding the token.")
    queue: str | None = None
    account_env: str | None = None
    working_directory: str | None = Field(
        default=None, description="May reference {account}, filled from account_env."
    )
    custom_attributes: dict[str, str] = Field(default_factory=dict)

    @field_validator("base_url")
    @classmethod
    def _require_https(cls, value: str) -> str:
        if urllib.parse.urlsplit(value).scheme != "https":
            raise ValueError("IRI base_url must use https")
        return value.rstrip("/")


KNOWN_FACILITIES: dict[str, IRIFacility] = {
    "nersc-perlmutter": IRIFacility(
        name="nersc-perlmutter",
        base_url="https://api.iri.nersc.gov/api/v1",
        compute_resource_id="94351904-6dba-4c16-b5cd-fbd280d8615b",
        token_env="IRI_TOKEN_NERSC",
        queue="regular",
        account_env="NERSC_PROJECT",
        custom_attributes={"constraint": "gpu"},
    ),
    "alcf-aurora": IRIFacility(
        name="alcf-aurora",
        base_url="https://api.alcf.anl.gov/api/v1",
        compute_resource_id="0325fc07-6fb7-4453-b772-3d5030b2df72",
        token_env="IRI_TOKEN_ALCF",
        account_env="ALCF_PROJECT",
        custom_attributes={"filesystems": "flare"},
    ),
    "olcf-frontier": IRIFacility(
        name="olcf-frontier",
        base_url="https://amsc-moderate.s3m.olcf.ornl.gov/api/v2",
        compute_resource_id="frontier",
        token_env="IRI_TOKEN_OLCF",
        queue="batch",
        account_env="OLCF_S3M_PROJECT",
        working_directory="/lustre/orion/{account}/proj-shared",
    ),
    "olcf-odo": IRIFacility(
        name="olcf-odo",
        base_url="https://amsc-open.s3m.olcf.ornl.gov/api/v2",
        compute_resource_id="odo",
        token_env="IRI_TOKEN_OLCF",
        queue="batch",
        account_env="OLCF_S3M_PROJECT",
        working_directory="/gpfs/wolf2/olcf/{account}/proj-shared",
    ),
}


def walltime_seconds(walltime: str) -> int:
    """Convert ``[[D-]HH:]MM:SS`` or plain seconds to seconds."""

    days = 0
    if "-" in walltime:
        day_text, walltime = walltime.split("-", 1)
        days = int(day_text)
    seconds = 0
    for part in walltime.split(":"):
        seconds = seconds * 60 + int(part)
    total = days * 86400 + seconds
    if total < 1:
        raise ValueError(f"walltime must be positive: {walltime!r}")
    return total


def _urllib_transport(
    method: str, url: str, headers: Mapping[str, str], body: bytes | None, timeout: float
) -> tuple[int, bytes]:
    request = urllib.request.Request(url, data=body, headers=dict(headers), method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


class IRIPlatform:
    """``ExecutionPlatform`` backed by an IRI Facility API compute resource."""

    def __init__(
        self,
        facility: IRIFacility | str,
        *,
        token: str | None = None,
        account: str | None = None,
        timeout: float = 30.0,
        transport: Transport | None = None,
    ) -> None:
        if isinstance(facility, str):
            if facility not in KNOWN_FACILITIES:
                known = ", ".join(sorted(KNOWN_FACILITIES))
                raise ValueError(f"unknown IRI facility {facility!r}; known: {known}")
            facility = KNOWN_FACILITIES[facility]
        self.facility = facility
        self.name = f"iri:{facility.name}"
        self._token = token
        self._account = account
        self._timeout = timeout
        self._transport = transport or _urllib_transport

    @property
    def account(self) -> str | None:
        if self._account:
            return self._account
        env = self.facility.account_env
        if not env:
            return None
        return os.environ.get(env) or None

    def _token_value(self) -> str:
        token = self._token or os.environ.get(self.facility.token_env)
        if not token:
            raise IRIError(f"no IRI token: set {self.facility.token_env}")
        return token

    def _request(
        self,
        method: str,
        path: str,
        *,
        payload: Mapping[str, Any] | None = None,
        authenticated: bool = True,
        extra_headers: Mapping[str, str] | None = None,
    ) -> Any:
        headers = {"Accept": "application/json"}
        if authenticated:
            headers["Authorization"] = f"Bearer {self._token_value()}"
        body = None
        if payload is not None:
            headers["Content-Type"] = "application/json"
            body = json.dumps(payload).encode()
        headers.update(extra_headers or {})
        url = f"{self.facility.base_url}{path}"
        try:
            status, raw = self._transport(method, url, headers, body, self._timeout)
        except OSError as error:
            raise IRIError(f"{method} {path} failed: {error}") from error
        if status >= 400:
            detail = raw.decode(errors="replace")[:500]
            raise IRIError(f"{method} {path} returned HTTP {status}: {detail}", status=status)
        return json.loads(raw) if raw else None

    def _resource_path(self, prefix: str, job_id: str | None = None) -> str:
        parts = [self.facility.compute_resource_id]
        if job_id is not None:
            parts.append(job_id)
        return f"/{prefix}/" + "/".join(urllib.parse.quote(part, safe="") for part in parts)

    def job_spec(
        self,
        command: Sequence[str],
        *,
        resources: ResourceRequest,
        workdir: str,
        env: dict[str, str] | None = None,
        name: str | None = None,
    ) -> dict[str, Any]:
        """Translate a scheduler-neutral request into an IRI ``JobSpec`` payload."""

        if not command:
            raise ValueError("command must not be empty")
        resource_spec: dict[str, Any] = {
            "node_count": resources.nodes,
            "processes_per_node": resources.tasks_per_node,
            "process_count": resources.nodes * resources.tasks_per_node,
        }
        if resources.gpus_per_node > 0:
            resource_spec["gpu_cores_per_process"] = max(
                1, resources.gpus_per_node // resources.tasks_per_node
            )
        attributes: dict[str, Any] = {
            "custom_attributes": dict(self.facility.custom_attributes),
        }
        if resources.walltime:
            attributes["duration"] = walltime_seconds(resources.walltime)
        queue = resources.queue or self.facility.queue
        if queue:
            attributes["queue_name"] = queue
        account = resources.account or self.account
        if account:
            attributes["account"] = account
        spec: dict[str, Any] = {
            "executable": command[0],
            "arguments": list(command[1:]),
            "directory": workdir,
            "resources": resource_spec,
            "attributes": attributes,
        }
        if env:
            spec["environment"] = dict(env)
        if name:
            spec["name"] = name
        return spec

    def submit(
        self,
        command: Sequence[str],
        *,
        resources: ResourceRequest,
        workdir: str,
        env: dict[str, str] | None = None,
        idempotency_key: str | None = None,
    ) -> str:
        """Launch *command* on the facility compute resource and return the job id."""

        spec = self.job_spec(command, resources=resources, workdir=workdir, env=env)
        response = self._request(
            "POST",
            self._resource_path("compute/job"),
            payload=spec,
            extra_headers={"Idempotency-Key": idempotency_key or str(uuid.uuid4())},
        )
        job_id = (response or {}).get("id") or (response or {}).get("job_id")
        if not job_id:
            raise IRIError(f"IRI submit response has no job id: {response!r}")
        return str(job_id)

    def job(self, job_id: str) -> dict[str, Any]:
        """Return the IRI ``Job`` record, including ``status.state`` and ``exit_code``."""

        return self._request("GET", self._resource_path("compute/status", job_id))

    def job_state(self, job_id: str) -> str:
        status = self.job(job_id).get("status") or {}
        return str(status.get("state", "unknown")).lower()

    def wait(
        self,
        job_id: str,
        *,
        timeout: float | None = None,
        poll_interval: float = 30.0,
        max_interval: float = 300.0,
        backoff: float = 1.5,
        max_transient_errors: int = 5,
    ) -> dict[str, Any]:
        """Poll until the job is completed, failed, or canceled and return its final record.

        Network failures and HTTP 5xx responses are retried up to
        ``max_transient_errors`` consecutive times; other errors propagate.
        Raises :class:`IRIWaitTimeout` if ``timeout`` seconds elapse first.
        """

        if poll_interval <= 0 or max_interval < poll_interval or backoff < 1:
            raise ValueError("need 0 < poll_interval <= max_interval and backoff >= 1")
        deadline = None if timeout is None else time.monotonic() + timeout
        interval = poll_interval
        state = "unknown"
        errors = 0
        while True:
            try:
                record = self.job(job_id) or {}
            except IRIError as error:
                transient = error.status is None or error.status >= 500
                errors += 1
                if not transient or errors > max_transient_errors:
                    raise
            else:
                errors = 0
                state = str((record.get("status") or {}).get("state", "unknown")).lower()
                if state in TERMINAL_STATES:
                    return record
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise IRIWaitTimeout(
                        f"job {job_id} still {state!r} after {timeout}s", last_state=state
                    )
                time.sleep(min(interval, remaining))
            else:
                time.sleep(interval)
            interval = min(interval * backoff, max_interval)

    def cancel(self, job_id: str) -> None:
        self._request("DELETE", self._resource_path("compute/cancel", job_id))

    def resource_status(self) -> str:
        """Return the public ``up``/``down``/``degraded``/``unknown`` status of the resource."""

        path = self._resource_path("status/resources")
        record = self._request("GET", path, authenticated=False)
        return str((record or {}).get("current_status", "unknown"))

    def available_resources(self) -> ResourceRequest:
        """Return the configured queue and account; IRI exposes no allocation sizes yet."""

        return ResourceRequest(queue=self.facility.queue, account=self.account)

    def resolved_working_directory(self) -> str | None:
        template = self.facility.working_directory
        if template is None:
            return None
        if "{account}" in template and not self.account:
            return None
        return template.format(account=self.account)


__all__ = [
    "IRIError",
    "IRIFacility",
    "IRIPlatform",
    "IRIWaitTimeout",
    "KNOWN_FACILITIES",
    "TERMINAL_STATES",
    "walltime_seconds",
]
