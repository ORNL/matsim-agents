from __future__ import annotations

import http.client
import json

import pytest

from matsim_agents.execution import ExecutionPlatform, iri
from matsim_agents.execution.iri import (
    KNOWN_FACILITIES,
    IRIError,
    IRIFacility,
    IRIPlatform,
    IRIWaitTimeout,
    walltime_seconds,
)
from matsim_agents.execution.resources import ResourceRequest


class FakeTransport:
    def __init__(self, responses: list[tuple[int, object]]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def __call__(self, method, url, headers, body, timeout):
        self.calls.append(
            {
                "method": method,
                "url": url,
                "headers": dict(headers),
                "body": json.loads(body) if body else None,
                "timeout": timeout,
            }
        )
        status, payload = self.responses.pop(0)
        raw = b"" if payload is None else json.dumps(payload).encode()
        return status, raw


def frontier(transport: FakeTransport, **kwargs) -> IRIPlatform:
    return IRIPlatform(
        "olcf-frontier", token="tok", account="mat123", transport=transport, **kwargs
    )


def test_platform_satisfies_protocol():
    assert isinstance(frontier(FakeTransport([])), ExecutionPlatform)


@pytest.mark.parametrize(
    ("text", "seconds"),
    [("300", 300), ("05:00", 300), ("02:00:00", 7200), ("1-00:00:30", 86430)],
)
def test_walltime_seconds(text, seconds):
    assert walltime_seconds(text) == seconds


def test_walltime_rejects_zero():
    with pytest.raises(ValueError):
        walltime_seconds("00:00")


def test_submit_posts_job_spec_to_frontier():
    transport = FakeTransport([(200, {"id": "4242", "status": {"state": "queued"}})])
    platform = frontier(transport)

    job_id = platform.submit(
        ["bash", "deployments/frontier/launchers/run-pw-gpu-frontier.sh", "si.in"],
        resources=ResourceRequest(nodes=2, tasks_per_node=8, gpus_per_node=8, walltime="01:00:00"),
        workdir="/lustre/orion/mat123/proj-shared/run1",
        env={"OMP_NUM_THREADS": "7"},
        idempotency_key="key-1",
    )

    assert job_id == "4242"
    call = transport.calls[0]
    assert call["method"] == "POST"
    assert call["url"] == "https://amsc-moderate.s3m.olcf.ornl.gov/api/v2/compute/job/frontier"
    assert call["headers"]["Authorization"] == "Bearer tok"
    assert call["headers"]["Idempotency-Key"] == "key-1"
    assert call["body"] == {
        "executable": "bash",
        "arguments": ["deployments/frontier/launchers/run-pw-gpu-frontier.sh", "si.in"],
        "directory": "/lustre/orion/mat123/proj-shared/run1",
        "resources": {
            "node_count": 2,
            "processes_per_node": 8,
            "process_count": 16,
            "gpu_cores_per_process": 1,
        },
        "attributes": {
            "custom_attributes": {},
            "duration": 3600,
            "queue_name": "batch",
            "account": "mat123",
        },
        "environment": {"OMP_NUM_THREADS": "7"},
    }


def test_request_overrides_facility_queue_and_account():
    platform = frontier(FakeTransport([]))
    spec = platform.job_spec(
        ["true"],
        resources=ResourceRequest(queue="extended", account="other"),
        workdir="/tmp",
    )
    assert spec["attributes"]["queue_name"] == "extended"
    assert spec["attributes"]["account"] == "other"
    assert "gpu_cores_per_process" not in spec["resources"]


@pytest.mark.parametrize(("tasks", "gpus"), [(8, 4), (3, 8)])
def test_job_spec_rejects_fractional_gpus_per_process(tasks, gpus):
    with pytest.raises(ValueError, match="whole number of GPUs"):
        frontier(FakeTransport([])).job_spec(
            ["true"],
            resources=ResourceRequest(tasks_per_node=tasks, gpus_per_node=gpus),
            workdir="/tmp",
        )


def test_redirect_is_an_error():
    transport = FakeTransport([(302, None)])
    with pytest.raises(IRIError) as excinfo:
        frontier(transport).job("7")
    assert excinfo.value.status == 302


def test_urllib_transport_does_not_follow_redirects():
    handler = iri._NoRedirect()
    assert handler.redirect_request(None, None, 302, "Found", {}, "http://evil") is None


def test_submit_omits_idempotency_key_by_default():
    transport = FakeTransport([(200, {"id": "1"})])
    frontier(transport).submit(["true"], resources=ResourceRequest(), workdir="/tmp")
    assert "Idempotency-Key" not in transport.calls[0]["headers"]


def test_submit_without_job_id_raises():
    with pytest.raises(IRIError, match="no job id"):
        frontier(FakeTransport([(200, {})])).submit(
            ["true"], resources=ResourceRequest(), workdir="/tmp"
        )


def test_http_error_carries_status():
    transport = FakeTransport([(403, {"detail": "forbidden"})])
    with pytest.raises(IRIError) as excinfo:
        frontier(transport).submit(["true"], resources=ResourceRequest(), workdir="/tmp")
    assert excinfo.value.status == 403
    assert "forbidden" in str(excinfo.value)


def test_job_state_and_cancel_paths():
    transport = FakeTransport(
        [(200, {"id": "7", "status": {"state": "COMPLETED", "exit_code": 0}}), (204, None)]
    )
    platform = frontier(transport)

    assert platform.job_state("7") == "completed"
    platform.cancel("7")

    base = "https://amsc-moderate.s3m.olcf.ornl.gov/api/v2"
    assert transport.calls[0]["url"] == f"{base}/compute/status/frontier/7"
    assert transport.calls[1]["method"] == "DELETE"
    assert transport.calls[1]["url"] == f"{base}/compute/cancel/frontier/7"


def test_job_id_is_path_escaped():
    transport = FakeTransport([(200, {"id": "x", "status": {"state": "active"}})])
    frontier(transport).job_state("../../account/whoami")
    assert transport.calls[0]["url"].endswith("/compute/status/frontier/..%2F..%2Faccount%2Fwhoami")


def test_resource_status_is_unauthenticated():
    transport = FakeTransport([(200, {"id": "frontier", "current_status": "up"})])
    platform = IRIPlatform("olcf-frontier", transport=transport)

    assert platform.resource_status() == "up"
    assert "Authorization" not in transport.calls[0]["headers"]
    assert transport.calls[0]["url"].endswith("/status/resources/frontier")


def test_resource_status_null_is_unknown():
    transport = FakeTransport([(200, {"id": "frontier", "current_status": None})])
    assert IRIPlatform("olcf-frontier", transport=transport).resource_status() == "unknown"


def test_job_state_null_is_unknown():
    transport = FakeTransport([(200, {"id": "7", "status": {"state": None}})])
    assert frontier(transport).job_state("7") == "unknown"


def truncated_transport(method, url, headers, body, timeout):
    raise http.client.IncompleteRead(b"{", 10)


def test_truncated_response_raises_status_less_iri_error():
    platform = IRIPlatform("olcf-frontier", token="tok", transport=truncated_transport)
    with pytest.raises(IRIError) as excinfo:
        platform.job("7")
    assert excinfo.value.status is None


def test_missing_token_names_environment_variable(monkeypatch):
    monkeypatch.delenv("IRI_TOKEN_OLCF", raising=False)
    platform = IRIPlatform("olcf-frontier", transport=FakeTransport([]))
    with pytest.raises(IRIError, match="IRI_TOKEN_OLCF"):
        platform.submit(["true"], resources=ResourceRequest(), workdir="/tmp")


def test_account_and_working_directory_from_environment(monkeypatch):
    monkeypatch.setenv("OLCF_S3M_PROJECT", "mat999")
    platform = IRIPlatform("olcf-frontier", transport=FakeTransport([]))

    assert platform.available_resources() == ResourceRequest(queue="batch", account="mat999")
    assert platform.resolved_working_directory() == "/lustre/orion/mat999/proj-shared"


def test_unknown_facility_lists_known_names():
    with pytest.raises(ValueError, match="olcf-frontier"):
        IRIPlatform("nowhere")


def test_facility_requires_https():
    with pytest.raises(ValueError, match="https"):
        IRIFacility(name="x", base_url="http://example.org", compute_resource_id="c", token_env="T")


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def clock(monkeypatch) -> FakeClock:
    fake = FakeClock()
    monkeypatch.setattr(iri.time, "monotonic", fake.monotonic)
    monkeypatch.setattr(iri.time, "sleep", fake.sleep)
    return fake


def state(name: str, **extra) -> tuple[int, dict]:
    return 200, {"id": "7", "status": {"state": name, **extra}}


def test_wait_returns_terminal_record_with_backoff(clock):
    transport = FakeTransport(
        [state("queued"), state("active"), state("active"), state("completed", exit_code=0)]
    )
    record = frontier(transport).wait("7", poll_interval=10, max_interval=20, backoff=2)

    assert record["status"] == {"state": "completed", "exit_code": 0}
    assert clock.sleeps == [10, 20, 20]


@pytest.mark.parametrize("terminal", ["failed", "CANCELED"])
def test_wait_stops_on_any_terminal_state(clock, terminal):
    record = frontier(FakeTransport([state(terminal)])).wait("7")
    assert record["status"]["state"] == terminal
    assert clock.sleeps == []


def test_wait_timeout_reports_last_state(clock):
    transport = FakeTransport([state("queued")] * 10)
    with pytest.raises(IRIWaitTimeout) as excinfo:
        frontier(transport).wait("7", timeout=25, poll_interval=10, backoff=1)

    assert excinfo.value.last_state == "queued"
    assert clock.sleeps == [10, 10, 5]
    assert len(transport.calls) == 3


def test_wait_caps_request_timeout_to_remaining_budget(clock):
    transport = FakeTransport([state("queued"), state("completed")])
    frontier(transport).wait("7", timeout=12, poll_interval=10)
    assert [call["timeout"] for call in transport.calls] == [12, 2]


def test_wait_without_token_fails_immediately(clock, monkeypatch):
    monkeypatch.delenv("IRI_TOKEN_OLCF", raising=False)
    platform = IRIPlatform("olcf-frontier", transport=FakeTransport([]))
    with pytest.raises(IRIError, match="IRI_TOKEN_OLCF"):
        platform.wait("7")
    assert clock.sleeps == []


def test_wait_retries_transient_errors(clock):
    transport = FakeTransport([(503, {"detail": "busy"}), state("completed")])
    assert frontier(transport).wait("7", poll_interval=1)["status"]["state"] == "completed"


def test_wait_retries_truncated_response(clock):
    responses = FakeTransport([state("completed")])
    calls = []

    def flaky(method, url, headers, body, timeout):
        calls.append(url)
        if len(calls) == 1:
            raise http.client.IncompleteRead(b"{", 10)
        return responses(method, url, headers, body, timeout)

    platform = IRIPlatform("olcf-frontier", token="tok", transport=flaky)
    assert platform.wait("7", poll_interval=1)["status"]["state"] == "completed"
    assert len(calls) == 2


def test_wait_gives_up_after_repeated_transient_errors(clock):
    transport = FakeTransport([(500, None)] * 3)
    with pytest.raises(IRIError) as excinfo:
        frontier(transport).wait("7", poll_interval=1, max_transient_errors=2)
    assert excinfo.value.status == 500


def test_wait_does_not_retry_client_errors(clock):
    transport = FakeTransport([(404, {"detail": "no such job"})])
    with pytest.raises(IRIError) as excinfo:
        frontier(transport).wait("7")
    assert excinfo.value.status == 404
    assert clock.sleeps == []


def test_wait_rejects_bad_intervals():
    with pytest.raises(ValueError):
        frontier(FakeTransport([])).wait("7", poll_interval=0)


def test_known_facilities_use_https():
    assert all(f.base_url.startswith("https://") for f in KNOWN_FACILITIES.values())
