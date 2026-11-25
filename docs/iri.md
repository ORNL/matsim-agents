# Running jobs through the DOE IRI Facility API

matsim-agents can submit jobs through the DOE Integrated Research Infrastructure
(IRI) Facility API, a facility-neutral REST interface to NERSC, ALCF, and OLCF
compute resources:

- IRI Facility API: <https://github.com/doe-iri/iri-facility-api-python>

IRI is an **optional** alternative to manual submission. The existing
`sbatch`/`qsub` scripts under [`deployments/`](../deployments/README.md) for
Frontier, Aurora, and Perlmutter are unchanged and remain the default.

## Facility profiles

The package carries no facility endpoints, queues, or filesystem paths. They live in a
profile file owned by the deployment:
[`deployments/common/iri-profiles.yaml`](../deployments/common/iri-profiles.yaml). Pass it as
`IRIPlatform(name, profiles=path)`, or pass an explicit `IRIFacility` object. A profile
may name the variable holding the run directory (`working_directory_env`, for example
`OLCF_IRI_WORKDIR`) or give a `working_directory` template (`{account}` filled from
`account_env`); when neither resolves, `resolved_working_directory()` returns `None` and the
caller must pass `workdir` explicitly. Naming a facility without a profile file raises `ValueError`.

## Supported facilities (shipped profiles)

| Name | Facility | Base URL | Token variable | Project variable |
|---|---|---|---|---|
| `nersc-perlmutter` | NERSC Perlmutter | `https://api.iri.nersc.gov/api/v1` | `IRI_TOKEN_NERSC` | `NERSC_PROJECT` |
| `alcf-aurora` | ALCF Aurora | `https://api.alcf.anl.gov/api/v1` | `IRI_TOKEN_ALCF` | `ALCF_PROJECT` |
| `olcf-frontier` | OLCF Frontier (S3M, moderate) | `https://amsc-moderate.s3m.olcf.ornl.gov/api/v2` | `IRI_TOKEN_OLCF` | `OLCF_S3M_PROJECT` |
| `olcf-odo` | OLCF Odo (S3M, open) | `https://amsc-open.s3m.olcf.ornl.gov/api/v2` | `IRI_TOKEN_OLCF` | `OLCF_S3M_PROJECT` |

## Environment variables

Set the token and project for each facility you use. Obtain tokens from the
facility (for OLCF, myOLCF → S3M Access → New Token with IRI permissions).

```bash
# OLCF (Frontier / Odo via S3M)
export IRI_TOKEN_OLCF="<token>"
export OLCF_S3M_PROJECT="<project id, e.g. abc123>"

# NERSC (Perlmutter)
export IRI_TOKEN_NERSC="<token>"
export NERSC_PROJECT="<project>"

# ALCF (Aurora)
export IRI_TOKEN_ALCF="<token>"
export ALCF_PROJECT="<project>"
```

- Tokens are bearer credentials: never commit them or pass them on the command line.
- Add the exports to the startup file of the shell you actually use
  (`~/.zshrc` on macOS by default, `~/.bashrc` for bash).
- The project variable fills the job `account`. Export `OLCF_IRI_WORKDIR` for the OLCF
  profiles; without it the caller passes `workdir`.

## Basic usage

```python
from matsim_agents.execution import IRIPlatform
from matsim_agents.execution.resources import ResourceRequest

platform = IRIPlatform("olcf-frontier", profiles="deployments/common/iri-profiles.yaml")
print(platform.resource_status())  # "up", "down", "degraded", or "unknown"

job_id = platform.submit(
    ["/bin/bash", "-lc", "hostname"],
    resources=ResourceRequest(nodes=1, walltime="00:05:00"),
    workdir=platform.resolved_working_directory() or "/path/to/run/directory",
)
record = platform.wait(job_id, timeout=3600)
print(record["status"])
```

`submit`, `job`, `job_state`, `wait`, and `cancel` map to the IRI
`/compute/job`, `/compute/status`, and `/compute/cancel` endpoints. Failures
raise `IRIError` (with the HTTP `status`); `wait` raises `IRIWaitTimeout` when
its deadline passes.

## Status

The client is covered by unit tests. Live submission to OLCF Frontier via S3M
has not yet been validated end to end.
