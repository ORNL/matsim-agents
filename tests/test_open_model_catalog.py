from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "deployments" / "common" / "open-model-catalog.json"
MANIFESTS = (
    ROOT / "deployments" / "frontier" / "config" / "all_model_specs.json",
    ROOT
    / "deployments"
    / "frontier"
    / "config"
    / "six_model_specs.vllm-only.json",
)
DOWNLOADERS = tuple(
    ROOT / "deployments" / facility / "download" / script
    for facility in ("frontier", "aurora", "perlmutter")
    for script in (
        "download-models-" + facility + ".sh",
        "download-open-models-" + facility + ".sh",
    )
)
BENCHMARK = (
    ROOT
    / "deployments"
    / "frontier"
    / "jobs"
    / "job-sequential-benchmark-frontier.sh"
)


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_first_class_open_models_are_present_in_every_deployment_path() -> None:
    """Backend-compatible catalog entries must have all first-class assets.

    A generic vLLM endpoint can technically serve many checkpoints.  This
    invariant prevents the documented first-class set from drifting behind:
    every catalog model must be benchmark-addressable and downloadable on all
    three supported DOE deployment layouts.
    """
    catalog = _read_json(CATALOG)
    models = {entry["model"] for entry in catalog}
    assert len(models) == len(catalog), "catalog model IDs must be unique"
    assert len({entry["name"] for entry in catalog}) == len(catalog)
    assert len({entry["base_url_env"] for entry in catalog}) == len(catalog)

    for manifest in MANIFESTS:
        configured = {
            entry["model"]
            for entry in _read_json(manifest)
            if entry["provider"] == "vllm"
        }
        assert models <= configured, f"{manifest} misses {sorted(models - configured)}"

    for downloader in DOWNLOADERS:
        contents = downloader.read_text(encoding="utf-8")
        missing = sorted(model for model in models if f'"{model}"' not in contents)
        assert not missing, f"{downloader} misses {missing}"

    benchmark = BENCHMARK.read_text(encoding="utf-8")
    missing = sorted(model for model in models if model not in benchmark)
    assert not missing, f"{BENCHMARK} misses {missing}"
