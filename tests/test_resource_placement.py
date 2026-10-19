"""Resolution and conflict rules of the resource-dispatch planner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from matsim_agents.execution.allocation import Allocation
from matsim_agents.execution.placement import ResourcePlan
from matsim_agents.execution.presets import PRESETS, get_preset

ROOT = Path(__file__).resolve().parents[1]
LAYOUTS = sorted((ROOT / "deployments").glob("*/resources/*.yaml"))


def _alloc(nodes: int = 1, gpus: int | None = None, scheduler: str = "slurm") -> Allocation:
    return Allocation(
        scheduler=scheduler,
        nodes=tuple(f"nid{index:06d}" for index in range(nodes)),
        gpus_per_node=gpus,
    )


def _plan(pools: dict, preset: str | None = "perlmutter-gpu", **extra) -> ResourcePlan:
    data = {"pools": pools, **extra}
    if preset is not None:
        data["preset"] = preset
    return ResourcePlan.model_validate(data)


VLLM = {"name": "proposer", "kind": "managed-vllm", "model": "/models/Qwen2.5-14B-Instruct"}


# -- schema -------------------------------------------------------------------


def test_unknown_pool_is_rejected():
    with pytest.raises(ValidationError, match="unknown pool 'gpu'"):
        _plan({"gpu": {}})


def test_unknown_field_is_rejected():
    with pytest.raises(ValidationError, match="ranks_per_node"):
        _plan({"dft": {"ranks_per_node": 4}})


def test_unknown_preset_is_rejected():
    with pytest.raises(ValidationError, match="unknown resource preset 'summit'"):
        _plan({"mlip": {}}, preset="summit")


def test_share_with_must_be_symmetric():
    with pytest.raises(ValidationError, match="sharing must be declared on both pools"):
        _plan({"mlip": {"share_with": ["dft"]}, "dft": {}})


def test_share_with_requires_declared_pool():
    with pytest.raises(ValidationError, match="undeclared pool 'dft'"):
        _plan({"mlip": {"share_with": ["dft"]}})


def test_only_one_rest_pool():
    with pytest.raises(ValidationError, match="only one pool may use nodes: rest"):
        _plan({"mlip": {"nodes": "rest"}, "dft": {"nodes": "rest"}})


def test_server_names_are_unique():
    with pytest.raises(ValidationError, match="duplicate server names: proposer"):
        _plan({"llm": {"servers": [VLLM, VLLM]}})


@pytest.mark.parametrize(
    ("server", "message"),
    [
        ({"name": "x", "kind": "external"}, "requires base_url"),
        (
            {"name": "x", "kind": "external", "base_url": "http://h/v1", "devices": [0]},
            "takes no devices",
        ),
        ({"name": "x", "kind": "managed-vllm"}, "requires model"),
        (
            {"name": "x", "kind": "managed-vllm", "model": "/m", "base_url": "http://h/v1"},
            "remove base_url",
        ),
        (
            {"name": "x", "kind": "in-process", "model": "/m", "max_model_len": 8192},
            "does not accept max_model_len",
        ),
    ],
)
def test_server_kind_fields(server, message):
    with pytest.raises(ValidationError, match=message):
        _plan({"llm": {"servers": [server]}})


def test_external_only_llm_pool_takes_no_nodes():
    server = {"name": "remote", "kind": "external", "base_url": "https://llm.example.org/v1"}
    with pytest.raises(ValidationError, match="only external servers and takes no nodes"):
        _plan({"llm": {"nodes": 1, "servers": [server]}})


# -- geometry -----------------------------------------------------------------


def test_preset_geometry_and_launcher():
    placement = _plan({"mlip": {}}, preset="aurora").resolve(_alloc(gpus=None, scheduler="pbs"))
    assert placement.geometry == get_preset("aurora").geometry
    assert placement.launcher == "mpiexec"
    env = placement.pool("mlip").slices[0].env
    assert env["ZE_AFFINITY_MASK"] == ",".join(str(i) for i in range(12))
    assert env["ZE_FLAT_DEVICE_HIERARCHY"] == "FLAT"


def test_frontier_mask_variable():
    placement = _plan({"mlip": {"devices": [2, 3]}}, preset="frontier").resolve(_alloc(gpus=8))
    assert placement.pool("mlip").slices[0].env == {"ROCR_VISIBLE_DEVICES": "2,3"}


def test_scheduler_device_count_overrides_preset():
    placement = _plan({"mlip": {}}).resolve(_alloc(gpus=2))
    assert placement.pool("mlip").slices[0].devices == (0, 1)


def test_explicit_geometry_contradicting_scheduler_fails():
    plan = _plan({"mlip": {}}, geometry={"devices_per_node": 4})
    with pytest.raises(ValueError, match="contradicts the 2 device"):
        plan.resolve(_alloc(gpus=2))


def test_geometry_required_without_preset():
    with pytest.raises(ValueError, match="geometry.vendor"):
        _plan({"mlip": {}}, preset=None).resolve(_alloc(gpus=1))


def test_cpu_geometry_claims_whole_nodes():
    plan = _plan(
        {"mlip": {"nodes": 1}, "dft": {"nodes": "rest"}},
        preset=None,
        geometry={"vendor": "cpu", "devices_per_node": 0, "cores_per_node": 128},
    )
    placement = plan.resolve(_alloc(nodes=3, gpus=None))
    assert placement.pool("mlip").hosts == ["nid000000"]
    dft = placement.pool("dft")
    assert dft.hosts == ["nid000001", "nid000002"]
    assert dft.ranks_per_node == 128 and dft.threads_per_rank == 1
    assert dft.slices[0].env == {}


def test_cpu_geometry_rejects_device_lists():
    plan = _plan(
        {"mlip": {"devices": [0]}},
        preset=None,
        geometry={"vendor": "cpu", "devices_per_node": 0, "cores_per_node": 8},
    )
    with pytest.raises(ValueError, match="no accelerator devices"):
        plan.resolve(_alloc(gpus=None))


# -- nodes and devices ----------------------------------------------------------


def test_disjoint_pools_on_one_node():
    plan = _plan(
        {
            "llm": {"devices": [0], "servers": [VLLM]},
            "mlip": {"devices": [1, 2, 3]},
        }
    )
    placement = plan.resolve(_alloc(gpus=4))
    assert placement.pool("llm").slices[0].env == {"CUDA_VISIBLE_DEVICES": "0"}
    assert placement.pool("mlip").slices[0].devices == (1, 2, 3)
    assert placement.pool("mlip").max_concurrent == 3
    assert placement.idle == [] and placement.warnings == []


def test_overlapping_pools_without_share_with_fail():
    plan = _plan({"llm": {"servers": [VLLM]}, "mlip": {"devices": [0]}})
    with pytest.raises(ValueError, match="device 0 on nid000000 overlaps pool 'llm'"):
        plan.resolve(_alloc(gpus=4))


def test_integer_nodes_skip_hosts_without_free_devices():
    plan = _plan({"llm": {"servers": [VLLM]}, "mlip": {"nodes": 1}})
    placement = plan.resolve(_alloc(nodes=2, gpus=4))
    assert placement.pool("llm").hosts == ["nid000000"]
    assert placement.pool("mlip").hosts == ["nid000001"]


def test_integer_devices_take_lowest_free_indices():
    plan = _plan({"training": {"devices": [0]}, "mlip": {"devices": 2}})
    placement = plan.resolve(_alloc(gpus=4))
    assert placement.pool("mlip").slices[0].devices == (1, 2)
    assert "nid000000: [3]" in placement.warnings[0]


def test_rest_takes_untouched_nodes_last():
    plan = _plan({"dft": {"nodes": "rest"}, "mlip": {"nodes": 1, "devices": [0]}})
    placement = plan.resolve(_alloc(nodes=3, gpus=4))
    assert placement.pool("mlip").hosts == ["nid000000"]
    assert placement.pool("dft").hosts == ["nid000001", "nid000002"]
    assert list(placement.pools) == ["dft", "mlip"]


def test_rest_with_no_nodes_left_fails():
    plan = _plan({"mlip": {"nodes": "all"}, "dft": {"nodes": "rest"}})
    with pytest.raises(ValueError, match="every node is used by another pool"):
        plan.resolve(_alloc(nodes=2, gpus=4))


def test_explicit_hosts():
    plan = _plan({"mlip": {"nodes": ["nid000001"]}})
    placement = plan.resolve(_alloc(nodes=2, gpus=4))
    assert placement.pool("mlip").hosts == ["nid000001"]


def test_unknown_host_fails():
    with pytest.raises(ValueError, match="host 'nid999999' is not available"):
        _plan({"mlip": {"nodes": ["nid999999"]}}).resolve(_alloc(nodes=2, gpus=4))


def test_too_many_nodes_fails():
    with pytest.raises(ValueError, match="requests 3 node"):
        _plan({"mlip": {"nodes": 3}}).resolve(_alloc(nodes=2, gpus=4))


def test_missing_device_index_fails():
    with pytest.raises(ValueError, match="device 4 does not exist on nid000000"):
        _plan({"mlip": {"devices": [4]}}).resolve(_alloc(gpus=4))


def test_duplicate_device_index_fails():
    with pytest.raises(ValueError, match="lists a device more than once"):
        _plan({"mlip": {"devices": [1, 1]}}).resolve(_alloc(gpus=4))


def test_allocation_must_have_unique_nodes():
    allocation = Allocation(scheduler="slurm", nodes=("a", "a"), gpus_per_node=4)
    with pytest.raises(ValueError, match="more than once"):
        _plan({"mlip": {}}).resolve(allocation)


# -- sharing and memory budgets -------------------------------------------------


def _shared_plan(llm_fraction, mlip_fraction, kind="in-process"):
    server = {"name": "proposer", "kind": kind, "model": "/models/Qwen2.5-72B-Instruct"}
    llm = {"devices": "all", "share_with": ["mlip"], "servers": [server]}
    mlip = {"devices": [0], "share_with": ["llm"]}
    if llm_fraction is not None:
        llm["memory_fraction"] = llm_fraction
    if mlip_fraction is not None:
        mlip["memory_fraction"] = mlip_fraction
    return _plan({"llm": llm, "mlip": mlip})


def test_declared_sharing_is_allowed():
    placement = _shared_plan(0.75, 0.2).resolve(_alloc(gpus=4))
    assert placement.pool("llm").slices[0].devices == (0, 1, 2, 3)
    assert placement.pool("mlip").slices[0].devices == (0,)
    assert placement.pool("llm").server("proposer").memory_fraction == 0.75


def test_shared_memory_budget_limit():
    with pytest.raises(ValueError, match=r"sum to 1\.00, above the 0\.95 limit"):
        _shared_plan(0.8, 0.2).resolve(_alloc(gpus=4))


def test_managed_vllm_on_shared_device_requires_budget():
    with pytest.raises(ValueError, match="set memory_fraction on pool 'llm'"):
        _shared_plan(None, 0.2, kind="managed-vllm").resolve(_alloc(gpus=4))


def test_in_process_on_shared_device_does_not_require_budget():
    placement = _shared_plan(None, None).resolve(_alloc(gpus=4))
    assert placement.pool("llm").memory_fraction is None


# -- LLM servers ----------------------------------------------------------------


def test_single_server_defaults_to_whole_pool():
    placement = _plan({"llm": {"devices": [0, 1], "servers": [VLLM]}}).resolve(_alloc(gpus=4))
    server = placement.pool("llm").server("proposer")
    assert server.slices[0].devices == (0, 1)
    assert server.tensor_parallel_size == 2
    assert server.pipeline_parallel_size == 1
    assert server.served_name == "Qwen2.5-14B-Instruct"


def test_campaign_panel_servers_take_slices_in_order():
    servers = [
        {"name": "kimi", "kind": "managed-vllm", "model": "/models/Kimi-K2.5", "nodes": 4},
        {"name": "glm", "kind": "managed-vllm", "model": "/models/GLM-4.7", "nodes": 2},
        {
            "name": "gemma",
            "kind": "managed-vllm",
            "model": "/models/gemma-4-31B-it",
            "nodes": 1,
            "devices": [0, 1],
        },
        {
            "name": "qwen",
            "kind": "managed-vllm",
            "model": "/models/Qwen2.5-14B-Instruct",
            "nodes": 1,
            "devices": [2],
        },
        {"name": "remote", "kind": "external", "base_url": "https://llm.example.org/v1"},
    ]
    plan = _plan(
        {
            "llm": {"nodes": 7, "servers": servers},
            "mlip": {"nodes": 1},
            "dft": {"nodes": "rest"},
        }
    )
    placement = plan.resolve(_alloc(nodes=16, gpus=4))
    llm = placement.pool("llm")
    hosts = [f"nid{index:06d}" for index in range(16)]
    assert [s.host for s in llm.server("kimi").slices] == hosts[0:4]
    assert llm.server("kimi").tensor_parallel_size == 4
    assert llm.server("kimi").pipeline_parallel_size == 4
    assert [s.host for s in llm.server("glm").slices] == hosts[4:6]
    assert llm.server("gemma").slices[0].host == hosts[6]
    assert llm.server("qwen").slices[0].host == hosts[6]
    assert llm.server("qwen").slices[0].devices == (2,)
    assert llm.server("remote").slices == []
    assert placement.pool("mlip").hosts == [hosts[7]]
    assert placement.pool("dft").hosts == hosts[8:]
    assert any("nid000006: [3]" in warning for warning in placement.warnings)


def test_several_servers_require_explicit_slices():
    second = {**VLLM, "name": "critic"}
    with pytest.raises(ValueError, match="set nodes"):
        _plan({"llm": {"servers": [VLLM, second]}}).resolve(_alloc(gpus=4))


def test_servers_cannot_overlap():
    first = {**VLLM, "nodes": 1, "devices": [0, 1]}
    second = {**VLLM, "name": "critic", "nodes": ["nid000000"], "devices": [1]}
    with pytest.raises(ValueError, match="already used by LLM server 'proposer'"):
        _plan({"llm": {"servers": [first, second]}}).resolve(_alloc(gpus=4))


def test_server_devices_must_lie_in_pool():
    server = {**VLLM, "devices": [3]}
    with pytest.raises(ValueError, match="outside pool 'llm'"):
        _plan({"llm": {"devices": [0, 1], "servers": [server]}}).resolve(_alloc(gpus=4))


def test_in_process_server_is_single_node():
    server = {"name": "hf", "kind": "in-process", "model": "/models/x"}
    with pytest.raises(ValueError, match="in-process and must use exactly 1 node"):
        _plan({"llm": {"nodes": 2, "servers": [server]}}).resolve(_alloc(nodes=2, gpus=4))


def test_tensor_parallel_must_divide_devices():
    server = {**VLLM, "tensor_parallel_size": 3}
    with pytest.raises(ValueError, match="tensor_parallel_size=3 does not divide its 4"):
        _plan({"llm": {"servers": [server]}}).resolve(_alloc(gpus=4))


def test_tensor_parallel_must_divide_attention_heads(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"num_attention_heads": 6}))
    server = {"name": "x", "kind": "managed-vllm", "model": str(model)}
    with pytest.raises(ValueError, match="does not divide the model's 6 attention heads"):
        _plan({"llm": {"servers": [server]}}).resolve(_alloc(gpus=4))


def test_attention_heads_from_text_config(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"text_config": {"num_attention_heads": 32}}))
    server = {"name": "x", "kind": "managed-vllm", "model": str(model)}
    placement = _plan({"llm": {"servers": [server]}}).resolve(_alloc(gpus=4))
    assert placement.pool("llm").server("x").tensor_parallel_size == 4


def test_external_only_llm_pool_claims_nothing():
    server = {"name": "remote", "kind": "external", "base_url": "https://llm.example.org/v1"}
    placement = _plan({"llm": {"servers": [server]}, "mlip": {}}).resolve(_alloc(gpus=4))
    assert placement.pool("llm").slices == []
    assert placement.pool("llm").server("remote").base_url == "https://llm.example.org/v1"
    assert placement.pool("mlip").slices[0].devices == (0, 1, 2, 3)


# -- component rules ------------------------------------------------------------


def test_dft_derives_ranks_threads_and_groups():
    plan = _plan({"dft": {"nodes": "all", "nodes_per_job": 2}}, preset="frontier")
    dft = plan.resolve(_alloc(nodes=4, gpus=8)).pool("dft")
    assert dft.ranks_per_node == 8
    assert dft.threads_per_rank == 7
    assert dft.job_groups == [["nid000000", "nid000001"], ["nid000002", "nid000003"]]


def test_dft_nodes_per_job_must_divide_pool():
    plan = _plan({"dft": {"nodes": "all", "nodes_per_job": 2}})
    with pytest.raises(ValueError, match="nodes_per_job=2 does not divide its 3"):
        plan.resolve(_alloc(nodes=3, gpus=4))


def test_dft_ranks_and_threads_must_fit_cores():
    plan = _plan({"dft": {"ranks_per_device": 4, "threads_per_rank": 8}})
    with pytest.raises(ValueError, match="16 rank\\(s\\) x 8 thread\\(s\\)"):
        plan.resolve(_alloc(gpus=4))


def test_training_requires_one_device():
    with pytest.raises(ValueError, match="pool 'training' must use exactly 1 device"):
        _plan({"training": {"devices": 2}}).resolve(_alloc(gpus=4))


def test_training_requires_one_node():
    with pytest.raises(ValueError, match="pool 'training' must use exactly 1 node"):
        _plan({"training": {"nodes": 2}}).resolve(_alloc(nodes=2, gpus=4))


def test_mlip_max_concurrent_cannot_exceed_devices():
    with pytest.raises(ValueError, match="max_concurrent=5 exceeds its 4"):
        _plan({"mlip": {"max_concurrent": 5}}).resolve(_alloc(gpus=4))


# -- loading and output ---------------------------------------------------------


def test_from_yaml_reads_resources_block_with_variables(tmp_path, monkeypatch):
    monkeypatch.setenv("MLIP_DEVICES", "[1, 2]")
    monkeypatch.delenv("MODEL_ROOT", raising=False)
    config = tmp_path / "al.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "vars": {"MODEL_ROOT": "/models"},
                "dft": {"backend": "qe"},
                "resources": {
                    "preset": "perlmutter-gpu",
                    "pools": {
                        "llm": {
                            "devices": [0],
                            "servers": [
                                {
                                    "name": "proposer",
                                    "kind": "managed-vllm",
                                    "model": "${MODEL_ROOT}/Qwen2.5-14B-Instruct",
                                }
                            ],
                        },
                        "mlip": {"devices": "${MLIP_DEVICES}"},
                    },
                },
            }
        )
    )
    plan = ResourcePlan.from_yaml(config)
    assert plan.pools["llm"].servers[0].model == "/models/Qwen2.5-14B-Instruct"
    assert plan.pools["mlip"].devices == [1, 2]


def test_from_yaml_requires_resources_block(tmp_path):
    config = tmp_path / "plain.yaml"
    config.write_text("dft: {backend: qe}\n")
    with pytest.raises(ValueError, match="no top-level 'resources:' mapping"):
        ResourcePlan.from_yaml(config)


def test_placement_round_trips_as_json(tmp_path):
    placement = _plan({"mlip": {"devices": [1]}}).resolve(_alloc(gpus=4))
    written = placement.write(tmp_path)
    assert written == tmp_path / "placement.json"
    data = json.loads(written.read_text())
    assert data["pools"]["mlip"]["slices"][0]["env"] == {"CUDA_VISIBLE_DEVICES": "1"}
    assert data["idle"] == [{"host": "nid000000", "devices": [0, 2, 3], "env": {}}]


# -- shipped layouts ------------------------------------------------------------


def test_layouts_are_shipped_for_every_facility():
    facilities = {path.parts[-3] for path in LAYOUTS}
    assert {"perlmutter", "frontier", "aurora"} <= facilities


@pytest.mark.parametrize("path", LAYOUTS, ids=lambda path: f"{path.parts[-3]}/{path.name}")
def test_shipped_layout_resolves(path, monkeypatch):
    monkeypatch.setenv("MODEL_ROOT", "/models")
    document = yaml.safe_load(path.read_text())
    preview = document.get("preview", {})
    plan = ResourcePlan.from_yaml(path)
    assert plan.preset in PRESETS
    nodes = int(preview.get("nodes", 1))
    gpus = get_preset(plan.preset).geometry.devices_per_node
    placement = plan.resolve(_alloc(nodes=nodes, gpus=gpus))
    assert not placement.idle, placement.warnings


# -- CLI ------------------------------------------------------------------------


def _cli(*args: str):
    from typer.testing import CliRunner

    from matsim_agents.cli import app

    return CliRunner().invoke(app, ["resources", *args])


def test_cli_plan_prints_pools_and_writes_json(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_ROOT", "/models")
    layout = ROOT / "deployments/perlmutter/resources/campaign-panel.yaml"
    result = _cli("plan", str(layout), "--nodes", "16", "--output", str(tmp_path))
    assert result.exit_code == 0, result.output
    assert "perlmutter-gpu: 16 node(s)" in result.output
    data = json.loads((tmp_path / "placement.json").read_text())
    kimi = data["pools"]["llm"]["servers"][0]
    assert kimi["name"] == "kimi-k2.5"
    assert (kimi["tensor_parallel_size"], kimi["pipeline_parallel_size"]) == (4, 4)
    assert data["pools"]["dft"]["slices"][0]["host"] == "node015"


def test_cli_plan_json_on_explicit_hosts(tmp_path):
    plan = tmp_path / "plan.yaml"
    plan.write_text("resources:\n  preset: frontier\n  pools:\n    mlip: {nodes: [b]}\n")
    result = _cli("plan", str(plan), "--hosts", "a,b", "--json")
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["pools"]["mlip"]["slices"][0]["host"] == "b"
    assert data["idle"][0]["host"] == "a"


def test_cli_validate_reports_conflicts(tmp_path):
    plan = tmp_path / "plan.yaml"
    plan.write_text(
        "resources:\n  preset: perlmutter-gpu\n  pools:\n"
        "    mlip: {devices: [0]}\n    training: {devices: [0]}\n"
    )
    result = _cli("validate", str(plan), "--nodes", "1")
    assert result.exit_code == 1
    assert "overlaps pool 'mlip'" in result.output


def test_cli_rejects_nodes_and_hosts_together(tmp_path):
    plan = tmp_path / "plan.yaml"
    plan.write_text("resources:\n  preset: frontier\n  pools:\n    mlip: {}\n")
    result = _cli("validate", str(plan), "--nodes", "1", "--hosts", "a")
    assert result.exit_code != 0
