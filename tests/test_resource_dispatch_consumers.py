"""DFT, training, and relaxation consumers of the resource placement (phase 2)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from ase.build import bulk
from pydantic import ValidationError

from matsim_agents.active_learning.config import ALConfig, TrainerConfig
from matsim_agents.active_learning.dft_protocol import dft_execution_signature
from matsim_agents.backends.dft.base import DFTJobSpec
from matsim_agents.backends.dft.config import DFTConfig, QEBackendConfig, VASPConfig
from matsim_agents.execution.placement import ProcessLaunch
from matsim_agents.workflows.relaxation import DFTBackendConfig, ScientificRelaxationConfig

LAUNCH = ProcessLaunch(
    hosts=("nid000001", "nid000002"),
    devices=(0, 1, 2, 3),
    ranks_per_node=4,
    threads_per_rank=16,
    env={"CUDA_VISIBLE_DEVICES": "0,1,2,3"},
)


def _qe(tmp_path: Path, **extra) -> dict:
    wrapper = tmp_path / "wrapper.sh"
    wrapper.write_text("#!/bin/bash\n")
    return {"pw_bin": "/opt/pw.x", "pw_wrapper": str(wrapper), "pseudo_dir": "/opt/pseudo", **extra}


@pytest.mark.parametrize(
    ("block", "key", "target"),
    [
        ("qe", "nodes_per_job", "resources.pools.dft.nodes_per_job"),
        ("qe", "ranks_per_node", "ranks_per_device"),
        ("qe", "threads_per_rank", "resources.pools.dft.threads_per_rank"),
        ("vasp", "ranks_per_node", "ranks_per_device"),
    ],
)
def test_backend_blocks_reject_moved_placement_fields(tmp_path, block, key, target):
    model = QEBackendConfig if block == "qe" else VASPConfig
    data = (
        _qe(tmp_path)
        if block == "qe"
        else {
            "vasp_bin": "/opt/vasp_std",
            "vasp_wrapper": "/opt/wrap.sh",
            "incar_template": "/opt/INCAR",
            "potcar_dir": "/opt/potcar",
        }
    )
    with pytest.raises(ValidationError, match=f"dft.{block}.{key} -> .*{target}"):
        model.model_validate({**data, key: 4})


def test_dft_block_rejects_concurrency_cap(tmp_path):
    with pytest.raises(ValidationError, match="dft.max_concurrent_jobs -> resources.pools.dft"):
        DFTConfig.model_validate({"backend": "qe", "qe": _qe(tmp_path), "max_concurrent_jobs": 2})


def test_trainer_rejects_moved_placement_fields():
    with pytest.raises(ValidationError, match="trainer.nodes_for_train -> resources"):
        TrainerConfig.model_validate({"nodes_for_train": 1})


def _al(tmp_path: Path, *, resources=None, trainer=None) -> dict:
    data = {
        "mlip": {"backend": "uma", "uma": {"model_name": "uma-s-1p1"}},
        "md": {"seed_source": {"kind": "paths", "paths": ["/tmp/seed.vasp"]}},
        "acquisition": {"strategy": "random", "n_select": 1},
        "dft": {"backend": "qe", "qe": _qe(tmp_path)},
        "trainer": trainer or {"enabled": False},
        "loop": {"out_dir": str(tmp_path / "out")},
    }
    if resources is not None:
        data["resources"] = resources
    return data


def test_al_config_requires_resources(tmp_path):
    with pytest.raises(ValidationError, match="resources"):
        ALConfig.model_validate(_al(tmp_path))


def test_al_config_requires_dft_pool(tmp_path):
    resources = {"preset": "perlmutter-gpu", "pools": {"mlip": {}}}
    with pytest.raises(ValidationError, match="resources.pools.dft is required"):
        ALConfig.model_validate(_al(tmp_path, resources=resources))


def test_enabled_trainer_requires_training_pool(tmp_path):
    script = tmp_path / "finetune_uma.py"
    script.touch()
    resources = {"preset": "perlmutter-gpu", "pools": {"dft": {"nodes": "all"}}}
    trainer = {"enabled": True, "train_script": str(script)}
    with pytest.raises(ValidationError, match="requires resources.pools.training"):
        ALConfig.model_validate(_al(tmp_path, resources=resources, trainer=trainer))


def test_qe_backend_passes_launch_to_wrapper(tmp_path, monkeypatch):
    from matsim_agents.backends.dft import qe as qe_module

    observed = {}

    def run(argv, **kwargs):
        observed["argv"] = argv
        observed["env"] = kwargs["env"]
        Path(kwargs["cwd"], "pw.out").write_text("")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(qe_module.subprocess, "run", run)
    backend = qe_module.QEBackend(QEBackendConfig.model_validate(_qe(tmp_path, kpts=(1, 1, 1))))
    monkeypatch.setattr(backend, "_settings_for", lambda atoms: object())
    monkeypatch.setattr(qe_module, "write_pw_input", lambda *args, **kwargs: None)
    spec = DFTJobSpec("si", bulk("Si"), str(tmp_path / "job"), launch=LAUNCH)
    backend.run_one(spec)
    assert observed["argv"][-3:] == ["2", "4", "16"]
    assert observed["env"]["MATSIM_ASSIGNED_NODES"] == "nid000001,nid000002"
    assert observed["env"]["MATSIM_DEVICES_PER_NODE"] == "4"
    assert observed["env"]["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"


def test_execution_signature_tracks_layout_not_hosts(tmp_path):
    cfg = DFTConfig.model_validate({"backend": "qe", "qe": _qe(tmp_path)})
    other_hosts = LAUNCH.model_copy(update={"hosts": ("nid000009", "nid000010")})
    fewer_ranks = LAUNCH.model_copy(update={"ranks_per_node": 2})
    assert dft_execution_signature(cfg, LAUNCH) == dft_execution_signature(cfg, other_hosts)
    assert dft_execution_signature(cfg, LAUNCH) != dft_execution_signature(cfg, fewer_ranks)


def test_relaxation_launcher_fills_placement_placeholders():
    dft = DFTBackendConfig(
        backend="qe",
        pseudo_dir="/opt/pseudo",
        launcher=[
            "bash",
            "wrap.sh",
            "{input}",
            "{nodes}",
            "{ranks_per_node}",
            "{threads_per_rank}",
        ],
        launch=LAUNCH,
    )
    assert dft.launcher_argv("pw.x") == ["bash", "wrap.sh", "{input}", "2", "4", "16"]
    assert dft.launch_environment("pw.x")["MATSIM_ASSIGNED_NODES"] == "nid000001,nid000002"


def test_relaxation_launcher_without_placeholders_needs_no_placement():
    dft = DFTBackendConfig(backend="qe", pseudo_dir="/opt/pseudo")
    assert dft.launcher_argv("pw.x") == ["pw.x"]
    assert dft.launch_environment("pw.x") is None


def test_relaxation_placeholders_require_placement():
    dft = DFTBackendConfig(backend="qe", pseudo_dir="/opt/pseudo", launcher=["wrap.sh", "{nodes}"])
    with pytest.raises(ValueError, match="add a resources: plan with a dft pool"):
        dft.launcher_argv("pw.x")


def test_relaxation_resources_need_a_dft_pool():
    with pytest.raises(ValidationError, match="requires a dft pool"):
        ScientificRelaxationConfig.model_validate(
            {
                "mode": "dft",
                "structure_path": "Si.vasp",
                "dft": {"backend": "qe", "pseudo_dir": "/opt/pseudo"},
                "resources": {"preset": "perlmutter-gpu", "pools": {"mlip": {}}},
            }
        )


def test_shipped_sequential_layouts_share_one_training_device():
    root = Path(__file__).resolve().parents[1]
    for facility in ("perlmutter", "frontier", "aurora"):
        path = root / "deployments" / facility / "resources" / "active-learning-sequential.yaml"
        pools = yaml.safe_load(path.read_text())["resources"]["pools"]
        assert pools["dft"]["share_with"] == ["training", "mlip"]
        assert pools["training"] == {"nodes": 1, "devices": 1, "share_with": ["dft", "mlip"]}
        assert pools["mlip"] == {"nodes": 1, "devices": "all", "share_with": ["dft", "training"]}


def test_direct_dft_launcher_runs_on_this_node_with_translated_mask(monkeypatch):
    import socket

    here = LAUNCH.model_copy(
        update={
            "hosts": (socket.gethostname(),),
            "devices": (1,),
            "ranks_per_node": 1,
            "env": {"CUDA_VISIBLE_DEVICES": "1"},
        }
    )
    dft = DFTBackendConfig(backend="qe", pseudo_dir="/opt/pseudo", launcher="pw.x", launch=here)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "6,7")
    assert dft.launch_environment("pw.x")["CUDA_VISIBLE_DEVICES"] == "7"

    remote = LAUNCH.model_copy(update={"hosts": ("nid000009",), "ranks_per_node": 1})
    elsewhere = dft.model_copy(update={"launch": remote})
    with pytest.raises(ValueError, match="runs directly on this node"):
        elsewhere.launch_environment("pw.x")


def test_saved_workflow_configs_with_null_resources_rerun(tmp_path):
    from matsim_agents.cli.workflow_commands import _load_workflow_config
    from matsim_agents.workflows.llm_check import LLMCheckConfig

    saved = tmp_path / "resolved_config.json"
    saved.write_text(
        '{"provider": "vllm", "model": "m", "base_url": "http://h:8000/v1", '
        '"server": null, "resources": null}'
    )
    cfg = LLMCheckConfig.model_validate(_load_workflow_config(saved))
    assert cfg.model == "m" and cfg.resources is None


def test_direct_dft_launcher_rejects_multi_rank_placements():
    import socket

    four = LAUNCH.model_copy(update={"hosts": (socket.gethostname(),)})
    dft = DFTBackendConfig(backend="qe", pseudo_dir="/opt/pseudo", launcher="pw.x", launch=four)
    with pytest.raises(ValueError, match=r"starts one process.*1 node\(s\) x 4 rank"):
        dft.launch_environment("pw.x")


@pytest.mark.parametrize("variable", ["CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"])
def test_empty_device_mask_hides_every_device(monkeypatch, variable):
    from matsim_agents.execution.devices import child_mask_environment, visible_ordinals

    monkeypatch.setenv(variable, "")
    with pytest.raises(ValueError, match=r"sees only .*\(none\)"):
        child_mask_environment({variable: "0"}, owner="pool 'mlip'")
    vendor = "cuda" if variable.startswith("CUDA") else "rocm"
    with pytest.raises(ValueError, match=r"\(none\)"):
        visible_ordinals(vendor, (0,), owner="x")


def test_empty_level_zero_mask_applies_no_mask(monkeypatch):
    from matsim_agents.execution.devices import child_mask_environment

    monkeypatch.setenv("ZE_AFFINITY_MASK", "")
    assert child_mask_environment({"ZE_AFFINITY_MASK": "3"}, owner="x") == {"ZE_AFFINITY_MASK": "3"}


def test_direct_launches_take_threads_from_the_placement(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    assert LAUNCH.environment({}, local=True)["OMP_NUM_THREADS"] == "16"
    # Step wrappers set OMP_NUM_THREADS themselves from <threads_per_rank>.
    assert "OMP_NUM_THREADS" not in LAUNCH.environment({})


def test_vasp_launcher_work_dir_placeholder_is_resolved(tmp_path):
    dft = DFTBackendConfig(
        backend="vasp",
        potcar_dir="/opt/potcar",
        launcher=["bash", "wrap.sh", "{work_dir}", "vasp_std", "{nodes}"],
        launch=LAUNCH,
    )
    argv = dft.launcher_argv("vasp_std", work_dir=tmp_path / "calc")
    assert argv == ["bash", "wrap.sh", str((tmp_path / "calc").resolve()), "vasp_std", "2"]


def test_rocm_worker_masks_follow_hip_only_driver_masks(monkeypatch):
    from matsim_agents.execution.devices import (
        apply_mask_environment,
        child_mask_environment,
        visible_ordinals,
    )

    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "4,5")
    mask = child_mask_environment({"ROCR_VISIBLE_DEVICES": "0"}, owner="pool 'mlip'")
    assert mask == {"ROCR_VISIBLE_DEVICES": "4"}
    env = {"HIP_VISIBLE_DEVICES": "4,5"}
    apply_mask_environment(env, mask)
    assert env == {"ROCR_VISIBLE_DEVICES": "4"}
    with pytest.raises(ValueError, match="sees only ROCR_VISIBLE_DEVICES=4,5"):
        visible_ordinals("rocm", (2,), owner="x")


def test_rocm_worker_masks_compose_hip_with_rocr(monkeypatch):
    from matsim_agents.execution.devices import child_mask_environment

    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "2,3,6,7")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "1,3")
    assert child_mask_environment({"ROCR_VISIBLE_DEVICES": "1"}, owner="x") == {
        "ROCR_VISIBLE_DEVICES": "7"
    }
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "5")
    with pytest.raises(ValueError, match="selects devices outside"):
        child_mask_environment({"ROCR_VISIBLE_DEVICES": "0"}, owner="x")


@pytest.mark.parametrize("module", ["qe_relax", "vasp_relax"])
def test_direct_rocm_dft_child_keeps_the_placed_device(monkeypatch, module):
    import importlib

    child_environment = importlib.import_module(
        f"matsim_agents.backends.dft.{module}"
    )._child_environment
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "2,3,6,7")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "1,3")
    env = child_environment({"ROCR_VISIBLE_DEVICES": "7", "OMP_NUM_THREADS": "7"})
    assert env["ROCR_VISIBLE_DEVICES"] == "7"
    assert "HIP_VISIBLE_DEVICES" not in env
    assert child_environment(None) is None
