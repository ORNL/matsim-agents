"""Unit tests for the active-learning configuration loader.

Covers:
* `ALConfig.from_yaml` end-to-end round-trip on a minimal valid config.
* Variable substitution: `${VAR}`, `${VAR:-default}`, `${VAR:?msg}`,
  the inline `vars:` block, env-overrides-vars precedence, and
  iterative resolution of nested references.
* Validators: `dft.backend` requires the matching sub-block;
  `acquisition.strategy='ensemble'` requires `hydragnn.ensemble_paths`.
* Legacy YAML compatibility (top-level `vasp:` and `md.seed_structures:`).
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest
import yaml

from matsim_agents.active_learning import ALConfig
from matsim_agents.active_learning.config import DFTConfig, QEBackendConfig
from matsim_agents.active_learning.loop import _dft_method_signature


def _write(tmp_path: Path, name: str, body: str) -> Path:
    p = tmp_path / name
    p.write_text(dedent(body))
    return p


def _minimal_yaml(
    *,
    backend: str = "vasp",
    strategy: str = "random",
    ensemble_paths: str = "",
    seeds_block: str = "      - __SEED_PATH__",
) -> str:
    """Return a syntactically valid AL config YAML body.

    Uses ``${...}`` placeholders for env-resolved paths and a literal
    ``__SEED_PATH__`` sentinel for the on-disk seed file (so callers can
    swap it in via ``str.replace()`` without colliding with f-string /
    .format() escaping of ``${...}``).
    """
    ensemble = ""
    if ensemble_paths:
        ensemble = "  ensemble_paths:\n" + ensemble_paths

    return (
        "hydragnn:\n"
        "  logdir: ${LOGDIR}\n"
        f"{ensemble}"
        "md:\n"
        "  seed_source:\n"
        "    kind: paths\n"
        "    paths:\n"
        f"{seeds_block}\n"
        "  n_steps: 50\n"
        "acquisition:\n"
        f"  strategy: {strategy}\n"
        "  n_select: 4\n"
        "dft:\n"
        f"  backend: {backend}\n"
        "  vasp:\n"
        "    vasp_bin: ${VASP_BIN}\n"
        "    vasp_wrapper: ${VASP_WRAPPER}\n"
        "    incar_template: ${INCAR}\n"
        "    potcar_dir: ${POTCAR_DIR}\n"
        "trainer:\n"
        "  enabled: false\n"
        "  train_script: ${TRAIN_SCRIPT}\n"
        "loop:\n"
        "  n_iterations: 1\n"
        "  out_dir: ${OUT_DIR}\n"
    )


# --------------------------------------------------------------------------- #
# Helpers shared by tests                                                     #
# --------------------------------------------------------------------------- #


@pytest.fixture
def required_paths(tmp_path: Path) -> dict[str, str]:
    """A set of stub files / dirs that satisfy pydantic Path validation."""
    seed = tmp_path / "seed.vasp"
    seed.write_text("dummy POSCAR\n")
    vasp_bin = tmp_path / "vasp_std"
    vasp_bin.write_text("#!/bin/bash\n")
    wrapper = tmp_path / "wrap.sh"
    wrapper.write_text("#!/bin/bash\n")
    incar = tmp_path / "INCAR.template"
    incar.write_text("ENCUT = 520\n")
    potcar_dir = tmp_path / "potcars"
    potcar_dir.mkdir()
    train = tmp_path / "train.py"
    train.write_text("# stub\n")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    return {
        "LOGDIR": str(tmp_path / "logdir"),
        "VASP_BIN": str(vasp_bin),
        "VASP_WRAPPER": str(wrapper),
        "INCAR": str(incar),
        "POTCAR_DIR": str(potcar_dir),
        "TRAIN_SCRIPT": str(train),
        "OUT_DIR": str(out_dir),
        "seed_path": str(seed),
    }


# --------------------------------------------------------------------------- #
# Round-trip                                                                  #
# --------------------------------------------------------------------------- #


def test_from_yaml_minimal_round_trip(
    tmp_path: Path,
    required_paths: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = _minimal_yaml().replace("__SEED_PATH__", required_paths["seed_path"])
    cfg_path = _write(tmp_path, "al.yaml", body)

    for k, v in required_paths.items():
        if k != "seed_path":
            monkeypatch.setenv(k, v)

    cfg = ALConfig.from_yaml(cfg_path)

    assert cfg.dft.backend == "vasp"
    assert cfg.dft.vasp is not None
    assert str(cfg.dft.vasp.vasp_bin) == required_paths["VASP_BIN"]
    assert str(cfg.mlip.hydragnn.logdir) == required_paths["LOGDIR"]
    assert cfg.acquisition.strategy == "random"
    assert cfg.md.seed_source.kind == "paths"
    assert len(cfg.md.seed_source.paths) == 1
    assert cfg.loop.n_iterations == 1


PROMOTION_EXAMPLES = [
    path
    for directory in ("active_learning", "paper_cases")
    for path in sorted(
        (Path(__file__).resolve().parents[1] / "examples" / directory).glob("*.yaml")
    )
    if yaml.safe_load(path.read_text()).get("trainer", {}).get("promote_model")
]


@pytest.mark.parametrize("example", PROMOTION_EXAMPLES, ids=lambda path: path.name)
def test_promotion_examples_require_elemental_manifest(
    example, tmp_path, required_paths, monkeypatch, elemental_manifest
):
    trainer = yaml.safe_load(example.read_text())["trainer"]
    assert trainer["validation_reference_set"] == (
        "${ELEMENTAL_REFERENCE_MANIFEST:?provide DFT-labelled elemental JSON manifest}"
    )
    assert trainer.get("validation_fraction", 0) > 0 or trainer.get("validation_set")
    trainer["train_script"] = required_paths["TRAIN_SCRIPT"]
    trainer.pop("train_launcher", None)
    body = _minimal_yaml().replace("__SEED_PATH__", required_paths["seed_path"])
    config_path = _write(
        tmp_path, "promotion.yaml", body + yaml.safe_dump({"trainer": trainer}, width=1000)
    )
    for name, value in required_paths.items():
        if name != "seed_path":
            monkeypatch.setenv(name, value)
    monkeypatch.delenv("ELEMENTAL_REFERENCE_MANIFEST", raising=False)
    with pytest.raises(ValueError, match="provide DFT-labelled elemental JSON manifest"):
        ALConfig.from_yaml(config_path)
    manifest = elemental_manifest({"Si": -1.0})
    monkeypatch.setenv("ELEMENTAL_REFERENCE_MANIFEST", str(manifest))
    cfg = ALConfig.from_yaml(config_path)
    assert cfg.trainer.validation_reference_set == manifest
    assert cfg.trainer.promote_model


# --------------------------------------------------------------------------- #
# Variable substitution                                                       #
# --------------------------------------------------------------------------- #


def test_default_syntax_used_when_env_unset(
    tmp_path: Path,
    required_paths: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`${VAR:-default}` falls back when env is unset."""
    body = _minimal_yaml().replace("__SEED_PATH__", required_paths["seed_path"])
    # Replace ${OUT_DIR} with the default-syntax form
    body = body.replace("${OUT_DIR}", "${OUT_DIR:-" + required_paths["OUT_DIR"] + "}")
    cfg_path = _write(tmp_path, "al.yaml", body)

    monkeypatch.delenv("OUT_DIR", raising=False)
    for k, v in required_paths.items():
        if k not in ("seed_path", "OUT_DIR"):
            monkeypatch.setenv(k, v)

    cfg = ALConfig.from_yaml(cfg_path)
    assert str(cfg.loop.out_dir) == required_paths["OUT_DIR"]


def test_required_syntax_raises_when_unset(
    tmp_path: Path,
    required_paths: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`${VAR}` (no modifier) must raise if unset."""
    body = _minimal_yaml().replace("__SEED_PATH__", required_paths["seed_path"])
    cfg_path = _write(tmp_path, "al.yaml", body)

    monkeypatch.delenv("OUT_DIR", raising=False)
    for k, v in required_paths.items():
        if k not in ("seed_path", "OUT_DIR"):
            monkeypatch.setenv(k, v)

    with pytest.raises(ValueError, match=r"undefined variable \$\{OUT_DIR\}"):
        ALConfig.from_yaml(cfg_path)


def test_error_syntax_raises_with_message(
    tmp_path: Path,
    required_paths: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`${VAR:?msg}` raises with the user-supplied message."""
    body = _minimal_yaml().replace("__SEED_PATH__", required_paths["seed_path"])
    body = body.replace("${OUT_DIR}", "${OUT_DIR:?please set OUT_DIR}")
    cfg_path = _write(tmp_path, "al.yaml", body)

    monkeypatch.delenv("OUT_DIR", raising=False)
    for k, v in required_paths.items():
        if k not in ("seed_path", "OUT_DIR"):
            monkeypatch.setenv(k, v)

    with pytest.raises(ValueError, match=r"please set OUT_DIR"):
        ALConfig.from_yaml(cfg_path)


def test_inline_vars_block_and_iterative_resolution(
    tmp_path: Path,
    required_paths: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`vars:` block is consumed before validation and resolves nested refs."""
    # Force the vars block to define VASP_BIN as ${PROJ_ROOT}/vasp_std so we
    # confirm iterative substitution works.
    proj_root = tmp_path
    body = _minimal_yaml().replace("__SEED_PATH__", required_paths["seed_path"])
    body = (
        f"vars:\n"
        f"  PROJ_ROOT: {proj_root}\n"
        f"  VASP_BIN: ${{PROJ_ROOT}}/vasp_std\n"
        f"  VASP_WRAPPER: ${{PROJ_ROOT}}/wrap.sh\n"
        f"  INCAR: ${{PROJ_ROOT}}/INCAR.template\n"
        f"  POTCAR_DIR: ${{PROJ_ROOT}}/potcars\n"
        f"  TRAIN_SCRIPT: ${{PROJ_ROOT}}/train.py\n"
        f"  OUT_DIR: ${{PROJ_ROOT}}/out\n"
        f"  LOGDIR: ${{PROJ_ROOT}}/logdir\n"
    ) + body
    cfg_path = _write(tmp_path, "al.yaml", body)

    # Make sure no env var leaks in.
    for k in ("PROJ_ROOT", "VASP_BIN", "OUT_DIR", "LOGDIR"):
        monkeypatch.delenv(k, raising=False)

    cfg = ALConfig.from_yaml(cfg_path)
    assert str(cfg.dft.vasp.vasp_bin) == str(proj_root / "vasp_std")
    assert str(cfg.loop.out_dir) == str(proj_root / "out")
    # `vars` itself must NOT appear in the parsed model.
    assert "vars" not in cfg.model_dump()


def test_env_overrides_vars_block(
    tmp_path: Path,
    required_paths: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """os.environ takes precedence over the inline `vars:` defaults."""
    body = _minimal_yaml().replace("__SEED_PATH__", required_paths["seed_path"])
    body = ("vars:\n  OUT_DIR: /this/should/be/overridden\n") + body
    cfg_path = _write(tmp_path, "al.yaml", body)

    for k, v in required_paths.items():
        if k != "seed_path":
            monkeypatch.setenv(k, v)

    cfg = ALConfig.from_yaml(cfg_path)
    assert str(cfg.loop.out_dir) == required_paths["OUT_DIR"]


# --------------------------------------------------------------------------- #
# Validators                                                                  #
# --------------------------------------------------------------------------- #


def test_qe_backend_requires_qe_block(
    tmp_path: Path,
    required_paths: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Selecting `dft.backend: qe` without a `dft.qe:` block must fail."""
    body = _minimal_yaml(backend="qe").replace("__SEED_PATH__", required_paths["seed_path"])
    cfg_path = _write(tmp_path, "al.yaml", body)

    for k, v in required_paths.items():
        if k != "seed_path":
            monkeypatch.setenv(k, v)

    with pytest.raises(ValueError, match=r"dft\.backend='qe' requires a dft\.qe block"):
        ALConfig.from_yaml(cfg_path)


def test_ensemble_strategy_requires_ensemble_paths(
    tmp_path: Path,
    required_paths: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`acquisition.strategy='ensemble'` without ensemble_paths must fail."""
    body = _minimal_yaml(strategy="ensemble").replace("__SEED_PATH__", required_paths["seed_path"])
    cfg_path = _write(tmp_path, "al.yaml", body)

    for k, v in required_paths.items():
        if k != "seed_path":
            monkeypatch.setenv(k, v)

    with pytest.raises(ValueError, match=r"ensemble"):
        ALConfig.from_yaml(cfg_path)


def test_dft_method_signature_excludes_operational_settings(
    tmp_path: Path,
    required_paths: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = _minimal_yaml().replace("__SEED_PATH__", required_paths["seed_path"])
    cfg_path = _write(tmp_path, "al.yaml", body)
    for key, value in required_paths.items():
        if key != "seed_path":
            monkeypatch.setenv(key, value)
    cfg = ALConfig.from_yaml(cfg_path)
    potcar_dir = Path(required_paths["POTCAR_DIR"])
    for element in ("Nb", "O"):
        element_dir = potcar_dir / element
        element_dir.mkdir()
        (element_dir / "POTCAR").write_text(element)
    operational_change = cfg.model_copy(deep=True)
    assert operational_change.dft.vasp is not None
    operational_change.dft.vasp.vasp_wrapper = tmp_path / "relocated-wrapper.sh"
    operational_change.dft.vasp.timeout_sec = 99
    operational_change.dft.vasp.ranks_per_node = 2
    operational_change.dft.max_concurrent_jobs = 3

    elements = {"Nb", "O"}
    assert _dft_method_signature(operational_change, elements) == _dft_method_signature(
        cfg, elements
    )

    scientific_change = cfg.model_copy(deep=True)
    assert scientific_change.dft.vasp is not None
    scientific_change.dft.vasp.extra_incar["ENCUT"] = "700"

    assert _dft_method_signature(scientific_change, elements) != _dft_method_signature(
        cfg, elements
    )


def test_dft_method_signature_hashes_only_selected_potcars(
    tmp_path: Path,
    required_paths: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = _minimal_yaml().replace("__SEED_PATH__", required_paths["seed_path"])
    cfg_path = _write(tmp_path, "al.yaml", body)
    for key, value in required_paths.items():
        if key != "seed_path":
            monkeypatch.setenv(key, value)
    cfg = ALConfig.from_yaml(cfg_path)
    potcar_dir = Path(required_paths["POTCAR_DIR"])
    for element in ("Nb", "O", "Ta"):
        element_dir = potcar_dir / element
        element_dir.mkdir()
        (element_dir / "POTCAR").write_text(element)

    original = _dft_method_signature(cfg, {"Nb", "O"})
    (potcar_dir / "Ta" / "POTCAR").write_text("changed unrelated potential")
    assert _dft_method_signature(cfg, {"Nb", "O"}) == original

    (potcar_dir / "Nb" / "POTCAR").write_text("changed selected potential")
    assert _dft_method_signature(cfg, {"Nb", "O"}) != original


def test_dft_method_signature_includes_backend_executable_identity(
    tmp_path: Path,
    required_paths: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = _minimal_yaml().replace("__SEED_PATH__", required_paths["seed_path"])
    cfg_path = _write(tmp_path, "al.yaml", body)
    for key, value in required_paths.items():
        if key != "seed_path":
            monkeypatch.setenv(key, value)
    cfg = ALConfig.from_yaml(cfg_path)
    potcar_dir = Path(required_paths["POTCAR_DIR"])
    for element in ("Nb", "O"):
        element_dir = potcar_dir / element
        element_dir.mkdir()
        (element_dir / "POTCAR").write_text(element)

    vasp_signature = _dft_method_signature(cfg, {"Nb", "O"})
    Path(required_paths["VASP_BIN"]).write_text("#!/bin/bash\n# different build\n")
    assert _dft_method_signature(cfg, {"Nb", "O"}) != vasp_signature

    qe_executable = tmp_path / "pw.x"
    qe_executable.write_text("QE build one")
    pseudo_dir = tmp_path / "qe-pseudo"
    pseudo_dir.mkdir()
    for element in ("Nb", "O"):
        (pseudo_dir / f"{element}.upf").write_text(element)
    qe = QEBackendConfig(
        pw_bin=qe_executable,
        pw_wrapper=Path(required_paths["VASP_WRAPPER"]),
        pseudo_dir=pseudo_dir,
        pseudopotentials={"Nb": "Nb.upf", "O": "O.upf"},
    )
    qe_cfg = cfg.model_copy(update={"dft": DFTConfig(backend="qe", qe=qe)})

    qe_signature = _dft_method_signature(qe_cfg, {"Nb", "O"})
    qe_executable.write_text("QE build two")
    assert _dft_method_signature(qe_cfg, {"Nb", "O"}) != qe_signature


# --------------------------------------------------------------------------- #
# Backward-compat shims                                                       #
# --------------------------------------------------------------------------- #


def test_legacy_top_level_vasp_block_promoted(
    tmp_path: Path,
    required_paths: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Older YAMLs with top-level `vasp:` (no `dft:`) must keep working."""
    body = dedent(
        """\
        hydragnn:
          logdir: {LOGDIR}
        md:
          seed_source:
            kind: paths
            paths:
              - {seed_path}
        acquisition:
          strategy: random
          n_select: 4
        vasp:
          vasp_bin: {VASP_BIN}
          vasp_wrapper: {VASP_WRAPPER}
          incar_template: {INCAR}
          potcar_dir: {POTCAR_DIR}
        trainer:
          enabled: false
          train_script: {TRAIN_SCRIPT}
        loop:
          n_iterations: 1
          out_dir: {OUT_DIR}
        """
    ).format(**required_paths)
    cfg_path = _write(tmp_path, "al.yaml", body)

    cfg = ALConfig.from_yaml(cfg_path)
    assert cfg.dft.backend == "vasp"
    assert cfg.dft.vasp is not None
    assert str(cfg.dft.vasp.vasp_bin) == required_paths["VASP_BIN"]


def test_legacy_seed_structures_promoted(
    tmp_path: Path,
    required_paths: dict[str, str],
) -> None:
    """`md.seed_structures: [...]` must be promoted to seed_source.kind=paths."""
    body = dedent(
        """\
        hydragnn:
          logdir: {LOGDIR}
        md:
          seed_structures:
            - {seed_path}
        acquisition:
          strategy: random
          n_select: 4
        dft:
          backend: vasp
          vasp:
            vasp_bin: {VASP_BIN}
            vasp_wrapper: {VASP_WRAPPER}
            incar_template: {INCAR}
            potcar_dir: {POTCAR_DIR}
        trainer:
          enabled: false
          train_script: {TRAIN_SCRIPT}
        loop:
          n_iterations: 1
          out_dir: {OUT_DIR}
        """
    ).format(**required_paths)
    cfg_path = _write(tmp_path, "al.yaml", body)

    cfg = ALConfig.from_yaml(cfg_path)
    assert cfg.md.seed_source.kind == "paths"
    assert len(cfg.md.seed_source.paths) == 1
    assert str(cfg.md.seed_source.paths[0]) == required_paths["seed_path"]
