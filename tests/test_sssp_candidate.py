"""SSSP candidate protocol: manifest generator and isolated qualification config."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest
import yaml

from matsim_agents.workflows import full_supercell_campaign as fsc

ROOT = Path(__file__).parents[1]
CANDIDATE = ROOT / "deployments/frontier/campaigns/qe-1024-sssp-candidate/campaign.yaml"
PRODUCTION = ROOT / "deployments/frontier/campaigns/qe-1024/campaign.yaml"
MANIFEST = ROOT / "deployments/common/pseudopotentials/sssp-1.3.0-pbe-precision.yaml"
ELEMENTS = {"W", "Ta", "Cr", "V", "Hf", "Mo", "Re", "Os", "H", "He"}


def _load_script():
    path = ROOT / "scripts/utils/build_pseudo_manifest.py"
    spec = importlib.util.spec_from_file_location("build_pseudo_manifest", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def manifest():
    return yaml.safe_load(MANIFEST.read_text())


@pytest.fixture
def candidate(monkeypatch, manifest):
    monkeypatch.setenv("MATSIM_SSSP_PSEUDO_DIR", os.environ.get("MATSIM_SSSP_PSEUDO_DIR", "/sssp"))
    return fsc.FullSupercellCampaign.from_yaml(CANDIDATE)


def test_manifest_covers_campaign_elements_with_verified_md5(manifest):
    pp = manifest["pseudopotentials"]
    assert set(pp) == ELEMENTS
    for el, e in pp.items():
        assert e["md5"] == e["md5_official"] and e["upf_element"] == el
        assert not e["upf_has_so"] and e["upf_relativistic"] == "scalar"
    assert manifest["pseudo_types"] == ["NC", "US"]


def test_candidate_is_isolated_from_production(candidate, manifest, monkeypatch):
    monkeypatch.setenv("MATSIM_PSEUDO_DIR", "/pslibrary")
    production = fsc.FullSupercellCampaign.from_yaml(PRODUCTION)
    assert candidate.campaign_id != production.campaign_id
    assert candidate.parent_supercell == production.parent_supercell
    assert candidate.parent_supercell.expected_parent_atoms == 1024
    assert "pslibrary" not in candidate.qe.pseudo_dir
    assert set(candidate.qe.pseudopotentials) == ELEMENTS
    assert not set(candidate.qe.pseudopotentials.values()) & set(
        production.qe.pseudopotentials.values()
    )
    files = {el: e["filename"] for el, e in manifest["pseudopotentials"].items()}
    assert dict(candidate.qe.pseudopotentials) == files
    minimum = manifest["campaign_minimum_cutoffs_ry"]
    assert candidate.qe.ecutwfc_ry >= minimum["ecutwfc"]
    assert candidate.qe.ecutrho_ry >= minimum["ecutrho"]


def test_installed_files_match_manifest(manifest):
    library = Path(os.path.expandvars(manifest["library_dir"]))
    if not library.is_dir():
        pytest.skip("SSSP library not installed on this host")
    for e in manifest["pseudopotentials"].values():
        assert hashlib.sha256((library / e["filename"]).read_bytes()).hexdigest() == e["sha256"]


def _fake_library(tmp_path, upf_element="W"):
    upf = (
        f'<UPF version="2.0.1">\n<PP_HEADER element="{upf_element}" pseudo_type="NC" '
        'relativistic="scalar" is_paw="false" core_correction="false" functional="PBE" '
        'z_valence="14.0"/>\n</UPF>\n'
    ).encode()
    (tmp_path / "w.upf").write_bytes(upf)
    meta = {
        "W": {
            "filename": "w.upf",
            "md5": hashlib.md5(upf).hexdigest(),  # noqa: S324 - test fixture
            "pseudopotential": "SG15",
            "cutoff_wfc": 35.0,
            "cutoff_rho": 140.0,
        }
    }
    (tmp_path / "sssp.json").write_text(json.dumps(meta))
    return [
        str(tmp_path),
        "--sssp-json",
        str(tmp_path / "sssp.json"),
        "--library-id",
        "fake",
        "--source-url",
        "https://example.invalid",
        "--element",
        "W",
        "--out",
        str(tmp_path / "m.yaml"),
    ]


def test_manifest_script_writes_verified_entry(tmp_path):
    assert _load_script().main(_fake_library(tmp_path)) == 0
    m = yaml.safe_load((tmp_path / "m.yaml").read_text())
    assert m["pseudopotentials"]["W"]["upf_pseudo_type"] == "NC"
    assert m["campaign_minimum_cutoffs_ry"]["ecutwfc"] == 35.0


@pytest.mark.parametrize("problem", ["md5", "element", "missing"])
def test_manifest_script_rejects_bad_library(tmp_path, problem):
    argv = _fake_library(tmp_path, upf_element="Ta" if problem == "element" else "W")
    if problem == "md5":
        (tmp_path / "w.upf").write_bytes((tmp_path / "w.upf").read_bytes() + b"\n")
    if problem == "missing":
        os.remove(tmp_path / "w.upf")
    assert _load_script().main(argv) == 1
    assert not (tmp_path / "m.yaml").exists()
