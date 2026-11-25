"""Channels derived from the composition instead of a hand-written list."""

from __future__ import annotations

import pytest
import yaml

from matsim_agents.backends.external.spectra_pka.config import ConfigError, load_config_file


def _write(tmp_path, campaign: dict, data_root: bool = True):
    tool = {"enabled": True, "workdir": "out"}
    if data_root:
        tool["nuclear_data_root"] = "pka"
    spectrum = tmp_path / "flux.dat"
    spectrum.write_text("x\n")
    base = {
        "scenario_id": "derived",
        "parent_alloy_id": "alloy",
        "nuclear_data_identity": "TENDL2023n",
        "flux": {"file": str(spectrum), "identity": "s", "normalization": "n/cm2/s"},
        "derive_channels": True,
    }
    path = tmp_path / "c.yaml"
    path.write_text(
        yaml.safe_dump({"external_tools": {"spectra_pka": tool}, "campaign": {**base, **campaign}})
    )
    return path


def test_natural_abundance_channels_match_the_composition(tmp_path):
    cfg = load_config_file(_write(tmp_path, {"composition": {"W": 0.9, "Ta": 0.1}}))
    campaign = cfg.campaign
    assert campaign is not None and not campaign.derive_channels
    files = {(c.parent_ele, c.parent_num): c.pka_file.name for c in campaign.channels}
    assert files[("W", 184)] == "W184s.asc"
    assert files[("Ta", 181)] == "Ta181s.asc"
    assert set(files) == {("W", m) for m in (180, 182, 183, 184, 186)} | {("Ta", 181)}
    assert all(c.pka_file.parent == (tmp_path / "pka").resolve() for c in campaign.channels)
    per_element = {}
    for c in campaign.channels:
        per_element[c.parent_ele] = per_element.get(c.parent_ele, 0.0) + c.atom_fraction
    assert per_element["W"] == pytest.approx(0.9)
    assert per_element["Ta"] == pytest.approx(0.1)


def test_isotope_override_replaces_natural_abundance(tmp_path):
    cfg = load_config_file(
        _write(
            tmp_path,
            {"composition": {"W": 1.0}, "isotope_overrides": {"W": {"184": 0.6, "186": 0.4}}},
        )
    )
    assert {(c.parent_ele, c.parent_num) for c in cfg.campaign.channels} == {
        ("W", 184),
        ("W", 186),
    }


def test_derived_channels_need_a_data_root(tmp_path):
    path = _write(tmp_path, {"composition": {"W": 1.0}}, data_root=False)
    with pytest.raises(ConfigError, match="nuclear_data_root"):
        load_config_file(path)


def test_explicit_channels_and_derivation_are_exclusive(tmp_path):
    channel = {"pka_file": "W184s.asc", "parent_ele": "W", "parent_num": 184, "atom_fraction": 1.0}
    path = _write(tmp_path, {"composition": {"W": 1.0}, "channels": [channel]})
    with pytest.raises(ConfigError, match="exclusive"):
        load_config_file(path)


def test_channels_are_required_without_derivation(tmp_path):
    path = _write(tmp_path, {"composition": {"W": 1.0}, "derive_channels": False})
    with pytest.raises(ConfigError, match="derive_channels"):
        load_config_file(path)
