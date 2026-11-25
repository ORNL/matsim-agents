from __future__ import annotations

import pytest
from pydantic import ValidationError

from matsim_agents.backends.external.spectra_pka.config import FluxSpec
from matsim_agents.backends.external.spectra_pka.input_writer import (
    FLUX_LINK,
    InputError,
    native_flux_text,
    stage_inputs,
)
from matsim_agents.backends.external.spectra_pka.spectrum import from_groups, monoenergetic
from tests.spectra_pka_fixtures import make_campaign


def test_flux_spec_needs_exactly_one_source(tmp_path):
    common = {"identity": "s", "normalization": "n"}
    with pytest.raises(ValidationError):
        FluxSpec(**common)
    with pytest.raises(ValidationError):
        FluxSpec(file=tmp_path / "a", spectrum_file=tmp_path / "b", **common)


def test_native_flux_text_layout_in_mev():
    spec = from_groups([1.0e6, 2.0e6, 4.0e6], [3.0, 1.0], identity="demo")
    lines = native_flux_text(spec, 2).splitlines()
    assert lines[1] == "2 0 2 0  1.0 1.0" and lines[2] == "2 -2"
    assert [float(v) for v in lines[3:6]] == [1.0, 2.0, 4.0]
    assert [float(v) for v in lines[6:]] == [3.0, 1.0]


def test_native_flux_rejects_line_and_shape_only_absolute_mode():
    with pytest.raises(InputError):
        native_flux_text(monoenergetic(14.1, total_flux=1e14), 2)
    shape = from_groups([1.0, 2.0, 4.0], [3.0, 1.0], identity="d", absolute=False)
    with pytest.raises(InputError):
        native_flux_text(shape, 3)


def test_stage_inputs_writes_native_flux_from_csv(tmp_path):
    campaign = make_campaign(tmp_path)
    csv = tmp_path / "data" / "spec.csv"
    csv.write_text("e_low,e_high,flux\n1e5,1e6,2e12\n1e6,2e7,1e12\n")
    flux = campaign.flux.model_copy(update={"file": None, "spectrum_file": csv})
    staged = stage_inputs(campaign.model_copy(update={"flux": flux}), tmp_path / "run")
    text = (tmp_path / "run" / FLUX_LINK).read_text().splitlines()
    assert text[2] == "2 -2" and float(text[3]) == pytest.approx(0.1)
    assert {"spectrum_source", "spectrum_content", FLUX_LINK} <= set(staged.data_sha256)
    assert not (tmp_path / "run" / FLUX_LINK).is_symlink()


def test_mode_three_is_rejected_for_a_generated_spectrum():
    spec = from_groups([1.0, 2.0, 4.0], [3.0, 1.0], identity="d")
    with pytest.raises(InputError, match="norm_type 3"):
        native_flux_text(spec, 3)


def test_shape_only_spectrum_needs_an_explicit_rescale(tmp_path):
    campaign = make_campaign(tmp_path)
    csv = tmp_path / "data" / "shape.csv"
    csv.write_text("e_low,e_high,flux\n1e5,1e6,0.5\n1e6,2e7,0.5\n")
    flux = campaign.flux.model_copy(update={"file": None, "spectrum_file": csv, "absolute": False})
    with pytest.raises(InputError, match="rescale_value"):
        stage_inputs(campaign.model_copy(update={"flux": flux}), tmp_path / "run")
    flux = flux.model_copy(update={"rescale_value": 1e14})
    stage_inputs(campaign.model_copy(update={"flux": flux}), tmp_path / "run2")
