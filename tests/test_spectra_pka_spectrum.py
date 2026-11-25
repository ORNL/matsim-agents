from __future__ import annotations

import json
import math

import pytest

from matsim_agents.backends.external.spectra_pka.spectrum import (
    NeutronSpectrum,
    SpectrumError,
    from_groups,
    load_spectrum_file,
    load_transport_tally,
    monoenergetic,
    rebin,
    validate_spectrum,
)


def test_monoenergetic_units_and_absolute():
    spec = monoenergetic(14.1, total_flux=1e14)
    assert spec.line_energy_eV == pytest.approx(14.1e6)
    assert spec.absolute and spec.total_flux == pytest.approx(1e14)
    assert monoenergetic(14.1).absolute is False
    assert monoenergetic(1.0, total_flux=1e10, flux_units="n/m2/s").total_flux == pytest.approx(1e6)


def test_unknown_units_rejected():
    with pytest.raises(SpectrumError):
        monoenergetic(1.0, energy_units="furlong")


def test_from_groups_unit_conversion_and_value_kinds():
    spec = from_groups([1.0, 2.0, 4.0], [1.0, 2.0], energy_units="MeV", identity="s")
    assert spec.boundaries_eV == (1e6, 2e6, 4e6)
    per_e = from_groups(
        [1.0, 3.0], [5.0], energy_units="MeV", value_kind="per_energy", identity="s"
    )
    assert per_e.group_flux[0] == pytest.approx(10.0)  # 5 per MeV x 2 MeV
    leth = from_groups([1.0, math.e], [3.0], value_kind="per_lethargy", identity="s")
    assert leth.group_flux[0] == pytest.approx(3.0)


@pytest.mark.parametrize(
    ("bounds", "vals"),
    [
        ([1.0, 2.0], [1.0, 1.0]),  # count mismatch
        ([2.0, 1.0, 3.0], [1.0, 1.0]),  # not increasing
        ([1.0, 2.0, 3.0], [1.0, -1.0]),  # negative flux
        ([1.0, 2.0, 3.0], [0.0, 0.0]),  # zero total
        ([0.0, 1.0, 2.0], [1.0, 1.0]),  # non-positive boundary
    ],
)
def test_invalid_spectra_rejected(bounds, vals):
    with pytest.raises(SpectrumError):
        from_groups(bounds, vals, identity="bad")


def test_normalized_has_no_absolute_flag_and_scaling_restores_it():
    spec = from_groups([1, 2, 3], [2.0, 6.0], absolute=False, identity="shape")
    assert not spec.absolute and spec.total_flux == pytest.approx(1.0)
    absolute = spec.scaled_to(1e12)
    assert absolute.absolute and absolute.total_flux == pytest.approx(1e12)
    assert absolute.group_flux[1] / absolute.group_flux[0] == pytest.approx(3.0)


def test_rebin_conserves_flux_and_is_exact_for_aligned_grids():
    spec = from_groups([1, 2, 4, 8], [1.0, 2.0, 4.0], identity="s")
    fine, lost = rebin(spec, [1, 2, 3, 4, 6, 8])
    assert lost == 0.0
    assert fine.total_flux == pytest.approx(7.0, rel=1e-12)
    assert fine.group_flux == pytest.approx((1.0, 1.0, 1.0, 2.0, 2.0))
    coarse, _ = rebin(fine, [1, 8])
    assert coarse.group_flux[0] == pytest.approx(7.0)


def test_rebin_lethargy_conserves_total():
    spec = from_groups([1, 10, 100], [1.0, 1.0], identity="s")
    out, _ = rebin(spec, [1, 3, 30, 100], interpolation="lethargy")
    assert out.total_flux == pytest.approx(2.0, rel=1e-12)


def test_rebin_out_of_range_flux_is_reported_not_hidden():
    spec = from_groups([1, 2, 3], [1.0, 1.0], identity="s")
    with pytest.raises(SpectrumError, match="outside the library"):
        rebin(spec, [1, 2])
    out, lost = rebin(spec, [1, 2], max_lost_fraction=0.6)
    assert lost == pytest.approx(0.5) and out.total_flux == pytest.approx(1.0)


def test_rebin_line_goes_to_containing_group():
    out, lost = rebin(monoenergetic(14.0, total_flux=5.0), [1e6, 1e7, 1.5e7, 2e7])
    assert lost == 0.0 and out.group_flux == (0.0, 5.0, 0.0)[:0] + (0.0, 5.0, 0.0)
    with pytest.raises(SpectrumError):
        rebin(monoenergetic(30.0, total_flux=5.0), [1e6, 2e7])


def test_csv_and_json_roundtrip(tmp_path):
    csv_path = tmp_path / "s.csv"
    csv_path.write_text("e_low,e_high,flux\n# c\n1,2,1.0\n2,4,3.0\n")
    spec = load_spectrum_file(csv_path, energy_units="MeV")
    assert spec.boundaries_eV == (1e6, 2e6, 4e6) and spec.source == "file"
    assert spec.detail["file_sha256"]
    json_path = tmp_path / "s.json"
    json_path.write_text(
        json.dumps({"energy_boundaries": [1, 2, 4], "flux": [1.0, 3.0], "energy_units": "MeV"})
    )
    assert load_spectrum_file(json_path).group_flux == spec.group_flux


def test_csv_gap_rejected(tmp_path):
    path = tmp_path / "gap.csv"
    path.write_text("e_low,e_high,flux\n1,2,1\n3,4,1\n")
    with pytest.raises(SpectrumError, match="gaps"):
        load_spectrum_file(path)


def test_transport_tally_shape_only_without_rate(tmp_path):
    path = tmp_path / "t.json"
    path.write_text(
        json.dumps({"energy_boundaries": [1, 2, 3], "flux_per_source_particle": [1.0, 3.0]})
    )
    shape = load_transport_tally(path)
    assert not shape.absolute and shape.source == "external_transport"
    absolute = load_transport_tally(path, source_rate_per_s=1e10, volume_cm3=2.0)
    assert absolute.absolute and absolute.total_flux == pytest.approx(2e10)


def test_content_hash_ignores_identity_and_paths():
    a = from_groups([1, 2, 3], [1.0, 2.0], identity="a")
    b = from_groups([1, 2, 3], [1.0, 2.0], identity="b", detail={"file": "x"})
    c = from_groups([1, 2, 3], [1.0, 2.5], identity="a")
    assert a.sha256() == b.sha256() != c.sha256()


def test_validate_spectrum_on_hand_built_object():
    bad = NeutronSpectrum("x", "file", (1.0,), True, boundaries_eV=(2.0, 1.0))
    assert validate_spectrum(bad)
