"""SPECTRA-PKA adapter tests.

These use a mock executable and synthetic fixtures. They verify file handling, status
logic, and contracts. They are NOT nuclear-physics verification of SPECTRA-PKA.
"""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from matsim_agents.integrations.external.spectra_pka import adapter, availability
from matsim_agents.integrations.external.spectra_pka.availability import Status
from matsim_agents.integrations.external.spectra_pka.config import (
    CampaignSpec,
    ConfigError,
    load_config_file,
)
from matsim_agents.integrations.external.spectra_pka.input_writer import (
    build_input_text,
    stage_inputs,
)
from matsim_agents.integrations.external.spectra_pka.parser import (
    BlockKind,
    ParseError,
    parse_number,
    parse_output_text,
    to_pka_spectrum,
)
from matsim_agents.integrations.external.spectra_pka.validation import validate_output
from matsim_agents.irradiation.contracts import (
    ContractError,
    DefectStatus,
    NoCascadeModel,
    PKASpectrum,
    ValidationStatus,
)
from matsim_agents.irradiation.pka_sampling import sample_pka_events
from tests.spectra_pka_fixtures import (
    SYNTHETIC_INDEXES,
    SYNTHETIC_OUT,
    make_campaign,
    make_config,
    write_mock_executable,
)


@pytest.fixture
def campaign(tmp_path):
    return make_campaign(tmp_path)


@pytest.fixture
def mock_exe(tmp_path):
    return write_mock_executable(tmp_path / "spectra-pka")


def _spectrum(campaign, tmp_path):
    parsed = parse_output_text(SYNTHETIC_OUT, indexes_text=SYNTHETIC_INDEXES)
    prov = adapter.PKASourceProvenance(source="t", nuclear_data_identity="t")
    return to_pka_spectrum(parsed, adapter.scenario_from_campaign(campaign), prov)


# --- config ---------------------------------------------------------------------------------


def test_config_rejects_unknown_field(tmp_path):
    f = tmp_path / "c.yaml"
    f.write_text(yaml.safe_dump({"external_tools": {"spectra_pka": {"bogus": 1}}}))
    with pytest.raises(ConfigError):
        load_config_file(f)


def test_config_defaults_disabled(tmp_path):
    f = tmp_path / "c.yaml"
    f.write_text("external_tools: {}\n")
    assert load_config_file(f).external_tools.spectra_pka.enabled is False


def test_slurm_requires_account():
    with pytest.raises(ValueError):
        adapter.SpectraPKAConfig(execution={"mode": "slurm"})


def test_campaign_isotope_fractions_must_match_composition(campaign):
    data = campaign.model_dump()
    data["channels"][0]["atom_fraction"] = 0.5
    with pytest.raises(ValueError, match="isotope fractions"):
        CampaignSpec(**data)


def test_campaign_rejects_duplicate_and_foreign_isotopes(campaign):
    data = campaign.model_dump()
    data["channels"] = [*data["channels"], dict(data["channels"][0])]
    with pytest.raises(ValueError):
        CampaignSpec(**data)
    data = campaign.model_dump()
    data["channels"][0]["parent_ele"] = "Zr"
    with pytest.raises(ValueError):
        CampaignSpec(**data)


def test_relative_paths_resolve_against_config_dir(tmp_path):
    f = tmp_path / "c.yaml"
    f.write_text("external_tools:\n  spectra_pka:\n    workdir: out\n")
    cfg = load_config_file(f).external_tools.spectra_pka
    assert cfg.workdir == (tmp_path / "out").resolve()


# --- input writer ---------------------------------------------------------------------------


def test_input_uses_only_known_labels(campaign):
    text = build_input_text(campaign)
    known = {
        "flux_filename", "results_stub", "num_columns", "columns", "number_pka_files",
        "flux_norm_type", "pka_filetype", "do_mtd_sums", "do_global_sums",
        "do_exclude_light_from_total", "energies_once_perfile", "max_global_recoils",
        "flux_rescale_value",
    }  # fmt: skip
    for line in text.splitlines():
        if line.startswith(("#", '"')) or "=" not in line:
            continue
        assert line.split("=")[0] in known
    assert "number_pka_files=2" in text and "pka_filetype=2" in text


def test_isotope_ratios_within_element_sum_to_one(campaign):
    iso = campaign.isotopic_composition()
    assert iso == {"W": {182: 1.0}, "Ta": {181: 1.0}}


def test_staging_never_modifies_user_data(campaign, tmp_path):
    before = {c.pka_file: c.pka_file.read_bytes() for c in campaign.channels}
    staged = stage_inputs(campaign, tmp_path / "run")
    assert staged.input_file.is_file()
    assert all(c.pka_file.read_bytes() == before[c.pka_file] for c in campaign.channels)
    assert len(staged.data_sha256) == 3


def test_stage_missing_data_raises(campaign, tmp_path):
    campaign.channels[0].pka_file.unlink()
    with pytest.raises(Exception, match="not found"):
        stage_inputs(campaign, tmp_path / "run")


# --- availability ---------------------------------------------------------------------------


def test_not_installed(tmp_path, campaign):
    cfg = make_config(tmp_path, tmp_path / "missing")
    assert availability.check_availability(cfg, campaign).status is Status.NOT_INSTALLED
    cfg = make_config(tmp_path, None)
    assert availability.check_availability(cfg, campaign).status is Status.NOT_INSTALLED


def test_installed_without_campaign(tmp_path, mock_exe):
    result = availability.check_availability(make_config(tmp_path, mock_exe))
    assert result.status is Status.INSTALLED
    assert result.executable_sha256
    assert result.check("invocation").ok is True


def test_inputs_missing(tmp_path, mock_exe, campaign):
    campaign.channels[0].pka_file.unlink()
    result = availability.check_availability(make_config(tmp_path, mock_exe), campaign)
    assert result.status is Status.INPUTS_MISSING
    assert result.check("data_coverage").ok is False


def test_ready_but_smoke_test_not_run(tmp_path, mock_exe, campaign):
    result = availability.check_availability(make_config(tmp_path, mock_exe), campaign)
    assert result.status is Status.READY
    assert result.smoke_test == "not_run"


def test_wrong_program_is_execution_failed(tmp_path, campaign):
    exe = tmp_path / "other"
    exe.write_text(f"#!{sys.executable}\nprint('hello')\n")
    exe.chmod(0o755)
    result = availability.check_availability(make_config(tmp_path, exe), campaign)
    assert result.status is Status.EXECUTION_FAILED


def test_non_executable_file_is_not_installed(tmp_path, campaign):
    exe = tmp_path / "plain"
    exe.write_text("x")
    exe.chmod(0o644)
    assert (
        availability.check_availability(make_config(tmp_path, exe), campaign).status
        is Status.NOT_INSTALLED
    )


def test_stale_smoke_record_is_flagged(tmp_path, mock_exe, campaign):
    cfg = make_config(tmp_path, mock_exe)
    availability.write_smoke_record(
        cfg.workdir, executable_sha256="deadbeef", passed=True, evidence="x"
    )
    assert availability.check_availability(cfg, campaign).smoke_test == "stale"


# --- parser / validation --------------------------------------------------------------------


def test_parse_number_fortran_forms():
    assert parse_number("1.5E-03") == 1.5e-3
    assert parse_number("1.5-103") == 1.5e-103
    with pytest.raises(ParseError):
        parse_number("abc")
    with pytest.raises(ParseError):
        parse_number("nan")


def test_parse_blocks_and_unknown_daughter():
    parsed = parse_output_text(SYNTHETIC_OUT, indexes_text=SYNTHETIC_INDEXES)
    assert len(parsed.of_kind(BlockKind.CHANNEL)) == 2
    ta = [b for b in parsed.of_kind(BlockKind.CHANNEL) if b.recoil_element == "Ta"][0]
    assert ta.daughter == "unknown"
    assert parsed.indexes[1].pka_file == "pka_W182.asc"
    assert parsed.of_kind(BlockKind.INPUT_SPECTRUM)[0].footer["total_flux"] == 1.0e10


REAL_DATA = Path(__file__).parent / "data" / "spectra_pka"


def test_parse_real_output_excerpt():
    """Layout check against an excerpt of a real run (see tests/data/spectra_pka/README.md)."""
    parsed = parse_output_text(
        (REAL_DATA / "real_w182_excerpt.out").read_text(),
        indexes_text=(REAL_DATA / "real_w182_excerpt.indexes").read_text(),
    )
    assert [b.kind for b in parsed.blocks] == [
        BlockKind.INPUT_SPECTRUM,
        BlockKind.CHANNEL,
        BlockKind.GLOBAL_ISOTOPE,
        BlockKind.GLOBAL_ELEMENT,
        BlockKind.GLOBAL_TOTAL,
    ]
    spectrum, channel, iso, elem, _ = parsed.blocks
    assert spectrum.footer == {"total_flux": 2.4608e14, "fluence": 2.4608e14}
    assert channel.reaction == "(n,np)"
    assert (channel.recoil_element, channel.recoil_mass_number) == ("H", 1)
    assert (channel.parent_element, channel.parent_mass_number) == ("W", 182)
    assert channel.footer["average_pka_energy_ev"] == 2.8131e6
    assert (iso.recoil_element, iso.recoil_mass_number) == ("W", 182)
    assert iso.rows[0] == (1.0e-11, 1.0471e-11, 1.6399e-12, 3.6581e-04, 3.6581e-04)
    assert elem.recoil_element == "W"
    assert parsed.indexes[1].description == "(n,np) proton matrix [ H1 ] from [ W182 ]"
    assert parsed.indexes[1].pka_file == "pka_W182.asc"
    assert parsed.indexes[2].pka_file is None


def test_input_columns_on_one_line(campaign):
    # read_input.f90 reads the column headers from the value of the columns= label itself.
    lines = build_input_text(campaign).splitlines()
    cols = [ln for ln in lines if ln.startswith("columns=")]
    assert len(cols) == 1 and len(cols[0].split()) > 1


def test_parser_rejects_wrong_column_count():
    bad = SYNTHETIC_OUT.replace(
        " 5.0E-07 1.0E-06 2.0E-03 5.0E-01\n", " 5.0E-07 1.0E-06 2.0E-03\n", 1
    )
    with pytest.raises(ParseError, match="columns"):
        parse_output_text(bad)


def test_parser_rejects_empty_and_unrecognised():
    with pytest.raises(ParseError):
        parse_output_text("")
    with pytest.raises(ParseError):
        parse_output_text("### index 1 #####\n# something else entirely\n 1 2 3\n")
    with pytest.raises(ParseError):
        parse_output_text("garbage before blocks\n### index 1 #####\n")


def test_validation_ok_and_failures():
    parsed = parse_output_text(SYNTHETIC_OUT)
    assert (
        validate_output(parsed, end_marker_seen=True).status is ValidationStatus.STRUCTURALLY_VALID
    )
    assert (
        validate_output(parsed, end_marker_seen=False).status
        is ValidationStatus.STRUCTURALLY_INVALID
    )
    neg = parse_output_text(
        SYNTHETIC_OUT.replace("3.0E-03 5.0E-01 5.0E-01", "-3.0E-03 5.0E-01 5.0E-01")
    )
    assert not validate_output(neg, end_marker_seen=True).ok
    overlap = parse_output_text(
        SYNTHETIC_OUT.replace("1.0E-06 2.0E-06 2.0E-03", "9.0E-07 2.0E-06 2.0E-03")
    )
    assert not validate_output(overlap, end_marker_seen=True).ok


def test_validation_rejects_truncated_global_matrix():
    truncated = parse_output_text(
        SYNTHETIC_OUT.replace(" 1.0E-06 2.0E-06 3.0E-03 1.0E+00 1.0E+00\n", "")
    )
    w182 = next(
        b for b in truncated.blocks if b.kind is not BlockKind.CHANNEL and "W182" in b.title
    )
    assert len(w182.rows) == 1 and w182.rows[-1][4] == 0.5
    report = validate_output(truncated, end_marker_seen=True)
    assert not report.ok
    assert any("truncated" in e for e in report.errors)


def test_to_spectrum_requires_global_blocks():
    only = SYNTHETIC_OUT.split("### index 3")[0]
    parsed = parse_output_text(only)
    with pytest.raises(ParseError, match="do_global_sums"):
        to_pka_spectrum(parsed, None, None)  # type: ignore[arg-type]


# --- contracts / sampling -------------------------------------------------------------------


def test_spectrum_aggregation_and_probabilities(campaign, tmp_path):
    spec = _spectrum(campaign, tmp_path)
    probs = spec.species_probabilities()
    assert math.isclose(sum(probs.values()), 1.0)
    assert math.isclose(probs["W182"], 0.75)
    assert set(spec.element_rates()) == {"W", "Ta"}
    assert spec.provenance.numerical_validation is ValidationStatus.NOT_VALIDATED


def test_empty_spectrum_rejected(campaign):
    prov = adapter.PKASourceProvenance(source="t", nuclear_data_identity="t")
    from matsim_agents.irradiation.contracts import RecoilSpecies, SpeciesSpectrum

    zero = SpeciesSpectrum(
        species=RecoilSpecies(element="W"), bin_low_mev=(1.0,), bin_high_mev=(2.0,), rates=(0.0,)
    )
    with pytest.raises(ValueError):
        PKASpectrum(
            scenario=adapter.scenario_from_campaign(campaign),
            spectra=(zero,),
            rate_unit="x",
            provenance=prov,
        )


def test_sampling_is_reproducible_and_respects_sites(campaign, tmp_path):
    spec = _spectrum(campaign, tmp_path)
    lattice = ["W"] * 9 + ["Ta"]
    a = sample_pka_events(spec, 400, seed=7, lattice_elements=lattice)
    b = sample_pka_events(spec, 400, seed=7, lattice_elements=lattice)
    c = sample_pka_events(spec, 400, seed=8, lattice_elements=lattice)
    assert a.events == b.events
    assert a.events != c.events
    for ev in a.events:
        assert ev.site_element == ev.species.element
        assert ev.time_s is not None and ev.interval_s > 0
    assert [e.time_s for e in a.events] == sorted(e.time_s for e in a.events)
    w_fraction = sum(e.species.element == "W" for e in a.events) / 400
    assert abs(w_fraction - 0.75) < 0.1


def test_sampling_reports_unplaceable_species(campaign, tmp_path):
    spec = _spectrum(campaign, tmp_path)
    result = sample_pka_events(spec, 50, seed=1, lattice_elements=["W"] * 5)
    assert result.excluded_species == ("Ta181",)
    assert math.isclose(result.excluded_rate_fraction, 0.25)
    with pytest.raises(ContractError):
        sample_pka_events(spec, 5, seed=1, lattice_elements=["Zr"] * 5)


def test_neighbor_configuration_recorded(campaign, tmp_path):
    spec = _spectrum(campaign, tmp_path)
    lattice = ["W", "Ta", "W"]
    nb = [[1, 2], [0, 2], [0, 1]]
    ev = sample_pka_events(spec, 5, seed=3, lattice_elements=lattice, neighbors=nb).events[0]
    assert len(ev.chemical_configuration) == 2


def test_pka_is_not_defects():
    from matsim_agents.irradiation.contracts import PKAEvent, RecoilSpecies

    ev = PKAEvent(index=0, species=RecoilSpecies(element="W"), energy_mev=1e-6, probability=0.5)
    assert NoCascadeModel().surviving_defects(ev).status is DefectStatus.NOT_ESTABLISHED


def test_irradiation_package_does_not_import_adapter():
    code = (
        "import sys, matsim_agents.irradiation;"
        "assert not any('spectra_pka' in m for m in sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


# --- adapter end to end with the mock executable --------------------------------------------


def test_dry_run_stages_but_does_not_execute(tmp_path, mock_exe, campaign):
    cfg = make_config(tmp_path, mock_exe)
    result = adapter.run(cfg, campaign)
    assert result.status == "DRY_RUN"
    assert (result.run_dir / "spectra_pka.in").is_file()
    assert not (result.run_dir / "spka.out").exists()


def test_disabled_and_not_installed_refuse(tmp_path, campaign):
    cfg = make_config(tmp_path, tmp_path / "nope")
    with pytest.raises(adapter.AdapterError, match="not installed"):
        adapter.run(cfg, campaign)
    with pytest.raises(adapter.AdapterError, match="enabled"):
        adapter.run(cfg.model_copy(update={"enabled": False}), campaign)


def test_execute_collects_validates_and_keeps_raw(tmp_path, mock_exe, campaign):
    cfg = make_config(tmp_path, mock_exe)
    result = adapter.run(cfg, campaign, execute=True)
    assert result.status == "STRUCTURALLY_VALID"
    assert result.outcome.process_succeeded
    assert result.spectrum is not None
    assert (result.run_dir / "spka.out").read_text() == SYNTHETIC_OUT
    prov = result.spectrum.provenance
    assert prov.executable_sha256 and "spka.out" in prov.output_sha256
    assert prov.numerical_validation is ValidationStatus.NOT_VALIDATED
    # restart: collecting again from the preserved raw files gives the same spectrum
    again = adapter.collect(cfg, campaign, result.run_dir)
    assert again.spectrum.species_probabilities() == result.spectrum.species_probabilities()
    assert adapter.status(cfg, campaign)["output_present"] is True


def test_incomplete_output_is_not_valid(tmp_path, campaign):
    exe = write_mock_executable(
        tmp_path / "bad", out="### index 0 #####\n#INPUT SPECTRUM\n 1 2 3\n"
    )
    result = adapter.run(make_config(tmp_path, exe), campaign, execute=True)
    assert result.status == "STRUCTURALLY_INVALID"
    assert result.spectrum is None


def test_failing_executable_reports_failure(tmp_path, campaign):
    exe = tmp_path / "fail"
    exe.write_text(
        f"#!{sys.executable}\nimport sys\nif sys.argv[1].startswith('matsim_agents_probe'):\n"
        "    print('error opening parameter input file')\nelse:\n    sys.exit(3)\n"
    )
    exe.chmod(0o755)
    result = adapter.run(make_config(tmp_path, exe), campaign, execute=True)
    assert result.status == "EXECUTION_FAILED"
    assert result.outcome.returncode == 3


def test_timeout_is_reported(tmp_path, campaign):
    exe = tmp_path / "slow"
    exe.write_text(
        f"#!{sys.executable}\nimport sys, time\nif sys.argv[1].startswith('matsim_agents_probe'):\n"
        "    print('error opening parameter input file')\nelse:\n    time.sleep(30)\n"
    )
    exe.chmod(0o755)
    cfg = make_config(tmp_path, exe)
    cfg = cfg.model_copy(
        update={"execution": cfg.execution.model_copy(update={"timeout_seconds": 1})}
    )
    result = adapter.run(cfg, campaign, execute=True)
    assert result.status == "EXECUTION_FAILED" and result.outcome.timed_out


def test_slurm_mode_submits_once_via_slurm_jobs(tmp_path, mock_exe, campaign, monkeypatch):
    from matsim_agents.execution.slurm_jobs import CommandResult
    from matsim_agents.integrations.external.spectra_pka import runner

    calls = []

    def fake(argv):
        calls.append(argv)
        return CommandResult(0, "4242\n", "")

    monkeypatch.setattr(
        runner,
        "submit_job",
        lambda spec, manifest, **kw: __import__(
            "matsim_agents.execution.slurm_jobs", fromlist=["submit_job"]
        ).submit_job(spec, manifest, runner=fake),
    )
    cfg = make_config(tmp_path, mock_exe)
    cfg = cfg.model_copy(
        update={"execution": cfg.execution.model_copy(update={"mode": "slurm", "account": "ACCT"})}
    )
    first = adapter.run(cfg, campaign, execute=True)
    second = adapter.run(cfg, campaign, execute=True)
    assert first.status == "SUBMITTED" and first.outcome.job_id == "4242"
    assert second.outcome.submitted is False
    assert len(calls) == 1
    assert "--wrap" in calls[0] and calls[0][0] == "sbatch"


def test_no_shell_true_in_package():
    root = Path(adapter.__file__).parent
    for p in root.glob("*.py"):
        assert "shell=True" not in p.read_text()
