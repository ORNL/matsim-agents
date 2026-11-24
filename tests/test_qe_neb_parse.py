"""Neb.x output parsing on synthetic fixtures built from the QE print formats."""

import math
from pathlib import Path

import pytest

from matsim_agents.backends.dft.qe_neb_parse import (
    parse_neb_dat,
    parse_neb_output,
)
from matsim_agents.backends.dft.qe_neb_run import NebRunRecord
from matsim_agents.workflows.neb_mapping import to_neb_result

ROOT = Path(__file__).resolve().parents[1]
ENERGIES = [-100.0, -99.6, -99.2, -99.7, -100.1]  # saddle at image 3: 0.8 up, 0.9 back
ERRORS = [0.0, 0.03, 0.02, 0.04, 0.0]


def _table(energies, errors, frozen=None):
    frozen = frozen or [True] + [False] * (len(energies) - 2) + [True]
    rows = "\n".join(
        f"     {i:5d}    {e:15.7f}          {err:10.6f}            {'T' if f else 'F'}"
        for i, (e, err, f) in enumerate(zip(energies, errors, frozen, strict=True), start=1)
    )
    return "     image        energy (eV)        error (eV/A)        frozen\n\n" + rows + "\n"


def _iteration(n, energies, errors, ci=None, frozen=None):
    top = max(energies)
    fwd = top - energies[0]
    rev = top - energies[-1]
    text = (
        f"\n     ------------------------------ iteration {n:3d} "
        "------------------------------\n\n"
        "     tcpu =      1.0    self-consistency for image   2\n"
        f"\n     activation energy (->) = {fwd:10.6f} eV\n"
        f"     activation energy (<-) = {rev:10.6f} eV\n\n"
    )
    text += _table(energies, errors, frozen)
    if ci is not None:
        text += f"\n     climbing image = {ci:2d}\n"
    return text + "\n     path length            =  3.100 bohr\n"


CONVERGED_TAIL = (
    "\n     -------------------------------------------------------------------------\n"
    "\n     neb: convergence achieved in   7 iterations\n"
)
LIMIT_TAIL = "\n     neb: reached the maximum number of steps\n"


def _converged_text():
    return (
        _iteration(6, ENERGIES, [0, 0.2, 0.2, 0.2, 0])
        + _iteration(7, ENERGIES, ERRORS, ci=3)
        + CONVERGED_TAIL
    )


def _record(status, code):
    return NebRunRecord(
        argv=("neb.x",),
        executable="neb.x",
        cwd="/w",
        input_path="/w/neb.in",
        input_sha256="0" * 64,
        config_hash="1" * 64,
        stdout_path="/w/neb.out",
        stdout_sha256=None,
        started_at="t0",
        finished_at="t1",
        exit_code=code,
        status=status,
    )


def test_converged_run_with_climbing_image():
    parse = parse_neb_output(
        _converged_text(), expected_images=5, path_thr_ev_ang=0.05, record=_record("exited_zero", 0)
    )
    assert parse.outcome == "converged" and parse.iterations_completed == 7
    assert parse.image_indices == (1, 2, 3, 4, 5)
    assert parse.relative_energies_eV == pytest.approx([0.0, 0.4, 0.8, 0.3, -0.1])
    assert parse.forward_barrier_eV == pytest.approx(0.8)
    assert parse.reverse_barrier_eV == pytest.approx(0.9)
    assert parse.endpoint_energy_difference_eV == pytest.approx(-0.1)
    assert parse.highest_image_index == 3
    assert parse.climbing_image_used and parse.climbing_images == (3,)
    assert parse.max_interior_error_eV_per_A == pytest.approx(0.04)
    assert any("not a refined saddle" in w for w in parse.warnings)


def test_forward_minus_reverse_barrier_is_the_endpoint_energy_difference():
    # (E_s - E_0) - (E_s - E_N) = E_N - E_0
    parse = parse_neb_output(_converged_text())
    assert parse.forward_barrier_eV - parse.reverse_barrier_eV == pytest.approx(
        parse.endpoint_energy_difference_eV
    )


def test_uses_the_last_iteration_table_only():
    text = _converged_text()
    parse = parse_neb_output(text)
    assert parse.errors_eV_per_A == pytest.approx(ERRORS)  # not the iteration-6 values


def test_iteration_limit_is_not_convergence():
    text = _iteration(50, ENERGIES, [0, 0.3, 0.2, 0.3, 0]) + LIMIT_TAIL
    parse = parse_neb_output(text, record=_record("exited_zero", 0))
    assert parse.outcome == "iteration_limit"
    assert parse.has_energies and parse.forward_barrier_eV == pytest.approx(0.8)


def test_exit_zero_without_message_is_incomplete():
    parse = parse_neb_output(_iteration(3, ENERGIES, ERRORS), record=_record("exited_zero", 0))
    assert parse.outcome == "incomplete_output"


def test_no_table_is_incomplete_and_has_no_numbers():
    parse = parse_neb_output("     some banner only\n", record=_record("exited_zero", 0))
    assert parse.outcome == "incomplete_output" and not parse.has_energies
    assert parse.forward_barrier_eV is None


def test_truncated_table_header_only():
    text = _iteration(2, ENERGIES, ERRORS).split("\n\n")[0] + "\n"
    text += "\n     activation energy (->) =   0.800000 eV\n"
    parse = parse_neb_output(text)
    assert parse.outcome == "incomplete_output" and parse.forward_barrier_eV is None


def test_process_failures_are_classified():
    parse = parse_neb_output("", record=_record("timed_out", 124))
    assert parse.outcome == "process_failure"
    assert parse_neb_output("", exit_code=137).outcome == "process_failure"
    mid = _iteration(3, ENERGIES, ERRORS)
    assert parse_neb_output(mid, record=_record("exited_nonzero", 1)).outcome == "process_failure"
    assert parse_neb_output(mid, record=_record("exited_nonzero", 1)).has_energies


def test_convergence_message_with_failed_process_is_a_process_failure():
    parse = parse_neb_output(_converged_text(), record=_record("exited_nonzero", 2))
    assert parse.outcome == "process_failure"
    assert any("convergence message present" in w for w in parse.warnings)


def test_electronic_failure_is_distinguished():
    text = "     convergence NOT achieved after 100 iterations: stopping\n"
    assert parse_neb_output(text, exit_code=1).outcome == "electronic_failure"
    mid = _iteration(3, ENERGIES, ERRORS) + text
    assert parse_neb_output(mid).outcome == "electronic_failure"


def test_routine_error_without_table_is_process_failure():
    text = "     Error in routine read_restart (1):\n     restart file not found\n"
    parse = parse_neb_output(text)
    assert parse.outcome == "process_failure"
    assert any("Error in routine" in w for w in parse.warnings)


def test_missing_image_rows_are_detected():
    energies = ENERGIES[:3]
    text = _iteration(4, energies, ERRORS[:3]) + CONVERGED_TAIL
    parse = parse_neb_output(text, expected_images=5)
    assert parse.outcome == "missing_images" and not parse.has_energies


def test_noncontiguous_image_indices_are_missing_images():
    text = _iteration(4, ENERGIES, ERRORS).replace("         3    ", "         4    ", 1)
    parse = parse_neb_output(text)
    assert parse.outcome == "missing_images"


def test_nan_energy_is_invalid_numbers():
    text = _iteration(4, ENERGIES, ERRORS).replace("-99.2000000", "NaN")
    parse = parse_neb_output(text + CONVERGED_TAIL)
    assert parse.outcome == "invalid_numbers" and not parse.has_energies
    assert parse.forward_barrier_eV is None


def test_convergence_claim_above_threshold_is_inconsistent():
    text = _iteration(7, ENERGIES, [0, 0.3, 0.2, 0.3, 0]) + CONVERGED_TAIL
    parse = parse_neb_output(text, path_thr_ev_ang=0.05)
    assert parse.outcome == "inconsistent_output"
    assert any("exceeds path_thr" in w for w in parse.warnings)
    assert parse_neb_output(text).outcome == "converged"  # no threshold supplied: not checked


def test_endpoint_maximum_gives_no_barrier():
    energies = [-99.0, -99.4, -99.5, -99.7, -100.1]
    text = _iteration(5, energies, ERRORS) + CONVERGED_TAIL
    parse = parse_neb_output(text)
    assert parse.outcome == "converged"
    assert parse.forward_barrier_eV is None and parse.reverse_barrier_eV is None
    assert parse.highest_image_index == 1
    assert any("endpoint" in w for w in parse.warnings)


def test_manual_climbing_images_from_summary():
    text = "     list of climbing images :   3,\n" + _converged_text().replace(
        "\n     climbing image =  3\n", ""
    )
    parse = parse_neb_output(text)
    assert parse.climbing_images == (3,) and parse.climbing_image_used


def test_no_climbing_image_when_none_reported():
    text = _iteration(7, ENERGIES, ERRORS) + CONVERGED_TAIL
    parse = parse_neb_output(text)
    assert not parse.climbing_image_used and parse.climbing_images == ()


def test_reads_output_from_a_path(tmp_path):
    path = tmp_path / "neb.out"
    path.write_text(_converged_text())
    assert parse_neb_output(path).outcome == "converged"


def test_dat_file_is_parsed_and_cross_checked(tmp_path):
    rel = [e - ENERGIES[0] for e in ENERGIES]
    dat = tmp_path / "p.dat"
    coords = [0.0, 0.25, 0.5, 0.75, 1.0]
    dat.write_text(
        "".join(
            f"  {c:16.10f}  {r:16.10f}  {x:16.10f}\n"
            for c, r, x in zip(coords, rel, ERRORS, strict=True)
        )
    )
    assert parse_neb_dat(dat)[1] == pytest.approx(rel)
    parse = parse_neb_output(_converged_text(), dat_path=dat)
    assert parse.dat_cross_checked
    dat.write_text(dat.read_text().replace("0.8000000000", "0.9000000000"))
    assert not parse_neb_output(_converged_text(), dat_path=dat).dat_cross_checked
    assert any("differ" in w for w in parse_neb_output(_converged_text(), dat_path=dat).warnings)


@pytest.mark.parametrize("content", ["", "1 2\n", "0.0 nan 0.1\n", "a b c\n", "0 inf 0\n"])
def test_bad_dat_files_raise(tmp_path, content):
    dat = tmp_path / "p.dat"
    dat.write_text(content)
    with pytest.raises(ValueError):
        parse_neb_dat(dat)


def test_bad_dat_is_a_warning_not_a_failure(tmp_path):
    parse = parse_neb_output(_converged_text(), dat_path=tmp_path / "missing.dat")
    assert parse.outcome == "converged" and not parse.dat_cross_checked
    assert any("prefix.dat unusable" in w for w in parse.warnings)


def test_to_neb_result_converged_is_never_qualified():
    parse = parse_neb_output(_converged_text(), expected_images=5)
    result = to_neb_result(parse, method_id="qe-pbe", n_images=5, endpoints_qualified=True)
    assert result.state == "converged_unqualified" and result.endpoints_qualified
    assert result.forward_barrier_eV == pytest.approx(0.8)
    assert result.climbing_image_used and result.provenance["parser_outcome"] == "converged"
    assert result.provenance["iterations_completed"] == "7"


def test_to_neb_result_maps_states_and_withholds_numbers():
    limit = parse_neb_output(_iteration(9, ENERGIES, ERRORS) + LIMIT_TAIL)
    assert to_neb_result(limit, method_id="m", n_images=5).state == "unconverged"
    assert to_neb_result(limit, method_id="m", n_images=5).relative_energies_eV
    failed = parse_neb_output("", exit_code=1)
    result = to_neb_result(failed, method_id="m", n_images=5)
    assert result.state == "failed" and not result.relative_energies_eV
    inc = parse_neb_output(_iteration(3, ENERGIES, ERRORS))
    assert to_neb_result(inc, method_id="m", n_images=5).state == "incomplete"
    assert to_neb_result(inc, method_id="m", n_images=5).forward_barrier_eV is None


def test_to_neb_result_image_count_mismatch_raises():
    parse = parse_neb_output(_converged_text())
    with pytest.raises(ValueError, match="expected 6"):
        to_neb_result(parse, method_id="m", n_images=6)


def test_parsed_numbers_are_finite():
    parse = parse_neb_output(_converged_text())
    assert all(math.isfinite(v) for v in parse.energies_eV + parse.errors_eV_per_A)


def test_prefix_dat_error_column_is_ev_per_angstrom_and_is_not_rescaled(tmp_path):
    dat = tmp_path / "p.dat"
    dat.write_text("".join(f"0.0 0.0 {x}\n" for x in ERRORS))
    assert parse_neb_dat(dat)[2] == pytest.approx(ERRORS)
    # A spurious eV/Bohr to eV/Angstrom factor (1/0.529177) would push 0.04 above 0.05.
    parse = parse_neb_output(
        _converged_text(), path_thr_ev_ang=0.05, dat_path=dat, record=_record("exited_zero", 0)
    )
    assert parse.outcome == "converged"
    assert parse.max_interior_error_eV_per_A == pytest.approx(max(ERRORS))


def test_qe_source_writes_neb_error_in_ev_per_angstrom():
    source = ROOT / "external/quantum-espresso/src/NEB/src/path_base.f90"
    if not source.is_file():
        pytest.skip("QE source is not checked out")
    text = " ".join(source.read_text().split())
    assert "error(i) = MAXVAL( ABS( grad(:,i) ) ) / bohr_radius_angs * autoev" in text
