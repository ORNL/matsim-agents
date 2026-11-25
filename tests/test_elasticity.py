"""MP3: elastic stiffness from stress-strain finite differences (synthetic harmonic models)."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from matsim_agents.campaign.elasticity import (
    ElasticSettings,
    apply_strain,
    ase_stress_fn,
    compute_elastic_tensor,
    elastic_property_evidence,
    isotropic_moduli,
    plan_calculations,
    strain_set,
)
from matsim_agents.execution.contracts import EvidenceLevel
from matsim_agents.execution.properties import PropertyStatus


def cubic_c(c11, c12, c44):
    c = np.zeros((6, 6))
    for i in range(3):
        for j in range(3):
            c[i, j] = c11 if i == j else c12
    for i in (3, 4, 5):
        c[i, i] = c44
    return c


def _cell():
    atoms = Atoms("W2", positions=[[0, 0, 0], [1.6, 1.6, 1.6]], cell=[3.2] * 3, pbc=True)
    return atoms


def harmonic(c, reference, *, offset=None, cubic_term=0.0, calls=None):
    """Stress = C e (+ optional nonlinearity) for the strain of the cell vs ``reference``."""
    ref = np.asarray(reference.cell)

    def stress(atoms):
        if calls is not None:
            calls.append(1)
        f_t = np.linalg.solve(ref, np.asarray(atoms.cell))
        f = f_t.T
        eps = (f + f.T) / 2 - np.eye(3)
        e = np.array([eps[0, 0], eps[1, 1], eps[2, 2], 2 * eps[1, 2], 2 * eps[0, 2], 2 * eps[0, 1]])
        sigma = c @ e + cubic_term * e**2 * 1000
        if offset is not None:
            sigma = sigma + offset
        return sigma, True

    return stress


def test_recovers_known_cubic_stiffness_and_moduli():
    c = cubic_c(250.0, 150.0, 100.0)
    atoms = _cell()
    result = compute_elastic_tensor(atoms, harmonic(c, atoms), relax_fn=lambda a: (a, True))
    assert np.allclose(result.stiffness_GPa, c, atol=1e-8)
    assert result.converged and result.mechanically_stable
    k = (250 + 2 * 150) / 3
    g_v = (250 - 150 + 3 * 100) / 5
    assert result.moduli["bulk_modulus_voigt"] == pytest.approx(k)
    assert result.moduli["bulk_modulus_reuss"] == pytest.approx(k)
    assert result.moduli["shear_modulus_voigt"] == pytest.approx(g_v)
    assert result.moduli["shear_modulus_reuss"] < g_v
    assert result.moduli["youngs_modulus"] == pytest.approx(
        9
        * result.moduli["bulk_modulus"]
        * result.moduli["shear_modulus"]
        / (3 * result.moduli["bulk_modulus"] + result.moduli["shear_modulus"])
    )
    assert result.calculations == 1 + len(strain_set(result.settings.amplitudes))


def test_isotropic_material_has_zero_anisotropy():
    lam, mu = 100.0, 60.0
    c = cubic_c(lam + 2 * mu, lam, mu)
    moduli = isotropic_moduli(c)
    assert moduli["elastic_anisotropy"] == pytest.approx(0.0, abs=1e-9)
    assert moduli["shear_modulus"] == pytest.approx(mu)
    assert moduli["poissons_ratio"] == pytest.approx(lam / (2 * (lam + mu)))


def test_positive_bulk_modulus_is_not_stability():
    c = cubic_c(200.0, 250.0, 50.0)
    assert (c[0, 0] + 2 * c[0, 1]) / 3 > 0
    atoms = _cell()
    result = compute_elastic_tensor(atoms, harmonic(c, atoms), relax_fn=lambda a: (a, True))
    assert not result.mechanically_stable
    assert result.eigenvalues_GPa[0] < 0
    assert result.moduli == {}
    evidence = elastic_property_evidence(
        result,
        structure_id="s",
        composition="W2",
        evidence_ids=["ev"],
        level=EvidenceLevel.CONVERGED_DFT,
        backend="qe",
    )
    by = {e.property: e for e in evidence}
    assert by["elastic_stability_min_eigenvalue"].value < 0
    assert by["bulk_modulus"].status == PropertyStatus.UNSUPPORTED
    assert by["bulk_modulus"].value is None


def test_residual_stress_blocks_convergence():
    atoms = _cell()
    c = cubic_c(250.0, 150.0, 100.0)
    offset = np.array([2.0, 2.0, 2.0, 0, 0, 0])
    result = compute_elastic_tensor(
        atoms, harmonic(c, atoms, offset=offset), relax_fn=lambda a: (a, True)
    )
    assert not result.converged
    assert any("residual stress" in p for p in result.problems)
    evidence = elastic_property_evidence(
        result,
        structure_id="s",
        composition="W2",
        evidence_ids=["ev"],
        level=EvidenceLevel.CONVERGED_DFT,
        backend="qe",
    )
    assert all(
        e.status == PropertyStatus.UNCONVERGED or e.status == PropertyStatus.UNSUPPORTED
        for e in evidence
    )


def test_nonlinear_response_is_flagged_by_amplitude_spread():
    atoms = _cell()
    c = cubic_c(250.0, 150.0, 100.0)

    # Cubic stress terms differ between amplitudes only when the response is not odd-linear.
    def stress(a):
        s, ok = harmonic(c, atoms)(a)
        f = np.linalg.solve(np.asarray(atoms.cell), np.asarray(a.cell)).T
        eps = (f + f.T) / 2 - np.eye(3)
        return s * (1 + 800 * np.trace(eps) ** 2) + 5e4 * np.trace(eps) ** 3, ok

    result = compute_elastic_tensor(atoms, stress, relax_fn=lambda a: (a, True))
    assert result.convergence["max_amplitude_spread"] > 0


def test_unconverged_relaxation_is_not_converged():
    atoms = _cell()
    c = cubic_c(250.0, 150.0, 100.0)
    result = compute_elastic_tensor(atoms, harmonic(c, atoms), relax_fn=lambda a: (a, False))
    assert not result.converged


def test_clamped_ion_constants_are_labelled():
    atoms = _cell()
    result = compute_elastic_tensor(atoms, harmonic(cubic_c(250, 150, 100), atoms))
    assert any("clamped-ion" in p for p in result.problems)
    assert not result.converged


def test_settings_require_multiple_small_amplitudes():
    with pytest.raises(ValueError, match="two distinct"):
        ElasticSettings(amplitudes=[0.01])
    with pytest.raises(ValueError, match="small-strain"):
        ElasticSettings(amplitudes=[0.01, 0.2])


def test_asymmetric_stress_response_is_symmetrized_and_reported():
    atoms = _cell()
    c = cubic_c(250.0, 150.0, 100.0)
    c_skew = c.copy()
    c_skew[0, 1] += 3.0
    result = compute_elastic_tensor(atoms, harmonic(c_skew, atoms), relax_fn=lambda a: (a, True))
    sym = np.array(result.stiffness_GPa)
    assert np.allclose(sym, sym.T)
    assert result.convergence["max_asymmetry"] > 0
    assert "symmetrized" in result.convergence["symmetrization"]


def test_plan_never_substitutes_a_smaller_cell_or_hides_missing_cost():
    plan = plan_calculations(128, ElasticSettings(), parent_sites=1024)
    assert plan["status"] == "RESOURCE_BLOCKED"
    assert any("smaller" in b for b in plan["blockers"])
    assert any("benchmark" in b for b in plan["blockers"])
    ok = plan_calculations(
        1024,
        ElasticSettings(),
        parent_sites=1024,
        node_hours_per_calculation=10.0,
        remaining_node_hours=1000.0,
    )
    assert ok["status"] == "READY" and ok["calculations"] == 25
    assert ok["estimated_node_hours"] == pytest.approx(250.0)
    over = plan_calculations(
        1024,
        ElasticSettings(),
        parent_sites=1024,
        node_hours_per_calculation=100.0,
        remaining_node_hours=1000.0,
    )
    assert over["status"] == "RESOURCE_BLOCKED"
    with pytest.raises(ValueError, match="smaller supercell"):
        compute_elastic_tensor(_cell(), lambda a: ([0] * 6, True), parent_sites=1024)


def test_mlip_without_stress_is_unsupported():
    class NoStress:
        implemented_properties = ["energy", "forces"]

    with pytest.raises(ValueError, match="does not implement stress"):
        ase_stress_fn(NoStress)(_cell())


def test_strain_is_applied_to_the_cell_and_scaled_positions():
    atoms = _cell()
    strained = apply_strain(atoms, [0.01, 0, 0, 0, 0, 0])
    assert strained.cell[0, 0] == pytest.approx(3.2 * 1.01)
    assert np.allclose(strained.get_scaled_positions(), atoms.get_scaled_positions())


def test_mlip_derived_evidence_is_surrogate_not_calculated():
    atoms = _cell()
    result = compute_elastic_tensor(
        atoms, harmonic(cubic_c(250, 150, 100), atoms), relax_fn=lambda a: (a, True)
    )
    evidence = elastic_property_evidence(
        result,
        structure_id="s",
        composition="W2",
        evidence_ids=["ev"],
        level=EvidenceLevel.MLIP_PREDICTION,
        backend="mace",
    )
    assert {e.status for e in evidence} == {PropertyStatus.SURROGATE_PREDICTED}
    assert all(e.uncertainty is not None for e in evidence)
    tensor = next(e for e in evidence if e.property == "elastic_stiffness_tensor")
    assert np.allclose(tensor.value, cubic_c(250, 150, 100))
    assert not any(e.property in {"yield_strength", "fracture_toughness"} for e in evidence)
