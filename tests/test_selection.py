"""MP6: deterministic multi-objective selection. All values are synthetic."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from matsim_agents.campaign.selection import (
    CandidateEvidence,
    Constraint,
    Correlation,
    Objective,
    SelectionSpec,
    SelectionStatus,
    select,
)
from matsim_agents.execution.contracts import PROPERTY_UNITS, EvidenceLevel
from matsim_agents.execution.properties import (
    PropertyConditions,
    PropertyEvidence,
    PropertyQualification,
    PropertyStatus,
)

T = 1000.0
KAPPA = "thermal_conductivity_total"
SIGMA = "electrical_conductivity"
BULK = "bulk_modulus"


def ev(prop, value, unc=0.0, *, status=PropertyStatus.CALCULATED, cid="A", **kw):
    cond = PropertyConditions(temperature_K=T if prop in {KAPPA, SIGMA} else None)
    base = dict(
        structure_id=f"{cid}-s",
        composition=cid,
        property=prop,
        unit=PROPERTY_UNITS[prop],
        conditions=cond,
        status=status,
    )
    if status == PropertyStatus.CALCULATED:
        base.update(
            value=value,
            uncertainty=unc,
            evidence_ids=[f"{cid}-{prop}"],
            evidence_level=EvidenceLevel.CONVERGED_DFT,
            converged=True,
            qualification=PropertyQualification.NUMERICALLY_VALIDATED,
        )
    elif status in {PropertyStatus.SURROGATE_PREDICTED, PropertyStatus.MEASURED}:
        base.update(
            value=value,
            uncertainty=unc,
            evidence_ids=[f"{cid}-{prop}-sur"],
            evidence_level=EvidenceLevel.MLIP_PREDICTION,
        )
    elif status in {PropertyStatus.MISSING, PropertyStatus.UNSUPPORTED}:
        base.update(reason="no calculation")
    base.update(kw)
    return PropertyEvidence(**base)


def cand(cid, kappa, bulk, sigma=None, unc=0.0, **kw):
    rows = [ev(KAPPA, kappa, unc, cid=cid, **kw), ev(BULK, bulk, unc, cid=cid, **kw)]
    if sigma is not None:
        rows.append(ev(SIGMA, sigma, unc, cid=cid, **kw))
    return CandidateEvidence(candidate_id=cid, evidence=rows)


def spec(**kw):
    base = dict(
        objectives=[
            Objective(property=KAPPA, direction="maximize", temperature_K=T),
            Objective(property=BULK, direction="maximize"),
        ],
        seed=0,
        n_samples=400,
    )
    base.update(kw)
    return SelectionSpec(**base)


def test_pareto_dominance_and_tradeoffs():
    # A: high kappa/low bulk. B: low kappa/high bulk. C: dominated by A. All synthetic.
    result = select(
        spec(),
        [cand("A", 30, 150), cand("B", 12, 250), cand("C", 20, 140)],
    )
    assert result.pareto_front == ["A", "B"]
    c = result.by_id("C")
    assert c.pareto == "dominated" and c.dominated_by == ["A"]
    assert result.by_id("A").best_in == [KAPPA] and result.by_id("C").worst_in == [BULK]
    assert result.by_id("B").best_in == [BULK]
    assert {r.status for r in result.candidates} == {SelectionStatus.QUALIFIED}
    assert result.by_id("A").desirability is None  # Pareto is the default; no scalar score


def test_missing_evidence_is_incomplete_never_zero():
    a = cand("A", 30, 150)
    b = CandidateEvidence(
        candidate_id="B",
        evidence=[ev(KAPPA, 40, cid="B"), ev(BULK, None, cid="B", status=PropertyStatus.MISSING)],
    )
    result = select(spec(), [a, b])
    sb = result.by_id("B")
    assert sb.status == SelectionStatus.INCOMPLETE and BULK in sb.missing
    assert sb.pareto == "not_ranked" and BULK not in sb.values
    assert result.pareto_front == ["A"]  # B's 40 is not allowed to win on a half-known record
    assert result.by_id("A").pareto == "nondominated"


def test_hard_constraint_makes_infeasible_and_excludes_from_front():
    s = spec(constraints=[Constraint(property=BULK, minimum=100.0)])
    result = select(s, [cand("A", 30, 90), cand("B", 12, 250)])
    assert result.by_id("A").status == SelectionStatus.INFEASIBLE
    assert "bulk_modulus" in result.by_id("A").violated[0]
    assert result.pareto_front == ["B"]


def test_constraint_requires_a_user_bound():
    with pytest.raises(ValidationError):
        Constraint(property=BULK)
    with pytest.raises(ValidationError):
        Objective(property=KAPPA, direction="maximize", worst=1.0)
    with pytest.raises(ValidationError):
        Objective(property=KAPPA, direction="maximize", worst=5.0, target=1.0)
    with pytest.raises(ValidationError):
        Objective(property="elastic_stiffness_tensor", direction="maximize")


def test_surrogate_values_never_qualify():
    a = CandidateEvidence(
        candidate_id="A",
        evidence=[
            ev(KAPPA, 30, 2.0, cid="A", status=PropertyStatus.SURROGATE_PREDICTED),
            ev(BULK, 150, 2.0, cid="A"),
        ],
    )
    result = select(spec(), [a])
    sa = result.by_id("A")
    assert sa.status == SelectionStatus.INSUFFICIENT_EVIDENCE
    assert sa.values[KAPPA].reliable is False and "surrogate_predicted" in sa.reasons[0]


def test_unvalidated_calculation_is_insufficient_when_validation_is_required():
    unvalidated = ev(KAPPA, 30, 1.0, qualification=PropertyQualification.UNVALIDATED, cid="A")
    a = CandidateEvidence(candidate_id="A", evidence=[unvalidated, ev(BULK, 150, 1.0, cid="A")])
    assert select(spec(), [a]).by_id("A").status == SelectionStatus.INSUFFICIENT_EVIDENCE
    relaxed = select(spec(min_qualification=PropertyQualification.UNVALIDATED), [a])
    assert relaxed.by_id("A").status == SelectionStatus.QUALIFIED


def test_unconverged_value_is_not_scored():
    bad = ev(
        KAPPA,
        50.0,
        cid="A",
        status=PropertyStatus.UNCONVERGED,
        reason="SCF did not converge",
        converged=False,
    )
    a = CandidateEvidence(candidate_id="A", evidence=[bad, ev(BULK, 150, cid="A")])
    sa = select(spec(), [a]).by_id("A")
    assert sa.status == SelectionStatus.INCOMPLETE and "unconverged" in sa.missing[KAPPA]


def test_best_record_is_preferred_and_alternatives_are_traced():
    measured = ev(
        KAPPA,
        28.0,
        1.0,
        cid="A",
        status=PropertyStatus.MEASURED,
        evidence_level=EvidenceLevel.EXPERIMENTAL,
        converged=None,
        qualification=PropertyQualification.NUMERICALLY_VALIDATED,
    )
    calc = ev(KAPPA, 30.0, 1.0, cid="A")
    a = CandidateEvidence(candidate_id="A", evidence=[calc, measured, ev(BULK, 150, cid="A")])
    used = select(spec(), [a]).by_id("A").values[KAPPA]
    assert used.status == PropertyStatus.MEASURED and used.value == 28.0
    assert used.alternatives == [calc.property_id()]


def test_uncertainty_gives_constraint_probability_and_pareto_robustness():
    s = spec(constraints=[Constraint(property=BULK, minimum=100.0)], n_samples=2000)
    near = cand("A", 30, 101, unc=0.0)  # bulk sits on the edge with real uncertainty
    near.evidence[1] = ev(BULK, 101.0, 10.0, cid="A")
    sure = cand("B", 12, 250)
    result = select(s, [near, sure])
    pa = result.by_id("A").constraint_probability
    assert 0.4 < pa < 0.65 and result.by_id("B").constraint_probability == 1.0
    assert result.by_id("A").pareto_robustness < 0.65
    assert result.by_id("B").pareto_robustness > 0.95


def test_constraint_confidence_demotes_a_marginal_candidate():
    s = spec(
        constraints=[Constraint(property=BULK, minimum=100.0)],
        min_constraint_confidence=0.9,
        n_samples=2000,
    )
    a = cand("A", 30, 101)
    a.evidence[1] = ev(BULK, 101.0, 10.0, cid="A")
    sa = select(s, [a, cand("B", 12, 250)]).by_id("A")
    assert sa.status == SelectionStatus.INSUFFICIENT_EVIDENCE
    assert "constraint probability" in sa.reasons[-1]


def test_weighted_desirability_is_optional_and_user_bounded():
    objectives = [
        Objective(
            property=KAPPA, direction="maximize", temperature_K=T, worst=5, target=35, weight=2
        ),
        Objective(property=BULK, direction="maximize", worst=100, target=300),
    ]
    result = select(
        spec(objectives=objectives, weighted_desirability=True),
        [cand("A", 30, 150), cand("B", 12, 250)],
    )
    a, b = result.by_id("A"), result.by_id("B")
    assert a.desirability is not None and 0 < a.desirability < 1
    assert {a.desirability_rank, b.desirability_rank} == {1, 2}
    assert a.probability_rank_1 is not None and sum(a.rank_distribution.values()) == 1
    with pytest.raises(ValidationError):
        spec(weighted_desirability=True)  # objectives lack worst/target


def test_correlated_conductivities_share_one_weight():
    objectives = [
        Objective(property=KAPPA, direction="maximize", temperature_K=T, worst=5, target=35),
        Objective(property=SIGMA, direction="maximize", temperature_K=T, worst=1e6, target=2e7),
        Objective(property=BULK, direction="maximize", worst=100, target=300),
    ]
    s = spec(
        objectives=objectives,
        weighted_desirability=True,
        correlations=[Correlation(a=KAPPA, b=SIGMA, rho=0.97)],
    )
    result = select(s, [cand("A", 30, 150, 1.5e7), cand("B", 12, 250, 5e6)])
    assert any("counted twice" in n for n in result.weight_notes)
    assert any(KAPPA in n and SIGMA in n for n in result.correlation_notes)


def test_derived_property_correlation_is_inferred():
    sigma = ev(SIGMA, 1.5e7, 1.0e6, cid="A", derived_from=[KAPPA])
    kappa = ev(KAPPA, 30, 3.0, cid="A")
    a = CandidateEvidence(candidate_id="A", evidence=[kappa, sigma, ev(BULK, 150, cid="A")])
    s = spec(
        objectives=[
            Objective(property=KAPPA, direction="maximize", temperature_K=T),
            Objective(property=SIGMA, direction="maximize", temperature_K=T),
        ]
    )
    result = select(s, [a])
    assert any("rho=+1.00" in n for n in result.correlation_notes)


def test_scores_are_deterministic_and_do_not_mutate_evidence():
    cands = [cand("A", 30, 150, unc=1.0), cand("B", 12, 250, unc=1.0)]
    before = [e.model_dump_json() for c in cands for e in c.evidence]
    r1, r2 = select(spec(), cands), select(spec(), cands)
    assert r1.score_digest == r2.score_digest and r1.score_digest.startswith("sel-")
    assert before == [e.model_dump_json() for c in cands for e in c.evidence]
    assert select(spec(seed=1), cands).score_digest != r1.score_digest


def test_duplicate_candidate_ids_are_rejected():
    with pytest.raises(ValueError, match="unique"):
        select(spec(), [cand("A", 1, 1), cand("A", 2, 2)])
