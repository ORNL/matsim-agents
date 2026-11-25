import math

import pytest
from pydantic import ValidationError

from matsim_agents.campaign.selection import Constraint, Correlation
from matsim_agents.execution.contracts import PROPERTY_UNITS, EvidenceLevel
from matsim_agents.execution.properties import (
    PropertyEvidence,
    PropertyQualification,
    PropertyStatus,
)
from matsim_agents.fusion import synthetic
from matsim_agents.fusion.policy import builtin_policy
from matsim_agents.fusion.scenario import FusionEvidence
from matsim_agents.fusion.scoring import (
    BenchmarkSet,
    ConstraintStatus,
    Dominance,
    RankingStatus,
    render_report,
    score_candidates,
)

POLICY = builtin_policy("fusion_pfc_initial_v1")


def _score(cands=None, **kw):
    return score_candidates(
        POLICY,
        synthetic.BENCHMARKS,
        synthetic.SCENARIO,
        cands if cands is not None else synthetic.candidates(),
        **kw,
    )


def _ab_only():
    c = synthetic.candidates()
    return {"A": c["A"], "B": c["B"]}


def test_weighted_sum_uses_fixed_benchmarks():
    r = _score(_ab_only())
    a = r.by_id("A")
    d = {o.objective: o.desirability for o in a.objectives}
    assert d["thermal_conductivity"] == pytest.approx((120 - 20) / 150)
    assert a.score == pytest.approx(math.fsum(POLICY.weights[k] * v for k, v in d.items()))
    # adding a candidate does not change A's desirabilities (no min/max normalization)
    r2 = _score()
    assert r2.by_id("A").score == pytest.approx(a.score)


def test_missing_objective_is_not_zero_filled_or_renormalized():
    r = _score()
    c = r.by_id("C")
    assert c.missing_objectives == ("low_activation_shutdown_dose",)
    assert c.score is None and c.rank is None
    assert c.ranking_status == RankingStatus.INCOMPLETE
    assert c.covered_weight == pytest.approx(1 - POLICY.weights["low_activation_shutdown_dose"])
    assert "C" not in r.ranking
    assert not r.pareto_authoritative
    assert any("partial" in n for n in r.notes)


def test_complete_set_has_authoritative_pareto_and_established_dominance():
    r = _score(_ab_only())
    assert r.pareto_authoritative
    assert r.pareto_front == ("A",)
    assert r.ranking == ("A", "B")
    pair = next(p for p in r.dominance if p.winner == "A")
    assert pair.loser == "B"
    assert pair.dominance in {Dominance.ESTABLISHED, Dominance.CONDITIONAL}
    assert r.by_id("A").ranking_status == RankingStatus.COMPLETE_PROVISIONAL


def test_dominance_undetermined_when_objectives_missing():
    r = _score()
    pairs = {(p.winner, p.loser): p.dominance for p in r.dominance}
    assert pairs[("C", "B")] == Dominance.UNDETERMINED


def test_conditional_dominance_when_intervals_overlap():
    cands = _ab_only()
    wide = []
    for item in cands["B"]:
        ev = item.evidence
        upd = ev.model_copy(update={"uncertainty": abs(ev.value) * 2})
        wide.append(FusionEvidence(evidence=upd, context=item.context))
    r = _score({"A": cands["A"], "B": wide})
    assert {(p.winner, p.loser): p.dominance for p in r.dominance}[("A", "B")] == (
        Dominance.CONDITIONAL
    )


def test_constraints_feasible_violated_unknown():
    r = _score(
        constraints=[
            Constraint(property="yield_strength", minimum=800.0),
            Constraint(property="creep_rate", maximum=1e-8),
        ]
    )
    assert r.by_id("B").feasibility == ConstraintStatus.VIOLATED
    assert r.by_id("B").ranking_status == RankingStatus.INFEASIBLE
    assert r.by_id("A").feasibility == ConstraintStatus.UNKNOWN
    assert "B" not in r.ranking


def _modified(items, prop, conditions=None, context=None):
    out = []
    for item in items:
        ev, ctx = item.evidence, item.context
        if ev.property == prop:
            if conditions:
                ev = ev.model_copy(
                    update={"conditions": ev.conditions.model_copy(update=conditions)}
                )
            if context:
                ctx = ctx.model_copy(update=context)
        out.append(FusionEvidence(evidence=ev, context=ctx))
    return out


def test_constraint_gate_uses_property_conditions():
    cands = _ab_only()
    # Activation records carry no temperature; the gate must not demand one.
    no_t = _modified(cands["A"], "shutdown_dose_rate", conditions={"temperature_K": None})
    # Wrong plasma energy must not decide feasibility.
    bad_e = _modified(cands["B"], "sputtering_yield", context={"plasma_ion_energy_eV": 1000.0})
    r = _score(
        {"A": no_t, "B": bad_e},
        constraints=[
            Constraint(property="shutdown_dose_rate", maximum=0.0),
            Constraint(property="sputtering_yield", maximum=0.0),
        ],
    )
    a, b = r.by_id("A"), r.by_id("B")
    assert a.feasibility == ConstraintStatus.VIOLATED
    sputter = next(c for c in b.constraints if c.property == "sputtering_yield")
    assert sputter.status == ConstraintStatus.UNKNOWN


def test_incomparable_evidence_is_excluded():
    cands = _ab_only()
    hot = []
    for item in cands["A"]:
        ev = item.evidence
        if ev.property == "thermal_conductivity_total":
            cond = ev.conditions.model_copy(update={"temperature_K": 300.0})
            ev = ev.model_copy(update={"conditions": cond})
        hot.append(FusionEvidence(evidence=ev, context=item.context))
    r = _score({"A": hot, "B": cands["B"]})
    tc = r.by_id("A").objective("thermal_conductivity")
    assert tc.desirability is None
    assert any("temperature" in x for x in tc.observables[0].excluded)


def test_proxy_derived_evidence_is_excluded():
    cands = _ab_only()
    proxied = []
    for item in cands["A"]:
        ev = item.evidence
        if ev.property == "defect_recovery_fraction":
            ev = ev.model_copy(update={"derived_from": ["dpa"]})
        proxied.append(FusionEvidence(evidence=ev, context=item.context))
    r = _score({"A": proxied, "B": cands["B"]})
    assert r.by_id("A").objective("radiation_damage_recovery").desirability is None


def test_unsupported_status_never_scored():
    cands = _ab_only()
    rec = PropertyEvidence(
        structure_id="x",
        composition="X",
        property="sputtering_yield",
        status=PropertyStatus.UNSUPPORTED,
        unit=PROPERTY_UNITS["sputtering_yield"],
        conditions={"temperature_K": 1200.0},
        reason="no BCA model available",
    )
    a = [i for i in cands["A"] if i.evidence.property != "sputtering_yield"]
    a.append(FusionEvidence(evidence=rec, context=cands["A"][0].context))
    r = _score({"A": a, "B": cands["B"]})
    o = r.by_id("A").objective("plasma_erosion_resistance")
    assert o.desirability is None and "unsupported" in o.observables[0].reason


def test_correlation_warning_and_weights_untouched():
    r = _score(
        _ab_only(),
        correlations=[Correlation(a="yield_strength", b="thermal_shock_damage_threshold", rho=0.9)],
    )
    assert any("correlated" in w for w in r.correlation_warnings)
    assert r.by_id("A").objective("radiation_damage_recovery").weight == 0.25


def test_weight_sensitivity_reported():
    r = _score(_ab_only())
    assert len(r.sensitivity) == 16
    assert all(s.top_candidate == "A" for s in r.sensitivity)


def test_benchmark_validation():
    data = synthetic.BENCHMARKS.model_dump()
    bad = dict(data)
    bad["objectives"] = dict(data["objectives"])
    bad["objectives"]["thermal_shock_fatigue"] = {
        "observables": {
            "thermal_conductivity_total": {"worst": 20.0, "target": 170.0, "source": "s"}
        }
    }
    with pytest.raises(ValidationError, match="required"):
        BenchmarkSet.model_validate(bad)
    bad["objectives"]["thermal_shock_fatigue"] = data["objectives"]["thermal_shock_fatigue"]
    bad["objectives"]["plasma_erosion_resistance"] = {
        "observables": {"sputtering_yield": {"worst": 1e-4, "target": 1e-2, "source": "s"}}
    }
    with pytest.raises(ValidationError, match="contradict direction"):
        BenchmarkSet.model_validate(bad)


def test_report_and_digest_deterministic():
    r1, r2 = _score(), _score()
    assert r1.digest() == r2.digest()
    text = render_report(r1)
    assert "SYNTHETIC" in text and "PARTIAL" in text and "MISSING" in text


def _qualified(items):
    upd = {
        "status": PropertyStatus.CALCULATED,
        "evidence_level": EvidenceLevel.CONVERGED_DFT,
        "converged": True,
        "qualification": PropertyQualification.NUMERICALLY_VALIDATED,
    }
    return [
        FusionEvidence(evidence=i.evidence.model_copy(update=upd), context=i.context) for i in items
    ]


def test_constraint_result_cites_spec_and_evidence():
    c = Constraint(property="yield_strength", minimum=100.0, temperature_K=1200.0)
    r = _score(_ab_only(), constraints=[c])
    cr = r.by_id("A").constraints[0]
    assert (cr.minimum, cr.maximum, cr.temperature_K) == (100.0, None, 1200.0)
    assert cr.property_id and cr.evidence_status and cr.qualification
    assert cr.context is not None
    looser = _score(_ab_only(), constraints=[c.model_copy(update={"minimum": 50.0})])
    assert looser.digest() != r.digest()


def test_qualified_ranking_needs_reliable_constraint_evidence():
    cands = {k: _qualified(v) for k, v in _ab_only().items()}
    assert _score(cands).by_id("A").ranking_status == RankingStatus.COMPLETE_QUALIFIED
    base = cands["A"][0]
    creep = base.evidence.model_copy(
        update={
            "property": "creep_rate",
            "unit": PROPERTY_UNITS["creep_rate"],
            "value": 1e-9,
            "uncertainty": 1e-10,
            "qualification": PropertyQualification.UNVALIDATED,
        }
    )
    weak = [*cands["A"], FusionEvidence(evidence=creep, context=base.context)]
    r = _score(
        {"A": weak, "B": cands["B"]}, constraints=[Constraint(property="creep_rate", maximum=1.0)]
    )
    a = r.by_id("A")
    assert a.feasibility == ConstraintStatus.FEASIBLE
    assert a.ranking_status == RankingStatus.COMPLETE_PROVISIONAL


def test_result_records_full_scenario():
    r = _score(_ab_only())
    assert r.scenario == synthetic.SCENARIO
    other = synthetic.SCENARIO.model_copy(update={"description": "changed"})
    assert score_candidates(POLICY, synthetic.BENCHMARKS, other, _ab_only()).digest() != r.digest()
