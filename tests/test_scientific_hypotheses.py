"""Shared hypothesis validation distinguishes proposals, MLIP, and DFT evidence."""

import pytest

from matsim_agents.backends.llm.hypotheses import validate_hypotheses


def _claim(status, evidence_id):
    return {
        "id": "h1",
        "claim": "The sampled structures differ in energy.",
        "prediction": "Identical-method energies will differ by more than numerical precision.",
        "status": status,
        "reason": "Support is limited to the recorded method and sampled structures.",
        "evidence_ids": [evidence_id],
    }


@pytest.mark.parametrize("status", ["supported_by_dft", "challenged_by_dft"])
def test_dft_claim_requires_explicit_measured_dft_evidence(status):
    with pytest.raises(ValueError, match="requires measured DFT evidence"):
        validate_hypotheses([_claim(status, "mlip")], {"mlip"}, [], mlip_evidence={"mlip"})
    assert validate_hypotheses(
        [_claim(status, "dft")],
        {"dft"},
        [],
        dft_evidence={"dft"},
    )


def test_operational_or_peer_feedback_is_not_mlip_support():
    with pytest.raises(ValueError, match="requires outcome evidence"):
        validate_hypotheses(
            [_claim("supported_by_mlip", "generation-shortfall")],
            {"generation-shortfall"},
            [],
        )
    with pytest.raises(ValueError, match="requires outcome evidence"):
        validate_hypotheses([_claim("supported_by_mlip", "peer")], {"peer"}, [])


def test_retired_hypothesis_still_has_to_be_recorded_in_next_revision():
    previous = [_claim("retired", "objective")]
    assert validate_hypotheses(previous, {"objective"}, previous)
    changed_id = [{**_claim("proposed", "objective"), "id": "replacement"}]
    with pytest.raises(ValueError, match="omit previous IDs"):
        validate_hypotheses(changed_id, {"objective"}, previous)
