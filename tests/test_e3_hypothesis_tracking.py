"""Scientific hypothesis history for E3-MLIP acquisitions."""

import json

import pytest

from benchmarks.e3 import hypothesis_tracking, production


def _hypothesis(evidence_id="campaign-design", *, status="proposed", claim="Claim"):
    return {
        "id": "material-1",
        "claim": claim,
        "prediction": "The sampled structures will exhibit a measurable energy difference.",
        "status": status,
        "reason": "The evidence does not establish a thermodynamic result.",
        "evidence_ids": [evidence_id],
    }


class _Chat:
    model_id = "test-model"

    def __init__(self, response):
        self.response = response
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return type("Response", (), {"content": json.dumps(self.response)})()


def _write_proposal(path):
    hypotheses = [_hypothesis()]
    evidence_ids = ["campaign-design"]
    production._atomic_json(
        path,
        {
            "status": "accepted",
            "prompt": f"Valid evidence IDs: {json.dumps(evidence_ids)}\n",
            "valid_evidence_ids": evidence_ids,
            "previous_hypotheses": [],
            "raw_response": json.dumps({"scientific_hypotheses": hypotheses}),
            "scientific_hypotheses": hypotheses,
        },
    )


def test_tracking_acquisition_persists_hypotheses_and_previous_context(tmp_path):
    actions = production._action_space("phosphorene")
    action = next(item for item in actions if production._finite_capacity_group(item) is None)
    previous = [_hypothesis()]
    chat = _Chat(
        {
            "allocations": [{"action_id": action["action_id"], "count": 1}],
            "cited_evidence_ids": ["campaign-design"],
            "rationale": "Sample an allowed structure.",
            "scientific_hypotheses": [_hypothesis(status="refined", claim="Revised claim")],
        }
    )
    path = tmp_path / "proposal.json"
    production._llm_allocation(
        chat,
        "phosphorene",
        actions,
        [],
        1,
        {},
        evidence_path=path,
        context={"hypothesis_tracking": True, "previous_hypotheses": previous},
    )
    saved = json.loads(path.read_text())
    assert saved["status"] == "accepted"
    assert saved["scientific_hypotheses"][0]["claim"] == "Revised claim"
    assert saved["previous_hypotheses"] == previous
    assert "Previous scientific hypotheses:" in chat.prompts[0]
    assert "scientific_hypotheses" in chat.prompts[0]


def test_tracking_rejects_missing_prior_hypotheses_and_invalid_citations(tmp_path):
    actions = production._action_space("phosphorene")
    action = next(item for item in actions if production._finite_capacity_group(item) is None)
    response = {
        "allocations": [{"action_id": action["action_id"], "count": 1}],
        "cited_evidence_ids": ["campaign-design"],
        "rationale": "Sample an allowed structure.",
        "scientific_hypotheses": [{**_hypothesis("fabricated"), "id": "new"}],
    }
    path = tmp_path / "proposal.json"
    with pytest.raises(ValueError, match="cite supplied evidence"):
        production._llm_allocation(
            _Chat(response),
            "phosphorene",
            actions,
            [],
            1,
            {},
            evidence_path=path,
            context={"hypothesis_tracking": True, "previous_hypotheses": [_hypothesis()]},
        )
    assert json.loads(path.read_text())["status"] == "rejected"
    with pytest.raises(ValueError, match="omit previous"):
        hypothesis_tracking.validate_hypotheses(
            [{**_hypothesis(), "id": "new"}], {"campaign-design"}, [_hypothesis()]
        )


def test_completed_round_assessment_is_recorded_once_and_used_next(tmp_path):
    stream = tmp_path / "phosphorene" / "llm_guided_adaptation" / "seed-20261113"
    round_dir = stream / "round-01"
    proposal = round_dir / "acquisition" / "proposal-001.json"
    _write_proposal(proposal)
    production._atomic_json(
        round_dir / "round-summary.json",
        {
            "status": "complete",
            "action_outcomes": [{"evidence_id": "round-1", "converged": 2, "attempted": 3}],
        },
    )
    updated = _hypothesis("round-1", status="refined", claim="Updated claim")
    chat = _Chat({"scientific_hypotheses": [updated]})
    production._round_hypothesis_assessment(stream, "phosphorene", 20261113, 1, chat)
    saved = json.loads((round_dir / "hypothesis-assessment.json").read_text())
    assert saved["status"] == "accepted"
    assert saved["scientific_hypotheses"] == [updated]
    assert '"round-1"' in saved["prompt"]
    assert production._previous_hypotheses(stream, 2) == [updated]
    production._round_hypothesis_assessment(stream, "phosphorene", 20261113, 1, chat)
    assert len(chat.prompts) == 1
    with pytest.raises(RuntimeError, match="different scientific stage"):
        production._round_hypothesis_assessment(
            stream,
            "phosphorene",
            20261113,
            1,
            chat,
            assessment_stage="post_dft_iteration",
        )
    assert len(chat.prompts) == 1


def test_rejected_assessment_keeps_raw_response_and_never_retries(tmp_path):
    stream = tmp_path / "phosphorene" / "llm_guided_adaptation" / "seed-20261113"
    round_dir = stream / "round-01"
    _write_proposal(round_dir / "acquisition" / "proposal-001.json")
    production._atomic_json(
        round_dir / "round-summary.json",
        {
            "status": "complete",
            "action_outcomes": [],
        },
    )
    chat = _Chat({"scientific_hypotheses": [_hypothesis("invented")]})
    with pytest.raises(ValueError, match="cite supplied evidence"):
        production._round_hypothesis_assessment(stream, "phosphorene", 20261113, 1, chat)
    saved = json.loads((round_dir / "hypothesis-assessment.json").read_text())
    assert saved["status"] == "rejected" and saved["raw_response"]
    with pytest.raises(RuntimeError, match="already attempted"):
        production._round_hypothesis_assessment(stream, "phosphorene", 20261113, 1, chat)
    assert len(chat.prompts) == 1


def test_three_round_history_passes_each_assessment_forward_and_records_final(tmp_path):
    material = "phosphorene"
    seed = 20261113
    stream = tmp_path / material / "llm_guided_adaptation" / f"seed-{seed}"
    actions = production._action_space(material)
    action = next(item for item in actions if production._finite_capacity_group(item) is None)
    for round_index in range(1, 4):
        round_dir = stream / f"round-{round_index:02d}"
        previous = production._previous_hypotheses(stream, round_index)
        if round_index > 1:
            assert previous[0]["claim"] == f"After round {round_index - 1}"
        proposal = _Chat(
            {
                "allocations": [{"action_id": action["action_id"], "count": 1}],
                "cited_evidence_ids": ["campaign-design"],
                "rationale": "Sample an allowed structure.",
                "scientific_hypotheses": [
                    _hypothesis(status="unresolved", claim=f"Before round {round_index}")
                ],
            }
        )
        production._llm_allocation(
            proposal,
            material,
            actions,
            [],
            1,
            {},
            evidence_path=round_dir / "acquisition" / "proposal-001.json",
            context={"previous_hypotheses": previous},
        )
        evidence_id = f"round-{round_index}"
        production._atomic_json(
            round_dir / "round-summary.json",
            {
                "status": "complete",
                "action_outcomes": [{"evidence_id": evidence_id, "attempted": 1, "converged": 1}],
            },
        )
        assessment = _Chat(
            {
                "scientific_hypotheses": [
                    _hypothesis(evidence_id, status="refined", claim=f"After round {round_index}")
                ]
            }
        )
        production._round_hypothesis_assessment(stream, material, seed, round_index, assessment)
        saved = json.loads((round_dir / "hypothesis-assessment.json").read_text())
        assert saved["after_round"] == round_index
        assert saved["scientific_hypotheses"][0]["claim"] == f"After round {round_index}"
        assert evidence_id in saved["prompt"]
    assert production._previous_hypotheses(stream, 4)[0]["claim"] == "After round 3"


def test_manifest_requires_hypothesis_protocol():
    manifest = production.build_campaign_manifest()
    manifest.pop("hypothesis_tracking")
    with pytest.raises(ValueError, match="hypothesis tracking"):
        production.validate_campaign_manifest(manifest)


def test_mlip_support_requires_observation():
    with pytest.raises(ValueError, match="requires outcome evidence"):
        hypothesis_tracking.validate_hypotheses(
            [_hypothesis(status="supported_by_mlip")], {"campaign-design"}, []
        )


def test_saved_proposal_must_match_prior_assessment_and_original_response(tmp_path):
    actions = production._action_space("phosphorene")
    action = next(item for item in actions if production._finite_capacity_group(item) is None)
    previous = [_hypothesis()]
    updated = _hypothesis(status="refined", claim="Revised claim")
    chat = _Chat(
        {
            "allocations": [{"action_id": action["action_id"], "count": 1}],
            "cited_evidence_ids": ["campaign-design"],
            "rationale": "Sample an allowed structure.",
            "scientific_hypotheses": [updated],
        }
    )
    path = tmp_path / "proposal.json"
    production._llm_allocation(
        chat,
        "phosphorene",
        actions,
        [],
        1,
        {},
        evidence_path=path,
        context={"previous_hypotheses": previous},
    )
    production._validate_saved_hypothesis_proposal(path, previous)
    with pytest.raises(RuntimeError, match="previous hypothesis assessment"):
        production._validate_saved_hypothesis_proposal(path, [])
    saved = json.loads(path.read_text())
    saved["scientific_hypotheses"] = [_hypothesis(claim="Altered claim")]
    production._atomic_json(path, saved)
    with pytest.raises(ValueError, match="differs from the original LLM response"):
        production._validate_saved_hypothesis_proposal(path, previous)


def test_saved_recovery_proposal_uses_exact_evidence_supplied_to_llm(tmp_path):
    actions = production._action_space("phosphorene")
    action = next(item for item in actions if production._finite_capacity_group(item) is None)
    evidence_id = "generation-shortfall-example"
    response = {
        "allocations": [{"action_id": action["action_id"], "count": 1}],
        "cited_evidence_ids": [evidence_id],
        "rationale": "Sample an allowed structure.",
        "scientific_hypotheses": [_hypothesis(evidence_id)],
    }
    path = tmp_path / "proposal.json"
    production._llm_allocation(
        _Chat(response),
        "phosphorene",
        actions,
        [{"evidence_id": evidence_id}],
        1,
        {},
        evidence_path=path,
    )
    production._validate_saved_hypothesis_proposal(path, [])


def test_assessment_rejects_modified_saved_hypotheses(tmp_path):
    stream = tmp_path / "phosphorene" / "llm_guided_adaptation" / "seed-20261113"
    round_dir = stream / "round-01"
    _write_proposal(round_dir / "acquisition" / "proposal-001.json")
    production._atomic_json(
        round_dir / "round-summary.json",
        {"status": "complete", "action_outcomes": []},
    )
    chat = _Chat({"scientific_hypotheses": [_hypothesis()]})
    production._round_hypothesis_assessment(stream, "phosphorene", 20261113, 1, chat)
    path = round_dir / "hypothesis-assessment.json"
    record = json.loads(path.read_text())
    record["scientific_hypotheses"] = [_hypothesis(claim="Altered claim")]
    production._atomic_json(path, record)
    with pytest.raises(ValueError, match="differs from the original LLM response"):
        production._round_hypothesis_assessment(stream, "phosphorene", 20261113, 1, chat)
    assert len(chat.prompts) == 1


def test_next_round_rejects_missing_or_modified_prior_proposal(tmp_path):
    stream = tmp_path / "phosphorene" / "llm_guided_adaptation" / "seed-20261113"
    round_dir = stream / "round-01"
    proposal_path = round_dir / "acquisition" / "proposal-001.json"
    _write_proposal(proposal_path)
    production._atomic_json(
        round_dir / "round-summary.json",
        {"status": "complete", "action_outcomes": []},
    )
    production._round_hypothesis_assessment(
        stream,
        "phosphorene",
        20261113,
        1,
        _Chat({"scientific_hypotheses": [_hypothesis()]}),
    )
    proposal = json.loads(proposal_path.read_text())
    proposal["scientific_hypotheses"] = [_hypothesis(claim="Altered claim")]
    production._atomic_json(proposal_path, proposal)
    with pytest.raises(ValueError, match="differs from the original LLM response"):
        production._previous_hypotheses(stream, 2)


def test_prepared_round_does_not_bypass_missing_previous_assessment(tmp_path, monkeypatch):
    manifest_path = tmp_path / "manifest.json"
    production.write_campaign_manifest(manifest_path)
    stream = tmp_path / "runs" / "phosphorene" / "llm_guided_adaptation" / "seed-20261113"
    _write_proposal(stream / "round-01" / "acquisition" / "proposal-001.json")
    production._atomic_json(
        stream / "round-01" / "round-summary.json",
        {
            "status": "complete",
            "action_outcomes": [],
        },
    )
    production._atomic_json(stream / "round-02" / "generation-preflight.json", {})
    monkeypatch.setattr(
        production,
        "_preflight_receipt",
        lambda *args: pytest.fail("must not accept preflight without hypothesis history"),
    )
    with pytest.raises(FileNotFoundError, match="hypothesis-assessment.json"):
        production.prepare_round(manifest_path, tmp_path / "runs", "phosphorene", 2)
