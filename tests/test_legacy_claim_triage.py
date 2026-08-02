from unittest import mock

import pytest

from agents.legacy_claim_triage import (
    build_initial_harness_problem,
    seed_initial_harness_problem,
    select_harness_evidence,
)


def _rows():
    return [
        {"id": 11, "paper_id": "p-1", "grounding_score": 0.95},
        {"id": 12, "paper_id": "p-1", "grounding_score": 0.9},
        {"id": 13, "paper_id": "p-2", "grounding_score": 0.8},
    ]


def test_triage_query_is_bounded_and_returns_no_claim_text():
    with mock.patch(
        "agents.legacy_claim_triage.db.fetchall", return_value=_rows()
    ) as fetchall:
        rows = select_harness_evidence(max_claims=9999)
    query, params = fetchall.call_args.args
    assert "claim_text" in query
    assert "LIMIT ?" in query
    assert params[-1] == 500
    assert rows == _rows()


def test_initial_problem_is_new_and_keeps_explicit_evidence_references():
    problem = build_initial_harness_problem(_rows(), agenda_id=5, max_papers=2)
    assert problem["paper_ids"] == ["p-1", "p-2"]
    assert problem["source_signal_ref"]["table"] == "claims"
    assert problem["source_signal_ref"]["evidence_claim_ids"] == [11, 12, 13]
    assert "self-improving LLM harnesses" in problem["problem_statement"]


def test_seed_rejects_inactive_agenda_without_reading_or_writing_claims():
    repo = mock.Mock()
    repo.get.return_value = mock.Mock(status="paused_manual", is_active=False)
    with pytest.raises(ValueError, match="active agenda"):
        seed_initial_harness_problem(agenda_id=5, repository=repo)
