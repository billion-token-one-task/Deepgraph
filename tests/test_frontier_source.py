from unittest.mock import patch

import pytest

from meta_harness.frontier import evaluate_frontier
from meta_harness.frontier_builder import FrontierBuildError
from meta_harness.frontier_source import (
    EvidenceGraphFrontierSource,
    FrontierAssessment,
)


def _assessment(**overrides):
    values = {
        "problem_status": "open",
        "contribution_delta": {"mechanism": "tests a distinct failure mode"},
        "why_not_obsolete": "The linked benchmark omits the proposed mechanism.",
        "minimum_falsification_experiment": {"metric": "accuracy", "stop": "<=0"},
        "evaluator": "frontier-reviewer",
        "provider": "review-provider",
        "model": "review-model",
        "prompt_version": "frontier-v1",
        "coverage_start": "2025-01-01",
        "coverage_end": "2026-07-31",
    }
    values.update(overrides)
    return FrontierAssessment(**values)


def _problem(status="open"):
    return {
        "id": 7,
        "agenda_id": 3,
        "problem_statement": "A scoped problem",
        "source_signal_ref": "signal:abc",
        "node_ids": '["ml.eval"]',
        "paper_ids": '["2601.00001", "2602.00002"]',
        "status": status,
        "ruled_out_approaches": "[]",
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-07-01T00:00:00+00:00",
    }


def _paper_rows():
    return [
        {
            "id": "2602.00002",
            "title": "Recent benchmark",
            "published_date": "2026-02-01",
            "status": "reasoned",
            "result_count": 2,
            "sota_result_count": 1,
            "grounded_claim_count": 2,
        },
        {
            "id": "2601.00001",
            "title": "Nearest method",
            "published_date": "2026-01-01",
            "status": "reasoned",
            "result_count": 1,
            "sota_result_count": 0,
            "grounded_claim_count": 1,
        },
    ]


def _benchmark_rows():
    return [
        {
            "id": 11,
            "paper_id": "2602.00002",
            "method_name": "baseline",
            "dataset_name": "heldout",
            "metric_name": "accuracy",
            "metric_value": 0.7,
            "metric_unit": None,
            "is_sota": 1,
            "evidence_location": "table:2",
            "grounding_status": "verified",
        }
    ]


@patch("meta_harness.frontier_source.db.fetchall")
@patch("meta_harness.frontier_source.db.fetchone")
def test_live_frontier_uses_scoped_graph_and_content_addressed_query(
    fetchone, fetchall
):
    fetchone.return_value = _problem()
    fetchall.side_effect = [_paper_rows(), _benchmark_rows(), []]

    packet = EvidenceGraphFrontierSource().build(
        agenda_id=3,
        research_problem_id=7,
        assessment=_assessment(),
        retrieved_at="2026-07-31T00:00:00+00:00",
    )

    assert evaluate_frontier(packet).allowed is True
    assert packet.coverage["source_indexes"] == [
        "deepgraph.postgresql.evidence_graph"
    ]
    assert packet.coverage["query_refs"][0].startswith(
        "deepgraph:evidence-graph:sha256:"
    )
    assert packet.strongest_recent_work[0]["paper_id"] == "2602.00002"
    assert packet.latest_benchmarks[0]["metric_value"] == 0.7
    query, params = fetchone.call_args.args
    assert "WHERE id=? AND agenda_id=?" in query
    assert params == (7, 3)


@patch("meta_harness.frontier_source.db.fetchall")
@patch("meta_harness.frontier_source.db.fetchone")
def test_persisted_obsolete_problem_cannot_be_overridden_by_assessment(
    fetchone, fetchall
):
    fetchone.return_value = _problem(status="obsolete")
    fetchall.side_effect = [_paper_rows(), _benchmark_rows(), []]

    packet = EvidenceGraphFrontierSource().build(
        agenda_id=3,
        research_problem_id=7,
        assessment=_assessment(problem_status="open"),
        retrieved_at="2026-07-31T00:00:00+00:00",
    )

    decision = evaluate_frontier(packet)
    assert packet.problem_status == "obsolete"
    assert decision.allowed is False
    assert "frontier_obsolete" in decision.reason_codes


@patch("meta_harness.frontier_source.db.fetchone", return_value=None)
def test_cross_agenda_problem_is_rejected(_fetchone):
    with pytest.raises(FrontierBuildError, match="agenda-scoped"):
        EvidenceGraphFrontierSource().build(
            agenda_id=3,
            research_problem_id=7,
            assessment=_assessment(),
        )
