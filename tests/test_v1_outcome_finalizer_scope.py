from types import SimpleNamespace
from unittest import mock

from meta_harness import outcome_finalizer
from scripts import auto_advance


def test_failed_run_candidate_requires_current_job_ownership():
    with mock.patch.object(outcome_finalizer.db, "fetchall", return_value=[]) as fetchall, mock.patch.object(
        outcome_finalizer.db, "commit"
    ):
        assert outcome_finalizer._candidate_rows(1) == []

    query = fetchall.call_args.args[0]
    assert "arj.resource_grant_id=er.resource_grant_id" in query
    assert "arj.experiment_run_id=er.id" in query


def test_mark_closed_scopes_job_update_to_the_same_grant():
    row = {
        "agenda_id": 1,
        "deep_insight_id": 2,
        "experiment_run_id": 3,
        "resource_grant_id": 4,
    }
    with (
        mock.patch.object(outcome_finalizer.db, "execute") as execute,
        mock.patch.object(outcome_finalizer.db, "commit"),
        mock.patch.object(outcome_finalizer, "apply_experiment_finished_deep"),
        mock.patch.object(outcome_finalizer.db, "emit_pipeline_event"),
    ):
        outcome_finalizer._mark_closed(row, outcome_id=5, verdict="inconclusive")

    query, params = execute.call_args.args
    assert "AND resource_grant_id=?" in query
    assert params[-1] == 4


def test_requeue_clears_terminal_run_before_assigning_replacement_grant():
    journal = SimpleNamespace(log=mock.Mock())
    args = SimpleNamespace(grant_token_cap=40000)
    with (
        mock.patch.object(auto_advance.db, "fetchone", return_value=None),
        mock.patch("orchestrator.auto_research._upsert_job") as upsert,
    ):
        auto_advance._requeue_for_consumer(1, 2, 4, journal, args, recycle=1)

    assert upsert.call_args.kwargs["resource_grant_id"] == 4
    assert upsert.call_args.kwargs["experiment_run_id"] is None
