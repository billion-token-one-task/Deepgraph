from unittest import mock

from agents import meta_learner


def test_meta_report_requires_explicit_agenda():
    with mock.patch.object(meta_learner.db, "fetchone") as fetch:
        report = meta_learner.get_full_meta_report()
    assert report["status"] == "agenda_required"
    fetch.assert_not_called()


def test_meta_report_counts_canonical_scientific_decisions_only():
    with (
        mock.patch.object(
            meta_learner.db,
            "fetchone",
            return_value={"c": 1},
        ) as fetchone,
        mock.patch.object(
            meta_learner,
            "get_track_record_summary",
            return_value={},
        ),
        mock.patch.object(meta_learner, "get_node_hit_rates", return_value=[]),
        mock.patch.object(
            meta_learner,
            "get_adversarial_calibration",
            return_value={},
        ),
        mock.patch.object(
            meta_learner,
            "get_method_type_analysis",
            return_value=[],
        ),
        mock.patch.object(meta_learner, "compute_signal_weights", return_value={}),
    ):
        report = meta_learner.get_full_meta_report(7)

    assert report["status"] == "ready"
    assert report["agenda_id"] == 7
    sql, params = fetchone.call_args.args
    assert "scientific_decision_records" in sql
    assert params == (7,)


def test_node_hit_rates_join_scientific_decisions_in_agenda():
    with (
        mock.patch.object(meta_learner.db, "use_postgres", return_value=True),
        mock.patch.object(
            meta_learner.db,
            "fetchall",
            return_value=[
                {
                    "node_id": "ml.reasoning",
                    "total": 3,
                    "confirmed": 2,
                    "refuted": 1,
                    "avg_effect": 0.1,
                }
            ],
        ) as fetchall,
    ):
        rows = meta_learner.get_node_hit_rates(9)

    assert rows[0]["hit_rate"] == 0.6667
    sql, params = fetchall.call_args.args
    assert "scientific_decision_records" in sql
    assert "er.agenda_id=?" in sql
    assert params == (9,)
