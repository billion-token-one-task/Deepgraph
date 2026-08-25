"""Contract tests for the independent processing-status-v3 truth domains."""

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from orchestrator import paper_worker
from web import app as web_app


class ProcessingStatusContractTests(unittest.TestCase):
    def test_legacy_worker_disabled_status_round_trips_through_get_status(self):
        with (
            mock.patch.object(paper_worker, "_worker_thread", None),
            mock.patch.object(paper_worker, "_last_status", {"status": "not_started"}),
        ):
            started = paper_worker.start()
            readback = paper_worker.get_status()

        self.assertEqual(started["status"], "disabled_resource_grant_required")
        self.assertEqual(readback["status"], started["status"])
        self.assertEqual(readback["reason"], started["reason"])
        self.assertFalse(readback["running"])

    def test_scoped_ingestion_decision_table(self):
        cases = (
            ("worker failure", {"running": True, "status": "failed"}, {}, {}, "worker_error"),
            ("offline queued", {"running": False, "status": "idle"}, {"queued": 1}, {}, "stalled"),
            ("offline running", {"running": False, "status": "idle"}, {"running": 1}, {}, "stalled"),
            ("online running", {"running": True, "status": "idle"}, {"running": 1}, {}, "running"),
            ("online queued", {"running": True, "status": "idle"}, {"retryable": 1}, {}, "queued"),
            ("authorized no work", {"running": True, "status": "idle"}, {}, {"active_grants": 1}, "authorized_idle"),
            ("unbound grant", {"running": True, "status": "idle"}, {}, {"orphan_active_grants": 1}, "halted"),
            ("orphan job", {"running": True, "status": "idle"}, {"queued": 1}, {"orphan_jobs": 1}, "halted"),
            ("manual reconciliation", {"running": True, "status": "idle"}, {}, {"unresolved_manual_reconciliation_jobs": 1}, "halted"),
            # A reconciliation backlog is historical work awaiting an operator's
            # decision; it does not stop new jobs. Nine such rows held this
            # domain at `halted` for eight days while the worker they describe
            # completed a canary and drained corpus backlog.
            ("manual reconciliation while running", {"running": True, "status": "idle"}, {"running": 1}, {"unresolved_manual_reconciliation_jobs": 9}, "running"),
            ("manual reconciliation while queued", {"running": True, "status": "idle"}, {"queued": 2}, {"unresolved_manual_reconciliation_jobs": 9}, "queued"),
            # An authority hazard is different: it is unaccounted spend, and it
            # halts the domain whatever else is happening.
            ("orphan grant while running", {"running": True, "status": "idle"}, {"running": 1}, {"orphan_active_grants": 1}, "halted"),
            ("ambiguous usage", {"running": True, "status": "idle"}, {}, {"unresolved_open_usage_reservations": 1}, "halted"),
            ("unknown grant lane", {"running": True, "status": "idle"}, {}, {"unclassified_grants": 1}, "failed"),
            ("online idle", {"running": True, "status": "idle"}, {}, {}, "idle"),
            ("offline empty", {"running": False, "status": "idle"}, {}, {}, "stopped"),
        )
        for label, worker, counts, authority, expected in cases:
            with self.subTest(label=label):
                self.assertEqual(
                    web_app._classify_scoped_ingestion(
                        worker, counts, **authority
                    ),
                    expected,
                )

    def test_scoped_authority_exposes_orphans_and_unresolved_usage(self):
        rows = [
            {
                "grant_id": 10,
                "grant_stage": "ingestion_backfill_canary",
                "grant_status": "active",
                "grant_live": 1,
                "job_id": None,
                "open_usage_count": 0,
            },
            {
                "grant_id": 11,
                "grant_stage": "ingestion_backfill_canary",
                "grant_status": "active",
                "grant_live": 1,
                "job_id": 25,
                "job_status": "manual_reconciliation",
                "job_result_json": "{}",
                "open_usage_count": 1,
            },
            {
                "grant_id": 12,
                "grant_stage": "ingestion_backfill_canary",
                "grant_status": "expired",
                "grant_live": 0,
                "job_id": 26,
                "job_status": "manual_reconciliation",
                "job_result_json": json.dumps(
                    {
                        "reconciliation": {
                            "version": "scoped-ingestion-reconciliation-v1",
                            "replacement_job_id": 27,
                        }
                    }
                ),
                "open_usage_count": 0,
            },
            {
                "grant_id": 13,
                "grant_stage": "pilot",
                "grant_status": "active",
                "grant_live": 1,
                "job_id": None,
                "open_usage_count": 0,
            },
            {
                "grant_id": 15,
                "grant_stage": "pilot",
                "grant_status": "active",
                "grant_live": 1,
                "job_id": 29,
                "job_status": "queued",
                "job_result_json": "{}",
                "open_usage_count": 0,
            },
            {
                "grant_id": 14,
                "grant_stage": "ingestion_backfill_canary",
                "grant_status": "active",
                "grant_live": 1,
                "job_id": 28,
                "job_status": "running",
                "job_result_json": "{}",
                "open_usage_count": 2,
            },
        ]

        snapshot = web_app._scoped_ingestion_authority_counts(rows)

        self.assertEqual(snapshot["active_grants"], 3)
        self.assertEqual(snapshot["bound_active_grants"], 2)
        self.assertEqual(snapshot["unbound_active_grants"], 1)
        self.assertEqual(snapshot["orphan_active_grants"], 2)
        self.assertEqual(snapshot["unresolved_manual_reconciliation_jobs"], 1)
        self.assertEqual(snapshot["unresolved_open_usage_reservations"], 1)
        self.assertEqual(snapshot["unclassified_grants"], 1)
        self.assertEqual(snapshot["orphan_jobs"], 0)

    def test_scoped_snapshot_halts_on_an_unbound_active_grant(self):
        with (
            mock.patch(
                "orchestrator.scoped_ingestion_worker.get_status",
                return_value={"running": True, "status": "idle"},
            ),
            mock.patch.object(
                web_app.db,
                "fetchall",
                side_effect=[
                    [{"status": "succeeded", "c": 8}],
                    [
                        {
                            "grant_id": 55,
                            "grant_stage": "ingestion_backfill_canary",
                            "grant_status": "active",
                            "grant_live": 1,
                            "job_id": None,
                            "open_usage_count": 0,
                        }
                    ],
                ],
            ),
        ):
            snapshot = web_app._scoped_ingestion_snapshot()

        self.assertEqual(snapshot["state"], "halted")
        self.assertEqual(snapshot["active_grants"], 1)
        self.assertEqual(snapshot["unbound_active_grants"], 1)
        self.assertEqual(snapshot["orphan_active_grants"], 1)
        self.assertEqual(snapshot["queued_jobs"], 0)
        self.assertEqual(snapshot["running_jobs"], 0)

    def test_open_usage_is_expected_only_for_one_live_running_binding(self):
        rows = [
            {
                "grant_id": 60,
                "grant_stage": "ingestion_backfill_canary",
                "grant_status": "active",
                "grant_live": 1,
                "job_id": 600,
                "job_status": "running",
                "job_result_json": "{}",
                "open_usage_count": 1,
            },
            {
                "grant_id": 61,
                "grant_stage": "ingestion_backfill_canary",
                "grant_status": "expired",
                "grant_live": 0,
                "job_id": 610,
                "job_status": "running",
                "job_result_json": "{}",
                "open_usage_count": 2,
            },
            {
                "grant_id": 62,
                "grant_stage": "ingestion_backfill_canary",
                "grant_status": "active",
                "grant_live": 1,
                "job_id": 620,
                "job_status": "running",
                "job_result_json": "{}",
                "open_usage_count": 3,
            },
            {
                "grant_id": 62,
                "grant_stage": "ingestion_backfill_canary",
                "grant_status": "active",
                "grant_live": 1,
                "job_id": 621,
                "job_status": "succeeded",
                "job_result_json": "{}",
                "open_usage_count": 3,
            },
        ]

        snapshot = web_app._scoped_ingestion_authority_counts(rows)

        self.assertEqual(snapshot["active_grants"], 2)
        self.assertEqual(snapshot["orphan_active_grants"], 1)
        self.assertEqual(snapshot["orphan_jobs"], 2)
        self.assertEqual(snapshot["unresolved_open_usage_reservations"], 5)

    def test_nonterminal_job_without_live_grant_is_orphaned_and_halted(self):
        with (
            mock.patch(
                "orchestrator.scoped_ingestion_worker.get_status",
                return_value={"running": True, "status": "idle"},
            ),
            mock.patch.object(
                web_app.db,
                "fetchall",
                side_effect=[
                    [{"status": "queued", "c": 1}],
                    [
                        {
                            "grant_id": 70,
                            "grant_stage": "ingestion_backfill_canary",
                            "grant_status": "expired",
                            "grant_live": 0,
                            "job_id": 700,
                            "job_status": "queued",
                            "job_result_json": "{}",
                            "open_usage_count": 0,
                        }
                    ],
                ],
            ),
        ):
            snapshot = web_app._scoped_ingestion_snapshot()

        self.assertEqual(snapshot["state"], "halted")
        self.assertEqual(snapshot["queued_jobs"], 1)
        self.assertEqual(snapshot["active_grants"], 0)
        self.assertEqual(snapshot["orphan_jobs"], 1)

    def test_research_runtime_decision_table(self):
        cases = (
            ("controller failure", {"running": False, "status": "failed"}, 0, 0, 0, 0, "error"),
            ("bounded work without controller", {"running": False, "status": "idle"}, 1, 1, 0, 0, "running"),
            ("authorized queue", {"running": True, "status": "idle"}, 1, 0, 1, 0, "queued"),
            ("stale running claim", {"running": True, "status": "idle"}, 1, 0, 0, 1, "halted"),
            ("authorized idle", {"running": True, "status": "idle"}, 1, 0, 0, 0, "authorized_idle"),
            ("no authority", {"running": True, "status": "idle"}, 0, 0, 0, 0, "idle_no_authorized_work"),
        )
        for label, controller, grants, running, queued, stale, expected in cases:
            with self.subTest(label=label):
                self.assertEqual(
                    web_app._classify_research_runtime(
                        controller,
                        active_grants=grants,
                        active_work_items=running + queued + stale,
                        running_work_items=running,
                        queued_authorized_work_items=queued,
                        stale_work_items=stale,
                    ),
                    expected,
                )

    def test_research_work_freshness_does_not_call_queues_running(self):
        now = datetime(2026, 8, 24, 20, 0, tzinfo=timezone.utc)
        rows = [
            {
                "grant_stage": "pilot",
                "job_status": "running_experiment",
                "updated_at": now - timedelta(minutes=2),
            },
            {
                "grant_stage": "pilot",
                "job_status": "running_cpu",
                "updated_at": now - timedelta(minutes=31),
            },
            {
                "grant_stage": "pilot",
                "job_status": "queued_gpu",
                "updated_at": now - timedelta(hours=3),
            },
            {
                "grant_stage": "pilot",
                "job_status": "review_pending",
                "updated_at": now - timedelta(hours=3),
            },
        ]

        counts = web_app._research_work_counts(rows, now=now)

        self.assertEqual(counts["running_work_items"], 1)
        self.assertEqual(counts["stale_work_items"], 1)
        self.assertEqual(counts["queued_authorized_work_items"], 2)
        self.assertEqual(counts["active_work_items"], 4)

    def test_stale_legacy_records_do_not_stall_research_runtime(self):
        legacy = web_app._legacy_paper_ingestion_snapshot(
            {
                "running": False,
                "status": "disabled_resource_grant_required",
                "reason": "agenda_scoped_ingestion_job_required",
            },
            processing_count=10,
            recent_processing_count=0,
        )
        research_state = web_app._classify_research_runtime(
            {"running": True}, active_grants=0, active_work_items=0
        )

        self.assertEqual(legacy["record_state"], "stale_records")
        self.assertEqual(legacy["stale_processing_count"], 10)
        self.assertEqual(legacy["state"], "disabled_resource_grant_required")
        self.assertEqual(research_state, "idle_no_authorized_work")

    def test_legacy_worker_source_unavailable_is_not_reported_idle(self):
        snapshot = web_app._legacy_paper_ingestion_snapshot(
            {
                "running": False,
                "status": "error",
                "available": False,
                "error": "worker status unavailable",
            },
            processing_count=0,
            recent_processing_count=0,
        )

        self.assertFalse(snapshot["available"])
        self.assertEqual(snapshot["state"], "worker_error")

    def test_queued_jobs_are_partitioned_without_becoming_authority(self):
        fetchall = mock.Mock(side_effect=[[], []])
        fetchone = mock.Mock(side_effect=[
            {"queued_active": 11, "queued_closed": 33, "queued_unclassified": 0},
        ])
        with (
            mock.patch("orchestrator.auto_research.get_status", return_value={"running": True}),
            mock.patch.object(web_app.db, "fetchall", fetchall),
            mock.patch.object(web_app.db, "fetchone", fetchone),
        ):
            snapshot = web_app._research_runtime_snapshot()

        self.assertEqual(snapshot["active_grants"], 0)
        self.assertEqual(snapshot["unclassified_active_grants"], 0)
        self.assertEqual(snapshot["active_work_items"], 0)
        self.assertEqual(snapshot["unclassified_active_work_items"], 0)
        self.assertEqual(snapshot["queued_active_agendas"], 11)
        self.assertEqual(snapshot["queued_closed_agendas"], 33)
        self.assertEqual(snapshot["queued_unclassified_agendas"], 0)
        self.assertEqual(snapshot["state"], "idle_no_authorized_work")
        queued_query = fetchone.call_args_list[0].args[0]
        self.assertIn("ra.is_active=1 AND ra.status='active'", queued_query)
        self.assertIn("ra.is_active=0 OR ra.status='closed'", queued_query)
        self.assertIn("active_rg.status='active'", queued_query)
        self.assertIn("active_rg.id IS NULL", queued_query)

    def test_authorized_queued_job_is_not_counted_as_demand_or_running(self):
        fetchall = mock.Mock(
            side_effect=[
                [{"stage": "pilot", "c": 1}],
                [
                    {
                        "grant_stage": "pilot",
                        "job_status": "queued",
                        "job_stage": "portfolio_granted",
                        "updated_at": datetime.now(timezone.utc),
                    }
                ],
            ]
        )
        with (
            mock.patch(
                "orchestrator.auto_research.get_status",
                return_value={"running": False, "status": "idle"},
            ),
            mock.patch.object(web_app.db, "fetchall", fetchall),
            mock.patch.object(
                web_app.db,
                "fetchone",
                return_value={
                    "queued_active": 10,
                    "queued_closed": 33,
                    "queued_unclassified": 0,
                },
            ),
        ):
            snapshot = web_app._research_runtime_snapshot()

        self.assertEqual(snapshot["state"], "queued")
        self.assertEqual(snapshot["active_grants"], 1)
        self.assertEqual(snapshot["active_work_items"], 1)
        self.assertEqual(snapshot["running_work_items"], 0)
        self.assertEqual(snapshot["queued_authorized_work_items"], 1)
        self.assertEqual(snapshot["queued_active_agendas"], 10)

    def test_canonical_ingestion_grants_and_work_are_not_research_authority(self):
        fetchall = mock.Mock(side_effect=[
            [
                {"stage": "ingestion", "c": 1},
                {"stage": "ingestion_backfill_canary", "c": 2},
                {"stage": "pilot", "c": 3},
            ],
            [
                {
                    "grant_stage": "ingestion_v1",
                    "job_status": "running_experiment",
                    "updated_at": datetime.now(timezone.utc),
                    "c": 4,
                },
                {
                    "grant_stage": "pilot",
                    "job_status": "running_experiment",
                    "updated_at": datetime.now(timezone.utc),
                    "c": 2,
                },
            ],
        ])
        with (
            mock.patch(
                "orchestrator.auto_research.get_status",
                return_value={"running": True, "status": "idle"},
            ),
            mock.patch.object(web_app.db, "fetchall", fetchall),
            mock.patch.object(
                web_app.db,
                "fetchone",
                return_value={
                    "queued_active": 0,
                    "queued_closed": 0,
                    "queued_unclassified": 0,
                },
            ),
        ):
            snapshot = web_app._research_runtime_snapshot()

        self.assertEqual(snapshot["active_grants"], 3)
        self.assertEqual(snapshot["active_work_items"], 2)
        self.assertEqual(snapshot["running_work_items"], 2)
        self.assertEqual(snapshot["queued_authorized_work_items"], 0)
        self.assertEqual(snapshot["stale_work_items"], 0)
        self.assertEqual(snapshot["unclassified_active_grants"], 0)
        self.assertEqual(snapshot["unclassified_active_work_items"], 0)
        self.assertEqual(snapshot["state"], "running")
        grant_query = fetchall.call_args_list[0].args[0]
        self.assertNotIn("NOT LIKE", grant_query)
        self.assertNotIn("scoped_ingestion_jobs_v1", grant_query)

    def test_unsupported_ingestion_like_grant_fails_closed(self):
        with (
            mock.patch(
                "orchestrator.auto_research.get_status",
                return_value={"running": True, "status": "idle"},
            ),
            mock.patch.object(
                web_app.db,
                "fetchall",
                side_effect=[[{"stage": "ingestionish", "c": 1}], []],
            ),
            mock.patch.object(
                web_app.db,
                "fetchone",
                return_value={
                    "queued_active": 0,
                    "queued_closed": 0,
                    "queued_unclassified": 0,
                },
            ),
        ):
            snapshot = web_app._research_runtime_snapshot()

        self.assertEqual(snapshot["active_grants"], 0)
        self.assertEqual(snapshot["unclassified_active_grants"], 1)
        self.assertEqual(snapshot["state"], "error")

    def test_corpus_uses_exact_status_counts(self):
        with mock.patch.object(
            web_app.db,
            "fetchone",
            return_value={"total": 24153, "pending": 16194, "processed": 7000, "error": 949},
        ):
            snapshot = web_app._corpus_snapshot()
        self.assertEqual(snapshot, {
            "available": True,
            "total": 24153,
            "pending": 16194,
            "processed": 7000,
            "error": 949,
        })

    @staticmethod
    def _harvest_line(finished_at, new, by_category):
        return json.dumps({
            "step": "harvest",
            "finished_at": finished_at,
            "new": new,
            "seen": 600,
            "by_category": by_category,
        })

    def test_harvest_uses_latest_zero_not_older_nonzero(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "harvest.jsonl"
            path.write_text("\n".join([
                self._harvest_line("2026-08-24T04:15:09Z", 207, {"cs.AI": 207}),
                self._harvest_line("2026-08-24T16:16:14Z", 0, {"cs.AI": 0}),
            ]) + "\n", encoding="utf-8")
            snapshot = web_app._harvest_snapshot(path)

        self.assertEqual(snapshot["state"], "idle")
        self.assertTrue(snapshot["available"])
        self.assertEqual(snapshot["last_success_at"], "2026-08-24T16:16:14Z")
        self.assertEqual(snapshot["last_new_count"], 0)
        self.assertEqual(snapshot["last_attempt_at"], snapshot["last_success_at"])
        self.assertIsNone(snapshot["diagnostic"])

    def test_harvest_category_error_is_degraded_and_retains_last_success(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "harvest.jsonl"
            path.write_text("\n".join([
                self._harvest_line("2026-08-24T04:15:09Z", 207, {"cs.AI": 207}),
                self._harvest_line(
                    "2026-08-24T16:16:14Z",
                    0,
                    {"cs.AI": 0, "cs.LG": "error: TimeoutError"},
                ),
            ]) + "\n", encoding="utf-8")
            snapshot = web_app._harvest_snapshot(path)

        self.assertEqual(snapshot["state"], "degraded")
        self.assertEqual(snapshot["last_success_at"], "2026-08-24T04:15:09Z")
        self.assertEqual(snapshot["last_new_count"], 207)
        self.assertEqual(snapshot["last_attempt_at"], "2026-08-24T16:16:14Z")
        self.assertEqual(snapshot["failed_categories"], ["cs.LG"])
        self.assertEqual(snapshot["diagnostic"], "category_errors")

    def test_harvest_malformed_tail_is_degraded_not_fabricated_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "harvest.jsonl"
            path.write_text(
                self._harvest_line("2026-08-24T16:16:14Z", 0, {"cs.AI": 0})
                + "\n{not-json}\n",
                encoding="utf-8",
            )
            snapshot = web_app._harvest_snapshot(path)

        self.assertEqual(snapshot["state"], "degraded")
        self.assertEqual(snapshot["last_new_count"], 0)
        self.assertEqual(snapshot["diagnostic"], "malformed_tail")

    def test_harvest_all_category_errors_is_failed_without_fake_success(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "harvest.jsonl"
            path.write_text(
                self._harvest_line(
                    "2026-08-24T16:16:14Z",
                    0,
                    {"cs.AI": "error: TimeoutError", "cs.LG": "error: HTTPError"},
                ) + "\n",
                encoding="utf-8",
            )
            snapshot = web_app._harvest_snapshot(path)

        self.assertEqual(snapshot["state"], "failed")
        self.assertIsNone(snapshot["last_success_at"])
        self.assertIsNone(snapshot["last_new_count"])
        self.assertEqual(snapshot["last_attempt_at"], "2026-08-24T16:16:14Z")
        self.assertEqual(snapshot["failed_categories"], ["cs.AI", "cs.LG"])

    def test_harvest_missing_source_is_unknown_with_null_truth(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = web_app._harvest_snapshot(Path(directory) / "missing.jsonl")
        self.assertEqual(snapshot["state"], "unknown")
        self.assertFalse(snapshot["available"])
        self.assertIsNone(snapshot["last_success_at"])
        self.assertIsNone(snapshot["last_new_count"])
        self.assertEqual(snapshot["diagnostic"], "source_missing")

    def test_harvest_file_without_valid_record_is_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "harvest.jsonl"
            path.write_text('{"step":"other"}\n{not-json}\n', encoding="utf-8")
            snapshot = web_app._harvest_snapshot(path)
        self.assertEqual(snapshot["state"], "unknown")
        self.assertFalse(snapshot["available"])
        self.assertEqual(snapshot["diagnostic"], "no_valid_record")

    def test_backfill_decision_table(self):
        now = datetime(2026, 8, 24, 20, 0, tzinfo=timezone.utc)
        cases = (
            ("fresh", "2026-08-24T19:00:00Z", "idle", None),
            ("exact threshold", "2026-08-24T17:00:00Z", "idle", None),
            ("stale", "2026-08-24T16:59:59Z", "halted", "no_progress_over_3h"),
            ("future", "2026-08-24T20:00:01Z", "failed", "progress_marker_in_future"),
            ("invalid", "not-a-time", "failed", "progress_marker_invalid"),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last_progress.txt"
            for label, value, state, reason in cases:
                with self.subTest(label=label):
                    path.write_text(value + "\n", encoding="utf-8")
                    snapshot = web_app._backfill_snapshot(path, now=now)
                    self.assertEqual(snapshot["state"], state)
                    self.assertEqual(snapshot["halt_reason"], reason)
                    self.assertEqual(snapshot["available"], label != "invalid")

    def test_backfill_missing_marker_is_unknown_not_idle(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = web_app._backfill_snapshot(Path(directory) / "missing")
        self.assertEqual(snapshot["state"], "unknown")
        self.assertFalse(snapshot["available"])
        self.assertIsNone(snapshot["last_progress_at"])
        self.assertEqual(snapshot["halt_reason"], "progress_marker_missing")

    def test_failed_domain_uses_null_counts_not_healthy_zeroes(self):
        def unavailable():
            raise RuntimeError("status source down")

        with mock.patch.object(web_app, "log_event"):
            snapshot = web_app._processing_domain_snapshot(
                "research_runtime",
                unavailable,
                {"active_grants": None, "active_work_items": None},
            )
        self.assertEqual(snapshot["state"], "failed")
        self.assertFalse(snapshot["available"])
        self.assertIsNone(snapshot["active_grants"])
        self.assertIsNone(snapshot["active_work_items"])

    def test_processing_endpoint_exposes_v3_without_conflating_domains(self):
        client = web_app.app.test_client()
        legacy = {
            "state": "disabled_resource_grant_required",
            "record_state": "stale_records",
            "stale_processing_count": 10,
        }
        scoped = {
            "state": "halted",
            "running": True,
            "queued_jobs": 0,
            "running_jobs": 0,
            "active_grants": 0,
            "bound_active_grants": 0,
            "unbound_active_grants": 0,
            "orphan_active_grants": 0,
            "orphan_jobs": 0,
            "unresolved_manual_reconciliation_jobs": 9,
            "unresolved_open_usage_reservations": 0,
            "unclassified_grants": 0,
        }
        research = {
            "state": "idle_no_authorized_work",
            "active_grants": 0,
            "unclassified_active_grants": 0,
            "active_work_items": 0,
            "running_work_items": 0,
            "queued_authorized_work_items": 0,
            "stale_work_items": 0,
            "unclassified_active_work_items": 0,
            "queued_active_agendas": 11,
            "queued_closed_agendas": 33,
            "queued_unclassified_agendas": 0,
        }
        corpus = {
            "available": True,
            "total": 24153,
            "pending": 16194,
            "processed": 7000,
            "error": 949,
        }
        harvest = {
            "available": True,
            "state": "idle",
            "last_success_at": "2026-08-24T16:16:14Z",
            "last_new_count": 0,
        }
        backfill = {
            "available": True,
            "state": "halted",
            "last_progress_at": "2026-08-20T08:54:18Z",
            "halt_reason": "no_progress_over_3h",
        }
        with (
            mock.patch.object(web_app.db, "fetchall", return_value=[]),
            mock.patch.object(web_app.db, "fetchone", side_effect=[{"c": 10}, {"c": 0}]),
            mock.patch(
                "orchestrator.paper_worker.get_status",
                return_value={"running": False, "status": "disabled_resource_grant_required"},
            ),
            mock.patch.object(web_app, "_legacy_paper_ingestion_snapshot", return_value=legacy),
            mock.patch.object(web_app, "_scoped_ingestion_snapshot", return_value=scoped),
            mock.patch.object(web_app, "_research_runtime_snapshot", return_value=research),
            mock.patch.object(web_app, "_corpus_snapshot", return_value=corpus),
            mock.patch.object(web_app, "_harvest_snapshot", return_value=harvest),
            mock.patch.object(web_app, "_backfill_snapshot", return_value=backfill),
        ):
            response = client.get("/api/processing")

        payload = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["contract_version"], "processing-status-v3")
        self.assertEqual(payload["legacy_paper_ingestion"], legacy)
        self.assertEqual(payload["scoped_ingestion"], {**scoped, "available": True})
        self.assertEqual(payload["research_runtime"], {**research, "available": True})
        self.assertEqual(payload["corpus"], corpus)
        self.assertEqual(payload["harvest"], harvest)
        self.assertEqual(payload["backfill"], backfill)
        # v2 compatibility aliases remain research-only: queued demand and a
        # halted backfill do not claim authorized/running research execution.
        self.assertEqual(payload["pipeline_state"], "idle_no_authorized_work")
        self.assertFalse(payload["pipeline_running"])
        self.assertEqual(payload["data_health"]["status"], "ok")

    def test_one_failed_v3_domain_does_not_fabricate_zero_or_fail_others(self):
        client = web_app.app.test_client()
        with (
            mock.patch.object(web_app.db, "fetchall", return_value=[]),
            mock.patch.object(web_app.db, "fetchone", side_effect=[{"c": 0}, {"c": 0}]),
            mock.patch(
                "orchestrator.paper_worker.get_status",
                return_value={"running": False, "status": "disabled_resource_grant_required"},
            ),
            mock.patch.object(
                web_app,
                "_scoped_ingestion_snapshot",
                return_value={"state": "idle", "queued_jobs": 0, "running_jobs": 0},
            ),
            mock.patch.object(
                web_app,
                "_research_runtime_snapshot",
                return_value={
                    "state": "idle_no_authorized_work",
                    "active_grants": 0,
                    "unclassified_active_grants": 0,
                    "active_work_items": 0,
                    "running_work_items": 0,
                    "queued_authorized_work_items": 0,
                    "stale_work_items": 0,
                    "unclassified_active_work_items": 0,
                    "queued_active_agendas": 11,
                    "queued_closed_agendas": 33,
                    "queued_unclassified_agendas": 0,
                },
            ),
            mock.patch.object(web_app, "_corpus_snapshot", side_effect=RuntimeError("db down")),
            mock.patch.object(
                web_app,
                "_harvest_snapshot",
                return_value={
                    "available": True,
                    "state": "idle",
                    "last_success_at": "2026-08-24T16:16:14Z",
                    "last_new_count": 0,
                },
            ),
            mock.patch.object(
                web_app,
                "_backfill_snapshot",
                return_value={
                    "available": True,
                    "state": "halted",
                    "last_progress_at": "2026-08-20T08:54:18Z",
                    "halt_reason": "no_progress_over_3h",
                },
            ),
            mock.patch.object(web_app, "log_event"),
        ):
            response = client.get("/api/processing")

        payload = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["corpus"]["state"], "failed")
        self.assertFalse(payload["corpus"]["available"])
        self.assertIsNone(payload["corpus"]["total"])
        self.assertIsNone(payload["corpus"]["pending"])
        self.assertEqual(payload["research_runtime"]["active_grants"], 0)
        self.assertEqual(payload["data_health"]["status"], "degraded")
        self.assertIn("corpus", payload["data_health"]["degraded_sources"])

    def test_fixed_fixture_is_a_complete_v3_example(self):
        path = Path(__file__).parent / "fixtures" / "processing_status_v3.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["contract_version"], "processing-status-v3")
        self.assertEqual(payload["research_runtime"]["queued_active_agendas"], 11)
        self.assertEqual(payload["research_runtime"]["queued_closed_agendas"], 33)
        self.assertEqual(payload["research_runtime"]["unclassified_active_grants"], 0)
        self.assertEqual(payload["research_runtime"]["running_work_items"], 0)
        self.assertEqual(
            payload["research_runtime"]["queued_authorized_work_items"], 0
        )
        self.assertEqual(
            payload["scoped_ingestion"][
                "unresolved_manual_reconciliation_jobs"
            ],
            9,
        )
        self.assertEqual(payload["scoped_ingestion"]["state"], "halted")
        self.assertTrue(
            {
                "active_grants",
                "bound_active_grants",
                "unbound_active_grants",
                "orphan_active_grants",
                "orphan_jobs",
                "unresolved_manual_reconciliation_jobs",
                "unresolved_open_usage_reservations",
                "unclassified_grants",
            }.issubset(payload["scoped_ingestion"])
        )
        self.assertTrue(
            {
                "active_work_items",
                "running_work_items",
                "queued_authorized_work_items",
                "stale_work_items",
            }.issubset(payload["research_runtime"])
        )
        self.assertEqual(payload["harvest"]["last_new_count"], 0)
        self.assertEqual(payload["backfill"]["state"], "halted")
        self.assertTrue({"total", "pending", "processed", "error"}.issubset(payload["corpus"]))
        for domain in (
            "legacy_paper_ingestion",
            "scoped_ingestion",
            "research_runtime",
            "corpus",
            "harvest",
            "backfill",
        ):
            self.assertIs(payload[domain]["available"], True)


if __name__ == "__main__":
    unittest.main()
