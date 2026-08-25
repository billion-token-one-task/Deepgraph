"""A status tile must report work, not a flag or a backlog.

Three tiles were saying the opposite of what was happening:

* `scoped ingestion` read `halted` for eight days because nine manual
  reconciliations from 2026-08-17 were unresolved, while the very worker it
  describes completed a five-paper canary and then drained corpus backlog;
* `graph construction` had no liveness rule at all and read idle while the
  corpus advanced from 7,014 to 7,023 papers;
* `idea generation` read the global auto-research controller, which is
  deliberately disabled in favour of agenda-scoped passes, so it read idle
  while ideas were being generated.
"""

import unittest
from unittest import mock

from db import database as db
from web import app as web_app


class ScopedIngestionClassificationTest(unittest.TestCase):
    def _classify(self, counts, worker, **kwargs):
        return web_app._classify_scoped_ingestion(worker, counts, **kwargs)

    def test_a_reconciliation_backlog_does_not_mask_a_running_job(self):
        state = self._classify({"running": 1}, {"running": True},
                               unresolved_manual_reconciliation_jobs=9)
        self.assertEqual(state, "running")

    def test_a_reconciliation_backlog_does_not_mask_a_queue(self):
        state = self._classify({"queued": 3}, {"running": True},
                               unresolved_manual_reconciliation_jobs=9)
        self.assertEqual(state, "queued")

    def test_a_reconciliation_backlog_with_nothing_running_still_halts(self):
        state = self._classify({}, {"running": False},
                               unresolved_manual_reconciliation_jobs=9)
        self.assertEqual(state, "halted")

    def test_an_orphan_grant_halts_even_while_work_runs(self):
        # Unaccounted authority is a money leak; continuing compounds it.
        state = self._classify({"running": 2}, {"running": True},
                               orphan_active_grants=1)
        self.assertEqual(state, "halted")

    def test_an_open_usage_reservation_halts_even_while_work_runs(self):
        state = self._classify({"running": 2}, {"running": True},
                               unresolved_open_usage_reservations=1)
        self.assertEqual(state, "halted")

    def test_an_unclassified_grant_is_still_a_failure(self):
        state = self._classify({"running": 1}, {"running": True},
                               unclassified_grants=1)
        self.assertEqual(state, "failed")


class FreshnessFragmentTest(unittest.TestCase):
    def test_it_accepts_the_columns_callers_use(self):
        for column in ("updated_at", "created_at", "recorded_at", "heartbeat_at"):
            self.assertIn(column, db.sql_updated_after_seconds(60, column=column))

    def test_it_refuses_an_arbitrary_column(self):
        with self.assertRaises(ValueError):
            db.sql_updated_after_seconds(60, column="1=1 OR x")

    def test_the_default_is_unchanged(self):
        self.assertIn("updated_at", db.sql_updated_after_seconds(60))

    def test_the_interval_is_an_integer(self):
        self.assertIn("60", db.sql_updated_after_seconds(60.9))


class RecentActivityCountTest(unittest.TestCase):
    def test_only_declared_tables_are_countable(self):
        self.assertEqual(web_app._recent_activity_count("papers; DROP", "updated_at", 60), 0)

    def test_a_failing_count_reports_zero_rather_than_breaking_the_view(self):
        with mock.patch.object(web_app.db, "fetchone", side_effect=RuntimeError("db down")):
            self.assertEqual(
                web_app._recent_activity_count("papers", "updated_at", 60), 0)

    def test_it_returns_the_count_it_was_given(self):
        with mock.patch.object(web_app.db, "fetchone", return_value={"n": 7}):
            self.assertEqual(
                web_app._recent_activity_count("deep_insights", "created_at", 60), 7)


if __name__ == "__main__":
    unittest.main()
