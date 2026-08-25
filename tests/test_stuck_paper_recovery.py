"""Papers abandoned mid-processing must be reclaimable without a full restart.

A paper is marked `processing` before extraction and cleared afterwards. A
worker that stops in between leaves a row nobody owns: no queue holds it, no
retry reaches it, and it shows up only as a count on a status page. Ten such
rows survived the 2026-06 and 2026-08-17 shutdowns and sat untouched for weeks
because the only recovery ran at pipeline startup.
"""

import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

from orchestrator import pipeline


class _FakeDb:
    def __init__(self, rows, updated):
        self.rows = rows
        self.updated = updated
        self.updates = []
        self.committed = False

    def fetchall(self, sql, params=()):
        return [{"id": paper_id} for paper_id in self.rows]

    def fetchone(self, sql, params=()):
        return {"updated_at": self.updated.get(params[0])}

    def execute(self, sql, params=()):
        self.updates.append(params[0])

    def commit(self):
        self.committed = True


class StuckPaperRecoveryTest(unittest.TestCase):
    def _run(self, rows, updated, **kwargs):
        fake = _FakeDb(rows, updated)
        with mock.patch.object(pipeline, "db", fake), \
             mock.patch.object(pipeline, "log_event", lambda *a, **k: None):
            recovered = pipeline.recover_stuck_processing_papers(**kwargs)
        return recovered, fake

    def test_an_abandoned_paper_returns_to_the_queue(self):
        old = datetime.now(timezone.utc) - timedelta(days=5)
        recovered, fake = self._run(["2603.17551"], {"2603.17551": old})
        self.assertEqual(recovered, ["2603.17551"])
        self.assertEqual(fake.updates, ["2603.17551"])
        self.assertTrue(fake.committed)

    def test_a_paper_a_live_worker_is_holding_is_left_alone(self):
        fresh = datetime.now(timezone.utc) - timedelta(seconds=30)
        recovered, fake = self._run(["2603.18892"], {"2603.18892": fresh})
        self.assertEqual(recovered, [])
        self.assertEqual(fake.updates, [])
        self.assertFalse(fake.committed)

    def test_startup_recovery_reclaims_everything(self):
        fresh = datetime.now(timezone.utc)
        recovered, _ = self._run(["a"], {"a": fresh}, older_than_seconds=0)
        self.assertEqual(recovered, ["a"])

    def test_a_naive_timestamp_is_read_as_utc_not_skipped(self):
        naive_old = datetime.utcnow() - timedelta(days=3)
        recovered, _ = self._run(["b"], {"b": naive_old})
        self.assertEqual(recovered, ["b"])

    def test_nothing_stuck_means_no_commit(self):
        recovered, fake = self._run([], {})
        self.assertEqual(recovered, [])
        self.assertFalse(fake.committed)


if __name__ == "__main__":
    unittest.main()
