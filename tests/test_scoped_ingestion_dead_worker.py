"""A killed ingestion worker is knowable now, not thirty minutes from now.

scoped_ingestion_jobs_v1.lease_owner carries host:pid, so a claim whose PID is
gone on this host is provably abandoned. Until 2026-08-19 the queue waited for
the lease to expire instead: web restarts (three that evening, for deploys)
killed the worker while its 30-minute lease stayed valid, and a 20-paper batch
finished 4 papers in 38 minutes -- most of the window spent waiting for
timeouts rather than working.

Two things had to change together. The reclaim itself, and the agenda selector
in the worker's startup recovery, which only looked at jobs whose lease had
ALREADY expired -- making the reclaim unreachable for the exact case it exists
to handle.
"""

import unittest
from unittest import mock

from meta_harness.ingestion_queue import ScopedIngestionRepository


class DeadWorkerReclaimTests(unittest.TestCase):
    def setUp(self):
        self.repo = ScopedIngestionRepository()

    def _run(self, rows, *, this_pid, live_pids):
        executed = []

        def fake_kill(pid, _sig):
            if pid not in live_pids:
                raise ProcessLookupError(pid)

        with mock.patch("meta_harness.ingestion_queue.db") as db, mock.patch(
            "meta_harness.ingestion_queue.socket.gethostname", return_value="h1"
        ), mock.patch(
            "meta_harness.ingestion_queue.os.getpid", return_value=this_pid
        ), mock.patch(
            "meta_harness.ingestion_queue.os.kill", side_effect=fake_kill
        ):
            db._use_pg.return_value = True
            db.fetchall.return_value = rows
            db.execute.side_effect = lambda *a: executed.append(a)
            released = self.repo.release_dead_worker_claims(agenda_id=10)
        return released, executed

    def test_claim_under_a_dead_pid_is_reclaimed_immediately(self):
        rows = [
            {
                "id": 13,
                "lease_owner": "h1:1131077:scoped-ingestion",
                "attempt_count": 1,
                "max_attempts": 3,
            }
        ]
        released, executed = self._run(rows, this_pid=1142894, live_pids={1142894})
        self.assertEqual(released, 1)
        self.assertEqual(executed[0][1][0], "retryable")
        self.assertEqual(
            executed[0][1][1], "worker_process_gone_checkpoint_resume"
        )

    def test_claim_under_a_live_pid_is_left_alone(self):
        rows = [
            {
                "id": 13,
                "lease_owner": "h1:999:scoped-ingestion",
                "attempt_count": 1,
                "max_attempts": 3,
            }
        ]
        released, executed = self._run(rows, this_pid=1000, live_pids={999, 1000})
        self.assertEqual(released, 0)
        self.assertEqual(executed, [])

    def test_our_own_claim_is_never_reclaimed(self):
        rows = [
            {
                "id": 13,
                "lease_owner": "h1:1000:scoped-ingestion",
                "attempt_count": 1,
                "max_attempts": 3,
            }
        ]
        released, _ = self._run(rows, this_pid=1000, live_pids=set())
        self.assertEqual(released, 0)

    def test_exhausted_attempts_go_to_manual_reconciliation_not_retry(self):
        rows = [
            {
                "id": 13,
                "lease_owner": "h1:5:scoped-ingestion",
                "attempt_count": 3,
                "max_attempts": 3,
            }
        ]
        released, executed = self._run(rows, this_pid=1000, live_pids={1000})
        self.assertEqual(released, 1)
        self.assertEqual(executed[0][1][0], "manual_reconciliation")

    def test_startup_recovery_considers_unexpired_leases(self):
        # the selector that made the reclaim unreachable after a restart
        import inspect

        from orchestrator import scoped_ingestion_worker

        source = inspect.getsource(scoped_ingestion_worker.start)
        self.assertIn("WHERE status='running'", source)
        self.assertNotIn("lease_expires_at <= CURRENT_TIMESTAMP", source)


if __name__ == "__main__":
    unittest.main()
