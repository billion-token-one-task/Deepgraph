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
to handle. A dead PID does not prove whether an already-reserved provider call
was billed, so open usage parks for operator reconciliation instead of being
silently released.
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
            fetches = [rows]
            for row in rows:
                parts = str(row.get("lease_owner") or "").split(":")
                if (
                    len(parts) >= 2
                    and parts[1].isdigit()
                    and int(parts[1]) not in live_pids | {this_pid}
                ):
                    fetches.extend(
                        ([{"id": row["id"], "status": "running"}], [])
                    )
            db.fetchall.side_effect = fetches

            def execute(*args):
                executed.append(args)
                return mock.Mock(rowcount=1)

            db.execute.side_effect = execute
            released = self.repo.release_dead_worker_claims(agenda_id=10)
        return released, executed

    def test_claim_under_a_dead_pid_is_reclaimed_immediately(self):
        rows = [
            {
                "id": 13,
                "resource_grant_id": 44,
                "lease_owner": "h1:1131077:scoped-ingestion",
                "attempt_count": 1,
                "max_attempts": 3,
            }
        ]
        released, executed = self._run(rows, this_pid=1142894, live_pids={1142894})
        self.assertEqual(released, 1)
        self.assertEqual(len(executed), 1)
        self.assertEqual(executed[0][1][0], "retryable")
        self.assertEqual(
            executed[0][1][1], "worker_process_gone_checkpoint_resume"
        )

    def test_claim_under_a_live_pid_is_left_alone(self):
        rows = [
            {
                "id": 13,
                "resource_grant_id": 44,
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
                "resource_grant_id": 44,
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
                "resource_grant_id": 44,
                "lease_owner": "h1:5:scoped-ingestion",
                "attempt_count": 3,
                "max_attempts": 3,
            }
        ]
        released, executed = self._run(rows, this_pid=1000, live_pids={1000})
        self.assertEqual(released, 1)
        self.assertEqual(executed[0][1][0], "manual_reconciliation")

    def test_dead_pid_with_open_usage_parks_without_releasing_spend(self):
        rows = [
            {
                "id": 13,
                "resource_grant_id": 44,
                "lease_owner": "h1:5:scoped-ingestion",
                "attempt_count": 1,
                "max_attempts": 3,
            }
        ]
        with mock.patch("meta_harness.ingestion_queue.db") as db, mock.patch(
            "meta_harness.ingestion_queue.socket.gethostname", return_value="h1"
        ), mock.patch(
            "meta_harness.ingestion_queue.os.getpid", return_value=1000
        ), mock.patch(
            "meta_harness.ingestion_queue.os.kill", side_effect=ProcessLookupError
        ):
            db._use_pg.return_value = True
            db.fetchall.side_effect = [
                rows,
                [{"id": 13, "status": "running"}],
                [{"id": 91}],
            ]
            db.execute.return_value = mock.Mock(rowcount=1)

            reclaimed = self.repo.release_dead_worker_claims(agenda_id=10)

        self.assertEqual(reclaimed, 1)
        self.assertEqual(db.execute.call_count, 1)
        statement = db.execute.call_args.args[0]
        self.assertIn("manual_reconciliation", statement)
        self.assertNotIn("resource_grant_usage_reservations", statement)

    def test_ambiguous_shared_grant_never_releases_usage(self):
        rows = [
            {
                "id": 13,
                "resource_grant_id": 44,
                "lease_owner": "h1:5:scoped-ingestion",
                "attempt_count": 1,
                "max_attempts": 3,
            }
        ]
        with mock.patch("meta_harness.ingestion_queue.db") as db, mock.patch(
            "meta_harness.ingestion_queue.socket.gethostname", return_value="h1"
        ), mock.patch(
            "meta_harness.ingestion_queue.os.getpid", return_value=1000
        ), mock.patch(
            "meta_harness.ingestion_queue.os.kill", side_effect=ProcessLookupError
        ):
            db._use_pg.return_value = True
            db.fetchall.side_effect = [
                rows,
                [
                    {"id": 13, "status": "running"},
                    {"id": 14, "status": "succeeded"},
                ],
            ]
            db.execute.return_value = mock.Mock(rowcount=1)

            released = self.repo.release_dead_worker_claims(agenda_id=10)

        self.assertEqual(released, 1)
        self.assertEqual(db.execute.call_count, 1)
        self.assertNotIn(
            "resource_grant_usage_reservations",
            db.execute.call_args.args[0],
        )
        self.assertIn("manual_reconciliation", db.execute.call_args.args[0])

    def test_uncertain_expired_lease_with_open_usage_fails_closed(self):
        with mock.patch.object(
            self.repo, "release_dead_worker_claims", return_value=0
        ), mock.patch("meta_harness.ingestion_queue.db") as db:
            db._use_pg.return_value = True
            db.execute.side_effect = [
                mock.Mock(rowcount=0),
                mock.Mock(rowcount=1),
                mock.Mock(rowcount=0),
            ]

            recovered = self.repo.recover_expired_leases(agenda_id=10)

        self.assertEqual(recovered["retryable"], 0)
        self.assertEqual(recovered["manual_reconciliation"], 1)
        retry_sql = db.execute.call_args_list[0].args[0]
        ambiguous_sql = db.execute.call_args_list[1].args[0]
        self.assertIn("NOT EXISTS", retry_sql)
        self.assertIn("rgu.status='reserved'", retry_sql)
        self.assertIn("EXISTS", ambiguous_sql)
        self.assertIn("open_usage_reconciliation_required", ambiguous_sql)

    def test_startup_recovery_considers_unexpired_leases(self):
        # the selector that made the reclaim unreachable after a restart
        import inspect

        from orchestrator import scoped_ingestion_worker

        source = inspect.getsource(scoped_ingestion_worker.start)
        self.assertIn("WHERE status='running'", source)
        self.assertNotIn("lease_expires_at <= CURRENT_TIMESTAMP", source)

    def test_claim_gate_excludes_any_grant_with_open_child_usage(self):
        with mock.patch("meta_harness.ingestion_queue.db") as db:
            db._use_pg.return_value = True
            db.fetchone.return_value = None

            claimed = self.repo.claim_next(worker_id="host:7:worker", lease_seconds=60)

        self.assertIsNone(claimed)
        claim_sql = db.fetchone.call_args.args[0]
        self.assertIn("resource_grant_usage_reservations", claim_sql)
        self.assertIn("rgu.status='reserved'", claim_sql)
        db.commit.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
