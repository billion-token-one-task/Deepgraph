"""Canonical GPU attempt accounting against an isolated PostgreSQL restore.

Run in its own process with the explicit isolated-test guards below.  The
fixture never accepts a production-looking database name or the configured
production URL.
"""

from __future__ import annotations

import os
import re
import threading
import time
import unittest
import uuid
from datetime import datetime, timedelta, timezone
from urllib.parse import urlsplit

from scripts.meta_harness_migration import apply_to_isolated_restore


URL = os.environ.get("DEEPGRAPH_ISOLATED_POSTGRES_URL", "").strip()
ACK = os.environ.get("DEEPGRAPH_ALLOW_ISOLATED_INTEGRATION_TESTS") == "1"
SOURCE_COMMIT = os.environ.get("META_HARNESS_CANDIDATE_COMMIT", "").strip()
ISOLATED_MARKERS = ("test", "ci", "canary", "sandbox", "restore", "shadow")


def _safe_url() -> bool:
    if not URL or not ACK or not re.fullmatch(r"[0-9a-f]{40}", SOURCE_COMMIT):
        return False
    parsed = urlsplit(URL)
    database = parsed.path.lstrip("/").lower()
    return bool(
        parsed.scheme in {"postgres", "postgresql"}
        and any(marker in database for marker in ISOLATED_MARKERS)
        and URL != os.environ.get("DEEPGRAPH_DATABASE_URL", "").strip()
    )


@unittest.skipUnless(_safe_url(), "explicit isolated PostgreSQL process required")
class IsolatedAttemptGPUUsageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["DEEPGRAPH_DATABASE_URL"] = ""
        apply_to_isolated_restore(
            URL,
            source_commit=SOURCE_COMMIT,
            migration_key="0003_attempt_gpu_usage",
        )
        os.environ["DEEPGRAPH_DATABASE_URL"] = URL
        os.environ["DEEPGRAPH_PG_IDLE_IN_TRANSACTION_TIMEOUT_MS"] = "200"

        from db import database
        from meta_harness.attempt_gpu_usage import GrantGPUUsageControl

        if not database._use_pg() or database.DATABASE_URL.strip() != URL:  # noqa: SLF001
            raise RuntimeError("database module captured a non-isolated URL")
        cls.db = database
        cls.Control = GrantGPUUsageControl

    def setUp(self):
        self.namespace = f"attempt_gpu_{uuid.uuid4().hex}"
        with self.db.get_conn().cursor() as cur:
            cur.execute(
                """
                INSERT INTO research_agendas
                    (name, token_budget, status, backlog_policy,
                     gpu_hours_budget, gpu_hours_reserved,
                     backend_allowlist_json)
                VALUES (%s, 100, 'active', 'explicit_import_only',
                        1.5, 1.5, '["ssh_gpu"]')
                RETURNING id
                """,
                (self.namespace,),
            )
            self.agenda_id = int(cur.fetchone()["id"])
            cur.execute(
                """
                INSERT INTO frontier_packets
                    (agenda_id, retrieved_at, coverage_json, problem_status,
                     why_not_obsolete, minimum_falsification_experiment_json,
                     content_hash)
                VALUES (%s, CURRENT_TIMESTAMP, '{}', 'open', 'test only',
                        '{}', %s)
                RETURNING id
                """,
                (self.agenda_id, self.namespace),
            )
            frontier_id = int(cur.fetchone()["id"])
            cur.execute(
                """
                INSERT INTO deep_insights (tier, title, agenda_id)
                VALUES (1, %s, %s) RETURNING id
                """,
                (f"isolated test {self.namespace}", self.agenda_id),
            )
            self.idea_id = int(cur.fetchone()["id"])
            cur.execute(
                """
                INSERT INTO idea_decision_packets
                    (agenda_id, idea_id, frontier_packet_id, decision,
                     estimates_json, candidate_family, correlation_keys_json,
                     reason_codes_json, policy_version)
                VALUES (%s, %s, %s, 'promote', '{}', 'isolated-test',
                        '["isolated-test"]', '["isolated-test"]', 'test-v1')
                RETURNING id
                """,
                (self.agenda_id, self.idea_id, frontier_id),
            )
            decision_id = int(cur.fetchone()["id"])
            cur.execute(
                """
                INSERT INTO agenda_resource_ledger
                    (agenda_id, operation, idempotency_key, token_reserved,
                     gpu_hours_reserved, status)
                VALUES (%s, 'resource_grant', %s, 0, 1.5, 'reserved')
                RETURNING id
                """,
                (self.agenda_id, f"ledger:{self.namespace}"),
            )
            self.ledger_id = int(cur.fetchone()["id"])
            cur.execute(
                """
                INSERT INTO resource_grants
                    (agenda_id, idea_id, decision_packet_id, stage, token_cap,
                     max_gpu_hours, backend_allowlist_json,
                     artifact_requirements_json, expires_at, grant_reason,
                     reservation_id, status, idempotency_key)
                VALUES (%s, %s, %s, 'pilot', 0, 1.5, '["ssh_gpu"]',
                        '["final_results"]', %s, 'isolated test', %s,
                        'active', %s)
                RETURNING id
                """,
                (
                    self.agenda_id,
                    self.idea_id,
                    decision_id,
                    datetime.now(timezone.utc) + timedelta(hours=2),
                    self.ledger_id,
                    f"grant:{self.namespace}",
                ),
            )
            self.grant_id = int(cur.fetchone()["id"])
        self.db.commit()

    def tearDown(self):
        self.db.execute(
            "DELETE FROM colab_work_requests_v1 WHERE agenda_id=?",
            (self.agenda_id,),
        )
        self.db.execute(
            "UPDATE compute_jobs_v1 SET gpu_attempt_reservation_id=NULL WHERE agenda_id=?",
            (self.agenda_id,),
        )
        self.db.execute(
            "UPDATE gpu_jobs SET gpu_attempt_reservation_id=NULL WHERE agenda_id=?",
            (self.agenda_id,),
        )
        self.db.execute(
            "DELETE FROM experiment_attempt_gpu_reservations_v1 WHERE agenda_id=?",
            (self.agenda_id,),
        )
        self.db.execute("DELETE FROM compute_jobs_v1 WHERE agenda_id=?", (self.agenda_id,))
        self.db.execute("DELETE FROM gpu_jobs WHERE agenda_id=?", (self.agenda_id,))
        self.db.execute("DELETE FROM experiment_runs WHERE agenda_id=?", (self.agenda_id,))
        for table in (
            "resource_grant_usage_reservations",
            "resource_grants",
            "agenda_resource_ledger",
            "idea_decision_packets",
            "frontier_packets",
        ):
            self.db.execute(
                f"DELETE FROM {table} WHERE agenda_id=?",
                (self.agenda_id,),
            )
        self.db.execute("DELETE FROM deep_insights WHERE id=?", (self.idea_id,))
        self.db.execute("DELETE FROM research_agendas WHERE id=?", (self.agenda_id,))
        self.db.commit()

    def _reserve(self, key: str, timeout: int = 60):
        return self.Control().reserve_attempt(
            agenda_id=self.agenda_id,
            idea_id=self.idea_id,
            resource_grant_id=self.grant_id,
            attempt_key=f"{self.namespace}:{key}",
            backend_kind="ssh_gpu",
            requested_timeout_seconds=timeout,
        )

    def test_concurrent_admission_serializes_without_overbooking(self):
        barrier = threading.Barrier(3)
        results = []
        errors = []

        def reserve(key: str) -> None:
            try:
                barrier.wait()
                results.append(self._reserve(key, timeout=3600))
            except BaseException as exc:  # surfaced in the main test thread
                errors.append(exc)

        threads = [
            threading.Thread(target=reserve, args=("concurrent-a",)),
            threading.Thread(target=reserve, args=("concurrent-b",)),
        ]
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join(timeout=10)

        self.assertFalse(errors)
        self.assertEqual(len(results), 2)
        rows = self.db.fetchall(
            """
            SELECT reserved_gpu_seconds, timeout_seconds
            FROM experiment_attempt_gpu_reservations_v1
            WHERE resource_grant_id=? ORDER BY id
            """,
            (self.grant_id,),
        )
        self.db.commit()
        self.assertAlmostEqual(
            sum(float(row["reserved_gpu_seconds"]) for row in rows),
            1.5 * 3600.0,
        )
        self.assertEqual(sorted(int(row["timeout_seconds"]) for row in rows), [1790, 3600])

    def test_controller_restart_and_duplicate_settlement_charge_once(self):
        reservation = self._reserve("controller-restart")
        started = datetime.now(timezone.utc) - timedelta(seconds=20)
        self.Control().start_attempt(reservation.reservation_id, started_at=started)

        # A new object represents a restarted controller: all timing authority
        # must come from PostgreSQL, never process memory.
        first = self.Control().settle_attempt(
            reservation.reservation_id,
            completed_at=started + timedelta(seconds=17),
            reason_code="controller_lost",
        )
        second = self.Control().settle_attempt(
            reservation.reservation_id,
            completed_at=started + timedelta(seconds=40),
            reason_code="controller_lost",
        )
        ledger = self.db.fetchone(
            "SELECT gpu_hours_used, gpu_hours_overrun FROM agenda_resource_ledger WHERE id=?",
            (self.ledger_id,),
        )
        self.db.commit()

        self.assertEqual(first.actual_gpu_seconds, 17.0)
        self.assertEqual(second.actual_gpu_seconds, 17.0)
        self.assertAlmostEqual(float(ledger["gpu_hours_used"]), 17.0 / 3600.0)
        self.assertEqual(float(ledger["gpu_hours_overrun"]), 0.0)

    def test_long_gpu_wait_has_no_idle_transaction_and_still_settles(self):
        reservation = self._reserve("short-idle-timeout")
        running = self.Control().start_attempt(reservation.reservation_id)
        backend_pid = int(self.db.fetchone("SELECT pg_backend_pid() AS pid")["pid"])
        self.db.commit()

        time.sleep(0.35)
        import psycopg

        with psycopg.connect(URL, autocommit=True) as observer:
            state = observer.execute(
                "SELECT state FROM pg_stat_activity WHERE pid=%s", (backend_pid,)
            ).fetchone()[0]
        self.assertEqual(state, "idle")

        settled = self.Control().settle_attempt(
            reservation.reservation_id,
            completed_at=running.started_at + timedelta(seconds=1),
            reason_code="attempt_completed",
        )
        self.assertEqual(settled.actual_gpu_seconds, 1.0)

    def test_transport_and_contract_failures_all_settle_real_wall_time(self):
        for index, reason in enumerate(
            (
                "ssh_disconnected",
                "attempt_timeout",
                "cuda_oom",
                "metric_missing",
            ),
            start=1,
        ):
            with self.subTest(reason=reason):
                reservation = self._reserve(reason)
                started = datetime.now(timezone.utc) - timedelta(seconds=index)
                self.Control().start_attempt(
                    reservation.reservation_id,
                    started_at=started,
                )
                settled = self.Control().settle_attempt(
                    reservation.reservation_id,
                    completed_at=started + timedelta(seconds=index),
                    reason_code=reason,
                )
                self.assertEqual(settled.actual_gpu_seconds, float(index))
                self.assertEqual(settled.reason_code, reason)

    def test_exact_exhaustion_consumes_grant_and_settles_ledgers(self):
        cap_hours = 60.0 / 3600.0
        self.db.execute(
            "UPDATE resource_grants SET max_gpu_hours=? WHERE id=?",
            (cap_hours, self.grant_id),
        )
        self.db.execute(
            "UPDATE agenda_resource_ledger SET gpu_hours_reserved=? WHERE id=?",
            (cap_hours, self.ledger_id),
        )
        self.db.execute(
            """
            UPDATE research_agendas
            SET gpu_hours_budget=?, gpu_hours_reserved=? WHERE id=?
            """,
            (cap_hours, cap_hours, self.agenda_id),
        )
        self.db.commit()

        reservation = self._reserve("exact-exhaustion", timeout=60)
        started = datetime.now(timezone.utc) - timedelta(seconds=60)
        self.Control().start_attempt(reservation.reservation_id, started_at=started)
        self.Control().settle_attempt(
            reservation.reservation_id,
            completed_at=started + timedelta(seconds=60),
            reason_code="attempt_completed",
        )

        grant = self.db.fetchone("SELECT status FROM resource_grants WHERE id=?", (self.grant_id,))
        ledger = self.db.fetchone(
            "SELECT status, gpu_hours_used, gpu_hours_overrun FROM agenda_resource_ledger WHERE id=?",
            (self.ledger_id,),
        )
        agenda = self.db.fetchone(
            "SELECT gpu_hours_reserved, gpu_hours_spent FROM research_agendas WHERE id=?",
            (self.agenda_id,),
        )
        self.db.commit()
        self.assertEqual(grant["status"], "consumed")
        self.assertEqual(ledger["status"], "settled")
        self.assertAlmostEqual(float(ledger["gpu_hours_used"]), cap_hours)
        self.assertEqual(float(ledger["gpu_hours_overrun"]), 0.0)
        self.assertAlmostEqual(float(agenda["gpu_hours_reserved"]), 0.0)
        self.assertAlmostEqual(float(agenda["gpu_hours_spent"]), cap_hours)

    def test_legacy_terminal_import_rebuilds_all_resource_truths_once(self):
        cap_hours = 60.0 / 3600.0
        self.db.execute(
            "UPDATE resource_grants SET max_gpu_hours=? WHERE id=?",
            (cap_hours, self.grant_id),
        )
        self.db.execute(
            "UPDATE agenda_resource_ledger SET gpu_hours_reserved=? WHERE id=?",
            (cap_hours, self.ledger_id),
        )
        self.db.execute(
            """
            UPDATE research_agendas
            SET gpu_hours_budget=?, gpu_hours_reserved=? WHERE id=?
            """,
            (cap_hours, cap_hours, self.agenda_id),
        )
        base = datetime.now(timezone.utc) - timedelta(minutes=5)
        for index, seconds in enumerate((30, 40), start=1):
            key = f"{self.namespace}:legacy:{index}"
            gpu_job_id = self.db.insert_returning_id(
                """
                INSERT INTO gpu_jobs
                    (deep_insight_id, resource_class, gpu_count, timeout_s,
                     status, started_at, completed_at, agenda_id,
                     resource_grant_id, meta_harness_idempotency_key)
                VALUES (?, 'gpu_small', 1, 60, 'failed', ?, ?, ?, ?, ?)
                RETURNING id
                """,
                (
                    self.idea_id,
                    base.isoformat(),
                    (base + timedelta(seconds=seconds)).isoformat(),
                    self.agenda_id,
                    self.grant_id,
                    key,
                ),
            )
            self.db.execute(
                """
                INSERT INTO compute_jobs_v1
                    (agenda_id, idea_id, resource_grant_id, stage,
                     backend_kind, backend_job_id, idempotency_key,
                     command_ref, artifact_namespace, requested_gpu_hours,
                     timeout_seconds, status, timeout_at, usage_json)
                VALUES (?, ?, ?, 'pilot', 'ssh_gpu', ?, ?, ?, ?, ?, 60,
                        'failed', ?, ?)
                """,
                (
                    self.agenda_id,
                    self.idea_id,
                    self.grant_id,
                    f"legacy-gpu-job:{gpu_job_id}",
                    key,
                    f"experiment-run:{index}",
                    f"legacy/{index}",
                    cap_hours,
                    (base + timedelta(hours=1)).isoformat(),
                    '{"backend_report":{"source":"experiment_iterations.duration_seconds"}}',
                ),
            )
        self.db.commit()

        imported = self.Control().import_legacy_terminal_attempts()
        imported_again = self.Control().import_legacy_terminal_attempts()
        attempts = self.db.fetchall(
            """
            SELECT actual_gpu_seconds, reason_code
            FROM experiment_attempt_gpu_reservations_v1
            WHERE resource_grant_id=? ORDER BY id
            """,
            (self.grant_id,),
        )
        ledger = self.db.fetchone(
            """
            SELECT status, gpu_hours_used, gpu_hours_overrun
            FROM agenda_resource_ledger WHERE id=?
            """,
            (self.ledger_id,),
        )
        agenda = self.db.fetchone(
            """
            SELECT gpu_hours_reserved, gpu_hours_spent
            FROM research_agendas WHERE id=?
            """,
            (self.agenda_id,),
        )
        compute_rows = self.db.fetchall(
            "SELECT usage_json FROM compute_jobs_v1 WHERE agenda_id=? ORDER BY id",
            (self.agenda_id,),
        )
        self.db.commit()

        self.assertEqual(imported, 2)
        self.assertEqual(imported_again, 0)
        self.assertEqual([row["actual_gpu_seconds"] for row in attempts], [30.0, 40.0])
        self.assertTrue(
            all(str(row["reason_code"]).startswith("legacy_terminal_import:") for row in attempts)
        )
        self.assertEqual(ledger["status"], "settled")
        self.assertAlmostEqual(float(ledger["gpu_hours_used"]), cap_hours)
        self.assertAlmostEqual(float(ledger["gpu_hours_overrun"]), 10.0 / 3600.0)
        self.assertAlmostEqual(float(agenda["gpu_hours_reserved"]), 0.0)
        self.assertAlmostEqual(float(agenda["gpu_hours_spent"]), 70.0 / 3600.0)
        self.assertTrue(
            all("experiment_attempt_gpu_reservations_v1" in row["usage_json"] for row in compute_rows)
        )

    def test_colab_terminal_recovery_uses_same_attempt_ledger(self):
        self.db.execute(
            "UPDATE resource_grants SET backend_allowlist_json='[\"colab_gpu\"]' WHERE id=?",
            (self.grant_id,),
        )
        run_id = self.db.insert_returning_id(
            """
            INSERT INTO experiment_runs
                (deep_insight_id, agenda_id, resource_grant_id, status)
            VALUES (?, ?, ?, 'running_gpu') RETURNING id
            """,
            (self.idea_id, self.agenda_id, self.grant_id),
        )
        reservation = self.Control().reserve_attempt(
            agenda_id=self.agenda_id,
            idea_id=self.idea_id,
            resource_grant_id=self.grant_id,
            attempt_key=f"{self.namespace}:colab",
            backend_kind="colab_gpu",
            requested_timeout_seconds=60,
            experiment_run_id=run_id,
        )
        compute_job_id = self.db.insert_returning_id(
            """
            INSERT INTO compute_jobs_v1
                (agenda_id, idea_id, resource_grant_id, stage, backend_kind,
                 backend_job_id, idempotency_key, command_ref,
                 artifact_namespace, requested_gpu_hours, timeout_seconds,
                 status, timeout_at, gpu_attempt_reservation_id)
            VALUES (?, ?, ?, 'pilot', 'colab_gpu', 'pending', ?, ?, ?, ?, ?,
                    'running', ?, ?) RETURNING id
            """,
            (
                self.agenda_id,
                self.idea_id,
                self.grant_id,
                f"{self.namespace}:colab",
                f"colab-run:{run_id}",
                f"colab/{run_id}",
                reservation.reserved_gpu_seconds / 3600.0,
                reservation.timeout_seconds,
                (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat(),
                reservation.reservation_id,
            ),
        )
        self.db.execute(
            """
            UPDATE experiment_attempt_gpu_reservations_v1
            SET compute_job_id=? WHERE id=?
            """,
            (compute_job_id, reservation.reservation_id),
        )
        request_id = self.db.insert_returning_id(
            """
            INSERT INTO colab_work_requests_v1
                (agenda_id, idea_id, experiment_run_id, resource_grant_id,
                 compute_job_id, stage, idempotency_key, code_dir,
                 command_tokens_json, artifact_map_json,
                 artifact_output_dir, timeout_seconds, status)
            VALUES (?, ?, ?, ?, ?, 'pilot', ?, 'code', '[\"python\"]',
                    '{\"final_results\":\"final.json\"}', 'artifacts', ?,
                    'running') RETURNING id
            """,
            (
                self.agenda_id,
                self.idea_id,
                run_id,
                self.grant_id,
                compute_job_id,
                f"{self.namespace}:colab",
                reservation.timeout_seconds,
            ),
        )
        self.db.execute(
            "UPDATE compute_jobs_v1 SET backend_job_id=? WHERE id=?",
            (f"colab-work-request:{request_id}", compute_job_id),
        )
        self.db.commit()

        started = datetime.now(timezone.utc) - timedelta(seconds=10)
        self.Control().start_attempt(reservation.reservation_id, started_at=started)
        self.db.execute(
            """
            UPDATE colab_work_requests_v1
            SET status='failed', failure_reason='controller_lost', completed_at=?
            WHERE id=?
            """,
            ((started + timedelta(seconds=7)).isoformat(), request_id),
        )
        self.db.commit()

        pending = self.Control().reconcile_terminal_colab_attempts()
        usage = self.Control().usage_for_compute_job(compute_job_id)
        self.assertEqual(pending, [request_id])
        self.assertEqual(usage["wall_seconds"], 7.0)
        self.assertEqual(usage["reason_code"], "attempt_failed")


if __name__ == "__main__":
    unittest.main()
