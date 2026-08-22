"""An execution-stage grant that never got a run must be releasable.

The finalizer settles AGAINST a run, revoke_grant refuses a grant that already
metered usage, and expire_grant_now used to admit proposal and evidence_audit
only. A pilot grant whose forge died before creating the run therefore had no
path at all and held an agenda concurrency slot for the rest of a 24-hour TTL:
seven of them on 2026-08-21, the oldest holding 13.5 hours of a slot for work
that had already failed.

Half of what is asserted here is the refusal. Widening a deliberately narrow
gate is only safe while it still refuses every grant that has live work behind
it, so those cases outnumber the one that now passes.

Run through scripts/run_isolated_postgres_tests.sh.
"""

from __future__ import annotations

import os
import re
import unittest
import uuid
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
class OrphanedGrantExpiryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["DEEPGRAPH_DATABASE_URL"] = ""
        apply_to_isolated_restore(URL, source_commit=SOURCE_COMMIT)
        os.environ["DEEPGRAPH_DATABASE_URL"] = URL

        from db import database
        from meta_harness.repository import MetaHarnessRepository

        if not database._use_pg() or database.DATABASE_URL.strip() != URL:  # noqa: SLF001
            raise RuntimeError("database module captured a non-isolated URL")
        cls.db = database
        cls.Repository = MetaHarnessRepository

    def setUp(self):
        self.namespace = f"grant_expiry_{uuid.uuid4().hex}"
        self.grant_sequence = 0
        self.repo = self.Repository()
        with self.db.get_conn().cursor() as cur:
            cur.execute(
                """
                INSERT INTO research_agendas (name, token_budget, status, backlog_policy)
                VALUES (%s, 1000000, 'active', 'explicit_import_only')
                RETURNING id
                """,
                (self.namespace,),
            )
            self.agenda_id = int(cur.fetchone()["id"])
            cur.execute(
                "INSERT INTO deep_insights (agenda_id, tier, title) VALUES (%s, 2, %s) RETURNING id",
                (self.agenda_id, self.namespace),
            )
            self.idea_id = int(cur.fetchone()["id"])
            cur.execute(
                """
                INSERT INTO frontier_packets
                    (agenda_id, retrieved_at, coverage_json, problem_status,
                     why_not_obsolete, minimum_falsification_experiment_json,
                     content_hash)
                VALUES (%s, CURRENT_TIMESTAMP, '{}', 'open', 'test only', '{}', %s)
                RETURNING id
                """,
                (self.agenda_id, self.namespace),
            )
            frontier_id = int(cur.fetchone()["id"])
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
            self.decision_id = int(cur.fetchone()["id"])
        self.db.commit()

    def tearDown(self):
        for sql in (
            "DELETE FROM colab_work_requests_v1 WHERE agenda_id=?",
            "DELETE FROM compute_jobs_v1 WHERE agenda_id=?",
            "DELETE FROM experiment_runs WHERE agenda_id=?",
            "DELETE FROM resource_grant_usage_reservations WHERE agenda_id=?",
            "DELETE FROM resource_grants WHERE agenda_id=?",
            "DELETE FROM agenda_resource_ledger WHERE agenda_id=?",
            "DELETE FROM auto_research_jobs WHERE agenda_id=?",
            "DELETE FROM idea_decision_packets WHERE agenda_id=?",
            "DELETE FROM frontier_packets WHERE agenda_id=?",
            "DELETE FROM deep_insights WHERE agenda_id=?",
            "DELETE FROM research_agendas WHERE id=?",
        ):
            try:
                self.db.execute(sql, (self.agenda_id,))
            except Exception:
                self.db.rollback()
        self.db.commit()

    def _grant(
        self,
        stage: str,
        *,
        metered_tokens: int = 0,
        gpu_hours_reserved: float = 0.0,
        gpu_hours_used: float = 0.0,
    ) -> int:
        self.grant_sequence += 1
        suffix = f"{stage}:{self.grant_sequence}"
        with self.db.get_conn().cursor() as cur:
            cur.execute(
                """
                INSERT INTO agenda_resource_ledger
                    (agenda_id, operation, idempotency_key, token_reserved,
                     gpu_hours_reserved, gpu_hours_used, status)
                VALUES (%s, 'resource_grant', %s, 40000, %s, %s, 'reserved')
                RETURNING id
                """,
                (
                    self.agenda_id,
                    f"ledger:{self.namespace}:{suffix}",
                    gpu_hours_reserved,
                    gpu_hours_used,
                ),
            )
            reservation_id = int(cur.fetchone()["id"])
            cur.execute(
                """
                INSERT INTO resource_grants
                    (agenda_id, idea_id, decision_packet_id, stage, token_cap,
                     max_gpu_hours, backend_allowlist_json,
                     artifact_requirements_json, expires_at, grant_reason,
                     reservation_id, status, idempotency_key)
                VALUES (%s, %s, %s, %s, 40000, %s, '["cpu"]', '["raw_metrics"]',
                        CURRENT_TIMESTAMP + INTERVAL '24 hours',
                        'isolated test', %s, 'active', %s)
                RETURNING id
                """,
                (
                    self.agenda_id,
                    self.idea_id,
                    self.decision_id,
                    stage,
                    gpu_hours_reserved,
                    reservation_id,
                    f"grant:{self.namespace}:{suffix}",
                ),
            )
            grant_id = int(cur.fetchone()["id"])
            cur.execute(
                """
                UPDATE research_agendas
                SET token_reserved=token_reserved+40000,
                    gpu_hours_reserved=gpu_hours_reserved+%s,
                    gpu_hours_spent=gpu_hours_spent+%s
                WHERE id=%s
                """,
                (
                    max(0.0, gpu_hours_reserved - gpu_hours_used),
                    gpu_hours_used,
                    self.agenda_id,
                ),
            )
            if metered_tokens:
                cur.execute(
                    """
                    INSERT INTO resource_grant_usage_reservations
                        (agenda_id, resource_grant_id, operation,
                         idempotency_key, token_reserved, tokens_used, status,
                         settled_at)
                    VALUES (%s, %s, 'forge_scaffold', %s, %s, %s, 'settled',
                            CURRENT_TIMESTAMP)
                    """,
                    (
                        self.agenda_id,
                        grant_id,
                        f"attempt:{self.namespace}:{suffix}",
                        metered_tokens,
                        metered_tokens,
                    ),
                )
        self.db.commit()
        return grant_id

    def _status(self, grant_id: int) -> str:
        row = self.db.fetchone(
            "SELECT status FROM resource_grants WHERE id=?", (grant_id,)
        )
        self.db.commit()
        return str(dict(row or {}).get("status") or "")

    # --- the case that now has a path -----------------------------------

    def test_metered_pilot_grant_with_no_run_can_be_expired(self):
        grant_id = self._grant("pilot", metered_tokens=40000)

        with self.assertRaises(Exception):
            # Withdrawal is still refused: it would erase a real spend.
            self.repo.revoke_grant(
                grant_id, agenda_id=self.agenda_id, reason="orphaned"
            )

        self.assertTrue(
            self.repo.expire_grant_now(
                grant_id, agenda_id=self.agenda_id, reason="orphaned pilot"
            )
        )
        self.assertEqual(self._status(grant_id), "expired")

    def test_expiry_keeps_the_spend_and_writes_no_outcome(self):
        grant_id = self._grant("pilot", metered_tokens=40000)

        self.repo.expire_grant_now(
            grant_id, agenda_id=self.agenda_id, reason="orphaned pilot"
        )

        settled = self.db.fetchone(
            """
            SELECT count(*) AS n, coalesce(sum(tokens_used), 0) AS tokens
            FROM resource_grant_usage_reservations
            WHERE resource_grant_id=? AND status='settled'
            """,
            (grant_id,),
        )
        self.db.commit()
        self.assertEqual(int(dict(settled)["n"]), 1)
        self.assertEqual(int(dict(settled)["tokens"]), 40000)

        outcomes = self.db.fetchone(
            "SELECT count(*) AS n FROM outcome_records WHERE resource_grant_id=?",
            (grant_id,),
        )
        self.db.commit()
        self.assertEqual(int(dict(outcomes)["n"]), 0)

    def test_expiry_settles_child_tokens_and_releases_only_unused_balance(self):
        grant_id = self._grant("pilot", metered_tokens=12500)

        self.repo.expire_grant_now(
            grant_id, agenda_id=self.agenda_id, reason="orphaned pilot"
        )

        agenda = self.db.fetchone(
            "SELECT token_reserved, token_spent FROM research_agendas WHERE id=?",
            (self.agenda_id,),
        )
        ledger = self.db.fetchone(
            """
            SELECT token_reserved, tokens_used, status, release_reason
            FROM agenda_resource_ledger
            WHERE id=(SELECT reservation_id FROM resource_grants WHERE id=?)
            """,
            (grant_id,),
        )
        self.db.commit()

        self.assertEqual(int(agenda["token_reserved"]), 0)
        self.assertEqual(int(agenda["token_spent"]), 12500)
        self.assertEqual(int(ledger["token_reserved"]), 40000)
        self.assertEqual(int(ledger["tokens_used"]), 12500)
        self.assertEqual(ledger["status"], "settled")
        self.assertEqual(ledger["release_reason"], "grant_expired")

    def test_three_metered_expiries_then_unused_revoke_clamp_agenda14_drift(self):
        expired_ids = [
            self._grant(
                "pilot",
                metered_tokens=30000,
                gpu_hours_reserved=4.0,
                gpu_hours_used=0.0,
            )
            for _ in range(3)
        ]
        unused_id = self._grant(
            "pilot",
            gpu_hours_reserved=4.0,
            gpu_hours_used=0.0,
        )
        self.db.execute(
            """
            UPDATE resource_grants
            SET expires_at=CURRENT_TIMESTAMP - INTERVAL '1 minute'
            WHERE id = ANY(?)
            """,
            (expired_ids,),
        )
        # Reproduce agenda 14: metered grants 262/266/271 expire first and
        # release 12 hours.  Unused grant 299 still claims four, while the
        # surviving agenda aggregate contains only 3.4813 of them.
        self.db.execute(
            "UPDATE research_agendas SET gpu_hours_reserved=? WHERE id=?",
            (15.4813097814, self.agenda_id),
        )
        self.db.commit()

        self.assertEqual(
            self.repo.reconcile_expired_grants(agenda_id=self.agenda_id), 3
        )
        self.assertEqual(
            self.repo.reconcile_expired_grants(agenda_id=self.agenda_id), 0
        )
        self.assertTrue(
            self.repo.revoke_grant(
                unused_id,
                agenda_id=self.agenda_id,
                reason="agenda14 closed with stale aggregate",
            )
        )

        agenda = self.db.fetchone(
            """
            SELECT token_reserved, token_spent,
                   gpu_hours_reserved, gpu_hours_spent
            FROM research_agendas WHERE id=?
            """,
            (self.agenda_id,),
        )
        ledgers = self.db.fetchall(
            """
            SELECT tokens_used, gpu_hours_reserved, gpu_hours_used,
                   status, release_reason
            FROM agenda_resource_ledger
            WHERE id IN (
                SELECT reservation_id FROM resource_grants WHERE id = ANY(?)
            )
            ORDER BY id
            """,
            (expired_ids + [unused_id],),
        )
        grants = self.db.fetchall(
            "SELECT status FROM resource_grants WHERE id = ANY(?) ORDER BY id",
            (expired_ids + [unused_id],),
        )
        self.db.commit()

        self.assertEqual(int(agenda["token_reserved"]), 0)
        self.assertEqual(int(agenda["token_spent"]), 90000)
        self.assertAlmostEqual(float(agenda["gpu_hours_reserved"]), 0.0, places=9)
        self.assertAlmostEqual(float(agenda["gpu_hours_spent"]), 0.0, places=9)
        self.assertEqual(
            [int(row["tokens_used"] or 0) for row in ledgers],
            [30000, 30000, 30000, 0],
        )
        # Preserve the ledger's measured truth. The aggregate shortfall is
        # disclosed, not hidden by inventing 0.5187 used GPU-hours.
        self.assertEqual([float(row["gpu_hours_used"]) for row in ledgers], [0.0] * 4)
        self.assertEqual(
            [row["status"] for row in ledgers],
            ["settled", "settled", "settled", "released"],
        )
        self.assertEqual(
            [row["status"] for row in grants],
            ["expired", "expired", "expired", "revoked"],
        )
        drift_reasons = [
            str(row["release_reason"])
            for row in ledgers
            if "agenda_gpu_reservation_shortfall_hours="
            in str(row["release_reason"])
        ]
        self.assertEqual(len(drift_reasons), 1)
        self.assertIn("grant_revoked:agenda14", drift_reasons[0])
        self.assertIn("0.5186902186", drift_reasons[0])

    def test_reconcile_clamps_and_discloses_token_reservation_drift(self):
        grant_id = self._grant("pilot", metered_tokens=30000)
        self.db.execute(
            """
            UPDATE resource_grants
            SET expires_at=CURRENT_TIMESTAMP - INTERVAL '1 minute'
            WHERE id=?
            """,
            (grant_id,),
        )
        self.db.execute(
            "UPDATE research_agendas SET token_reserved=25000 WHERE id=?",
            (self.agenda_id,),
        )
        self.db.commit()

        self.assertEqual(
            self.repo.reconcile_expired_grants(agenda_id=self.agenda_id), 1
        )

        agenda = self.db.fetchone(
            "SELECT token_reserved, token_spent FROM research_agendas WHERE id=?",
            (self.agenda_id,),
        )
        ledger = self.db.fetchone(
            """
            SELECT tokens_used, release_reason
            FROM agenda_resource_ledger
            WHERE id=(SELECT reservation_id FROM resource_grants WHERE id=?)
            """,
            (grant_id,),
        )
        self.db.commit()

        self.assertEqual(int(agenda["token_reserved"]), 0)
        self.assertEqual(int(agenda["token_spent"]), 30000)
        self.assertEqual(int(ledger["tokens_used"]), 30000)
        self.assertIn(
            "agenda_token_reservation_shortfall=15000",
            str(ledger["release_reason"]),
        )

    # --- and the cases it must still refuse ------------------------------

    def test_refuses_a_pilot_grant_that_has_a_run(self):
        grant_id = self._grant("pilot", metered_tokens=40000)
        self.db.execute(
            """
            INSERT INTO experiment_runs
                (agenda_id, deep_insight_id, resource_grant_id, status)
            VALUES (?, ?, ?, 'failed')
            """,
            (self.agenda_id, self.idea_id, grant_id),
        )
        self.db.commit()

        self.assertFalse(
            self.repo.expire_grant_now(
                grant_id, agenda_id=self.agenda_id, reason="should be refused"
            )
        )
        self.assertEqual(self._status(grant_id), "active")

    def test_refuses_a_pilot_grant_with_a_live_compute_job(self):
        grant_id = self._grant("pilot", metered_tokens=40000)
        self.db.execute(
            """
            INSERT INTO compute_jobs_v1
                (agenda_id, idea_id, resource_grant_id, stage, backend_kind,
                 backend_job_id, idempotency_key, command_ref,
                 artifact_namespace, requested_gpu_hours, timeout_seconds,
                 status, timeout_at)
            VALUES (?, ?, ?, 'pilot', 'cpu', ?, ?, 'run:test', 'isolated',
                    0, 60, 'running', CURRENT_TIMESTAMP + INTERVAL '1 hour')
            """,
            (
                self.agenda_id,
                self.idea_id,
                grant_id,
                f"cpu:{self.namespace}",
                f"compute:{self.namespace}",
            ),
        )
        self.db.commit()

        self.assertFalse(
            self.repo.expire_grant_now(
                grant_id, agenda_id=self.agenda_id, reason="should be refused"
            )
        )
        self.assertEqual(self._status(grant_id), "active")

    def test_refuses_a_pilot_grant_with_a_live_colab_request(self):
        """A request can name a grant its run does not, so check both.

        colab_work_requests_v1.experiment_run_id is NOT NULL, so a live request
        always has some run behind it -- but that run carries its own
        resource_grant_id, which need not be this grant. The run condition
        alone would then miss it, which is why the request is checked too. The
        run here belongs to a different grant on purpose.
        """
        other_grant_id = self._grant("full_benchmark")
        grant_id = self._grant("pilot", metered_tokens=40000)
        run_id = self.db.insert_returning_id(
            """
            INSERT INTO experiment_runs
                (agenda_id, deep_insight_id, resource_grant_id, status)
            VALUES (?, ?, ?, 'testing')
            RETURNING id
            """,
            (self.agenda_id, self.idea_id, other_grant_id),
        )
        self.db.execute(
            """
            INSERT INTO colab_work_requests_v1
                (agenda_id, idea_id, resource_grant_id, experiment_run_id,
                 stage, status, idempotency_key, code_dir,
                 command_tokens_json, artifact_map_json, artifact_output_dir,
                 timeout_seconds)
            VALUES (?, ?, ?, ?, 'pilot', 'queued', ?, '/isolated/code',
                    '["python","train.py"]', '{}', '/isolated/out', 60)
            """,
            (
                self.agenda_id,
                self.idea_id,
                grant_id,
                run_id,
                f"colab:{self.namespace}",
            ),
        )
        self.db.commit()

        self.assertFalse(
            self.repo.expire_grant_now(
                grant_id, agenda_id=self.agenda_id, reason="should be refused"
            )
        )
        self.assertEqual(self._status(grant_id), "active")

    def test_terminal_compute_job_does_not_keep_the_grant_alive(self):
        """A finished attempt is not live work; only a live one blocks."""
        grant_id = self._grant("pilot", metered_tokens=40000)
        self.db.execute(
            """
            INSERT INTO compute_jobs_v1
                (agenda_id, idea_id, resource_grant_id, stage, backend_kind,
                 backend_job_id, idempotency_key, command_ref,
                 artifact_namespace, requested_gpu_hours, timeout_seconds,
                 status, timeout_at)
            VALUES (?, ?, ?, 'pilot', 'cpu', ?, ?, 'run:test', 'isolated',
                    0, 60, 'failed', CURRENT_TIMESTAMP + INTERVAL '1 hour')
            """,
            (
                self.agenda_id,
                self.idea_id,
                grant_id,
                f"cpu:{self.namespace}",
                f"compute:{self.namespace}",
            ),
        )
        self.db.commit()

        self.assertTrue(
            self.repo.expire_grant_now(
                grant_id, agenda_id=self.agenda_id, reason="orphaned pilot"
            )
        )

    def test_proposal_stage_is_unchanged_by_the_widening(self):
        grant_id = self._grant("proposal", metered_tokens=32000)

        self.assertTrue(
            self.repo.expire_grant_now(
                grant_id, agenda_id=self.agenda_id, reason="exhausted proposal"
            )
        )
        self.assertEqual(self._status(grant_id), "expired")

    def test_a_reason_is_still_required(self):
        grant_id = self._grant("pilot", metered_tokens=40000)
        with self.assertRaises(Exception):
            self.repo.expire_grant_now(grant_id, agenda_id=self.agenda_id, reason="  ")
        self.assertEqual(self._status(grant_id), "active")

    def test_refuses_another_agendas_grant(self):
        grant_id = self._grant("pilot", metered_tokens=40000)
        self.assertFalse(
            self.repo.expire_grant_now(
                grant_id, agenda_id=self.agenda_id + 10_000, reason="wrong scope"
            )
        )
        self.assertEqual(self._status(grant_id), "active")


if __name__ == "__main__":
    unittest.main()
