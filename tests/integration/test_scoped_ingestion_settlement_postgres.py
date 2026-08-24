"""Real PostgreSQL proof for scoped-ingestion atomic grant settlement.

Run through ``scripts/run_isolated_postgres_tests.sh``.  The module refuses
any database that is not explicitly marked as disposable.
"""

from __future__ import annotations

import os
import re
import unittest
import uuid
from urllib.parse import urlsplit


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
class ScopedIngestionSettlementPostgresTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["DEEPGRAPH_DATABASE_URL"] = URL

        from db import database
        from meta_harness.ingestion_queue import (
            ScopedIngestionRepository,
            ScopedIngestionUsageDispositionRequest,
        )

        if not database._use_pg() or database.DATABASE_URL.strip() != URL:  # noqa: SLF001
            raise RuntimeError("database module captured a non-isolated URL")
        cls.db = database
        cls.Repository = ScopedIngestionRepository
        cls.UsageDispositionRequest = ScopedIngestionUsageDispositionRequest

    def setUp(self):
        self.namespace = f"ingestion_settlement_{uuid.uuid4().hex}"
        self.paper_id = f"test-{uuid.uuid4().hex}"
        self.worker_id = "isolated-test:1:scoped-ingestion"
        with self.db.get_conn().cursor() as cur:
            cur.execute(
                """
                INSERT INTO research_agendas
                    (name, token_budget, token_spent, token_reserved,
                     status, backlog_policy)
                VALUES (%s, 1000, 0, 100, 'active', 'explicit_import_only')
                RETURNING id
                """,
                (self.namespace,),
            )
            self.agenda_id = int(cur.fetchone()["id"])
            cur.execute(
                """
                INSERT INTO deep_insights (agenda_id, tier, title)
                VALUES (%s, 2, %s) RETURNING id
                """,
                (self.agenda_id, self.namespace),
            )
            self.idea_id = int(cur.fetchone()["id"])
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
                INSERT INTO idea_decision_packets
                    (agenda_id, idea_id, frontier_packet_id, decision,
                     estimates_json, candidate_family, correlation_keys_json,
                     reason_codes_json, policy_version)
                VALUES (%s, %s, %s, 'promote', '{}', 'isolated-test',
                        '[]', '[]', 'test-v1')
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
                VALUES (%s, 'scoped_ingestion', %s, 100, 0, 'reserved')
                RETURNING id
                """,
                (self.agenda_id, f"ledger:{self.namespace}"),
            )
            reservation_id = int(cur.fetchone()["id"])
            cur.execute(
                """
                INSERT INTO resource_grants
                    (agenda_id, idea_id, decision_packet_id, stage, token_cap,
                     max_gpu_hours, backend_allowlist_json,
                     artifact_requirements_json, expires_at, grant_reason,
                     reservation_id, status, idempotency_key)
                VALUES (%s, %s, %s, 'ingestion_backfill_canary', 100, 0,
                        '["llm"]', '[]', CURRENT_TIMESTAMP + INTERVAL '1 hour',
                        'isolated test', %s, 'active', %s)
                RETURNING id
                """,
                (
                    self.agenda_id,
                    self.idea_id,
                    decision_id,
                    reservation_id,
                    f"grant:{self.namespace}",
                ),
            )
            self.grant_id = int(cur.fetchone()["id"])
            cur.execute(
                """
                INSERT INTO papers (id, title, status, processing_stage)
                VALUES (%s, %s, 'reasoned', 'reasoned')
                """,
                (self.paper_id, self.namespace),
            )
            cur.execute(
                """
                INSERT INTO claims (paper_id, claim_text, claim_type)
                VALUES (%s, 'isolated grounded claim', 'finding')
                """,
                (self.paper_id,),
            )
            self.entity_ids = (
                f"{self.namespace}:entity-a",
                f"{self.namespace}:entity-b",
            )
            for entity_id, name in zip(self.entity_ids, ("entity a", "entity b")):
                cur.execute(
                    """
                    INSERT INTO graph_entities
                        (id, canonical_name, entity_type, normalized_name)
                    VALUES (%s, %s, 'concept', %s)
                    """,
                    (entity_id, name, name),
                )
                cur.execute(
                    """
                    INSERT INTO paper_entity_mentions
                        (paper_id, entity_id, mention_text, mention_role)
                    VALUES (%s, %s, %s, 'subject')
                    """,
                    (self.paper_id, entity_id, name),
                )
            cur.execute(
                """
                INSERT INTO graph_relations
                    (paper_id, subject_entity_id, predicate, object_entity_id)
                VALUES (%s, %s, 'supports', %s)
                """,
                (self.paper_id, self.entity_ids[0], self.entity_ids[1]),
            )
            for stage, payload in (
                ("extracted", '{"claims":[{"text":"grounded"}]}'),
                (
                    "graph_written",
                    '{"claim_count":1,"graph_entities":2,"graph_relations":1}',
                ),
                ("reasoned", '{"claims":1}'),
            ):
                cur.execute(
                    """
                    INSERT INTO paper_stage_checkpoints
                        (paper_id, stage, payload)
                    VALUES (%s, %s, %s)
                    """,
                    (self.paper_id, stage, payload),
                )
            cur.execute(
                """
                INSERT INTO resource_grant_usage_reservations
                    (agenda_id, resource_grant_id, operation, idempotency_key,
                     token_reserved, tokens_used, status, settled_at)
                VALUES (%s, %s, 'claims_methods', %s, 80, 60, 'settled',
                        CURRENT_TIMESTAMP)
                """,
                (
                    self.agenda_id,
                    self.grant_id,
                    f"usage:{self.namespace}",
                ),
            )
            cur.execute(
                """
                INSERT INTO scoped_ingestion_jobs_v1
                    (agenda_id, idea_id, resource_grant_id, stage,
                     idempotency_key, paper_ids_json, status, max_attempts,
                     attempt_count, lease_owner,
                     lease_expires_at, started_at)
                VALUES (%s, %s, %s, 'ingestion_backfill_canary', %s, %s,
                        'running', 3, 1, %s,
                        CURRENT_TIMESTAMP + INTERVAL '10 minutes',
                        CURRENT_TIMESTAMP)
                RETURNING id
                """,
                (
                    self.agenda_id,
                    self.idea_id,
                    self.grant_id,
                    f"job:{self.namespace}",
                    f'["{self.paper_id}"]',
                    self.worker_id,
                ),
            )
            self.job_id = int(cur.fetchone()["id"])
        self.db.commit()

    def tearDown(self):
        for sql, value in (
            (
                "DELETE FROM experiment_attempt_gpu_reservations_v1 "
                "WHERE agenda_id=?",
                self.agenda_id,
            ),
            ("DELETE FROM scoped_ingestion_jobs_v1 WHERE agenda_id=?", self.agenda_id),
            (
                "DELETE FROM resource_grant_usage_reservations WHERE agenda_id=?",
                self.agenda_id,
            ),
            ("DELETE FROM resource_grants WHERE agenda_id=?", self.agenda_id),
            ("DELETE FROM agenda_resource_ledger WHERE agenda_id=?", self.agenda_id),
            ("DELETE FROM idea_decision_packets WHERE agenda_id=?", self.agenda_id),
            ("DELETE FROM frontier_packets WHERE agenda_id=?", self.agenda_id),
            ("DELETE FROM deep_insights WHERE agenda_id=?", self.agenda_id),
            ("DELETE FROM paper_stage_checkpoints WHERE paper_id=?", self.paper_id),
            ("DELETE FROM graph_relations WHERE paper_id=?", self.paper_id),
            ("DELETE FROM paper_entity_mentions WHERE paper_id=?", self.paper_id),
            ("DELETE FROM claims WHERE paper_id=?", self.paper_id),
            ("DELETE FROM graph_entities WHERE id = ANY(?)", list(self.entity_ids)),
            ("DELETE FROM papers WHERE id=?", self.paper_id),
            ("DELETE FROM research_agendas WHERE id=?", self.agenda_id),
        ):
            self.db.execute(sql, (value,))
        self.db.commit()

    def _states(self) -> dict:
        job = self.db.fetchone(
            "SELECT status, result_json FROM scoped_ingestion_jobs_v1 WHERE id=?",
            (self.job_id,),
        )
        grant = self.db.fetchone(
            "SELECT status FROM resource_grants WHERE id=?", (self.grant_id,)
        )
        ledger = self.db.fetchone(
            """
            SELECT status, tokens_used
            FROM agenda_resource_ledger
            WHERE id=(SELECT reservation_id FROM resource_grants WHERE id=?)
            """,
            (self.grant_id,),
        )
        agenda = self.db.fetchone(
            """
            SELECT token_reserved, token_spent
            FROM research_agendas WHERE id=?
            """,
            (self.agenda_id,),
        )
        self.db.commit()
        return {
            "job": dict(job or {}),
            "grant": dict(grant or {}),
            "ledger": dict(ledger or {}),
            "agenda": dict(agenda or {}),
        }

    def test_success_and_duplicate_are_atomically_settled_and_idempotent(self):
        lifecycle_result = {
            "paper_id": self.paper_id,
            "claims": 1,
            "graph_entities": 2,
            "graph_relations": 1,
        }
        result = self.Repository().complete(
            self.job_id,
            agenda_id=self.agenda_id,
            worker_id=self.worker_id,
            results=[lifecycle_result],
        )
        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(result["tokens_used"], 60)
        states = self._states()
        self.assertEqual(states["job"]["status"], "succeeded")
        self.assertEqual(states["grant"]["status"], "consumed")
        self.assertEqual(states["ledger"], {"status": "settled", "tokens_used": 60})
        self.assertEqual(states["agenda"], {"token_reserved": 0, "token_spent": 60})

        duplicate = self.Repository().complete(
            self.job_id,
            agenda_id=self.agenda_id,
            worker_id=self.worker_id,
            results=[lifecycle_result],
        )
        self.assertEqual(duplicate["status"], "already_succeeded")
        self.assertEqual(self._states(), states)

    def test_budget_guard_failure_rolls_back_the_prior_job_write(self):
        self.db.execute(
            "UPDATE research_agendas SET token_budget=50 WHERE id=?",
            (self.agenda_id,),
        )
        self.db.commit()

        with self.assertRaisesRegex(Exception, "settle_ingestion_agenda_budget"):
            self.Repository().complete(
                self.job_id,
                agenda_id=self.agenda_id,
                worker_id=self.worker_id,
                results=[{
                    "paper_id": self.paper_id,
                    "claims": 1,
                    "graph_entities": 2,
                    "graph_relations": 1,
                }],
            )

        states = self._states()
        self.assertEqual(states["job"]["status"], "running")
        self.assertEqual(states["grant"]["status"], "active")
        self.assertEqual(states["ledger"]["status"], "reserved")
        self.assertEqual(states["agenda"], {"token_reserved": 100, "token_spent": 0})

    def test_non_reasoned_paper_refuses_settlement_before_any_write(self):
        self.db.execute(
            """
            UPDATE papers SET status='extracted', processing_stage='extracted'
            WHERE id=?
            """,
            (self.paper_id,),
        )
        self.db.commit()

        with self.assertRaisesRegex(Exception, "durably reasoned"):
            self.Repository().complete(
                self.job_id,
                agenda_id=self.agenda_id,
                worker_id=self.worker_id,
                results=[{
                    "paper_id": self.paper_id,
                    "claims": 1,
                    "graph_entities": 2,
                    "graph_relations": 1,
                }],
            )

        states = self._states()
        self.assertEqual(states["job"]["status"], "running")
        self.assertEqual(states["grant"]["status"], "active")
        self.assertEqual(states["ledger"]["status"], "reserved")
        self.assertEqual(states["agenda"], {"token_reserved": 100, "token_spent": 0})

    def test_exact_unbilled_disposition_then_retry_and_completion_has_no_orphan(self):
        self.db.execute(
            """
            UPDATE scoped_ingestion_jobs_v1
            SET status='manual_reconciliation', lease_owner=NULL,
                lease_expires_at=NULL,
                failure_reason='provider_delivery_unknown'
            WHERE id=?
            """,
            (self.job_id,),
        )
        usage_id = self.db.insert_returning_id(
            """
            INSERT INTO resource_grant_usage_reservations
                (agenda_id, resource_grant_id, operation, idempotency_key,
                 token_reserved, status)
            VALUES (?, ?, 'claims_methods_retry', ?, 20, 'reserved')
            RETURNING id
            """,
            (self.agenda_id, self.grant_id, f"crash:{self.namespace}:t1"),
        )
        self.db.commit()

        disposed = self.Repository().dispose_open_usage(
            self.UsageDispositionRequest(
                job_id=self.job_id,
                agenda_id=self.agenda_id,
                idea_id=self.idea_id,
                resource_grant_id=self.grant_id,
                stage="ingestion_backfill_canary",
                paper_ids=(self.paper_id,),
                usage_reservation_id=usage_id,
                operation="claims_methods_retry",
                usage_idempotency_key=f"crash:{self.namespace}:t1",
                expected_token_reserved=20,
                disposition="release_confirmed_unbilled",
                tokens_used=None,
                cost_usd=None,
                actor="isolated-test-operator",
                reason="provider confirms request was not accepted",
                evidence_ref=f"isolated-provider-log:{self.namespace}:missing",
                operator_request_id=f"dispose:{self.namespace}",
                resume=True,
            )
        )
        self.assertEqual(disposed["job_status"], "retryable")
        claimed = self.Repository().claim_next(
            worker_id=self.worker_id, lease_seconds=600
        )
        self.assertEqual(int(claimed["id"]), self.job_id)

        completed = self.Repository().complete(
            self.job_id,
            agenda_id=self.agenda_id,
            worker_id=self.worker_id,
            results=[{
                "paper_id": self.paper_id,
                "claims": 1,
                "graph_entities": 2,
                "graph_relations": 1,
            }],
        )
        self.assertEqual(completed["status"], "succeeded")
        self.assertEqual(
            self.db.fetchone(
                """
                SELECT COUNT(*) AS count
                FROM resource_grant_usage_reservations
                WHERE resource_grant_id=? AND status='reserved'
                """,
                (self.grant_id,),
            )["count"],
            0,
        )
        self.assertEqual(self._states()["grant"]["status"], "consumed")


if __name__ == "__main__":
    unittest.main()
