"""Closed-agenda manuscript recovery against an isolated PostgreSQL only."""

from __future__ import annotations

import hashlib
import json
import os
import re
import unittest
import uuid
from datetime import datetime, timedelta, timezone
from urllib.parse import urlsplit
from unittest.mock import patch

from scripts.meta_harness_migration import apply_to_isolated_restore


URL = os.environ.get("DEEPGRAPH_ISOLATED_POSTGRES_URL", "").strip()
ACK = os.environ.get("DEEPGRAPH_ALLOW_ISOLATED_INTEGRATION_TESTS") == "1"
SOURCE_COMMIT = os.environ.get("META_HARNESS_CANDIDATE_COMMIT", "").strip()
ISOLATED_MARKERS = ("test", "ci", "canary", "sandbox", "restore", "shadow")


def _safe_url() -> bool:
    parsed = urlsplit(URL)
    return bool(
        URL
        and ACK
        and re.fullmatch(r"[0-9a-f]{40}", SOURCE_COMMIT)
        and parsed.scheme in {"postgres", "postgresql"}
        and any(marker in parsed.path.lower() for marker in ISOLATED_MARKERS)
        and URL != os.environ.get("DEEPGRAPH_DATABASE_URL", "").strip()
    )


@unittest.skipUnless(_safe_url(), "explicit isolated PostgreSQL process required")
class ManuscriptRecoveryPostgresTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["DEEPGRAPH_DATABASE_URL"] = ""
        apply_to_isolated_restore(
            URL,
            source_commit=SOURCE_COMMIT,
            migration_key="0008_manuscript_gate_records",
        )
        os.environ["DEEPGRAPH_DATABASE_URL"] = URL
        from db import database as database
        from agents.llm_client import configured_role_prompt_version
        from meta_harness.grant_usage import GrantUsageLedger
        from meta_harness.llm_routing import RouteObservation
        from meta_harness.manuscript_gate import (
            MANUSCRIPT_REVIEWER_KEY_ID,
            _finish_terminal_result,
            run_manuscript_gate,
        )
        from meta_harness.repository import MetaHarnessRepository

        cls.db = database
        cls.Ledger = GrantUsageLedger
        cls.Observation = RouteObservation
        cls.Repository = MetaHarnessRepository
        cls.finish_terminal = staticmethod(_finish_terminal_result)
        cls.run_gate = staticmethod(run_manuscript_gate)
        cls.prompt_ref = configured_role_prompt_version("reviewer")
        cls.reviewer_key_id = MANUSCRIPT_REVIEWER_KEY_ID

    def setUp(self):
        self.namespace = f"manuscript_recovery_{uuid.uuid4().hex}"
        self.verdict_hash = hashlib.sha256(self.namespace.encode()).hexdigest()
        with self.db.get_conn().cursor() as cur:
            cur.execute(
                """
                INSERT INTO research_agendas
                    (name, token_budget, token_spent, token_reserved, status,
                     is_active, max_concurrency, backend_allowlist_json,
                     backlog_policy)
                VALUES (%s, 100000, 1085, 0, 'closed', 0, 2, '["llm"]',
                        'explicit_import_only') RETURNING id
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
                VALUES (%s, CURRENT_TIMESTAMP, '{}', 'open', 'test', '{}', %s)
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
                VALUES (%s, %s, %s, 'promote', '{}', 'test', '["test"]',
                        '["test"]', 'test-v1') RETURNING id
                """,
                (self.agenda_id, self.idea_id, frontier_id),
            )
            self.packet_id = int(cur.fetchone()["id"])
            cur.execute(
                """
                INSERT INTO agenda_resource_ledger
                    (agenda_id, operation, idempotency_key, token_reserved,
                     tokens_used, gpu_hours_reserved, gpu_hours_used, status,
                     settled_at)
                VALUES (%s, 'resource_grant', %s, 40000, 1085, 0, 0,
                        'settled', CURRENT_TIMESTAMP) RETURNING id
                """,
                (self.agenda_id, f"old-ledger:{self.namespace}"),
            )
            self.old_ledger_id = int(cur.fetchone()["id"])
            cur.execute(
                """
                INSERT INTO resource_grants
                    (agenda_id, idea_id, decision_packet_id, stage, token_cap,
                     gpu_class, max_gpu_hours, backend_allowlist_json,
                     artifact_requirements_json, expires_at, grant_reason,
                     reservation_id, status, idempotency_key)
                VALUES (%s, %s, %s, 'evidence_audit', 40000, 'none', 0,
                        '["llm"]', '["claim_ledger"]',
                        CURRENT_TIMESTAMP + INTERVAL '1 hour', 'old audit', %s,
                        'consumed', %s) RETURNING id
                """,
                (
                    self.agenda_id,
                    self.idea_id,
                    self.packet_id,
                    self.old_ledger_id,
                    f"old-grant:{self.namespace}",
                ),
            )
            self.old_grant_id = int(cur.fetchone()["id"])
            cur.execute(
                """
                INSERT INTO experiment_runs
                    (agenda_id, deep_insight_id, resource_grant_id, status,
                     scientific_evidence_state)
                VALUES (%s, %s, %s, 'completed', 'scientifically_decided')
                RETURNING id
                """,
                (self.agenda_id, self.idea_id, self.old_grant_id),
            )
            self.run_id = int(cur.fetchone()["id"])
            cur.execute(
                """
                INSERT INTO evidence_audit_records
                    (agenda_id, experiment_run_id, raw_artifacts_hash,
                     claim_ledger_hash, benchmark_contract_hash, evaluator_ref,
                     evaluator_hash, holdout_ref, holdout_hash)
                VALUES (%s, %s, %s, %s, %s, 'test:evaluator', %s,
                        'test:holdout', %s) RETURNING id
                """,
                (
                    self.agenda_id,
                    self.run_id,
                    "a" * 64,
                    "b" * 64,
                    "c" * 64,
                    "d" * 64,
                    "e" * 64,
                ),
            )
            audit_id = int(cur.fetchone()["id"])
            cur.execute(
                """
                INSERT INTO scientific_decision_records
                    (agenda_id, experiment_run_id, evidence_audit_record_id,
                     verdict, verdict_hash, evidence_decision_json)
                VALUES (%s, %s, %s, 'supported', %s, '{}')
                """,
                (self.agenda_id, self.run_id, audit_id, self.verdict_hash),
            )
            cur.execute(
                """
                INSERT INTO outcome_records
                    (agenda_id, idea_id, resource_grant_id, experiment_run_id,
                     actual_tokens, actual_gpu_hours, wall_seconds,
                     execution_result, verdict, new_information_json,
                     state_decision, prediction_error_json,
                     artifact_manifest_json)
                VALUES (%s, %s, %s, %s, 1085, 0, 1, 'completed',
                        'supported', '{}', 'scientifically_decided', '{}', '{}')
                RETURNING id
                """,
                (
                    self.agenda_id,
                    self.idea_id,
                    self.old_grant_id,
                    self.run_id,
                ),
            )
            self.old_outcome_id = int(cur.fetchone()["id"])
        self.db.commit()

    def tearDown(self):
        for sql in (
            "DELETE FROM manuscript_gate_records_v1 WHERE agenda_id=?",
            "DELETE FROM manuscript_gate_attempts_v1 WHERE agenda_id=?",
            "DELETE FROM llm_route_observations WHERE agenda_id=?",
            "DELETE FROM resource_grant_usage_reservations WHERE agenda_id=?",
            "DELETE FROM outcome_records WHERE agenda_id=?",
            "DELETE FROM reviewer_approval_records WHERE agenda_id=?",
            "DELETE FROM scientific_decision_records WHERE agenda_id=?",
            "DELETE FROM evidence_audit_records WHERE agenda_id=?",
            "DELETE FROM evidence_state_transitions WHERE agenda_id=?",
            "DELETE FROM experiment_runs WHERE agenda_id=?",
            "DELETE FROM resource_grants WHERE agenda_id=?",
            "DELETE FROM agenda_resource_ledger WHERE agenda_id=?",
            "DELETE FROM idea_decision_packets WHERE agenda_id=?",
            "DELETE FROM frontier_packets WHERE agenda_id=?",
            "DELETE FROM deep_insights WHERE agenda_id=?",
            "DELETE FROM research_agendas WHERE id=?",
        ):
            self.db.execute(sql, (self.agenda_id,))
        self.db.commit()

    def _issue(self) -> int:
        return self.Repository().issue_historical_manuscript_grant(
            agenda_id=self.agenda_id,
            experiment_run_id=self.run_id,
            expected_verdict_hash=self.verdict_hash,
            token_cap=40000,
            expires_at=(datetime.now(timezone.utc) + timedelta(hours=2)).isoformat(),
        )

    def _attempt(self, repo, grant_id: int) -> dict:
        return repo.begin_manuscript_gate_attempt(
            agenda_id=self.agenda_id,
            idea_id=self.idea_id,
            experiment_run_id=self.run_id,
            resource_grant_id=grant_id,
            verdict_hash=self.verdict_hash,
            prompt_ref=self.prompt_ref,
        )

    def _review_usage(self, repo, grant_id: int, attempt: dict, *, tokens=10) -> int:
        usage = self.Ledger(grant_id).reserve(
            agenda_id=self.agenda_id,
            operation="manuscript_gate_review",
            idempotency_key=str(attempt["idempotency_key"]),
            token_cap=100,
        )
        self.Ledger(grant_id).settle(usage.reservation_id, tokens_used=tokens)
        repo.save_route_observation(
            self.Observation(
                agenda_id=self.agenda_id,
                idea_id=self.idea_id,
                role="reviewer",
                provider="test-provider",
                model="test-model",
                model_family="test-family",
                prompt_version=self.prompt_ref,
                input_tokens=max(1, tokens - 4),
                output_tokens=min(4, tokens),
                cost_usd=0.0,
                status="succeeded",
                failure_reason=None,
                reservation_id=usage.reservation_id,
            )
        )
        return int(usage.reservation_id)

    def _terminal_kwargs(
        self,
        *,
        grant_id: int,
        disposition: str,
        usage_id: int,
    ) -> dict:
        return {
            "agenda_id": self.agenda_id,
            "idea_id": self.idea_id,
            "experiment_run_id": self.run_id,
            "resource_grant_id": grant_id,
            "verdict_hash": self.verdict_hash,
            "disposition": disposition,
            "prompt_ref": self.prompt_ref,
            "judgement": {
                "concur": disposition == "approved",
                "reasons": ["isolated test"],
            },
            "grant_usage_reservation_id": usage_id,
            "reviewer_ref": "test-provider:test-model",
            "reviewer_response_hash": "f" * 64,
        }

    def test_closed_agenda_issue_is_new_bounded_and_idempotent(self):
        grant_id = self._issue()
        self.assertEqual(self._issue(), grant_id)
        grant = dict(self.db.fetchone("SELECT * FROM resource_grants WHERE id=?", (grant_id,)))
        self.assertEqual(grant["stage"], "manuscript")
        self.assertEqual(grant["status"], "active")
        self.assertEqual(int(grant["token_cap"]), 40000)
        self.assertEqual(float(grant["max_gpu_hours"]), 0.0)
        run = dict(self.db.fetchone("SELECT resource_grant_id FROM experiment_runs WHERE id=?", (self.run_id,)))
        self.assertEqual(int(run["resource_grant_id"]), grant_id)
        agenda = dict(self.db.fetchone("SELECT status, is_active, token_reserved FROM research_agendas WHERE id=?", (self.agenda_id,)))
        self.assertEqual((agenda["status"], int(agenda["is_active"])), ("closed", 0))
        self.assertEqual(int(agenda["token_reserved"]), 40000)
        old = dict(self.db.fetchone("SELECT status, reservation_id FROM resource_grants WHERE id=?", (self.old_grant_id,)))
        self.assertEqual((old["status"], int(old["reservation_id"])), ("consumed", self.old_ledger_id))

    def test_reconciled_expired_grant_gets_a_new_generation(self):
        repo = self.Repository()
        first = self._issue()
        self._attempt(repo, first)
        self.db.execute(
            "UPDATE resource_grants SET expires_at=CURRENT_TIMESTAMP - "
            "INTERVAL '1 minute' WHERE id=?",
            (first,),
        )
        self.db.commit()

        self.assertEqual(
            repo.reconcile_expired_grants(agenda_id=self.agenda_id),
            1,
        )
        second = self._issue()
        self.assertNotEqual(second, first)
        self.assertEqual(self._issue(), second)

        grants = self.db.fetchall(
            """
            SELECT id, status, idempotency_key
            FROM resource_grants WHERE id IN (?, ?) ORDER BY id
            """,
            (first, second),
        )
        run = self.db.fetchone(
            "SELECT resource_grant_id FROM experiment_runs WHERE id=?",
            (self.run_id,),
        )
        self.db.commit()
        self.assertEqual([row["status"] for row in grants], ["expired", "active"])
        self.assertTrue(str(grants[0]["idempotency_key"]).endswith(":g1"))
        self.assertTrue(str(grants[1]["idempotency_key"]).endswith(":g2"))
        self.assertEqual(int(run["resource_grant_id"]), second)

    def test_two_pre_reservation_failures_are_terminal_and_settled(self):
        grant_id = self._issue()
        run = dict(
            self.db.fetchone("SELECT * FROM experiment_runs WHERE id=?", (self.run_id,))
        )
        self.db.commit()

        self.assertEqual(
            self.run_gate(run, secret="", log=lambda *args: None),
            "review_failed",
        )
        self.assertEqual(
            self.run_gate(run, secret="", log=lambda *args: None),
            "technical_failed",
        )

        attempts = self.db.fetchone(
            "SELECT COUNT(*) AS n FROM manuscript_gate_attempts_v1 "
            "WHERE resource_grant_id=?",
            (grant_id,),
        )
        usage = self.db.fetchone(
            "SELECT COUNT(*) AS n FROM resource_grant_usage_reservations "
            "WHERE resource_grant_id=?",
            (grant_id,),
        )
        terminal = self.db.fetchone(
            "SELECT disposition, grant_usage_reservation_id "
            "FROM manuscript_gate_records_v1 WHERE resource_grant_id=?",
            (grant_id,),
        )
        grant = self.db.fetchone(
            "SELECT status FROM resource_grants WHERE id=?", (grant_id,)
        )
        ledger = self.db.fetchone(
            "SELECT status, tokens_used FROM agenda_resource_ledger "
            "WHERE id=(SELECT reservation_id FROM resource_grants WHERE id=?)",
            (grant_id,),
        )
        self.db.commit()
        self.assertEqual(int(attempts["n"]), 2)
        self.assertEqual(int(usage["n"]), 0)
        self.assertEqual(terminal["disposition"], "technical_failed")
        self.assertIsNone(terminal["grant_usage_reservation_id"])
        self.assertEqual(grant["status"], "consumed")
        self.assertEqual((ledger["status"], int(ledger["tokens_used"])), ("settled", 0))

    def test_refusal_settles_tokens_without_a_new_outcome(self):
        grant_id = self._issue()
        repo = self.Repository()
        attempt = self._attempt(repo, grant_id)
        usage_id = self._review_usage(repo, grant_id, attempt)
        run = dict(
            self.db.fetchone("SELECT * FROM experiment_runs WHERE id=?", (self.run_id,))
        )
        self.db.commit()
        self.assertEqual(
            self.finish_terminal(
                repo=repo,
                run=run,
                terminal={
                    "disposition": "refused",
                    "resource_grant_id": grant_id,
                },
                verdict_hash=self.verdict_hash,
                secret="unused-for-refusal",
                log=lambda *args: None,
                record_kwargs=self._terminal_kwargs(
                    grant_id=grant_id,
                    disposition="refused",
                    usage_id=usage_id,
                ),
            ),
            "refused",
        )
        self.assertEqual(
            repo.complete_manuscript_grant(
                agenda_id=self.agenda_id,
                experiment_run_id=self.run_id,
                resource_grant_id=grant_id,
            ),
            10,
        )
        agenda = dict(self.db.fetchone("SELECT token_spent, token_reserved, status FROM research_agendas WHERE id=?", (self.agenda_id,)))
        self.assertEqual((int(agenda["token_spent"]), int(agenda["token_reserved"])), (1095, 0))
        self.assertEqual(agenda["status"], "closed")
        count = dict(self.db.fetchone("SELECT COUNT(*) AS n FROM outcome_records WHERE experiment_run_id=?", (self.run_id,)))
        self.assertEqual(int(count["n"]), 1)
        old = dict(self.db.fetchone("SELECT actual_tokens, verdict FROM outcome_records WHERE id=?", (self.old_outcome_id,)))
        self.assertEqual((int(old["actual_tokens"]), old["verdict"]), (1085, "supported"))

    def test_approval_transition_record_and_settlement_are_atomic(self):
        grant_id = self._issue()
        repo = self.Repository()
        attempt = self._attempt(repo, grant_id)
        usage_id = self._review_usage(repo, grant_id, attempt)
        run = dict(
            self.db.fetchone("SELECT * FROM experiment_runs WHERE id=?", (self.run_id,))
        )
        self.db.commit()
        secret = "isolated-manuscript-secret"
        env = {
            "DEEPGRAPH_REVIEWER_APPROVAL_KEYS_JSON": json.dumps(
                {self.reviewer_key_id: "env:_ISOLATED_MANUSCRIPT_SECRET"}
            ),
            "_ISOLATED_MANUSCRIPT_SECRET": secret,
        }
        with patch.dict(os.environ, env):
            self.assertEqual(
                self.finish_terminal(
                    repo=repo,
                    run=run,
                    terminal={
                        "disposition": "approved",
                        "resource_grant_id": grant_id,
                    },
                    verdict_hash=self.verdict_hash,
                    secret=secret,
                    log=lambda *args: None,
                    record_kwargs=self._terminal_kwargs(
                        grant_id=grant_id,
                        disposition="approved",
                        usage_id=usage_id,
                    ),
                ),
                "manuscript_allowed",
            )
        state = self.db.fetchone(
            "SELECT scientific_evidence_state FROM experiment_runs WHERE id=?",
            (self.run_id,),
        )
        counts = self.db.fetchone(
            """
            SELECT
              (SELECT COUNT(*) FROM manuscript_gate_records_v1
               WHERE experiment_run_id=?) AS terminals,
              (SELECT COUNT(*) FROM reviewer_approval_records
               WHERE agenda_id=? AND purpose='scientific_manuscript') AS approvals,
              (SELECT COUNT(*) FROM evidence_state_transitions
               WHERE experiment_run_id=? AND to_state='manuscript_allowed') AS transitions
            """,
            (self.run_id, self.agenda_id, self.run_id),
        )
        grant = self.db.fetchone(
            "SELECT status FROM resource_grants WHERE id=?", (grant_id,)
        )
        self.db.commit()
        self.assertEqual(state["scientific_evidence_state"], "manuscript_allowed")
        self.assertEqual(
            (int(counts["terminals"]), int(counts["approvals"]), int(counts["transitions"])),
            (1, 1, 1),
        )
        self.assertEqual(grant["status"], "consumed")

    def test_late_approval_failure_rolls_back_terminal_and_transition(self):
        grant_id = self._issue()
        repo = self.Repository()
        attempt = self._attempt(repo, grant_id)
        usage_id = self._review_usage(repo, grant_id, attempt)
        run = dict(
            self.db.fetchone("SELECT * FROM experiment_runs WHERE id=?", (self.run_id,))
        )
        self.db.commit()
        secret = "isolated-manuscript-secret"
        env = {
            "DEEPGRAPH_REVIEWER_APPROVAL_KEYS_JSON": json.dumps(
                {self.reviewer_key_id: "env:_ISOLATED_MANUSCRIPT_SECRET"}
            ),
            "_ISOLATED_MANUSCRIPT_SECRET": secret,
        }
        with patch.dict(os.environ, env), patch.object(
            repo,
            "complete_manuscript_grant",
            side_effect=RuntimeError("late settlement rejection"),
        ):
            with self.assertRaisesRegex(RuntimeError, "late settlement rejection"):
                self.finish_terminal(
                    repo=repo,
                    run=run,
                    terminal={
                        "disposition": "approved",
                        "resource_grant_id": grant_id,
                    },
                    verdict_hash=self.verdict_hash,
                    secret=secret,
                    log=lambda *args: None,
                    record_kwargs=self._terminal_kwargs(
                        grant_id=grant_id,
                        disposition="approved",
                        usage_id=usage_id,
                    ),
                )
        state = self.db.fetchone(
            "SELECT scientific_evidence_state FROM experiment_runs WHERE id=?",
            (self.run_id,),
        )
        counts = self.db.fetchone(
            """
            SELECT
              (SELECT COUNT(*) FROM manuscript_gate_records_v1
               WHERE experiment_run_id=?) AS terminals,
              (SELECT COUNT(*) FROM reviewer_approval_records
               WHERE agenda_id=? AND purpose='scientific_manuscript') AS approvals,
              (SELECT COUNT(*) FROM evidence_state_transitions
               WHERE experiment_run_id=? AND to_state='manuscript_allowed') AS transitions
            """,
            (self.run_id, self.agenda_id, self.run_id),
        )
        grant = self.db.fetchone(
            "SELECT status FROM resource_grants WHERE id=?", (grant_id,)
        )
        self.db.commit()
        self.assertEqual(state["scientific_evidence_state"], "scientifically_decided")
        self.assertEqual(
            (int(counts["terminals"]), int(counts["approvals"]), int(counts["transitions"])),
            (0, 0, 0),
        )
        self.assertEqual(grant["status"], "active")


if __name__ == "__main__":
    unittest.main()
