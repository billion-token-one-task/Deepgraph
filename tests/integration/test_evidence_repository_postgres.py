"""Scientific evidence persistence tests for isolated PostgreSQL only."""

from __future__ import annotations

import os
import re
import unittest
import uuid
import hashlib
import hmac
import json
from datetime import datetime, timezone
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
class IsolatedEvidenceRepositoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["DEEPGRAPH_DATABASE_URL"] = URL
        apply_to_isolated_restore(URL, source_commit=SOURCE_COMMIT)

        from db import database as database
        from meta_harness.evidence_state import EvidenceTransitionContext
        from meta_harness.repository import MetaHarnessRepository
        from meta_harness.scientific_authority import (
            positive_decision_authorized,
        )

        if not database._use_pg() or database.DATABASE_URL.strip() != URL:  # noqa: SLF001
            raise RuntimeError(
                "database module was initialized before isolated URL selection"
            )
        cls.db = database
        cls.Context = EvidenceTransitionContext
        cls.Repository = MetaHarnessRepository
        cls.positive_decision_authorized = staticmethod(
            positive_decision_authorized
        )
        os.environ["DEEPGRAPH_REVIEWER_APPROVAL_KEYS_JSON"] = json.dumps(
            {"isolated-reviewer-key": "env:DEEPGRAPH_ISOLATED_REVIEWER_SECRET"}
        )
        os.environ["DEEPGRAPH_ISOLATED_REVIEWER_SECRET"] = (
            "isolated-reviewer-test-secret"
        )

    def setUp(self):
        self.namespace = f"evidence_repo_{uuid.uuid4().hex}"
        with self.db.get_conn().cursor() as cur:
            cur.execute(
                """
                INSERT INTO research_agendas
                    (name, token_budget, status, backlog_policy)
                VALUES (%s, 10, 'active', 'explicit_import_only')
                RETURNING id
                """,
                (self.namespace,),
            )
            self.agenda_id = int(cur.fetchone()["id"])
            cur.execute(
                """
                INSERT INTO deep_insights (agenda_id, tier, title)
                VALUES (%s, 2, %s)
                RETURNING id
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
                        '["isolated-test"]', '["isolated-test"]', 'test-v1')
                RETURNING id
                """,
                (self.agenda_id, self.idea_id, frontier_id),
            )
            decision_id = int(cur.fetchone()["id"])
            self.grants = {}
            for stage in ("pilot", "full_benchmark", "evidence_audit"):
                cur.execute(
                    """
                    INSERT INTO agenda_resource_ledger
                        (agenda_id, operation, idempotency_key, token_reserved,
                         status)
                    VALUES (%s, 'resource_grant', %s, 1, 'reserved')
                    RETURNING id
                    """,
                    (self.agenda_id, f"ledger:{self.namespace}:{stage}"),
                )
                reservation_id = int(cur.fetchone()["id"])
                cur.execute(
                    """
                    INSERT INTO resource_grants
                        (agenda_id, idea_id, decision_packet_id, stage,
                         token_cap, max_gpu_hours, backend_allowlist_json,
                         artifact_requirements_json, expires_at, grant_reason,
                         reservation_id, status, idempotency_key)
                    VALUES (%s, %s, %s, %s, 1, 0, '["cpu"]',
                            '["raw_metrics"]',
                            CURRENT_TIMESTAMP + INTERVAL '10 minutes',
                            'isolated test', %s, 'active', %s)
                    RETURNING id
                    """,
                    (
                        self.agenda_id,
                        self.idea_id,
                        decision_id,
                        stage,
                        reservation_id,
                        f"grant:{self.namespace}:{stage}",
                    ),
                )
                self.grants[stage] = int(cur.fetchone()["id"])
            cur.execute(
                """
                INSERT INTO experiment_runs
                    (agenda_id, deep_insight_id, resource_grant_id, status)
                VALUES (%s, %s, %s, 'completed')
                RETURNING id
                """,
                (self.agenda_id, self.idea_id, self.grants["pilot"]),
            )
            self.run_id = int(cur.fetchone()["id"])
        self.db.commit()

    def tearDown(self):
        for sql in (
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

    def test_supported_decision_is_content_addressed_and_authoritative(self):
        digest = "a" * 64
        verdict_hash = "b" * 64
        repository = self.Repository()
        from meta_harness.reviewer_approval import (
            ReviewerApproval,
            scientific_manuscript_subject,
        )

        subject = scientific_manuscript_subject(
            agenda_id=self.agenda_id,
            experiment_run_id=self.run_id,
            verdict_hash=verdict_hash,
        )
        unsigned = ReviewerApproval(
            reviewer_id="isolated-reviewer",
            key_id="isolated-reviewer-key",
            purpose="scientific_manuscript",
            subject=subject,
            issued_at=datetime.now(timezone.utc).isoformat(),
            signature="pending",
        )
        approval = ReviewerApproval(
            **{
                **unsigned.__dict__,
                "signature": hmac.new(
                    os.environ["DEEPGRAPH_ISOLATED_REVIEWER_SECRET"].encode(),
                    unsigned.signing_payload(),
                    hashlib.sha256,
                ).hexdigest(),
            }
        )
        state = repository.advance_experiment_state(
            agenda_id=self.agenda_id,
            experiment_run_id=self.run_id,
            target="sanity_passed",
            context=self.Context(
                resource_grant_valid=True,
                resource_grant_id=self.grants["pilot"],
                execution_succeeded=True,
                raw_artifacts_present=True,
                raw_artifacts_hash=digest,
            ),
            actor="isolated-evaluator",
        )
        self.assertEqual(state, "sanity_passed")
        self.db.execute(
            "UPDATE experiment_runs SET resource_grant_id=? WHERE id=?",
            (self.grants["full_benchmark"], self.run_id),
        )
        self.db.commit()
        state = repository.advance_experiment_state(
            agenda_id=self.agenda_id,
            experiment_run_id=self.run_id,
            target="full_benchmark_complete",
            context=self.Context(
                resource_grant_valid=True,
                resource_grant_id=self.grants["full_benchmark"],
                execution_succeeded=True,
                full_benchmark_complete=True,
                benchmark_contract_hash=digest,
            ),
            actor="isolated-evaluator",
        )
        self.assertEqual(state, "full_benchmark_complete")
        self.db.execute(
            "UPDATE experiment_runs SET resource_grant_id=? WHERE id=?",
            (self.grants["evidence_audit"], self.run_id),
        )
        self.db.commit()
        state = repository.advance_experiment_state(
            agenda_id=self.agenda_id,
            experiment_run_id=self.run_id,
            target="evidence_audited",
            context=self.Context(
                resource_grant_valid=True,
                resource_grant_id=self.grants["evidence_audit"],
                execution_succeeded=True,
                raw_artifacts_present=True,
                claim_ledger_present=True,
                evaluator_passed=True,
                holdout_passed=True,
                raw_artifacts_hash=digest,
                claim_ledger_hash=digest,
                benchmark_contract_hash=digest,
                evaluator_ref="isolated:evaluator:v1",
                evaluator_hash=digest,
                holdout_ref="isolated:holdout:v1",
                holdout_hash=digest,
            ),
            actor="isolated-evaluator",
        )
        self.assertEqual(state, "evidence_audited")
        state = repository.advance_experiment_state(
            agenda_id=self.agenda_id,
            experiment_run_id=self.run_id,
            target="scientifically_decided",
            context=self.Context(
                resource_grant_valid=True,
                resource_grant_id=self.grants["evidence_audit"],
                execution_succeeded=True,
                verdict="supported",
                verdict_hash=verdict_hash,
                evidence_decision_passed=True,
                p_value=0.01,
                metric_value=0.71,
                baseline_value=0.70,
                raw_artifacts_hash=digest,
                claim_ledger_hash=digest,
                benchmark_contract_hash=digest,
                evaluator_hash=digest,
                holdout_hash=digest,
                evaluator_ref="isolated:evaluator:v1",
                holdout_ref="isolated:holdout:v1",
            ),
            actor="isolated-evaluator",
        )
        self.assertEqual(state, "scientifically_decided")
        self.assertTrue(
            self.positive_decision_authorized(
                agenda_id=self.agenda_id,
                run_id=self.run_id,
            )
        )
        state = repository.advance_experiment_state(
            agenda_id=self.agenda_id,
            experiment_run_id=self.run_id,
            target="manuscript_allowed",
            context=self.Context(
                resource_grant_valid=True,
                resource_grant_id=self.grants["evidence_audit"],
                execution_succeeded=True,
                verdict="supported",
                verdict_hash=verdict_hash,
                reviewer_approval=approval.__dict__,
            ),
            actor="isolated-reviewer",
        )
        self.assertEqual(state, "manuscript_allowed")
        counts = self.db.fetchone(
            """
            SELECT
                (SELECT COUNT(*) FROM evidence_audit_records
                 WHERE agenda_id=?) AS audits,
                (SELECT COUNT(*) FROM scientific_decision_records
                 WHERE agenda_id=?) AS decisions
            """,
            (self.agenda_id, self.agenda_id),
        )
        self.assertEqual(int(counts["audits"]), 1)
        self.assertEqual(int(counts["decisions"]), 1)


if __name__ == "__main__":
    unittest.main()
