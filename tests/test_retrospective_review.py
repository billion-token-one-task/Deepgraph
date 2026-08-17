"""Retrospective review: eligibility, verdict ceiling, signed audited apply."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from db import database
from meta_harness import retrospective_review as rr


def _reset_db(tmpdir: Path):
    for attr in ("pg_conn", "sqlite_conn", "conn"):
        if hasattr(database._local, attr):
            try:
                getattr(database._local, attr).close()
            except Exception:
                pass
            setattr(database._local, attr, None)
    database.DATABASE_URL = ""
    database.DB_PATH = tmpdir / "test.db"
    database.init_db()


def _create_meta_harness_tables():
    for statement in (
        """CREATE TABLE IF NOT EXISTS evidence_state_transitions (
            id INTEGER PRIMARY KEY, agenda_id INTEGER, experiment_run_id INTEGER,
            from_state TEXT, to_state TEXT, actor TEXT, context_json TEXT,
            created_at TEXT DEFAULT '2026-08-04 00:00:00')""",
        """CREATE TABLE IF NOT EXISTS evidence_audit_records (
            id INTEGER PRIMARY KEY, agenda_id INTEGER, experiment_run_id INTEGER,
            raw_artifacts_hash TEXT, claim_ledger_hash TEXT,
            benchmark_contract_hash TEXT, evaluator_ref TEXT, evaluator_hash TEXT,
            holdout_ref TEXT, holdout_hash TEXT,
            created_at TEXT DEFAULT '2026-08-04 00:00:00')""",
        # evidence_audit_record_id is NOT NULL in db/migrations/0001_meta_harness_v1.sql.
        # This fixture used to omit it, so the suite could pass while the real
        # insert raised NotNullViolation on every production run.
        """CREATE TABLE IF NOT EXISTS scientific_decision_records (
            id INTEGER PRIMARY KEY, agenda_id INTEGER, experiment_run_id INTEGER,
            evidence_audit_record_id INTEGER NOT NULL,
            verdict TEXT, verdict_hash TEXT, evidence_decision_json TEXT,
            created_at TEXT DEFAULT '2026-08-04 00:00:00')""",
        """CREATE TABLE IF NOT EXISTS reviewer_approval_records (
            id INTEGER PRIMARY KEY, agenda_id INTEGER, purpose TEXT, subject TEXT,
            reviewer_id TEXT, key_id TEXT, issued_at TEXT, signature_hash TEXT,
            created_at TEXT DEFAULT '2026-08-04 00:00:00')""",
    ):
        database.execute(statement)
    database.commit()


class RetrospectiveReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.old_db_path = database.DB_PATH
        self.old_database_url = database.DATABASE_URL
        _reset_db(Path(self.tmpdir.name))
        _create_meta_harness_tables()
        for table, col in (("experiment_runs", "agenda_id"),
                           ("experiment_runs", "scientific_evidence_state"),
                           ("experiment_artifacts", "agenda_id"),
                           ("experimental_claims", "agenda_id"),
                           ("deep_insights", "agenda_id")):
            try:
                database.execute(f"ALTER TABLE {table} ADD COLUMN {col} TEXT")
            except Exception:
                pass
        database.execute(
            "INSERT INTO deep_insights (id, agenda_id, tier, title) VALUES (1, 6, 2, 'Legacy idea')"
        )
        database.execute(
            """INSERT INTO experiment_runs
               (id, deep_insight_id, status, agenda_id, baseline_metric_value,
                best_metric_value, effect_pct, hypothesis_verdict, experiment_suite)
               VALUES (14, 1, 'completed', 6, 0.789122, 0.812058,
                       2.906521425077486, 'confirmed', 'main')"""
        )
        artifact = Path(self.tmpdir.name) / "metrics.json"
        artifact.write_text('{"metric": 0.812058}', encoding="utf-8")
        database.execute(
            """INSERT INTO experiment_artifacts (run_id, artifact_type, path)
               VALUES (14, 'metric', ?)""",
            (str(artifact),),
        )
        database.execute(
            """INSERT INTO experimental_claims
               (run_id, deep_insight_id, claim_text, verdict)
               VALUES (14, 1, 'improves metric', 'confirmed')"""
        )
        database.commit()
        self.env = {
            "DEEPGRAPH_REVIEWER_APPROVAL_KEYS_JSON": json.dumps(
                {"test-key": "env:TEST_REVIEWER_SECRET"}
            ),
            "TEST_REVIEWER_SECRET": "s3cret",
        }

    def tearDown(self):
        database.DB_PATH = self.old_db_path
        database.DATABASE_URL = self.old_database_url
        self.tmpdir.cleanup()

    def _approval(self, subject):
        return rr.sign_approval(
            reviewer_id="operator", key_id="test-key", subject=subject,
            secret="s3cret",
        )

    def test_eligibility_and_packet(self):
        rows = rr.eligible_run_rows()
        self.assertEqual([r["id"] for r in rows], [14])
        packet = rr.build_packet(14)
        self.assertEqual(packet["blockers"], [])
        self.assertEqual(packet["policy"]["verdict_ceiling"], "inconclusive")
        self.assertEqual(packet["artifacts"]["files_present"], 1)
        self.assertTrue(packet["evaluator_report"]["passed"])

    def test_verdict_ceiling_is_enforced(self):
        with self.assertRaises(rr.RetrospectiveReviewError):
            rr.apply_review(run_id=14, approval={}, verdict="supported")

    def test_apply_writes_full_audited_chain(self):
        subject = rr.retrospective_subject(agenda_id=6, experiment_run_id=14)
        with mock.patch.dict(os.environ, self.env):
            result = rr.apply_review(run_id=14, approval=self._approval(subject))
        self.assertEqual(result["verdict"], "inconclusive")

        transitions = database.fetchall(
            "SELECT from_state, to_state, actor, context_json FROM evidence_state_transitions WHERE experiment_run_id=14 ORDER BY id"
        )
        self.assertEqual(
            [(t["from_state"], t["to_state"]) for t in transitions],
            [("planned", "sanity_passed"),
             ("sanity_passed", "full_benchmark_complete"),
             ("full_benchmark_complete", "evidence_audited"),
             ("evidence_audited", "scientifically_decided")],
        )
        for t in transitions:
            self.assertEqual(t["actor"], "retrospective_review:operator")
            self.assertTrue(json.loads(t["context_json"])["legacy_review"])

        audit = database.fetchone(
            "SELECT * FROM evidence_audit_records WHERE experiment_run_id=14"
        )
        self.assertEqual(audit["evaluator_ref"], rr.EVALUATOR_REF)
        self.assertEqual(audit["holdout_ref"], rr.HOLDOUT_REF)

        decision = database.fetchone(
            "SELECT verdict, verdict_hash FROM scientific_decision_records WHERE experiment_run_id=14"
        )
        self.assertEqual(decision["verdict"], "inconclusive")
        self.assertEqual(len(decision["verdict_hash"]), 64)

        approval = database.fetchone(
            "SELECT reviewer_id, purpose FROM reviewer_approval_records WHERE agenda_id=6"
        )
        self.assertEqual(approval["reviewer_id"], "operator")
        self.assertEqual(approval["purpose"], rr.PURPOSE)

        state = database.fetchone(
            "SELECT scientific_evidence_state FROM experiment_runs WHERE id=14"
        )
        self.assertEqual(state["scientific_evidence_state"], "scientifically_decided")

        # The run is no longer eligible: double-apply is refused.
        self.assertEqual(rr.eligible_run_rows(), [])
        with mock.patch.dict(os.environ, self.env):
            with self.assertRaises(rr.RetrospectiveReviewError):
                rr.apply_review(run_id=14, approval=self._approval(subject))

    def test_bad_signature_is_refused(self):
        subject = rr.retrospective_subject(agenda_id=6, experiment_run_id=14)
        bad = rr.sign_approval(
            reviewer_id="operator", key_id="test-key", subject=subject,
            secret="wrong-secret",
        )
        with mock.patch.dict(os.environ, self.env):
            from meta_harness.reviewer_approval import ReviewerApprovalError
            with self.assertRaises(ReviewerApprovalError):
                rr.apply_review(run_id=14, approval=bad)
        self.assertIsNone(database.fetchone(
            "SELECT 1 as x FROM scientific_decision_records WHERE experiment_run_id=14"
        ))


if __name__ == "__main__":
    unittest.main()
