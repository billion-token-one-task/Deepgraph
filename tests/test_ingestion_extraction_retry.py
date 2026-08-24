"""Regression coverage for the scoped-ingestion crash/retry boundary."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import types
import unittest
from collections import Counter
from dataclasses import replace
from pathlib import Path
from unittest import mock

from agents import multi_agent_extraction
from db import database
from meta_harness.grant_usage import GrantUsageError, GrantUsageLedger
from meta_harness.ingestion_queue import (
    ScopedIngestionRepository,
    ScopedIngestionUsageDispositionRequest,
)
from meta_harness.scoped_llm import proposer_json
from orchestrator import pipeline


class TempUsageDbCase(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmpdir.name) / "test.db"
        self.old_db_path = database.DB_PATH
        self.old_database_url = database.DATABASE_URL
        self._close_connection()
        database.DATABASE_URL = ""
        database.DB_PATH = self.db_path
        database.init_db()
        database.execute(
            """
            CREATE TABLE resource_grant_usage_reservations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agenda_id INTEGER NOT NULL,
                resource_grant_id INTEGER NOT NULL,
                operation TEXT NOT NULL,
                idempotency_key TEXT NOT NULL,
                token_reserved INTEGER NOT NULL,
                tokens_used INTEGER,
                cost_usd REAL,
                status TEXT NOT NULL,
                release_reason TEXT,
                settled_at TEXT,
                UNIQUE (resource_grant_id, idempotency_key)
            )
            """
        )
        database.commit()

    def tearDown(self):
        self._close_connection()
        database.DATABASE_URL = self.old_database_url
        database.DB_PATH = self.old_db_path
        self.tmpdir.cleanup()

    @staticmethod
    def _close_connection():
        for attr in ("pg_conn", "sqlite_conn", "conn"):
            if hasattr(database._local, attr):
                try:
                    getattr(database._local, attr).close()
                except Exception:
                    pass
                setattr(database._local, attr, None)

    @staticmethod
    def _scope():
        return {
            "agenda_id": 10,
            "idea_id": 125,
            "resource_grant_id": 193,
            "stage": "ingestion_backfill_canary",
            "token_cap": 100_000,
        }

    @staticmethod
    def _fake_client(call):
        module = types.ModuleType("agents.llm_client")
        module.configured_role_prompt_version = lambda role: f"{role}_v1"
        module.call_llm_json_for_role = call
        return module

    @staticmethod
    def _record_usage(kwargs, *, status: str, tokens_used: int | None = None):
        database.execute(
            """
            INSERT INTO resource_grant_usage_reservations
                (agenda_id, resource_grant_id, operation, idempotency_key,
                 token_reserved, tokens_used, status)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                kwargs["agenda_id"],
                kwargs["resource_grant_id"],
                kwargs["operation"],
                kwargs["idempotency_key"],
                100,
                tokens_used,
                status,
            ),
        )
        database.commit()


class ScopedAttemptRegressionTests(TempUsageDbCase):
    def test_settled_attempt_can_retry_but_open_attempt_fails_closed(self):
        """A settled miss may retry; ambiguous delivery requires an operator."""

        scope = self._scope()
        operations = (
            ("paper_extraction:p:settled", "system-a", "user-a", "settled", 9),
            ("paper_extraction:p:reserved", "system-b", "user-b", "reserved", None),
        )
        legacy_keys = []
        for operation, system_prompt, user_prompt, status, tokens_used in operations:
            digest = hashlib.sha256(
                "\n".join(
                    (
                        str(scope["agenda_id"]),
                        str(scope["idea_id"]),
                        str(scope["resource_grant_id"]),
                        str(scope["stage"]),
                        operation,
                        system_prompt,
                        user_prompt,
                    )
                ).encode("utf-8")
            ).hexdigest()
            legacy_key = f"{operation}:{digest}"
            legacy_keys.append(legacy_key)
            self._record_usage(
                {
                    "agenda_id": scope["agenda_id"],
                    "resource_grant_id": scope["resource_grant_id"],
                    "operation": operation,
                    "idempotency_key": legacy_key,
                },
                status=status,
                tokens_used=tokens_used,
            )

        calls: list[str] = []

        def fake_call(_system, _user, **kwargs):
            calls.append(kwargs["idempotency_key"])
            self._record_usage(kwargs, status="settled", tokens_used=7)
            return {"recovered": True}, 7, {"provider": "test"}

        client = self._fake_client(fake_call)
        with mock.patch.dict(sys.modules, {"agents.llm_client": client}):
            settled_recovery = proposer_json(
                "system-a",
                "user-a",
                llm_scope=scope,
                operation="paper_extraction:p:settled",
            )
            with self.assertRaisesRegex(
                GrantUsageError, "exact operator usage disposition"
            ):
                proposer_json(
                    "system-b",
                    "user-b",
                    llm_scope=scope,
                    operation="paper_extraction:p:reserved",
                )

        self.assertEqual(calls, [f"{legacy_keys[0]}:t2"])
        self.assertEqual(settled_recovery[0], {"recovered": True})
        statuses = database.fetchall(
            """
            SELECT status FROM resource_grant_usage_reservations
            WHERE operation='paper_extraction:p:reserved'
            ORDER BY id
            """
        )
        self.assertEqual([row["status"] for row in statuses], ["reserved"])

    def _seed_disposition_scope(self) -> None:
        statements = (
            """
            CREATE TABLE agenda_resource_ledger (
                id INTEGER PRIMARY KEY, agenda_id INTEGER NOT NULL,
                status TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE resource_grants (
                id INTEGER PRIMARY KEY, agenda_id INTEGER NOT NULL,
                idea_id INTEGER NOT NULL, stage TEXT NOT NULL,
                token_cap INTEGER NOT NULL, max_gpu_hours REAL NOT NULL,
                backend_allowlist_json TEXT NOT NULL, reservation_id INTEGER NOT NULL,
                status TEXT NOT NULL, expires_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE scoped_ingestion_jobs_v1 (
                id INTEGER PRIMARY KEY, agenda_id INTEGER NOT NULL,
                idea_id INTEGER NOT NULL, resource_grant_id INTEGER NOT NULL,
                stage TEXT NOT NULL, paper_ids_json TEXT NOT NULL,
                status TEXT NOT NULL, attempt_count INTEGER NOT NULL,
                max_attempts INTEGER NOT NULL, lease_owner TEXT,
                lease_expires_at TEXT, completed_at TEXT, failure_reason TEXT,
                updated_at TEXT
            )
            """,
            """
            CREATE TABLE experiment_attempt_gpu_reservations_v1 (
                id INTEGER PRIMARY KEY, resource_grant_id INTEGER NOT NULL
            )
            """,
        )
        for statement in statements:
            database.execute(statement)
        database.execute(
            "INSERT INTO agenda_resource_ledger (id, agenda_id, status) "
            "VALUES (77, 10, 'reserved')"
        )
        database.execute(
            """
            INSERT INTO resource_grants
                (id, agenda_id, idea_id, stage, token_cap, max_gpu_hours,
                 backend_allowlist_json, reservation_id, status, expires_at)
            VALUES (193, 10, 125, 'ingestion_backfill_canary', 100000, 0,
                    '["llm"]', 77, 'active', datetime('now', '+1 hour'))
            """
        )
        database.execute(
            """
            INSERT INTO scoped_ingestion_jobs_v1
                (id, agenda_id, idea_id, resource_grant_id, stage,
                 paper_ids_json, status, attempt_count, max_attempts)
            VALUES (25, 10, 125, 193, 'ingestion_backfill_canary',
                    '["2608.18972"]', 'manual_reconciliation', 1, 3)
            """
        )
        database.commit()

    def test_exact_operator_disposition_is_required_before_t2_and_leaves_no_open_usage(self):
        self._seed_disposition_scope()
        scope = self._scope()
        operation = "paper_extraction:2608.18972:claims_methods"
        calls = 0

        def fake_call(_system, _user, **kwargs):
            nonlocal calls
            calls += 1
            ledger = GrantUsageLedger(kwargs["resource_grant_id"])
            reservation = ledger.reserve(
                agenda_id=kwargs["agenda_id"],
                operation=kwargs["operation"],
                idempotency_key=kwargs["idempotency_key"],
                token_cap=100,
            )
            if calls == 1:
                raise RuntimeError("provider delivery state lost")
            ledger.settle(reservation.reservation_id, tokens_used=11)
            return {"recovered": True}, 11, {"provider": "test"}

        client = self._fake_client(fake_call)
        with mock.patch.dict(sys.modules, {"agents.llm_client": client}):
            with self.assertRaisesRegex(RuntimeError, "delivery state lost"):
                proposer_json(
                    "system", "user", llm_scope=scope, operation=operation
                )
            with self.assertRaisesRegex(
                GrantUsageError, "exact operator usage disposition"
            ):
                proposer_json(
                    "system", "user", llm_scope=scope, operation=operation
                )

            open_row = database.fetchone(
                "SELECT * FROM resource_grant_usage_reservations "
                "WHERE resource_grant_id=193 AND status='reserved'"
            )
            disposition_request = ScopedIngestionUsageDispositionRequest(
                job_id=25,
                agenda_id=10,
                idea_id=125,
                resource_grant_id=193,
                stage="ingestion_backfill_canary",
                paper_ids=("2608.18972",),
                usage_reservation_id=int(open_row["id"]),
                operation=operation,
                usage_idempotency_key=str(open_row["idempotency_key"]),
                expected_token_reserved=100,
                disposition="release_confirmed_unbilled",
                tokens_used=None,
                cost_usd=None,
                actor="isolated-test-operator",
                reason="provider confirms request was not accepted",
                evidence_ref="test-provider-request-log:missing",
                operator_request_id="job25-usage-t1-disposition-v1",
                resume=True,
            )
            disposition = ScopedIngestionRepository().dispose_open_usage(
                disposition_request
            )
            recovered = proposer_json(
                "system", "user", llm_scope=scope, operation=operation
            )
            replay = ScopedIngestionRepository().dispose_open_usage(
                disposition_request
            )

        self.assertEqual(disposition["job_status"], "retryable")
        self.assertEqual(replay["status"], "already_disposed")
        self.assertEqual(recovered[0], {"recovered": True})
        self.assertEqual(calls, 2)
        rows = database.fetchall(
            "SELECT status, idempotency_key FROM "
            "resource_grant_usage_reservations ORDER BY id"
        )
        self.assertEqual([row["status"] for row in rows], ["released", "settled"])
        self.assertTrue(rows[0]["idempotency_key"].endswith(":t1"))
        self.assertTrue(rows[1]["idempotency_key"].endswith(":t2"))
        self.assertEqual(
            database.fetchone(
                "SELECT COUNT(*) AS c FROM resource_grant_usage_reservations "
                "WHERE status='reserved'"
            )["c"],
            0,
        )

    def test_measured_disposition_is_audited_and_different_replay_is_refused(self):
        self._seed_disposition_scope()
        reservation = GrantUsageLedger(193).reserve(
            agenda_id=10,
            operation="paper_reasoning:2608.18972",
            idempotency_key="reasoning-key:t1",
            token_cap=40,
        )
        request = ScopedIngestionUsageDispositionRequest(
            job_id=25,
            agenda_id=10,
            idea_id=125,
            resource_grant_id=193,
            stage="ingestion_backfill_canary",
            paper_ids=("2608.18972",),
            usage_reservation_id=reservation.reservation_id,
            operation="paper_reasoning:2608.18972",
            usage_idempotency_key="reasoning-key:t1",
            expected_token_reserved=40,
            disposition="settle_measured",
            tokens_used=25,
            cost_usd=0.0025,
            actor="isolated-test-operator",
            reason="provider usage record recovered",
            evidence_ref="provider-usage-log:req-25",
            operator_request_id="job25-measured-usage-v1",
            resume=False,
        )

        disposed = ScopedIngestionRepository().dispose_open_usage(request)
        replay = ScopedIngestionRepository().dispose_open_usage(request)
        with self.assertRaisesRegex(
            Exception, "already disposed differently"
        ):
            ScopedIngestionRepository().dispose_open_usage(
                replace(request, reason="different operator judgement")
            )

        self.assertEqual(disposed["usage_status"], "settled")
        self.assertEqual(replay["status"], "already_disposed")
        row = database.fetchone(
            "SELECT * FROM resource_grant_usage_reservations WHERE id=?",
            (reservation.reservation_id,),
        )
        self.assertEqual(row["tokens_used"], 25)
        self.assertAlmostEqual(row["cost_usd"], 0.0025)
        audit = json.loads(row["release_reason"])
        self.assertEqual(audit["schema"], "scoped-ingestion-usage-disposition-v1")
        self.assertEqual(audit["evidence_ref"], "provider-usage-log:req-25")

    def test_attempt_exhaustion_fails_before_a_fourth_provider_call(self):
        calls = 0

        def fake_call(_system, _user, **kwargs):
            nonlocal calls
            calls += 1
            self._record_usage(kwargs, status="settled", tokens_used=5)
            return {"attempt": calls}, 5, {"provider": "test"}

        client = self._fake_client(fake_call)
        with mock.patch.dict(sys.modules, {"agents.llm_client": client}):
            for _ in range(3):
                proposer_json(
                    "system",
                    "unusable-output-not-checkpointed",
                    llm_scope=self._scope(),
                    operation="paper_extraction:p:bounded",
                )
            with self.assertRaisesRegex(GrantUsageError, "all 3 attempts"):
                proposer_json(
                    "system",
                    "unusable-output-not-checkpointed",
                    llm_scope=self._scope(),
                    operation="paper_extraction:p:bounded",
                )

        self.assertEqual(calls, 3)


class RoleCheckpointRegressionTests(TempUsageDbCase):
    def test_retry_reuses_delivered_roles_and_only_settles_missing_role(self):
        paper_id = "2608.18972"
        ScopedAttemptRegressionTests._seed_disposition_scope(self)
        database.execute(
            "INSERT INTO papers (id, title) VALUES (?, ?)",
            (paper_id, "Canary paper"),
        )
        database.commit()
        calls: Counter[str] = Counter()

        def fake_call(_system, _user, **kwargs):
            role_name = kwargs["operation"].rsplit(":", 1)[-1]
            calls[role_name] += 1
            if role_name == "claims_methods" and calls[role_name] == 1:
                self._record_usage(kwargs, status="reserved")
                raise RuntimeError("worker crashed with an open reservation")
            self._record_usage(kwargs, status="settled", tokens_used=11)
            return {"delivered_by": role_name}, 11, {"provider": "test"}

        client = self._fake_client(fake_call)
        with mock.patch.dict(sys.modules, {"agents.llm_client": client}):
            with self.assertRaisesRegex(
                RuntimeError, "claims_methods=worker crashed"
            ):
                multi_agent_extraction.extract_paper_multi_agent(
                    paper_id,
                    "Canary paper",
                    "Available taxonomy leaf nodes:\n  ml.test",
                    "grounded body",
                    llm_scope=self._scope(),
                )

            checkpoint_count = database.fetchone(
                """
                SELECT COUNT(*) AS c FROM paper_stage_checkpoints
                WHERE paper_id=? AND stage LIKE 'paper_extraction_role:%'
                """,
                (paper_id,),
            )["c"]
            self.assertEqual(checkpoint_count, 4)
            delivered = database.get_paper_checkpoint(
                paper_id, "paper_extraction_role:empirical_results"
            )["payload"]
            self.assertEqual(delivered["schema"], "paper-extraction-role-v1")
            self.assertEqual(delivered["scope"]["resource_grant_id"], 193)
            self.assertEqual(delivered["usage"]["tokens"], 11)
            self.assertEqual(
                delivered["result"], {"delivered_by": "empirical_results"}
            )

            open_row = database.fetchone(
                """
                SELECT * FROM resource_grant_usage_reservations
                WHERE operation=? AND status='reserved'
                """,
                (f"paper_extraction:{paper_id}:claims_methods",),
            )
            ScopedIngestionRepository().dispose_open_usage(
                ScopedIngestionUsageDispositionRequest(
                    job_id=25,
                    agenda_id=10,
                    idea_id=125,
                    resource_grant_id=193,
                    stage="ingestion_backfill_canary",
                    paper_ids=(paper_id,),
                    usage_reservation_id=int(open_row["id"]),
                    operation=str(open_row["operation"]),
                    usage_idempotency_key=str(open_row["idempotency_key"]),
                    expected_token_reserved=100,
                    disposition="release_confirmed_unbilled",
                    tokens_used=None,
                    cost_usd=None,
                    actor="isolated-test-operator",
                    reason="simulated provider did not deliver the crashed call",
                    evidence_ref="test-provider-request-log:missing",
                    operator_request_id="role-checkpoint-crash-disposition-v1",
                    resume=True,
                )
            )

            _result, retry_tokens = (
                multi_agent_extraction.extract_paper_multi_agent(
                    paper_id,
                    "Canary paper",
                    "Available taxonomy leaf nodes:\n  ml.test",
                    "grounded body",
                    llm_scope=self._scope(),
                )
            )
            calls_after_recovery = calls.copy()
            _result, replay_tokens = (
                multi_agent_extraction.extract_paper_multi_agent(
                    paper_id,
                    "Canary paper",
                    "Available taxonomy leaf nodes:\n  ml.test",
                    "grounded body",
                    llm_scope=self._scope(),
                )
            )

        self.assertEqual(retry_tokens, 11)
        self.assertEqual(replay_tokens, 0)
        self.assertEqual(calls, calls_after_recovery)
        self.assertEqual(calls["claims_methods"], 2)
        for role_name in (
            "taxonomy_overview",
            "empirical_results",
            "graph_context",
            "research_facets",
        ):
            self.assertEqual(calls[role_name], 1)

        usage = database.fetchall(
            """
            SELECT operation, status, tokens_used
            FROM resource_grant_usage_reservations
            ORDER BY id
            """
        )
        self.assertEqual(len(usage), 6)
        self.assertEqual(
            Counter(row["status"] for row in usage),
            Counter({"settled": 5, "released": 1}),
        )
        self.assertEqual(
            sum(int(row["tokens_used"] or 0) for row in usage),
            55,
        )
        claims_keys = database.fetchall(
            """
            SELECT idempotency_key FROM resource_grant_usage_reservations
            WHERE operation=? ORDER BY id
            """,
            (f"paper_extraction:{paper_id}:claims_methods",),
        )
        self.assertTrue(claims_keys[0]["idempotency_key"].endswith(":t1"))
        self.assertTrue(claims_keys[1]["idempotency_key"].endswith(":t2"))


class ReasonedPaperReplayTests(unittest.TestCase):
    def test_reasoned_retry_returns_durable_lifecycle_counts(self):
        with mock.patch.object(
            pipeline, "require_active_scope", return_value={"resource_grant_id": 193}
        ), mock.patch.object(
            pipeline.db,
            "fetchone",
            return_value={
                "id": "p1",
                "title": "Recovered paper",
                "processing_stage": "reasoned",
            },
        ), mock.patch.object(
            pipeline,
            "_load_checkpoint_payload",
            side_effect=[
                {
                    "claim_count": 3,
                    "result_count": 1,
                    "graph_entities": 4,
                    "graph_relations": 2,
                    "taxonomy_nodes": ["ml.test"],
                },
                {
                    "claims": 3,
                    "results": 1,
                    "contradictions": 1,
                    "taxonomy_nodes": ["ml.test"],
                },
            ],
        ), mock.patch.object(pipeline, "log_event"):
            result = pipeline.process_single_paper(
                "p1", llm_scope={"resource_grant_id": 193}
            )

        self.assertNotIn("error", result)
        self.assertEqual(result["claims"], 3)
        self.assertEqual(result["graph_entities"], 4)
        self.assertEqual(result["graph_relations"], 2)
        self.assertEqual(result["tokens"], 0)

    def test_reasoned_retry_without_checkpoints_fails_closed(self):
        with mock.patch.object(
            pipeline, "require_active_scope", return_value={"resource_grant_id": 193}
        ), mock.patch.object(
            pipeline.db,
            "fetchone",
            return_value={
                "id": "p1",
                "title": "Broken paper",
                "processing_stage": "reasoned",
            },
        ), mock.patch.object(
            pipeline, "_load_checkpoint_payload", return_value=None
        ), mock.patch.object(pipeline, "log_event"):
            result = pipeline.process_single_paper(
                "p1", llm_scope={"resource_grant_id": 193}
            )

        self.assertIn("lacks durable lifecycle checkpoints", result["error"])


if __name__ == "__main__":
    unittest.main()
