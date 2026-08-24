"""Atomic settlement and exact operator reconciliation for scoped ingestion."""

from __future__ import annotations

import json
import os
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

from flask import Flask

from meta_harness.ingestion_queue import (
    ScopedIngestionReconciliationRequest,
    ScopedIngestionRepository,
    ScopedIngestionUsageDispositionRequest,
)
from meta_harness.scoped_llm import ScopedLLMError
from web.meta_harness_routes import blueprint


def _cursor(rowcount: int = 1) -> SimpleNamespace:
    return SimpleNamespace(rowcount=rowcount)


def _job(*, status: str = "running", result_json: str | None = None) -> dict:
    return {
        "id": 25,
        "agenda_id": 10,
        "idea_id": 125,
        "resource_grant_id": 193,
        "stage": "ingestion_backfill_canary",
        "paper_ids_json": '["p1", "p2"]',
        "status": status,
        "lease_owner": None if status == "succeeded" else "host:7:worker",
        "result_json": result_json,
    }


def _grant(*, status: str = "active", token_cap: int = 100) -> dict:
    return {
        "id": 193,
        "agenda_id": 10,
        "idea_id": 125,
        "stage": "ingestion_backfill_canary",
        "token_cap": token_cap,
        "reservation_id": 77,
        "status": status,
        "unexpired": 1,
        "max_gpu_hours": 0.0,
        "backend_allowlist_json": '["llm"]',
    }


def _ledger(
    *,
    status: str = "reserved",
    tokens_used=None,
    token_reserved: int = 100,
    gpu_hours_reserved: float = 0.0,
    gpu_hours_used: float = 0.0,
) -> dict:
    return {
        "id": 77,
        "agenda_id": 10,
        "token_reserved": token_reserved,
        "gpu_hours_reserved": gpu_hours_reserved,
        "gpu_hours_used": gpu_hours_used,
        "tokens_used": tokens_used,
        "status": status,
    }


def _lifecycle_checkpoints() -> list[dict]:
    rows = []
    for paper_id in ("p1", "p2"):
        rows.extend(
            [
                {
                    "paper_id": paper_id,
                    "stage": "extracted",
                    "payload": json.dumps({"claims": [{"text": "grounded"}]}),
                    "error_message": None,
                },
                {
                    "paper_id": paper_id,
                    "stage": "graph_written",
                    "payload": json.dumps(
                        {
                            "claim_count": 1,
                            "graph_entities": 2,
                            "graph_relations": 1,
                        }
                    ),
                    "error_message": None,
                },
                {
                    "paper_id": paper_id,
                    "stage": "reasoned",
                    "payload": json.dumps({"claims": 1}),
                    "error_message": None,
                },
            ]
        )
    return rows


class ScopedIngestionSettlementTests(unittest.TestCase):
    def setUp(self):
        self.repository = ScopedIngestionRepository()
        self.results = [
            {
                "paper_id": paper_id,
                "claims": 1,
                "graph_entities": 2,
                "graph_relations": 1,
            }
            for paper_id in ("p1", "p2")
        ]

    def _db(
        self,
        *,
        job=None,
        grant=None,
        paper_states=None,
        checkpoints=None,
        siblings=None,
        usage=None,
        gpu=None,
        ledger=None,
    ):
        patched = mock.patch("meta_harness.ingestion_queue.db")
        database = patched.start()
        self.addCleanup(patched.stop)
        database.fetchone.side_effect = [
            job or _job(),
            grant or _grant(),
            ledger or _ledger(),
        ]
        database.fetchall.side_effect = [
            paper_states
            if paper_states is not None
            else [
                {
                    "id": "p1",
                    "status": "reasoned",
                    "processing_stage": "reasoned",
                    "claim_count": 1,
                    "graph_entity_count": 2,
                    "graph_relation_count": 1,
                },
                {
                    "id": "p2",
                    "status": "reasoned",
                    "processing_stage": "reasoned",
                    "claim_count": 1,
                    "graph_entity_count": 2,
                    "graph_relation_count": 1,
                },
            ],
            checkpoints if checkpoints is not None else _lifecycle_checkpoints(),
            siblings if siblings is not None else [{"id": 25, "status": "running"}],
            usage
            if usage is not None
            else [
                {
                    "id": 1,
                    "status": "settled",
                    "token_reserved": 80,
                    "tokens_used": 60,
                },
                {
                    "id": 2,
                    "status": "released",
                    "token_reserved": 20,
                    "tokens_used": None,
                },
            ],
            gpu if gpu is not None else [],
        ]
        database.execute.return_value = _cursor()
        return database

    def test_success_settles_job_grant_ledger_and_agenda_in_one_commit(self):
        database = self._db()

        settled = self.repository.complete(
            25,
            agenda_id=10,
            worker_id="host:7:worker",
            results=self.results,
        )

        self.assertEqual(
            settled,
            {"status": "succeeded", "resource_grant_id": 193, "tokens_used": 60},
        )
        statements = [call.args[0] for call in database.execute.call_args_list]
        self.assertEqual(len(statements), 4)
        self.assertIn("UPDATE scoped_ingestion_jobs_v1", statements[0])
        self.assertIn("UPDATE research_agendas", statements[1])
        self.assertIn("UPDATE agenda_resource_ledger", statements[2])
        self.assertIn("UPDATE resource_grants", statements[3])
        database.commit.assert_called_once_with()
        database.rollback.assert_not_called()

    def test_duplicate_completion_is_read_only_and_idempotent(self):
        persisted = json.dumps(
            {"papers": self.results},
            ensure_ascii=False,
            sort_keys=True,
        )
        database = self._db(
            job=_job(status="succeeded", result_json=persisted),
            grant=_grant(status="consumed"),
            siblings=[{"id": 25, "status": "succeeded"}],
            ledger=_ledger(status="settled", tokens_used=60),
        )

        settled = self.repository.complete(
            25,
            agenda_id=10,
            worker_id="host:7:worker",
            results=self.results,
        )

        self.assertEqual(settled["status"], "already_succeeded")
        database.execute.assert_not_called()
        database.commit.assert_called_once_with()
        database.rollback.assert_not_called()

    def test_open_child_reservation_refuses_all_settlement(self):
        database = self._db(
            usage=[
                {
                    "id": 9,
                    "status": "reserved",
                    "token_reserved": 20,
                    "tokens_used": None,
                }
            ]
        )
        # No ledger lookup occurs after the open-child gate.
        database.fetchone.side_effect = [_job(), _grant()]

        with self.assertRaisesRegex(ScopedLLMError, "open child"):
            self.repository.complete(
                25,
                agenda_id=10,
                worker_id="host:7:worker",
                results=self.results,
            )

        database.execute.assert_not_called()
        database.commit.assert_not_called()
        database.rollback.assert_called_once_with()

    def test_non_success_sibling_refuses_all_settlement(self):
        database = self._db(
            siblings=[
                {"id": 25, "status": "running"},
                {"id": 26, "status": "queued"},
            ]
        )
        database.fetchone.side_effect = [_job(), _grant()]

        with self.assertRaisesRegex(ScopedLLMError, "one-to-one"):
            self.repository.complete(
                25,
                agenda_id=10,
                worker_id="host:7:worker",
                results=self.results,
            )

        database.execute.assert_not_called()
        database.rollback.assert_called_once_with()

    def test_even_succeeded_sibling_refuses_ambiguous_grant_settlement(self):
        database = self._db(
            siblings=[
                {"id": 24, "status": "succeeded"},
                {"id": 25, "status": "running"},
            ]
        )

        with self.assertRaisesRegex(ScopedLLMError, "one-to-one"):
            self.repository.complete(
                25,
                agenda_id=10,
                worker_id="host:7:worker",
                results=self.results,
            )

        database.execute.assert_not_called()
        database.commit.assert_not_called()
        database.rollback.assert_called_once_with()

    def test_caller_results_cannot_omit_duplicate_or_add_a_paper(self):
        invalid_results = (
            [self.results[0]],
            [self.results[0], self.results[0]],
            [self.results[0], {**self.results[1], "paper_id": "p3"}],
            [self.results[0], {**self.results[1], "error": "failed"}],
        )
        for results in invalid_results:
            with self.subTest(results=results):
                database = self._db()
                with self.assertRaisesRegex(ScopedLLMError, "exact paper set"):
                    self.repository.complete(
                        25,
                        agenda_id=10,
                        worker_id="host:7:worker",
                        results=results,
                    )
                database.execute.assert_not_called()
                database.commit.assert_not_called()
                database.rollback.assert_called_once_with()

    def test_database_paper_terminal_state_is_required_in_same_transaction(self):
        database = self._db(
            paper_states=[
                {
                    "id": "p1", "status": "reasoned",
                    "processing_stage": "reasoned", "claim_count": 1,
                    "graph_entity_count": 2, "graph_relation_count": 1,
                },
                {
                    "id": "p2", "status": "extracted",
                    "processing_stage": "extracted", "claim_count": 1,
                    "graph_entity_count": 2, "graph_relation_count": 1,
                },
            ]
        )

        with self.assertRaisesRegex(ScopedLLMError, "durably reasoned"):
            self.repository.complete(
                25,
                agenda_id=10,
                worker_id="host:7:worker",
                results=self.results,
            )

        query = database.fetchall.call_args_list[0].args[0]
        self.assertIn("FROM papers", query)
        self.assertIn("FOR UPDATE", query)
        database.execute.assert_not_called()
        database.commit.assert_not_called()
        database.rollback.assert_called_once_with()

    def test_caller_and_database_must_both_prove_claims_and_graph_rows(self):
        missing_result = [self.results[0], {**self.results[1], "graph_relations": 0}]
        database = self._db()
        with self.assertRaisesRegex(ScopedLLMError, "caller did not report"):
            self.repository.complete(
                25,
                agenda_id=10,
                worker_id="host:7:worker",
                results=missing_result,
            )
        database.fetchall.assert_not_called()

        database = self._db(
            paper_states=[
                {
                    "id": "p1", "status": "reasoned",
                    "processing_stage": "reasoned", "claim_count": 1,
                    "graph_entity_count": 2, "graph_relation_count": 1,
                },
                {
                    "id": "p2", "status": "reasoned",
                    "processing_stage": "reasoned", "claim_count": 1,
                    "graph_entity_count": 2, "graph_relation_count": 0,
                },
            ]
        )
        with self.assertRaisesRegex(ScopedLLMError, "persisted claims and graph"):
            self.repository.complete(
                25,
                agenda_id=10,
                worker_id="host:7:worker",
                results=self.results,
            )
        database.execute.assert_not_called()

    def test_every_paper_requires_extraction_claims_and_graph_checkpoints(self):
        incomplete = _lifecycle_checkpoints()
        for row in incomplete:
            if row["paper_id"] == "p2" and row["stage"] == "graph_written":
                row["payload"] = json.dumps(
                    {
                        "claim_count": 1,
                        "graph_entities": 2,
                        "graph_relations": 0,
                    }
                )
        database = self._db(checkpoints=incomplete)

        with self.assertRaisesRegex(ScopedLLMError, "full extraction/claims/graph"):
            self.repository.complete(
                25,
                agenda_id=10,
                worker_id="host:7:worker",
                results=self.results,
            )

        database.execute.assert_not_called()
        database.commit.assert_not_called()
        database.rollback.assert_called_once_with()

    def test_metered_usage_over_grant_cap_refuses_all_settlement(self):
        database = self._db(
            grant=_grant(token_cap=50),
            usage=[
                {
                    "id": 1,
                    "status": "settled",
                    "token_reserved": 80,
                    "tokens_used": 60,
                }
            ],
        )
        database.fetchone.side_effect = [_job(), _grant(token_cap=50)]

        with self.assertRaisesRegex(ScopedLLMError, "exceed ResourceGrant"):
            self.repository.complete(
                25,
                agenda_id=10,
                worker_id="host:7:worker",
                results=self.results,
            )

        database.execute.assert_not_called()
        database.rollback.assert_called_once_with()

    def test_parent_ledger_must_exactly_match_ingestion_grant_reservation(self):
        for ledger in (
            _ledger(token_reserved=101),
            _ledger(gpu_hours_reserved=0.25),
            _ledger(gpu_hours_reserved=float("nan")),
        ):
            with self.subTest(ledger=ledger):
                database = self._db(ledger=ledger)

                with self.assertRaisesRegex(
                    ScopedLLMError, "parent reservation does not exactly match"
                ):
                    self.repository.complete(
                        25,
                        agenda_id=10,
                        worker_id="host:7:worker",
                        results=self.results,
                    )

                database.execute.assert_not_called()
                database.commit.assert_not_called()
                database.rollback.assert_called_once_with()

    def test_ingestion_settlement_rejects_nonfinite_gpu_usage(self):
        database = self._db(ledger=_ledger(gpu_hours_used=float("nan")))

        with self.assertRaisesRegex(ScopedLLMError, "contains GPU usage"):
            self.repository.complete(
                25,
                agenda_id=10,
                worker_id="host:7:worker",
                results=self.results,
            )

        database.execute.assert_not_called()
        database.commit.assert_not_called()
        database.rollback.assert_called_once_with()

    def test_any_gpu_attempt_refuses_ingestion_settlement(self):
        database = self._db(gpu=[{"id": 4, "status": "released"}])
        database.fetchone.side_effect = [_job(), _grant()]

        with self.assertRaisesRegex(ScopedLLMError, "GPU attempt"):
            self.repository.complete(
                25,
                agenda_id=10,
                worker_id="host:7:worker",
                results=self.results,
            )

        database.execute.assert_not_called()
        database.rollback.assert_called_once_with()

    def test_write_failure_rolls_back_job_and_all_ledgers(self):
        database = self._db()
        database.execute.side_effect = [_cursor(), RuntimeError("agenda write failed")]

        with self.assertRaisesRegex(RuntimeError, "agenda write failed"):
            self.repository.complete(
                25,
                agenda_id=10,
                worker_id="host:7:worker",
                results=self.results,
            )

        database.commit.assert_not_called()
        database.rollback.assert_called_once_with()

    def test_failed_replacement_retains_its_reconciliation_origin(self):
        origin = {
            "version": "scoped-ingestion-reconciliation-v1",
            "source_job_id": 25,
        }
        with mock.patch("meta_harness.ingestion_queue.db") as database:
            database.fetchone.return_value = {
                "result_json": json.dumps({"recovery_origin": origin})
            }
            database.execute.return_value = _cursor()

            self.repository._finish(  # noqa: SLF001 - persistence invariant
                44,
                agenda_id=10,
                worker_id="host:7:worker",
                status="failed",
                result={"papers": []},
                failure_reason="test failure",
            )

        persisted = json.loads(database.execute.call_args.args[1][1])
        self.assertEqual(persisted["recovery_origin"], origin)
        database.commit.assert_called_once_with()
        database.rollback.assert_not_called()

    def test_retryable_failure_keeps_parent_grant_open_for_the_next_attempt(self):
        with mock.patch("meta_harness.ingestion_queue.db") as database, mock.patch.object(
            self.repository, "_finish"
        ) as finish, mock.patch.object(
            self.repository,
            "_settle_terminal_failure",
            side_effect=AssertionError("retryable work must not settle the grant"),
        ):
            database.fetchone.side_effect = [
                {
                    "attempt_count": 1,
                    "max_attempts": 3,
                    "resource_grant_id": 193,
                },
                None,
            ]

            status = self.repository.fail(
                25,
                agenda_id=10,
                worker_id="host:7:worker",
                reason="temporary provider timeout",
                retryable=True,
                partial_results=[{"paper_id": "p1"}],
            )

        self.assertEqual(status, "retryable")
        finish.assert_called_once()
        self.assertEqual(finish.call_args.kwargs["status"], "retryable")

    def test_retryable_failure_with_open_usage_parks_for_operator(self):
        with mock.patch("meta_harness.ingestion_queue.db") as database, mock.patch.object(
            self.repository, "_finish"
        ) as finish:
            database.fetchone.side_effect = [
                {
                    "attempt_count": 1,
                    "max_attempts": 3,
                    "resource_grant_id": 193,
                },
                {"id": 91},
            ]

            status = self.repository.fail(
                25,
                agenda_id=10,
                worker_id="host:7:worker",
                reason="provider delivery unknown",
                retryable=True,
                partial_results=[],
            )

        self.assertEqual(status, "manual_reconciliation")
        self.assertEqual(
            finish.call_args.kwargs["status"], "manual_reconciliation"
        )
        self.assertIn(
            "disposition_required", finish.call_args.kwargs["failure_reason"]
        )

    def test_terminal_failure_settles_metered_usage_and_closes_the_grant(self):
        with mock.patch("meta_harness.ingestion_queue.db") as database:
            database.fetchone.side_effect = [_job(), _grant(), _ledger()]
            database.fetchall.side_effect = [
                [{"id": 25}],
                [
                    {"id": 1, "status": "settled", "tokens_used": 60},
                    {"id": 2, "status": "released", "tokens_used": None},
                ],
                [],
            ]
            database.execute.return_value = _cursor()

            status = self.repository._settle_terminal_failure(  # noqa: SLF001
                25,
                agenda_id=10,
                worker_id="host:7:worker",
                result={"papers": [{"paper_id": "p1"}]},
                failure_reason="nonretryable extraction failure",
            )

        self.assertEqual(status, "failed")
        statements = [call.args[0] for call in database.execute.call_args_list]
        self.assertEqual(len(statements), 4)
        self.assertIn("status='failed'", statements[0])
        self.assertIn("UPDATE research_agendas", statements[1])
        self.assertIn("terminal_ingestion_failure", statements[2])
        self.assertIn("status='consumed'", statements[3])
        database.commit.assert_called_once_with()
        database.rollback.assert_not_called()

    def test_terminal_failure_with_open_usage_parks_without_guessing_spend(self):
        with mock.patch("meta_harness.ingestion_queue.db") as database:
            database.fetchone.side_effect = [_job(), _grant()]
            database.fetchall.side_effect = [
                [{"id": 25}],
                [{"id": 9, "status": "reserved", "tokens_used": None}],
            ]
            database.execute.return_value = _cursor()

            status = self.repository._settle_terminal_failure(  # noqa: SLF001
                25,
                agenda_id=10,
                worker_id="host:7:worker",
                result={"papers": []},
                failure_reason="worker lost provider delivery state",
            )

        self.assertEqual(status, "manual_reconciliation")
        self.assertEqual(database.execute.call_count, 1)
        statement = database.execute.call_args.args[0]
        self.assertIn("status='manual_reconciliation'", statement)
        self.assertNotIn("agenda_resource_ledger", statement)
        database.commit.assert_called_once_with()
        database.rollback.assert_not_called()

    def test_terminal_failure_refuses_parent_reservation_mismatch(self):
        for ledger in (
            _ledger(token_reserved=99),
            _ledger(gpu_hours_reserved=0.5),
            _ledger(gpu_hours_reserved=float("nan")),
        ):
            with self.subTest(ledger=ledger), mock.patch(
                "meta_harness.ingestion_queue.db"
            ) as database:
                database.fetchone.side_effect = [_job(), _grant(), ledger]
                database.fetchall.side_effect = [
                    [{"id": 25}],
                    [{"id": 1, "status": "settled", "tokens_used": 60}],
                    [],
                ]

                with self.assertRaisesRegex(
                    ScopedLLMError, "parent reservation does not exactly"
                ):
                    self.repository._settle_terminal_failure(  # noqa: SLF001
                        25,
                        agenda_id=10,
                        worker_id="host:7:worker",
                        result={"papers": []},
                        failure_reason="nonretryable extraction failure",
                    )

                database.execute.assert_not_called()
                database.commit.assert_not_called()
                database.rollback.assert_called_once_with()

    def test_terminal_failure_rejects_nonfinite_gpu_usage(self):
        with mock.patch("meta_harness.ingestion_queue.db") as database:
            database.fetchone.side_effect = [
                _job(),
                _grant(),
                _ledger(gpu_hours_used=float("nan")),
            ]
            database.fetchall.side_effect = [
                [{"id": 25}],
                [{"id": 1, "status": "settled", "tokens_used": 60}],
                [],
            ]

            with self.assertRaisesRegex(ScopedLLMError, "contains GPU usage"):
                self.repository._settle_terminal_failure(  # noqa: SLF001
                    25,
                    agenda_id=10,
                    worker_id="host:7:worker",
                    result={"papers": []},
                    failure_reason="nonretryable extraction failure",
                )

        database.execute.assert_not_called()
        database.commit.assert_not_called()
        database.rollback.assert_called_once_with()


class ScopedIngestionReconciliationTests(unittest.TestCase):
    def setUp(self):
        self.repository = ScopedIngestionRepository()
        self.request = ScopedIngestionReconciliationRequest(
            source_job_id=25,
            agenda_id=10,
            idea_id=125,
            source_resource_grant_id=193,
            replacement_resource_grant_id=240,
            stage="ingestion_backfill_canary",
            paper_ids=("p1", "p2"),
            actor="operator@example",
            reason="job 25 deterministic reservation collision",
            idempotency_key="reconcile-job-25-v1",
            max_attempts=3,
        )

    def test_exact_failed_source_creates_one_audited_replacement(self):
        source = _job(
            status="failed",
            result_json=json.dumps({"papers": [{"paper_id": "p1"}]}),
        )
        source["lease_owner"] = None
        old_grant = {
            "agenda_id": 10,
            "idea_id": 125,
            "stage": "ingestion_backfill_canary",
            "status": "expired",
            "ledger_status": "settled",
        }
        new_grant = {
            "agenda_id": 10,
            "idea_id": 125,
            "stage": "ingestion_backfill_canary",
            "status": "active",
            "backend_allowlist_json": '["llm"]',
            "ledger_status": "reserved",
        }
        with mock.patch("meta_harness.ingestion_queue.db") as database:
            database._use_pg.return_value = True
            database.fetchone.side_effect = [
                source,
                old_grant,
                None,
                new_grant,
                None,
                None,
                {"count": 2},
            ]
            database.fetchall.return_value = []
            database.execute.side_effect = [_cursor(1)]
            database.insert_returning_id.return_value = 44

            result = self.repository.reconcile_failed_job(self.request)

        self.assertEqual(result["status"], "queued")
        self.assertEqual(result["replacement_job_id"], 44)
        inserted = database.insert_returning_id.call_args.args[1]
        origin = json.loads(inserted[-1])["recovery_origin"]
        self.assertEqual(origin["source_job_id"], 25)
        self.assertEqual(origin["paper_ids"], ["p1", "p2"])
        self.assertEqual(origin["actor"], "operator@example")
        source_update = database.execute.call_args_list[-1].args[1][0]
        audit = json.loads(source_update)["reconciliation"]
        self.assertEqual(audit["replacement_job_id"], 44)
        self.assertEqual(audit["released_source_reservations"], 0)
        database.commit.assert_called_once_with()
        database.rollback.assert_not_called()

    def test_reconciliation_request_rejects_noninteger_exact_scope(self):
        for field, value in (
            ("agenda_id", 10.5),
            ("source_resource_grant_id", True),
            ("max_attempts", 3.5),
        ):
            with self.subTest(field=field), self.assertRaisesRegex(
                ScopedLLMError, "must be integers"
            ):
                replace(self.request, **{field: value}).validate()

    def test_reconciliation_is_idempotent_only_for_the_exact_same_audit(self):
        audit = {
            "version": "scoped-ingestion-reconciliation-v1",
            "action": "replace_failed_job",
            "source_job_id": 25,
            "source_resource_grant_id": 193,
            "replacement_resource_grant_id": 240,
            "agenda_id": 10,
            "idea_id": 125,
            "stage": "ingestion_backfill_canary",
            "paper_ids": ["p1", "p2"],
            "actor": "operator@example",
            "reason": "job 25 deterministic reservation collision",
            "idempotency_key": "reconcile-job-25-v1",
            "max_attempts": 3,
            "recorded_at": "2026-08-24T00:00:00+00:00",
            "replacement_job_id": 44,
            "released_source_reservations": 0,
        }
        source = _job(
            status="failed",
            result_json=json.dumps({"reconciliation": audit}),
        )
        replacement = {
            "id": 44,
            "agenda_id": 10,
            "idea_id": 125,
            "resource_grant_id": 240,
            "stage": "ingestion_backfill_canary",
            "idempotency_key": "reconcile-job-25-v1",
            "paper_ids_json": '["p1", "p2"]',
            "max_attempts": 3,
        }
        with mock.patch("meta_harness.ingestion_queue.db") as database:
            database._use_pg.return_value = True
            database.fetchone.side_effect = [source, replacement]

            result = self.repository.reconcile_failed_job(self.request)

        self.assertEqual(result["status"], "already_reconciled")
        self.assertEqual(result["replacement_job_id"], 44)
        database.execute.assert_not_called()
        database.insert_returning_id.assert_not_called()
        database.commit.assert_called_once_with()
        database.rollback.assert_not_called()

    def test_historical_shared_source_grant_requires_manual_reconciliation(self):
        source = _job(
            status="failed",
            result_json=json.dumps({"papers": [{"paper_id": "p1"}]}),
        )
        source["lease_owner"] = None
        old_grant = {
            "agenda_id": 10,
            "idea_id": 125,
            "stage": "ingestion_backfill_canary",
            "status": "expired",
            "ledger_status": "settled",
        }
        with mock.patch("meta_harness.ingestion_queue.db") as database:
            database._use_pg.return_value = True
            database.fetchone.side_effect = [source, old_grant, {"id": 24}]

            with self.assertRaisesRegex(ScopedLLMError, "not one-to-one"):
                self.repository.reconcile_failed_job(self.request)

        database.execute.assert_not_called()
        database.insert_returning_id.assert_not_called()
        database.commit.assert_not_called()
        database.rollback.assert_called_once_with()

    def test_open_source_usage_must_be_disposed_before_replacement(self):
        source = _job(status="failed")
        source["lease_owner"] = None
        old_grant = {
            "agenda_id": 10,
            "idea_id": 125,
            "stage": "ingestion_backfill_canary",
            "status": "expired",
            "ledger_status": "settled",
        }
        new_grant = {
            "agenda_id": 10,
            "idea_id": 125,
            "stage": "ingestion_backfill_canary",
            "status": "active",
            "backend_allowlist_json": '["llm"]',
            "ledger_status": "reserved",
        }
        with mock.patch("meta_harness.ingestion_queue.db") as database:
            database._use_pg.return_value = True
            database.fetchone.side_effect = [
                source,
                old_grant,
                None,
                new_grant,
                None,
                None,
                {"count": 2},
            ]
            database.fetchall.return_value = [{"id": 91}]

            with self.assertRaisesRegex(ScopedLLMError, "exact usage disposition"):
                self.repository.reconcile_failed_job(self.request)

        database.execute.assert_not_called()
        database.insert_returning_id.assert_not_called()
        database.rollback.assert_called_once_with()


class ScopedIngestionReconciliationRouteTests(unittest.TestCase):
    def setUp(self):
        app = Flask(__name__)
        app.register_blueprint(blueprint)
        self.client = app.test_client()
        self.path = "/api/meta-harness/v1/ingestion/jobs/25/reconcile"
        self.payload = {
            "agenda_id": 10,
            "idea_id": 125,
            "source_resource_grant_id": 193,
            "replacement_resource_grant_id": 240,
            "stage": "ingestion_backfill_canary",
            "paper_ids": ["p1", "p2"],
            "actor": "operator@example",
            "reason": "job 25 deterministic reservation collision",
            "idempotency_key": "reconcile-job-25-v1",
            "max_attempts": 3,
        }

    def test_route_refuses_unauthenticated_reconciliation(self):
        with mock.patch.dict(
            os.environ,
            {"DEEPGRAPH_META_HARNESS_OPERATOR_TOKEN": "secret"},
        ), mock.patch(
            "web.meta_harness_routes.ScopedIngestionRepository.reconcile_failed_job"
        ) as reconcile:
            response = self.client.post(self.path, json=self.payload)

        self.assertEqual(response.status_code, 403)
        reconcile.assert_not_called()

    def test_route_forwards_every_exact_scope_field_after_auth(self):
        with mock.patch.dict(
            os.environ,
            {"DEEPGRAPH_META_HARNESS_OPERATOR_TOKEN": "secret"},
        ), mock.patch(
            "web.meta_harness_routes.ScopedIngestionRepository.reconcile_failed_job",
            return_value={
                "status": "queued",
                "source_job_id": 25,
                "replacement_job_id": 44,
                "released_source_reservations": 0,
            },
        ) as reconcile:
            response = self.client.post(
                self.path,
                json=self.payload,
                headers={"X-DeepGraph-Operator-Token": "secret"},
            )

        self.assertEqual(response.status_code, 202)
        request = reconcile.call_args.args[0]
        self.assertEqual(request.source_job_id, 25)
        self.assertEqual(request.source_resource_grant_id, 193)
        self.assertEqual(request.replacement_resource_grant_id, 240)
        self.assertEqual(request.paper_ids, ("p1", "p2"))
        self.assertEqual(request.actor, "operator@example")
        self.assertEqual(request.reason, self.payload["reason"])

    def test_route_rejects_noninteger_exact_scope_without_truncation(self):
        for field, value in (
            ("agenda_id", 10.5),
            ("source_resource_grant_id", True),
            ("max_attempts", 3.5),
        ):
            with self.subTest(field=field), mock.patch.dict(
                os.environ,
                {"DEEPGRAPH_META_HARNESS_OPERATOR_TOKEN": "secret"},
            ), mock.patch(
                "web.meta_harness_routes.ScopedIngestionRepository.reconcile_failed_job"
            ) as reconcile:
                response = self.client.post(
                    self.path,
                    json={**self.payload, field: value},
                    headers={"X-DeepGraph-Operator-Token": "secret"},
                )

                self.assertEqual(response.status_code, 400)
                reconcile.assert_not_called()


class ScopedIngestionUsageDispositionRouteTests(unittest.TestCase):
    def setUp(self):
        app = Flask(__name__)
        app.register_blueprint(blueprint)
        self.client = app.test_client()
        self.path = "/api/meta-harness/v1/ingestion/jobs/25/usage/91/disposition"
        self.payload = {
            "agenda_id": 10,
            "idea_id": 125,
            "resource_grant_id": 193,
            "stage": "ingestion_backfill_canary",
            "paper_ids": ["p1", "p2"],
            "operation": "paper_extraction:p1:claims_methods",
            "usage_idempotency_key": "exact-attempt:t1",
            "expected_token_reserved": 100,
            "disposition": "release_confirmed_unbilled",
            "tokens_used": None,
            "cost_usd": None,
            "actor": "operator@example",
            "reason": "provider request log confirms no accepted request",
            "evidence_ref": "provider-log/request-123:not-found",
            "operator_request_id": "job25-usage91-v1",
            "resume": True,
        }

    def test_route_refuses_unauthenticated_disposition(self):
        with mock.patch.dict(
            os.environ,
            {"DEEPGRAPH_META_HARNESS_OPERATOR_TOKEN": "secret"},
        ), mock.patch(
            "web.meta_harness_routes.ScopedIngestionRepository.dispose_open_usage"
        ) as dispose:
            response = self.client.post(self.path, json=self.payload)

        self.assertEqual(response.status_code, 403)
        dispose.assert_not_called()

    def test_route_forwards_the_complete_exact_scope(self):
        with mock.patch.dict(
            os.environ,
            {"DEEPGRAPH_META_HARNESS_OPERATOR_TOKEN": "secret"},
        ), mock.patch(
            "web.meta_harness_routes.ScopedIngestionRepository.dispose_open_usage",
            return_value={
                "status": "disposed",
                "usage_status": "released",
                "job_status": "retryable",
            },
        ) as dispose:
            response = self.client.post(
                self.path,
                json=self.payload,
                headers={"X-DeepGraph-Operator-Token": "secret"},
            )

        self.assertEqual(response.status_code, 200)
        request_value = dispose.call_args.args[0]
        self.assertIsInstance(request_value, ScopedIngestionUsageDispositionRequest)
        self.assertEqual(request_value.job_id, 25)
        self.assertEqual(request_value.usage_reservation_id, 91)
        self.assertEqual(request_value.resource_grant_id, 193)
        self.assertEqual(request_value.paper_ids, ("p1", "p2"))
        self.assertEqual(request_value.evidence_ref, self.payload["evidence_ref"])
        self.assertTrue(request_value.resume)

    def test_request_rejects_noninteger_or_nonfinite_evidence_values(self):
        request_value = ScopedIngestionUsageDispositionRequest(
            job_id=25,
            agenda_id=10,
            idea_id=125,
            resource_grant_id=193,
            stage="ingestion_backfill_canary",
            paper_ids=("p1", "p2"),
            usage_reservation_id=91,
            operation="paper_extraction:p1:claims_methods",
            usage_idempotency_key="exact-attempt:t1",
            expected_token_reserved=100,
            disposition="settle_measured",
            tokens_used=10,
            cost_usd=0.01,
            actor="operator@example",
            reason="provider invoice confirms usage",
            evidence_ref="provider-log/request-123:measured",
            operator_request_id="job25-usage91-v1",
            resume=False,
        )
        for field, value in (
            ("agenda_id", 10.5),
            ("expected_token_reserved", True),
            ("tokens_used", 1.5),
            ("cost_usd", float("inf")),
        ):
            with self.subTest(field=field), self.assertRaises(ScopedLLMError):
                replace(request_value, **{field: value}).validate()

    def test_route_rejects_noninteger_or_nonfinite_values_without_truncation(self):
        for field, value in (
            ("agenda_id", 10.5),
            ("expected_token_reserved", 100.5),
            ("tokens_used", 1.5),
            ("cost_usd", True),
            ("resume", "false"),
        ):
            with self.subTest(field=field), mock.patch.dict(
                os.environ,
                {"DEEPGRAPH_META_HARNESS_OPERATOR_TOKEN": "secret"},
            ), mock.patch(
                "web.meta_harness_routes.ScopedIngestionRepository.dispose_open_usage"
            ) as dispose:
                response = self.client.post(
                    self.path,
                    json={**self.payload, field: value},
                    headers={"X-DeepGraph-Operator-Token": "secret"},
                )

                self.assertEqual(response.status_code, 400)
                dispose.assert_not_called()


if __name__ == "__main__":
    unittest.main()
