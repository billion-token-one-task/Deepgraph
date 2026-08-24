"""Scoped-ingestion grants cannot mutate or impersonate research authority."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import unittest
from unittest import mock

from contracts.meta_harness import ResourceGrant
from meta_harness.grant_stages import (
    INGESTION_GRANT_LANE,
    INITIAL_RESEARCH_GRANT_LANE,
    LATER_RESEARCH_GRANT_LANE,
    PROPOSAL_GRANT_LANE,
    ResourceGrantStageError,
    classify_resource_grant_stage,
)
from meta_harness.ingestion_queue import (
    ScopedIngestionReconciliationRequest,
    ScopedIngestionRepository,
    ScopedIngestionRequest,
)
from meta_harness.repository import (
    MetaHarnessPersistenceError,
    MetaHarnessRepository,
)


def _grant(stage: str, **overrides) -> ResourceGrant:
    values = {
        "agenda_id": 2,
        "idea_id": 115,
        "decision_packet_id": 41,
        "stage": stage,
        "token_cap": 1_000,
        "gpu_class": "none",
        "max_gpu_hours": 0.0,
        "backend_allowlist": ["llm"],
        "artifact_requirements": ["claims", "graph_delta"],
        "expires_at": (
            datetime.now(timezone.utc) + timedelta(hours=1)
        ).isoformat(),
        "grant_reason": "controlled_canary",
        "idempotency_key": f"grant-{stage}",
    }
    values.update(overrides)
    return ResourceGrant(**values)


def _agenda() -> dict:
    return {
        "id": 2,
        "is_active": 1,
        "status": "active",
        "max_concurrency": 4,
        "token_budget": 50_000,
        "token_spent": 0,
        "token_reserved": 0,
        "gpu_hours_budget": 0.0,
        "gpu_hours_spent": 0.0,
        "gpu_hours_reserved": 0.0,
        "backend_allowlist_json": '["cpu","llm"]',
        "prefer_json": "{}",
    }


def _decision_row() -> dict:
    return {"agenda_id": 2, "idea_id": 115, "decision": "promote"}


def _existing_grant_row(grant: ResourceGrant, *, grant_id: int = 193) -> dict:
    return {
        "id": grant_id,
        "agenda_id": grant.agenda_id,
        "idea_id": grant.idea_id,
        "decision_packet_id": grant.decision_packet_id,
        "stage": grant.stage,
        "token_cap": grant.token_cap,
        "gpu_class": grant.gpu_class,
        "max_gpu_hours": grant.max_gpu_hours,
        "backend_allowlist_json": json.dumps(grant.backend_allowlist),
        "artifact_requirements_json": json.dumps(grant.artifact_requirements),
        # Exercise the PostgreSQL TIMESTAMPTZ representation rather than only
        # comparing identical strings.
        "expires_at": datetime.fromisoformat(grant.expires_at),
        "grant_reason": grant.grant_reason,
        "reservation_id": 71,
        "preflight_result_id": grant.preflight_result_id,
    }


def _issue_and_capture(stage: str):
    agenda = {**_agenda(), "backend_allowlist_json": '["llm"]'}
    decision = _decision_row()
    with (
        mock.patch("meta_harness.repository.db._use_pg", return_value=False),
        mock.patch(
            "meta_harness.repository.db.fetchone",
            side_effect=[agenda, decision, None, {"count": 0}],
        ),
        mock.patch(
            "meta_harness.repository.db.insert_returning_id",
            side_effect=[71, 193],
        ),
        mock.patch("meta_harness.repository.db.execute") as execute,
        mock.patch("meta_harness.repository.db.commit"),
        mock.patch("meta_harness.repository.db.rollback"),
    ):
        grant_id = MetaHarnessRepository().issue_grant(_grant(stage))
    return grant_id, execute.call_args_list


class GrantStageClassificationTests(unittest.TestCase):
    def test_supported_stages_have_one_exact_lane(self):
        self.assertEqual(
            classify_resource_grant_stage("proposal"), PROPOSAL_GRANT_LANE
        )
        self.assertEqual(
            classify_resource_grant_stage("pilot"), INITIAL_RESEARCH_GRANT_LANE
        )
        for stage in ("validation", "full_benchmark", "evidence_audit", "manuscript"):
            self.assertEqual(
                classify_resource_grant_stage(stage), LATER_RESEARCH_GRANT_LANE
            )
        for stage in ("ingestion", "ingestion_backfill_canary", "ingestion_v1"):
            self.assertEqual(
                classify_resource_grant_stage(stage), INGESTION_GRANT_LANE
            )

    def test_fuzzy_or_misspelled_stages_are_rejected(self):
        for stage in (
            "ingestionish",
            "ingestion-backfill",
            "ingestion_",
            "ingestion__canary",
            "Ingestion",
            " ingestion",
            "pilot_ingestion",
            "experimental",
        ):
            with self.subTest(stage=stage), self.assertRaises(ResourceGrantStageError):
                classify_resource_grant_stage(stage)


class GrantIssueIsolationTests(unittest.TestCase):
    def test_logically_closed_agenda_cannot_receive_a_grant(self):
        with mock.patch("meta_harness.repository.db") as database:
            database._use_pg.return_value = False
            database.fetchone.return_value = {
                **_agenda(),
                "is_active": 0,
                "status": "active",
            }
            with self.assertRaisesRegex(
                MetaHarnessPersistenceError, "agenda is not active"
            ):
                MetaHarnessRepository().issue_grant(
                    _grant("ingestion_backfill_canary")
                )

        database.insert_returning_id.assert_not_called()
        database.rollback.assert_called_once_with()

    def test_new_grant_cannot_start_in_a_terminal_state(self):
        for status in ("consumed", "expired", "revoked"):
            with self.subTest(status=status), mock.patch(
                "meta_harness.repository.db.fetchone",
                side_effect=AssertionError("database must not be touched"),
            ):
                with self.assertRaisesRegex(
                    MetaHarnessPersistenceError, "must start active"
                ):
                    MetaHarnessRepository().issue_grant(
                        _grant("ingestion_backfill_canary", status=status)
                    )

    def test_ingestion_grant_does_not_update_auto_research_job(self):
        grant_id, calls = _issue_and_capture("ingestion_backfill_canary")

        self.assertEqual(grant_id, 193)
        statements = " ".join(str(call.args[0]) for call in calls)
        self.assertNotIn("UPDATE auto_research_jobs", statements)

    def test_only_pilot_performs_initial_research_binding(self):
        _, calls = _issue_and_capture("pilot")

        statements = " ".join(str(call.args[0]) for call in calls)
        self.assertIn("UPDATE auto_research_jobs", statements)
        self.assertIn("stage='portfolio_granted'", statements)

    def test_later_research_grant_does_not_rebind_an_initial_job(self):
        _, calls = _issue_and_capture("full_benchmark")

        statements = " ".join(str(call.args[0]) for call in calls)
        self.assertNotIn("UPDATE auto_research_jobs", statements)

    def test_unsupported_stage_fails_before_any_database_access(self):
        with mock.patch(
            "meta_harness.repository.db.fetchone", side_effect=AssertionError
        ):
            with self.assertRaisesRegex(
                MetaHarnessPersistenceError, "unsupported ResourceGrant stage"
            ):
                MetaHarnessRepository().issue_grant(_grant("ingestionish"))

    def test_ingestion_lane_cannot_hide_compute_authority(self):
        grant = _grant(
            "ingestion_backfill_canary",
            gpu_class="a10",
            max_gpu_hours=0.25,
            backend_allowlist=["llm", "ssh_gpu"],
        )
        with mock.patch(
            "meta_harness.repository.db.fetchone", side_effect=AssertionError
        ):
            with self.assertRaisesRegex(
                MetaHarnessPersistenceError, "token-only and LLM-only"
            ):
                MetaHarnessRepository().issue_grant(grant)


class ScopedIngestionAdmissionTests(unittest.TestCase):
    def test_fresh_enqueue_refuses_a_previously_reasoned_paper(self):
        request = ScopedIngestionRequest(
            agenda_id=2,
            idea_id=115,
            resource_grant_id=193,
            stage="ingestion_backfill_canary",
            idempotency_key="paper-canary-1",
            paper_ids=("2608.00001",),
        )
        grant = {
            "agenda_id": 2,
            "idea_id": 115,
            "stage": "ingestion_backfill_canary",
            "status": "active",
            "backend_allowlist_json": '["llm"]',
            "max_gpu_hours": 0.0,
            "ledger_status": "reserved",
        }
        with mock.patch("meta_harness.ingestion_queue.db") as database:
            database._use_pg.return_value = True
            database.fetchone.side_effect = [
                None,
                grant,
                None,
                {"count": 1, "reasoned_count": 1},
            ]

            with self.assertRaisesRegex(Exception, "already reasoned"):
                ScopedIngestionRepository().enqueue(request)

        database.insert_returning_id.assert_not_called()
        database.rollback.assert_called_once_with()

    def test_queue_refuses_a_research_grant_stage(self):
        request = ScopedIngestionRequest(
            agenda_id=2,
            idea_id=115,
            resource_grant_id=193,
            stage="pilot",
            idempotency_key="paper-canary-1",
            paper_ids=("2608.00001",),
        )
        with self.assertRaisesRegex(Exception, "requires an ingestion"):
            request.validate()

    def test_reconciliation_refuses_a_fuzzy_ingestion_stage(self):
        request = ScopedIngestionReconciliationRequest(
            source_job_id=25,
            agenda_id=2,
            idea_id=115,
            source_resource_grant_id=193,
            replacement_resource_grant_id=194,
            stage="ingestionish",
            paper_ids=("2608.18972",),
            actor="controlled-recovery",
            reason="replace exact failed job",
            idempotency_key="reconcile-25-1",
        )
        with self.assertRaisesRegex(Exception, "unsupported ResourceGrant stage"):
            request.validate()


if __name__ == "__main__":
    unittest.main()
