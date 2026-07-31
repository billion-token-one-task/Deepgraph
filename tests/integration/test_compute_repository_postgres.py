"""Durable compute lifecycle tests for an isolated PostgreSQL process only.

Run this file by itself after restoring production data into a disposable
database. Test discovery on a production host is explicitly unsupported.
"""

from __future__ import annotations

import os
import re
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
    if parsed.scheme not in {"postgres", "postgresql"}:
        return False
    if not any(marker in database for marker in ISOLATED_MARKERS):
        return False
    return URL != os.environ.get("DEEPGRAPH_DATABASE_URL", "").strip()


@unittest.skipUnless(_safe_url(), "explicit isolated PostgreSQL process required")
class IsolatedComputeRepositoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # This test must be its own process so config/db cannot have captured a
        # different database URL during earlier test imports.
        os.environ["DEEPGRAPH_DATABASE_URL"] = URL
        apply_to_isolated_restore(URL, source_commit=SOURCE_COMMIT)

        from db import database as database
        from meta_harness.compute import (
            ArtifactCollection,
            ComputeBackendError,
            ComputeJob,
            ComputeSubmission,
            UsageAccounting,
        )
        from meta_harness.compute_repository import ComputeJobRepository

        if not database._use_pg() or database.DATABASE_URL.strip() != URL:  # noqa: SLF001
            raise RuntimeError(
                "database module was initialized before isolated URL selection"
            )
        cls.db = database
        cls.ArtifactCollection = ArtifactCollection
        cls.ComputeBackendError = ComputeBackendError
        cls.ComputeJob = ComputeJob
        cls.ComputeSubmission = ComputeSubmission
        cls.UsageAccounting = UsageAccounting
        cls.ComputeJobRepository = ComputeJobRepository

    def setUp(self):
        self.namespace = f"compute_repo_{uuid.uuid4().hex}"
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
                (self.agenda_id, 700001, frontier_id),
            )
            decision_id = int(cur.fetchone()["id"])
            cur.execute(
                """
                INSERT INTO agenda_resource_ledger
                    (agenda_id, operation, idempotency_key, token_reserved,
                     status)
                VALUES (%s, 'resource_grant', %s, 1, 'reserved')
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
                VALUES (%s, %s, %s, 'pilot', 1, 0, '["cpu"]',
                        '["raw_metrics"]', %s, 'isolated test', %s, 'active',
                        %s)
                RETURNING id
                """,
                (
                    self.agenda_id,
                    700001,
                    decision_id,
                    datetime.now(timezone.utc) + timedelta(minutes=10),
                    reservation_id,
                    f"grant:{self.namespace}",
                ),
            )
            self.grant_id = int(cur.fetchone()["id"])
        self.db.commit()

    def tearDown(self):
        # The target is disposable, but cleanup makes repeat execution easier.
        for table in (
            "compute_jobs_v1",
            "resource_grants",
            "agenda_resource_ledger",
            "idea_decision_packets",
            "frontier_packets",
            "research_agendas",
        ):
            self.db.execute(
                f"DELETE FROM {table} WHERE agenda_id=?",
                (self.agenda_id,),
            )
        self.db.commit()

    def _submission(self, suffix: str = "main"):
        return self.ComputeSubmission(
            agenda_id=self.agenda_id,
            idea_id=700001,
            stage="pilot",
            resource_grant_id=self.grant_id,
            idempotency_key=f"{self.namespace}:{suffix}",
            command_ref=f"artifact:commands/{suffix}.json",
            artifact_namespace=f"{self.namespace}/{suffix}",
            timeout_seconds=60,
        )

    def test_restart_reuses_live_job_and_success_requires_artifacts_and_usage(self):
        request = self._submission()
        first_repository = self.ComputeJobRepository()
        claim = first_repository.claim(request, backend_kind="cpu")
        self.assertTrue(claim.is_new)
        first_repository.bind_submitted_job(
            claim.record_id,
            self.ComputeJob(
                "cpu",
                f"backend:{self.namespace}",
                request.idempotency_key,
                "submitted",
            ),
        )

        after_restart = self.ComputeJobRepository()
        duplicate = after_restart.claim(request, backend_kind="cpu")
        self.assertFalse(duplicate.is_new)
        self.assertEqual(
            duplicate.existing_job().backend_job_id,
            f"backend:{self.namespace}",
        )
        self.assertEqual(
            after_restart.record_backend_state(
                self.ComputeJob(
                    "cpu",
                    f"backend:{self.namespace}",
                    request.idempotency_key,
                    "succeeded",
                )
            ),
            "collecting",
        )
        after_restart.finalize_success(
            claim.record_id,
            artifacts=self.ArtifactCollection(
                {"raw_metrics": {"uri": "artifact:raw/metrics.json"}},
                True,
            ),
            usage=self.UsageAccounting(2.0, 0.0, 0.01),
        )
        row = self.db.fetchone(
            "SELECT status, artifact_manifest_json, usage_json "
            "FROM compute_jobs_v1 WHERE id=?",
            (claim.record_id,),
        )
        self.assertEqual(row["status"], "succeeded")
        self.assertTrue(row["artifact_manifest_json"])
        self.assertTrue(row["usage_json"])

    def test_unknown_submission_is_not_reclaimed_for_resubmission(self):
        request = self._submission("unknown")
        repository = self.ComputeJobRepository()
        claim = repository.claim(request, backend_kind="cpu")
        repository.mark_submission_unknown(
            claim.record_id,
            reason="connection_lost_after_submit",
        )

        duplicate = self.ComputeJobRepository().claim(
            request,
            backend_kind="cpu",
        )
        self.assertFalse(duplicate.is_new)
        self.assertEqual(duplicate.status, "submission_unknown")
        with self.assertRaisesRegex(
            self.ComputeBackendError,
            "submission_reconciliation_required",
        ):
            duplicate.existing_job()


if __name__ == "__main__":
    unittest.main()
