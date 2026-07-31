"""Pure/mocked contracts for durable Colab and scoped ingestion."""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from meta_harness.backends.colab_durable import (
    ColabWorkSpec,
    DurableColabTransport,
)
from meta_harness.compute import ColabAccount, ComputeSubmission
from meta_harness.ingestion_queue import ScopedIngestionRequest


class _Executor:
    class _Config:
        gpu_type = "T4"

    config = _Config()


class DurableColabContractTests(unittest.TestCase):
    def test_work_spec_rejects_artifact_escape(self):
        spec = ColabWorkSpec(
            agenda_id=1,
            idea_id=2,
            experiment_run_id=3,
            resource_grant_id=4,
            stage="pilot",
            idempotency_key="colab-1",
            code_dir="/isolated/code",
            command_tokens=("python", "train.py"),
            environment={},
            artifact_map={"raw_metrics": "../metrics.json"},
            artifact_output_dir="/isolated/artifacts/1",
            timeout_seconds=60,
        )
        with self.assertRaisesRegex(Exception, "safe relative"):
            spec.validate()

    @mock.patch("meta_harness.backends.colab_durable.db.fetchone")
    def test_submit_only_binds_durable_identity(self, fetchone):
        fetchone.return_value = {
            "id": 9,
            "agenda_id": 1,
            "idea_id": 2,
            "resource_grant_id": 4,
            "stage": "pilot",
            "idempotency_key": "colab-1",
            "status": "admitting",
        }
        account = ColabAccount(
            account_ref="account-1",
            credential_ref="env:COLAB_ACCOUNT_1",
            isolated_home="/isolated/home/account-1",
            oauth_store="/isolated/home/account-1/oauth",
            session_namespace="account-1",
            quota_gpu_hours=1,
        )
        transport = DurableColabTransport(
            executor=_Executor(),
            accounts=(account,),
        )
        job = transport.submit(
            ComputeSubmission(
                agenda_id=1,
                idea_id=2,
                stage="pilot",
                resource_grant_id=4,
                idempotency_key="colab-1",
                command_ref="colab-work-request:9",
                artifact_namespace="agenda-1/idea-2/colab-9",
                timeout_seconds=60,
                requested_gpu_hours=1 / 60,
            )
        )
        self.assertEqual(job.backend_job_id, "colab-work-request:9")
        self.assertEqual(job.status, "submitted")


class ScopedIngestionContractTests(unittest.TestCase):
    def test_request_requires_bounded_existing_identity_list(self):
        request = ScopedIngestionRequest(
            agenda_id=1,
            idea_id=2,
            resource_grant_id=3,
            stage="ingestion",
            idempotency_key="ingest-1",
            paper_ids=tuple(str(value) for value in range(101)),
        )
        with self.assertRaisesRegex(Exception, "100-paper"):
            request.validate()


if __name__ == "__main__":
    unittest.main()
