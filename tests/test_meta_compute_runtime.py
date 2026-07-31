"""Isolated-CI contracts for legacy runtime adapters.

These tests use mocks only. They are intentionally not executed on the
production host because importing application modules can initialize config.
"""

from __future__ import annotations

import unittest
from unittest import mock

from meta_harness.compute import ComputeBackendError, ComputeSubmission
from orchestrator import meta_compute_runtime


class LegacyCPUValidationTransportTests(unittest.TestCase):
    def _submission(self) -> ComputeSubmission:
        return ComputeSubmission(
            agenda_id=2,
            idea_id=3,
            stage="pilot",
            resource_grant_id=5,
            idempotency_key="cpu-a2-i3-r7",
            command_ref="experiment-run:7",
            artifact_namespace="agenda-2/idea-3/run-7",
            timeout_seconds=600,
        )

    @mock.patch.object(meta_compute_runtime.db, "fetchone")
    def test_submit_requires_exact_run_scope(self, fetchone):
        fetchone.return_value = {
            "id": 7,
            "agenda_id": 2,
            "deep_insight_id": 3,
            "resource_grant_id": 5,
            "resource_class": "cpu",
        }

        job = meta_compute_runtime.LegacyCPUValidationTransport().submit(
            self._submission()
        )

        self.assertEqual(job.backend_kind, "cpu")
        self.assertEqual(job.backend_job_id, "cpu-experiment-run:7")
        self.assertEqual(job.status, "submitted")

    @mock.patch.object(meta_compute_runtime.db, "fetchone")
    def test_submit_fails_closed_on_cross_agenda_run(self, fetchone):
        fetchone.return_value = {
            "id": 7,
            "agenda_id": 99,
            "deep_insight_id": 3,
            "resource_grant_id": 5,
            "resource_class": "cpu",
        }

        with self.assertRaisesRegex(ComputeBackendError, "scope mismatch"):
            meta_compute_runtime.LegacyCPUValidationTransport().submit(
                self._submission()
            )

    @mock.patch.object(meta_compute_runtime.db, "fetchone")
    def test_failed_run_never_reports_backend_success(self, fetchone):
        fetchone.return_value = {
            "id": 7,
            "agenda_id": 2,
            "deep_insight_id": 3,
            "resource_grant_id": 5,
            "status": "failed",
            "error_message": "no metric",
        }

        job = meta_compute_runtime.LegacyCPUValidationTransport().status(
            "cpu-experiment-run:7"
        )

        self.assertEqual(job.status, "failed")
        self.assertEqual(job.failure_reason, "no metric")

    @mock.patch.object(meta_compute_runtime.db, "fetchone")
    def test_usage_is_measured_from_scoped_iterations(self, fetchone):
        fetchone.side_effect = [
            {
                "id": 7,
                "agenda_id": 2,
                "deep_insight_id": 3,
                "resource_grant_id": 5,
                "status": "completed",
            },
            {"measured_seconds": 1800, "peak_memory_mb": 256},
        ]

        usage = meta_compute_runtime.LegacyCPUValidationTransport().usage(
            "cpu-experiment-run:7"
        )

        self.assertEqual(usage.wall_seconds, 1800)
        self.assertEqual(usage.gpu_hours, 0)
        self.assertEqual(usage.cpu_core_hours, 0.5)
        self.assertEqual(fetchone.call_args_list[-1].args[1], (2, 7))


if __name__ == "__main__":
    unittest.main()
