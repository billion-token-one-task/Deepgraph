"""Contract tests for the three independent runtime status domains."""

import unittest
from unittest import mock

from orchestrator import paper_worker
from web import app as web_app


class ProcessingStatusContractTests(unittest.TestCase):
    def test_legacy_worker_disabled_status_round_trips_through_get_status(self):
        with (
            mock.patch.object(paper_worker, "_worker_thread", None),
            mock.patch.object(paper_worker, "_last_status", {"status": "not_started"}),
        ):
            started = paper_worker.start()
            readback = paper_worker.get_status()

        self.assertEqual(started["status"], "disabled_resource_grant_required")
        self.assertEqual(readback["status"], started["status"])
        self.assertEqual(readback["reason"], started["reason"])
        self.assertFalse(readback["running"])

    def test_scoped_ingestion_can_be_healthy_and_idle(self):
        state = web_app._classify_scoped_ingestion(
            {"running": True, "status": "idle"},
            {"queued": 0, "retryable": 0, "running": 0, "succeeded": 8},
        )
        self.assertEqual(state, "idle")

    def test_stale_legacy_records_do_not_stall_research_runtime(self):
        legacy = web_app._legacy_paper_ingestion_snapshot(
            {
                "running": False,
                "status": "disabled_resource_grant_required",
                "reason": "agenda_scoped_ingestion_job_required",
            },
            processing_count=10,
            recent_processing_count=0,
        )
        research_state = web_app._classify_research_runtime(
            {"running": True}, active_grants=0, active_work_items=0
        )

        self.assertEqual(legacy["record_state"], "stale_records")
        self.assertEqual(legacy["stale_processing_count"], 10)
        self.assertEqual(
            legacy["state"], "disabled_resource_grant_required"
        )
        self.assertEqual(research_state, "idle_no_authorized_work")

    def test_research_control_plane_without_active_grant_is_idle(self):
        self.assertEqual(
            web_app._classify_research_runtime(
                {"running": True}, active_grants=0, active_work_items=0
            ),
            "idle_no_authorized_work",
        )

    def test_real_worker_loss_with_authorized_work_is_stalled(self):
        self.assertEqual(
            web_app._classify_research_runtime(
                {"running": False}, active_grants=1, active_work_items=1
            ),
            "stalled",
        )
        self.assertEqual(
            web_app._classify_scoped_ingestion(
                {"running": False, "status": "idle"},
                {"queued": 0, "retryable": 0, "running": 1},
            ),
            "stalled",
        )

    def test_processing_endpoint_exposes_three_separate_domains(self):
        client = web_app.app.test_client()
        legacy = {
            "state": "disabled_resource_grant_required",
            "record_state": "stale_records",
            "stale_processing_count": 10,
        }
        scoped = {"state": "idle", "running": True, "running_jobs": 0}
        research = {
            "state": "idle_no_authorized_work",
            "active_grants": 0,
            "active_work_items": 0,
        }
        with (
            mock.patch.object(web_app.db, "fetchall", return_value=[]),
            mock.patch.object(
                web_app.db,
                "fetchone",
                side_effect=[{"c": 10}, {"c": 0}],
            ),
            mock.patch(
                "orchestrator.paper_worker.get_status",
                return_value={
                    "running": False,
                    "status": "disabled_resource_grant_required",
                },
            ),
            mock.patch.object(
                web_app, "_legacy_paper_ingestion_snapshot", return_value=legacy
            ),
            mock.patch.object(
                web_app, "_scoped_ingestion_snapshot", return_value=scoped
            ),
            mock.patch.object(
                web_app, "_research_runtime_snapshot", return_value=research
            ),
        ):
            response = client.get("/api/processing")

        payload = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["contract_version"], "processing-status-v2")
        self.assertEqual(payload["legacy_paper_ingestion"], legacy)
        self.assertEqual(payload["scoped_ingestion"], scoped)
        self.assertEqual(payload["research_runtime"], research)
        self.assertEqual(payload["pipeline_state"], "idle_no_authorized_work")
        self.assertFalse(payload["pipeline_running"])


if __name__ == "__main__":
    unittest.main()
