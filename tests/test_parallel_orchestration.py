import unittest
from unittest import mock

from orchestrator import auto_research, discovery_scheduler


class AutoResearchLoopTests(unittest.TestCase):
    def test_run_once_keeps_backlog_progressing_even_with_events(self):
        with (
            mock.patch.object(auto_research.db, "init_db"),
            mock.patch.object(auto_research, "consume_pipeline_events_once", return_value={"events": 7}),
            mock.patch.object(auto_research, "run_cycle", return_value={"status": "processed"}),
            mock.patch.object(auto_research, "_active_job_count", return_value=1),
        ):
            stats = auto_research._run_once()

        self.assertEqual(stats["events"], 7)
        self.assertEqual(stats["cycle_status"], "processed")
        self.assertEqual(stats["active_jobs"], 1)


class AutoResearchSchedulingTests(unittest.TestCase):
    def test_process_candidate_blocks_underspecified_verification(self):
        candidate = {"id": 12, "tier": 1, "novelty_status": "unchecked"}
        upserts = []

        def _capture_upsert(insight_id, **fields):
            upserts.append((insight_id, fields))

        with (
            mock.patch.object(auto_research, "assess_experiment_route", return_value=("cpu", "ready")),
            mock.patch.object(auto_research, "evosci_available", return_value=True),
            mock.patch.object(
                auto_research,
                "launch_verification",
                return_value={
                    "error": "Deep insight 12 is missing required fields for novelty verification: Field A.",
                    "error_code": auto_research.INSIGHT_INPUT_MISSING_ERROR_CODE,
                    "missing_fields": ["Field A"],
                },
            ),
            mock.patch.object(auto_research, "_upsert_job", side_effect=_capture_upsert),
            mock.patch.object(auto_research, "log_event"),
        ):
            auto_research._process_candidate(candidate)

        self.assertEqual(len(upserts), 2)
        self.assertEqual(upserts[0][0], 12)
        self.assertEqual(upserts[0][1]["cpu_eligible"], 1)
        self.assertEqual(upserts[1][1]["status"], "blocked")
        self.assertEqual(upserts[1][1]["stage"], "verification_input_missing")
        self.assertEqual(upserts[1][1]["cpu_eligible"], 0)
        self.assertIn("Field A", upserts[1][1]["last_note"])

    def test_next_candidate_requeues_blocked_input_missing_after_repair(self):
        repaired_candidate = {
            "id": 13,
            "tier": 2,
            "novelty_status": "unchecked",
            "auto_status": "blocked",
            "auto_stage": "verification_input_missing",
        }

        with (
            mock.patch.object(auto_research, "_candidate_pool", return_value=[repaired_candidate]),
            mock.patch.object(auto_research, "_execution_active_job_count", return_value=0),
            mock.patch.object(auto_research, "_verification_job_count", return_value=0),
            mock.patch.object(auto_research, "evosci_available", return_value=True),
            mock.patch.object(auto_research, "get_evosci_input_issue", return_value=None),
        ):
            candidate = auto_research._next_candidate()

        self.assertEqual(candidate["id"], 13)

    def test_process_candidate_marks_smoke_only_runs_outside_formal_path(self):
        candidate = {"id": 21, "tier": 2, "novelty_status": "novel"}
        upserts = []

        def _capture_upsert(insight_id, **fields):
            upserts.append((insight_id, fields))

        with (
            mock.patch.object(auto_research, "assess_experiment_route", return_value=("cpu", "ready")),
            mock.patch.object(auto_research, "evosci_available", return_value=False),
            mock.patch.object(auto_research.db, "fetchone", side_effect=[None, {"id": 5, "status": "scaffolding", "proxy_config": '{"formal_experiment": false, "smoke_test_only": true}'}]),
            mock.patch.object(auto_research, "forge_experiment", return_value={"run_id": 5, "smoke_test_only": True, "formal_experiment": False, "judgement": {"summary": "smoke only"}}),
            mock.patch.object(auto_research, "_upsert_job", side_effect=_capture_upsert),
            mock.patch.object(auto_research, "log_event"),
            mock.patch.object(auto_research.db, "execute"),
            mock.patch.object(auto_research.db, "commit"),
        ):
            auto_research._process_candidate(candidate)

        self.assertEqual(upserts[-1][1]["status"], "smoke_only")
        self.assertEqual(upserts[-1][1]["stage"], "experiment_review_smoke_only")


class ParallelTier2LaunchTests(unittest.TestCase):
    def setUp(self):
        self.old_thread = discovery_scheduler._tier2_thread
        self.old_last = discovery_scheduler._last_parallel_tier2_at

    def tearDown(self):
        discovery_scheduler._tier2_thread = self.old_thread
        discovery_scheduler._last_parallel_tier2_at = self.old_last

    def test_launches_parallel_tier2_when_backlog_empty(self):
        fake_thread = mock.Mock()
        fake_thread.is_alive.return_value = False

        with (
            mock.patch.object(discovery_scheduler, "_eligible_tier2_backlog", return_value=0),
            mock.patch.object(discovery_scheduler, "_reasoned_paper_count", return_value=128),
            mock.patch.object(discovery_scheduler, "log_event"),
            mock.patch.object(discovery_scheduler.threading, "Thread", return_value=fake_thread),
        ):
            discovery_scheduler._tier2_thread = None
            discovery_scheduler._last_parallel_tier2_at = 0.0
            result = discovery_scheduler._maybe_launch_parallel_tier2_discovery("test")

        fake_thread.start.assert_called_once()
        self.assertEqual(result["status"], "started")


if __name__ == "__main__":
    unittest.main()
