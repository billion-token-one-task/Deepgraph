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

    def test_start_refuses_second_process_when_lock_is_held_elsewhere(self):
        old_thread = auto_research._worker_thread
        try:
            auto_research._worker_thread = None
            with (
                mock.patch.object(auto_research.db, "init_db"),
                mock.patch.object(auto_research, "_try_acquire_process_lock", return_value=False),
            ):
                result = auto_research.start()
        finally:
            auto_research._worker_thread = old_thread

        self.assertEqual(result["status"], "already_running_elsewhere")


class AutoResearchSchedulingTests(unittest.TestCase):
    def test_candidate_pool_query_does_not_treat_review_pending_as_ready_candidate(self):
        with mock.patch.object(auto_research.db, "fetchall", return_value=[] ) as fetchall:
            auto_research._candidate_pool()

        sql = fetchall.call_args.args[0]
        self.assertNotIn("'review_pending'", sql)

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

    def test_refresh_running_jobs_converts_review_pending_with_error_to_blocked(self):
        job = {
            "deep_insight_id": 14,
            "status": "review_pending",
            "experiment_run_id": None,
            "last_error": "review failed",
            "last_note": "review failed",
        }
        upserts = []

        def _capture_upsert(insight_id, **fields):
            upserts.append((insight_id, fields))

        with (
            mock.patch.object(auto_research.db, "fetchall", return_value=[job]),
            mock.patch.object(auto_research, "_upsert_job", side_effect=_capture_upsert),
        ):
            auto_research._refresh_running_jobs()

        self.assertEqual(upserts[-1][0], 14)
        self.assertEqual(upserts[-1][1]["status"], "blocked")
        self.assertEqual(upserts[-1][1]["stage"], "experiment_review_blocked")

    def test_refresh_running_jobs_requeues_stale_review_pending_without_run(self):
        stale_job = {
            "deep_insight_id": 15,
            "status": "review_pending",
            "experiment_run_id": None,
            "last_error": None,
            "last_note": "still reviewing",
            "updated_at": "2026-04-21T00:00:00",
        }
        upserts = []

        def _capture_upsert(insight_id, **fields):
            upserts.append((insight_id, fields))

        with (
            mock.patch.object(auto_research.db, "fetchall", return_value=[stale_job]),
            mock.patch.object(auto_research, "_upsert_job", side_effect=_capture_upsert),
            mock.patch.object(auto_research, "log_event"),
        ):
            auto_research._refresh_running_jobs()

        self.assertEqual(upserts[-1][0], 15)
        self.assertEqual(upserts[-1][1]["status"], "queued")
        self.assertEqual(upserts[-1][1]["stage"], "review_retry")

    def test_process_candidate_runs_cpu_validation_for_smoke_only_forge(self):
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
            mock.patch.object(auto_research, "run_validation_loop", return_value={"verdict": "inconclusive"}),
            mock.patch.object(auto_research, "process_completed_run"),
            mock.patch.object(auto_research, "generate_submission_bundle", return_value={"bundle_ids": [99], "error": "fail"}),
        ):
            auto_research._process_candidate(candidate)

        smoke_upserts = [u for u in upserts if u[1].get("status") == "smoke_only"]
        self.assertTrue(smoke_upserts)
        self.assertEqual(smoke_upserts[-1][1]["stage"], "experiment_review_smoke_only")

        self.assertEqual(upserts[-1][1]["status"], "completed")
        self.assertEqual(upserts[-1][1]["stage"], "closed_loop_complete")

    def test_process_candidate_keeps_scaffolding_run_in_review_pending_until_decision_ready(self):
        candidate = {"id": 22, "tier": 2, "novelty_status": "novel", "canonical_run_id": 8}
        upserts = []

        def _capture_upsert(insight_id, **fields):
            upserts.append((insight_id, fields))

        existing_run = {"id": 8, "status": "scaffolding", "proxy_config": None}

        with (
            mock.patch.object(auto_research, "assess_experiment_route", return_value=("cpu", "ready")),
            mock.patch.object(auto_research, "evosci_available", return_value=False),
            mock.patch.object(auto_research.db, "fetchone", return_value=existing_run),
            mock.patch.object(auto_research, "_upsert_job", side_effect=_capture_upsert),
        ):
            auto_research._process_candidate(candidate)

        self.assertEqual(upserts[-1][1]["status"], "review_pending")
        self.assertEqual(upserts[-1][1]["stage"], "experiment_review")
        self.assertEqual(upserts[-1][1]["experiment_run_id"], 8)

    def test_process_candidate_keeps_formal_run_pending_until_scaffold_ready(self):
        candidate = {"id": 23, "tier": 2, "novelty_status": "novel", "canonical_run_id": 9}
        upserts = []

        def _capture_upsert(insight_id, **fields):
            upserts.append((insight_id, fields))

        existing_run = {
            "id": 9,
            "status": "scaffolding",
            "workdir": "/tmp/run_9",
            "proxy_config": '{"formal_experiment": true, "smoke_test_only": false}',
            "program_md": "",
            "success_criteria": None,
        }

        with (
            mock.patch.object(auto_research, "assess_experiment_route", return_value=("cpu", "ready")),
            mock.patch.object(auto_research, "evosci_available", return_value=False),
            mock.patch.object(auto_research.db, "fetchone", return_value=existing_run),
            mock.patch.object(auto_research, "_upsert_job", side_effect=_capture_upsert),
        ):
            auto_research._process_candidate(candidate)

        self.assertEqual(upserts[-1][1]["status"], "review_pending")
        self.assertEqual(upserts[-1][1]["stage"], "experiment_review")
        self.assertEqual(upserts[-1][1]["experiment_run_id"], 9)

    def test_next_candidate_allows_experiment_when_only_research_jobs_are_active(self):
        candidate = {
            "id": 31,
            "tier": 2,
            "novelty_status": "novel",
            "auto_status": None,
            "auto_stage": None,
        }

        with (
            mock.patch.object(auto_research, "_candidate_pool", return_value=[candidate]),
            mock.patch.object(auto_research, "_execution_active_job_count", return_value=0),
            mock.patch.object(auto_research, "_verification_job_count", return_value=0),
            mock.patch.object(auto_research, "_research_job_count", return_value=3),
            mock.patch.object(auto_research, "evosci_available", return_value=False),
        ):
            selected = auto_research._next_candidate()

        self.assertEqual(selected["id"], 31)

    def test_active_job_count_still_includes_researching_jobs(self):
        with (
            mock.patch.object(auto_research, "_execution_active_job_count", return_value=1),
            mock.patch.object(auto_research, "_verification_job_count", return_value=2),
            mock.patch.object(auto_research, "_research_job_count", return_value=3),
        ):
            active = auto_research._active_job_count()

        self.assertEqual(active, 6)

    def test_process_candidate_tier2_continues_to_experiment_while_research_starts(self):
        candidate = {"id": 31, "tier": 2, "novelty_status": "novel"}
        upserts = []

        def _capture_upsert(insight_id, **fields):
            upserts.append((insight_id, fields))

        with (
            mock.patch.object(auto_research, "assess_experiment_route", return_value=("gpu_small", "ready")),
            mock.patch.object(auto_research, "evosci_available", return_value=True),
            mock.patch.object(auto_research, "launch_full_research", return_value={"workdir": "/tmp/deep-research-31"}),
            mock.patch.object(auto_research.db, "fetchone", side_effect=[None, {"id": 5, "status": "scaffolding", "proxy_config": '{"formal_experiment": true}'}, None]),
            mock.patch.object(auto_research, "forge_experiment", return_value={"run_id": 5, "smoke_test_only": False, "formal_experiment": True}),
            mock.patch.object(auto_research, "_upsert_job", side_effect=_capture_upsert),
            mock.patch.object(auto_research.gpu_scheduler, "start"),
            mock.patch.object(auto_research.gpu_scheduler, "queue_run", return_value=99),
            mock.patch.object(auto_research, "log_event"),
            mock.patch.object(auto_research.db, "execute"),
            mock.patch.object(auto_research.db, "commit"),
        ):
            auto_research._process_candidate(candidate)

        self.assertTrue(any(fields.get("stage") == "deep_research_background" for _, fields in upserts))
        self.assertEqual(upserts[-1][1]["status"], "queued_gpu")
        self.assertEqual(upserts[-1][1]["experiment_run_id"], 5)

    def test_process_candidate_tier1_continues_to_experiment_while_research_starts(self):
        candidate = {"id": 41, "tier": 1, "novelty_status": "novel", "predictions": '["p1"]'}
        upserts = []

        def _capture_upsert(insight_id, **fields):
            upserts.append((insight_id, fields))

        with (
            mock.patch.object(auto_research, "assess_experiment_route", return_value=("gpu_small", "ready")),
            mock.patch.object(auto_research, "evosci_available", return_value=True),
            mock.patch.object(auto_research, "launch_full_research", return_value={"workdir": "/tmp/deep-research-41"}),
            mock.patch.object(auto_research.db, "fetchone", side_effect=[None, {"id": 6, "status": "scaffolding", "proxy_config": '{"formal_experiment": true}'} , None]),
            mock.patch.object(auto_research, "forge_experiment", return_value={"run_id": 6, "smoke_test_only": False, "formal_experiment": True}),
            mock.patch.object(auto_research, "_upsert_job", side_effect=_capture_upsert),
            mock.patch.object(auto_research.gpu_scheduler, "start"),
            mock.patch.object(auto_research.gpu_scheduler, "queue_run", return_value=100),
            mock.patch.object(auto_research, "log_event"),
            mock.patch.object(auto_research.db, "execute"),
            mock.patch.object(auto_research.db, "commit"),
        ):
            auto_research._process_candidate(candidate)

        self.assertTrue(any(fields.get("stage") == "deep_research_background" for _, fields in upserts))
        self.assertEqual(upserts[-1][1]["status"], "queued_gpu")
        self.assertEqual(upserts[-1][1]["experiment_run_id"], 6)

    def test_next_candidate_requeues_legacy_tier1_completed_without_runs(self):
        candidate = {
            "id": 51,
            "tier": 1,
            "novelty_status": "novel",
            "auto_status": "completed",
            "auto_stage": "tier1_research_complete",
        }

        with (
            mock.patch.object(auto_research, "_candidate_pool", return_value=[candidate]),
            mock.patch.object(auto_research, "_execution_active_job_count", return_value=0),
            mock.patch.object(auto_research, "_verification_job_count", return_value=0),
            mock.patch.object(auto_research, "evosci_available", return_value=False),
        ):
            selected = auto_research._next_candidate()

        self.assertEqual(selected["id"], 51)


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
            mock.patch.object(discovery_scheduler, "_warm_tier2_backlog", return_value=0),
            mock.patch.object(discovery_scheduler, "DISCOVERY_MIN_TIER2_BACKLOG", 3),
            mock.patch.object(discovery_scheduler, "_reasoned_paper_count", return_value=128),
            mock.patch.object(discovery_scheduler, "log_event"),
            mock.patch.object(discovery_scheduler.threading, "Thread", return_value=fake_thread),
        ):
            discovery_scheduler._tier2_thread = None
            discovery_scheduler._last_parallel_tier2_at = 0.0
            result = discovery_scheduler._maybe_launch_parallel_tier2_discovery("test")

        fake_thread.start.assert_called_once()
        self.assertEqual(result["status"], "started")

    def test_skips_parallel_tier2_when_warm_backlog_meets_target(self):
        with (
            mock.patch.object(discovery_scheduler, "_warm_tier2_backlog", return_value=3),
            mock.patch.object(discovery_scheduler, "DISCOVERY_MIN_TIER2_BACKLOG", 3),
        ):
            result = discovery_scheduler._maybe_launch_parallel_tier2_discovery("test")

        self.assertEqual(result["status"], "backlog_ready")

    def test_run_parallel_tier2_discovery_fills_backlog_deficit(self):
        with (
            mock.patch.object(discovery_scheduler, "_warm_tier2_backlog", return_value=1),
            mock.patch.object(discovery_scheduler, "DISCOVERY_MIN_TIER2_BACKLOG", 4),
            mock.patch.object(discovery_scheduler, "harvest_signals"),
            mock.patch.object(discovery_scheduler, "run_tier2_discovery", return_value=[{"id": 1}, {"id": 2}, {"id": 3}]) as run_tier2,
            mock.patch.object(discovery_scheduler, "log_event"),
        ):
            discovery_scheduler._run_parallel_tier2_discovery()

        run_tier2.assert_called_once_with(max_problems=3, max_papers=discovery_scheduler.DISCOVERY_TIER2_PAPERS)


if __name__ == "__main__":
    unittest.main()
