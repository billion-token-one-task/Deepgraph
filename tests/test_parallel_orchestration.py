import json
import unittest
from unittest import mock
from datetime import timedelta

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


class AutoResearchRoutingTests(unittest.TestCase):
    def test_real_llm_benchmark_plan_upgrades_cpu_route_to_gpu_large(self):
        insight = {
            "id": 31,
            "resource_class": "cpu",
            "title": "Closed-loop benchmark routing",
            "experimental_plan": json.dumps(
                {
                    "datasets": [{"name": "GSM8K"}],
                    "baselines": [
                        {"name": "Vanilla Direct Answering"},
                        {"name": "Always-Reason Chain-of-Thought"},
                    ],
                }
            ),
        }

        with mock.patch.object(auto_research, "gpu_resource_allowed", return_value=(True, "")):
            resource_class, reason = auto_research.assess_experiment_route(insight)

        self.assertEqual(resource_class, "gpu_large")
        self.assertIn("upgraded cpu route to gpu_large", reason)


class AutoResearchSchedulingTests(unittest.TestCase):
    def test_candidate_pool_query_does_not_treat_review_pending_as_ready_candidate(self):
        with mock.patch.object(auto_research.db, "fetchall", return_value=[] ) as fetchall:
            auto_research._candidate_pool()

        sql = fetchall.call_args.args[0]
        self.assertNotIn("'review_pending'", sql)

    def test_candidate_pool_query_does_not_let_generic_failed_jobs_block_queue(self):
        with mock.patch.object(auto_research.db, "fetchall", return_value=[] ) as fetchall:
            auto_research._candidate_pool()

        sql = fetchall.call_args.args[0]
        self.assertNotIn("'queued', 'eligible', 'failed'", sql)
        self.assertIn("arj.status='failed'", sql)
        self.assertIn("'retry_failed_run'", sql)

    def test_candidate_pool_query_does_not_retry_all_blocked_cpu_eligible_jobs(self):
        with mock.patch.object(auto_research.db, "fetchall", return_value=[] ) as fetchall:
            auto_research._candidate_pool()

        sql = fetchall.call_args.args[0]
        self.assertNotIn("arj.cpu_eligible=1", sql)
        self.assertIn("arj.stage='cpu_ineligible'", sql)
        self.assertIn("'verification_input_missing'", sql)
        self.assertIn("'experiment_review_blocked'", sql)

    def test_process_candidate_repairs_preexisting_review_block(self):
        candidate = {
            "id": 21,
            "tier": 2,
            "novelty_status": "novel",
            "auto_status": "blocked",
            "auto_stage": "experiment_review_blocked",
            "auto_last_error": "missing benchmark targets",
        }
        upserts = []

        def _capture_upsert(insight_id, **fields):
            upserts.append((insight_id, fields))

        with (
            mock.patch.object(auto_research, "assess_experiment_route", return_value=("gpu_small", "ready")),
            mock.patch.object(auto_research, "repair_experiment_plan_from_review", return_value={
                "status": "repaired",
                "attempt": 1,
                "repair_summary": "Added benchmark targets.",
                "llm_repair_used": True,
            }),
            mock.patch.object(auto_research.db, "fetchone", return_value={"last_note": "", "last_error": ""}),
            mock.patch.object(auto_research, "_upsert_job", side_effect=_capture_upsert),
            mock.patch.object(auto_research, "log_event"),
        ):
            auto_research._process_candidate(candidate)

        self.assertEqual(upserts[-1][0], 21)
        self.assertEqual(upserts[-1][1]["status"], "queued")
        self.assertEqual(upserts[-1][1]["stage"], "experiment_review_repair")
        self.assertIn("auto_repair:experiment_review", upserts[-1][1]["last_note"])

    def test_review_repair_attempt_count_survives_review_pending_note_overwrite(self):
        upserts = []

        def _capture_upsert(insight_id, **fields):
            upserts.append((insight_id, fields))

        fetches = [
            {"last_note": "Running structured experiment review before forge.", "last_error": ""},
            {"experimental_plan": json.dumps({"review_repair_history": [{"attempt": 1}, {"attempt": 1}]})},
        ]

        with (
            mock.patch.object(auto_research.db, "fetchone", side_effect=fetches),
            mock.patch.object(auto_research, "record_harness_required", return_value={
                "harness_job_id": 11,
                "benchmark_name": "custom benchmark",
                "paths": {},
            }),
            mock.patch.object(auto_research, "repair_experiment_plan_from_review") as repair,
            mock.patch.object(auto_research, "_upsert_job", side_effect=_capture_upsert),
            mock.patch.object(auto_research, "log_event"),
        ):
            auto_research._handle_experiment_review_blocked(
                24,
                {
                    "error": "blocked: missing baseline methods",
                    "route": "blocked",
                    "judgement": {
                        "summary": "Experimental plan lacks required baseline methods.",
                        "blockers": ["Experimental plan lacks required baseline methods."],
                        "warnings": [],
                    },
                },
            )

        repair.assert_not_called()
        self.assertEqual(upserts[-1][0], 24)
        self.assertEqual(upserts[-1][1]["status"], "harness_required")
        self.assertEqual(upserts[-1][1]["stage"], "benchmark_harness_required")
        self.assertIn("Benchmark harness job", upserts[-1][1]["last_note"])

    def test_review_blocked_with_harness_requirement_queues_harness_job(self):
        upserts = []

        def _capture_upsert(insight_id, **fields):
            upserts.append((insight_id, fields))

        forged = {
            "error": "blocked: dedicated benchmark harness required",
            "route": "blocked",
            "harness_required": True,
            "judgement": {
                "summary": "Generated runner cannot execute the custom attention-audit benchmark.",
                "blockers": ["Generated real-benchmark runner does not support LongBench v2; a dedicated benchmark harness/recipe is required before GPU execution."],
                "warnings": [],
                "environment_review": {
                    "benchmark_harness_required": True,
                    "harness_queue": "benchmark_harness_jobs",
                },
            },
        }

        with (
            mock.patch.object(auto_research.db, "fetchone", return_value={"last_note": "", "last_error": ""}),
            mock.patch.object(auto_research, "record_harness_required", return_value={
                "harness_job_id": 9,
                "benchmark_name": "LongBench v2",
                "paths": {"benchmark_harness_task.json": "/tmp/task.json"},
            }) as record,
            mock.patch.object(auto_research, "repair_experiment_plan_from_review") as repair,
            mock.patch.object(auto_research, "_upsert_job", side_effect=_capture_upsert),
            mock.patch.object(auto_research, "log_event"),
        ):
            auto_research._handle_experiment_review_blocked(24, forged, source="forge_review")

        record.assert_called_once()
        repair.assert_not_called()
        self.assertEqual(upserts[-1][0], 24)
        self.assertEqual(upserts[-1][1]["status"], "harness_required")
        self.assertEqual(upserts[-1][1]["stage"], "benchmark_harness_required")
        self.assertIn("Main experiment scheduling is released", upserts[-1][1]["last_note"])

    def test_process_candidate_does_not_reforge_after_review_repair_exhausted(self):
        candidate = {
            "id": 24,
            "tier": 2,
            "novelty_status": "novel",
            "auto_status": "queued",
            "auto_stage": "experiment_review_repair",
            "experimental_plan": json.dumps({"review_repair_history": [{"attempt": 1}, {"attempt": 1}]}),
        }
        upserts = []

        def _capture_upsert(insight_id, **fields):
            upserts.append((insight_id, fields))

        def _fake_fetchone(sql, params=()):
            if "SELECT experimental_plan FROM deep_insights" in sql:
                return {"experimental_plan": json.dumps({"review_repair_history": [{"attempt": 1}, {"attempt": 1}]})}
            return None

        with (
            mock.patch.object(auto_research, "assess_experiment_route", return_value=("gpu_small", "ready")),
            mock.patch.object(auto_research.db, "fetchone", side_effect=_fake_fetchone),
            mock.patch.object(auto_research, "record_harness_required", return_value={
                "harness_job_id": 12,
                "benchmark_name": "custom benchmark",
                "paths": {},
            }),
            mock.patch.object(auto_research, "forge_experiment") as forge,
            mock.patch.object(auto_research, "_upsert_job", side_effect=_capture_upsert),
            mock.patch.object(auto_research, "log_event"),
        ):
            auto_research._process_candidate(candidate)

        forge.assert_not_called()
        self.assertEqual(upserts[-1][0], 24)
        self.assertEqual(upserts[-1][1]["status"], "harness_required")
        self.assertEqual(upserts[-1][1]["stage"], "benchmark_harness_required")

    def test_refresh_keeps_benchmark_completion_queued_for_completed_run(self):
        job = {
            "deep_insight_id": 30,
            "status": "queued_gpu",
            "stage": auto_research.BENCHMARK_COMPLETION_STAGE,
            "experiment_run_id": 7,
            "resource_class": "gpu_large",
        }
        run = {
            "id": 7,
            "status": "completed",
            "resource_class": "gpu_large",
            "hypothesis_verdict": "confirmed",
            "effect_pct": 4.2,
        }

        with (
            mock.patch.object(auto_research.db, "fetchall", return_value=[job]),
            mock.patch.object(auto_research.db, "fetchone", return_value=run),
            mock.patch.object(auto_research, "_queue_benchmark_completion_run") as queue_completion,
            mock.patch.object(auto_research, "_upsert_job") as upsert,
            mock.patch.object(auto_research, "apply_experiment_finished_deep") as finish,
        ):
            auto_research._refresh_running_jobs()

        queue_completion.assert_called_once_with(30, run, "gpu_large")
        finish.assert_not_called()
        completed_calls = [call for call in upsert.call_args_list if call.kwargs.get("stage") == "closed_loop_complete"]
        self.assertEqual(completed_calls, [])

    def test_process_candidate_blocks_underspecified_verification(self):
        candidate = {"id": 12, "tier": 1, "novelty_status": "unchecked"}
        upserts = []

        def _capture_upsert(insight_id, **fields):
            upserts.append((insight_id, fields))

        with (
            mock.patch.object(auto_research, "REQUIRE_EVOSCIENTIST_FOR_EXPERIMENTS", True),
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
            mock.patch.object(auto_research.db, "fetchone", return_value=None),
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

    def test_process_candidate_blocks_gpu_unavailable_route(self):
        candidate = {"id": 15, "tier": 2, "novelty_status": "novel"}
        upserts = []

        def _capture_upsert(insight_id, **fields):
            upserts.append((insight_id, fields))

        with (
            mock.patch.object(
                auto_research,
                "assess_experiment_route",
                return_value=("gpu_unavailable", "inferred gpu_small but no GPU lane"),
            ),
            mock.patch.object(auto_research, "_upsert_job", side_effect=_capture_upsert),
            mock.patch.object(auto_research, "log_event") as log_event,
        ):
            auto_research._process_candidate(candidate)

        self.assertEqual(len(upserts), 1)
        self.assertEqual(upserts[0][0], 15)
        self.assertEqual(upserts[0][1]["status"], "blocked")
        self.assertEqual(upserts[0][1]["stage"], "gpu_unavailable")
        self.assertEqual(upserts[0][1]["cpu_eligible"], 0)
        self.assertIn("no GPU lane", upserts[0][1]["last_error"])
        log_event.assert_called_once()

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
            mock.patch.object(auto_research, "_review_pending_job_count", return_value=0),
            mock.patch.object(auto_research, "evosci_available", return_value=True),
            mock.patch.object(auto_research, "get_evosci_input_issue", return_value=None),
        ):
            candidate = auto_research._next_candidate()

        self.assertEqual(candidate["id"], 13)

    def test_refresh_running_jobs_repairs_review_pending_with_error(self):
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
            mock.patch.object(auto_research.db, "fetchone", return_value={"last_note": "", "last_error": ""}),
            mock.patch.object(auto_research, "repair_experiment_plan_from_review", return_value={
                "status": "repaired",
                "attempt": 1,
                "repair_summary": "Added missing fields.",
                "llm_repair_used": False,
            }),
            mock.patch.object(auto_research, "_upsert_job", side_effect=_capture_upsert),
            mock.patch.object(auto_research, "log_event"),
        ):
            auto_research._refresh_running_jobs()

        self.assertEqual(upserts[-1][0], 14)
        self.assertEqual(upserts[-1][1]["status"], "queued")
        self.assertEqual(upserts[-1][1]["stage"], "experiment_review_repair")

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

    def test_job_age_seconds_handles_naive_local_database_timestamps(self):
        stale = (auto_research.datetime.now() - timedelta(minutes=26)).replace(microsecond=0)
        age = auto_research._job_age_seconds({"updated_at": stale.isoformat()})

        self.assertGreater(age, 15 * 60)

    def test_refresh_running_jobs_surfaces_testing_progress(self):
        job = {
            "deep_insight_id": 16,
            "status": "running_gpu",
            "experiment_run_id": 9,
        }
        run = {
            "id": 9,
            "status": "testing",
            "phase": "hypothesis_testing",
            "best_metric_value": 1.5,
            "baseline_metric_value": 1.2,
        }
        upserts = []

        def _capture_upsert(insight_id, **fields):
            upserts.append((insight_id, fields))

        with (
            mock.patch.object(auto_research.db, "fetchall", return_value=[job]),
            mock.patch.object(auto_research.db, "fetchone", return_value=run),
            mock.patch.object(auto_research, "_is_execution_live_in_process", return_value=True),
            mock.patch.object(auto_research, "_upsert_job", side_effect=_capture_upsert),
        ):
            auto_research._refresh_running_jobs()

        self.assertEqual(upserts[-1][0], 16)
        self.assertEqual(upserts[-1][1]["status"], "running_gpu")
        self.assertEqual(upserts[-1][1]["stage"], "hypothesis_testing")
        self.assertIn("best=1.5", upserts[-1][1]["last_note"])
        self.assertFalse(upserts[-1][1].get("touch_updated_at", True))

    def test_failed_run_is_repaired_and_requeued_once(self):
        run = {
            "id": 31,
            "status": "failed",
            "error_message": "reproduction failed: no metric obtained",
        }
        upserts = []

        def _capture_upsert(insight_id, **fields):
            upserts.append((insight_id, fields))

        with (
            mock.patch.object(auto_research.db, "fetchone", return_value={"last_note": "", "last_error": ""}),
            mock.patch.object(auto_research, "repair_experiment_plan_from_review", return_value={
                "status": "repaired",
                "attempt": 1,
                "repair_summary": "Adjusted runnable benchmark contract.",
            }),
            mock.patch.object(auto_research, "_supersede_stale_scaffold_run") as supersede,
            mock.patch.object(auto_research, "_upsert_job", side_effect=_capture_upsert),
            mock.patch.object(auto_research, "log_event"),
        ):
            repaired = auto_research._retry_failed_run_with_repair(17, run, "gpu_small")

        self.assertTrue(repaired)
        supersede.assert_called_once()
        self.assertEqual(upserts[-1][0], 17)
        self.assertEqual(upserts[-1][1]["status"], "queued")
        self.assertEqual(upserts[-1][1]["stage"], "retry_failed_run")

    def test_recover_stale_execution_jobs_requeues_zombie_running_cpu(self):
        job = {
            "deep_insight_id": 6,
            "status": "running_cpu",
            "experiment_run_id": 6,
            "run_status": "testing",
        }
        upserts = []

        def _capture_upsert(insight_id, **fields):
            upserts.append((insight_id, fields))

        with (
            mock.patch.object(auto_research, "_active_execution_run_id", return_value=None),
            mock.patch.object(auto_research.db, "fetchall", return_value=[job]),
            mock.patch.object(auto_research, "_upsert_job", side_effect=_capture_upsert),
            mock.patch.object(auto_research, "log_event"),
        ):
            recovered = auto_research.recover_stale_execution_jobs()

        self.assertEqual(recovered, 1)
        self.assertEqual(upserts[-1][0], 6)
        self.assertEqual(upserts[-1][1]["status"], "queued")
        self.assertEqual(upserts[-1][1]["stage"], "execution_retry")
        self.assertEqual(upserts[-1][1]["experiment_run_id"], 6)

    def test_recover_stale_execution_jobs_skips_active_in_process_run(self):
        job = {
            "deep_insight_id": 12,
            "status": "running_cpu",
            "experiment_run_id": 14,
            "run_status": "reproducing",
        }

        with (
            mock.patch.object(auto_research, "_active_execution_run_id", return_value=14),
            mock.patch.object(auto_research.db, "fetchall", return_value=[job]),
            mock.patch.object(auto_research, "_requeue_stale_execution_job") as requeue,
        ):
            recovered = auto_research.recover_stale_execution_jobs()

        self.assertEqual(recovered, 0)
        requeue.assert_not_called()

    def test_refresh_running_jobs_requeues_interrupted_testing_run(self):
        job = {
            "deep_insight_id": 16,
            "status": "running_cpu",
            "experiment_run_id": 9,
        }
        run = {
            "id": 9,
            "status": "testing",
            "phase": "hypothesis_testing",
            "best_metric_value": 0.0,
            "baseline_metric_value": 0.0,
        }

        with (
            mock.patch.object(auto_research.db, "fetchall", return_value=[job]),
            mock.patch.object(auto_research.db, "fetchone", return_value=run),
            mock.patch.object(auto_research, "_is_execution_live_in_process", return_value=False),
            mock.patch.object(auto_research, "_requeue_stale_execution_job") as requeue,
        ):
            auto_research._refresh_running_jobs()

        requeue.assert_called_once()
        self.assertIn("interrupted", requeue.call_args.args[1])

    def test_launch_dispatches_review_queue_to_worker(self):
        candidate = {
            "id": 70,
            "tier": 2,
            "novelty_status": "novel",
            "auto_status": "queued",
            "auto_stage": "idea_ready",
        }

        with (
            mock.patch.object(auto_research, "recover_stale_execution_jobs", return_value=0),
            mock.patch.object(auto_research, "_refresh_running_jobs"),
            mock.patch.object(auto_research, "_select_candidate_from_queues", side_effect=[
                (candidate, {"selected_queue": auto_research.QUEUE_REVIEW, "queue_counts": {auto_research.QUEUE_REVIEW: 1}}),
                (None, {"selected_queue": None, "queue_counts": {}}),
            ]),
            mock.patch.object(auto_research, "_start_candidate_worker", return_value=True) as start_worker,
            mock.patch.object(auto_research, "_process_candidate") as process,
            mock.patch.object(auto_research, "_queue_active_counts", return_value={
                auto_research.QUEUE_EXECUTION: 0,
                auto_research.QUEUE_VERIFICATION: 0,
                auto_research.QUEUE_RESEARCH: 0,
                auto_research.QUEUE_REVIEW: 1,
            }),
            mock.patch.object(auto_research, "_active_job_count", return_value=0),
        ):
            stats = auto_research._launch_candidates_to_capacity()

        start_worker.assert_called_once_with(candidate, auto_research.QUEUE_REVIEW)
        process.assert_not_called()
        self.assertEqual(stats["scheduled"], [70])

    def test_process_candidate_requeues_cpu_when_execution_lane_busy(self):
        candidate = {"id": 71, "tier": 2, "novelty_status": "novel"}
        existing_run = {
            "id": 8,
            "status": "pending",
            "proxy_config": '{"formal_experiment": true, "smoke_test_only": false}',
            "resource_class": "cpu",
        }
        upserts = []

        def _capture_upsert(insight_id, **fields):
            upserts.append((insight_id, fields))

        with (
            mock.patch.object(auto_research, "assess_experiment_route", return_value=("cpu", "ready")),
            mock.patch.object(auto_research, "evosci_available", return_value=False),
            mock.patch.object(auto_research, "_existing_run_for_candidate", return_value=existing_run),
            mock.patch.object(auto_research, "_active_execution_run_id", return_value=99),
            mock.patch.object(auto_research, "_upsert_job", side_effect=_capture_upsert),
            mock.patch.object(auto_research, "run_validation_loop") as validation,
        ):
            auto_research._process_candidate(candidate)

        validation.assert_not_called()
        self.assertEqual(upserts[-1][1]["status"], "queued_cpu")
        self.assertEqual(upserts[-1][1]["stage"], "cpu_execution_wait")

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

    def test_existing_run_lookup_skips_superseded_canonical_run(self):
        with mock.patch.object(
            auto_research.db,
            "fetchone",
            side_effect=[
                {"id": 7, "status": "superseded"},
                {"id": 8, "status": "scaffolding"},
            ],
        ):
            run = auto_research._existing_run_for_candidate({"id": 21, "canonical_run_id": 7})

        self.assertEqual(run["id"], 8)

    def test_reset_completed_stage_reforges_terminal_runs(self):
        insight = {"id": 21, "auto_stage": "reset_completed_experiments"}

        self.assertTrue(auto_research._manual_reforge_requested(insight, {"id": 7, "status": "completed"}))
        self.assertTrue(auto_research._manual_reforge_requested(insight, {"id": 8, "status": "bundle_ready"}))

    def test_benchmark_completion_stage_does_not_reforge_completed_run(self):
        insight = {"id": 21, "auto_stage": auto_research.BENCHMARK_COMPLETION_STAGE}

        self.assertFalse(auto_research._manual_reforge_requested(insight, {"id": 7, "status": "completed"}))

    def test_process_candidate_queues_completed_run_for_benchmark_completion(self):
        candidate = {
            "id": 25,
            "tier": 2,
            "novelty_status": "novel",
            "canonical_run_id": 11,
            "auto_stage": auto_research.BENCHMARK_COMPLETION_STAGE,
        }
        existing_run = {
            "id": 11,
            "status": "completed",
            "proxy_config": '{"formal_experiment": true, "smoke_test_only": false}',
            "resource_class": "gpu_large",
        }

        with (
            mock.patch.object(auto_research, "assess_experiment_route", return_value=("gpu_large", "ready")),
            mock.patch.object(auto_research, "evosci_available", return_value=False),
            mock.patch.object(auto_research, "_existing_run_for_candidate", return_value=existing_run),
            mock.patch.object(auto_research, "_auto_job_stage", return_value=auto_research.BENCHMARK_COMPLETION_STAGE),
            mock.patch.object(auto_research, "_upsert_job"),
            mock.patch.object(auto_research, "_queue_benchmark_completion_run", return_value=True) as queue_completion,
            mock.patch.object(auto_research, "log_event"),
        ):
            auto_research._process_candidate(candidate)

        queue_completion.assert_called_once_with(25, existing_run, "gpu_large")

    def test_recover_soft_benchmark_completion_jobs_requeues_non_benchmark_rows(self):
        rows = [
            {
                "id": 7,
                "deep_insight_id": 4,
                "experiment_run_id": 335,
                "last_error": "Only manuscript polish blockers remain after benchmark evidence passed.",
            }
        ]

        with (
            mock.patch.object(auto_research.db, "fetchall", return_value=rows),
            mock.patch.object(auto_research.db, "execute") as execute,
            mock.patch.object(auto_research.db, "commit") as commit,
            mock.patch.object(auto_research, "log_event"),
        ):
            recovered = auto_research.recover_soft_benchmark_completion_jobs()

        self.assertEqual(recovered, 1)
        self.assertIn("manuscript_retry_after_soft_benchmark_gate", execute.call_args.args[0])
        commit.assert_called_once()

    def test_recover_soft_benchmark_completion_jobs_keeps_full_benchmark_gaps(self):
        rows = [
            {
                "id": 7,
                "deep_insight_id": 4,
                "experiment_run_id": 335,
                "last_error": "benchmark_summary.full_benchmark_completed is false; required baselines missing: Extra",
            }
        ]

        with (
            mock.patch.object(auto_research.db, "fetchall", return_value=rows),
            mock.patch.object(auto_research.db, "execute") as execute,
            mock.patch.object(auto_research.db, "commit") as commit,
            mock.patch.object(auto_research, "log_event"),
        ):
            recovered = auto_research.recover_soft_benchmark_completion_jobs()

        self.assertEqual(recovered, 0)
        execute.assert_not_called()
        commit.assert_not_called()

    def test_recover_soft_benchmark_completion_jobs_keeps_hard_blockers(self):
        rows = [
            {
                "id": 8,
                "deep_insight_id": 9,
                "experiment_run_id": 278,
                "last_error": "Benchmark summary must include at least two methods/baselines.",
            }
        ]

        with (
            mock.patch.object(auto_research.db, "fetchall", return_value=rows),
            mock.patch.object(auto_research.db, "execute") as execute,
            mock.patch.object(auto_research.db, "commit") as commit,
        ):
            recovered = auto_research.recover_soft_benchmark_completion_jobs()

        self.assertEqual(recovered, 0)
        execute.assert_not_called()
        commit.assert_not_called()

    def test_recover_blocked_manuscript_jobs_requeues_latest_blocked_run(self):
        rows = [
            {
                "deep_insight_id": 8,
                "experiment_run_id": 13,
                "manuscript_status": "manuscript_blocked",
                "manuscript_workdir": "/tmp/idea8/papers/current",
                "auto_status": "completed",
                "auto_stage": "experiment_confirmed",
                "auto_resource_class": "cpu",
                "run_resource_class": "gpu_small",
                "run_error_message": "Manuscript quality gate failed",
            }
        ]
        upserts = []

        def _capture_upsert(insight_id, **fields):
            upserts.append((insight_id, fields))

        with (
            mock.patch.object(auto_research.db, "fetchall", return_value=rows),
            mock.patch.object(auto_research, "_upsert_job", side_effect=_capture_upsert),
            mock.patch.object(auto_research, "log_event"),
        ):
            recovered = auto_research.recover_blocked_manuscript_jobs()

        self.assertEqual(recovered, 1)
        self.assertEqual(upserts[-1][0], 8)
        self.assertEqual(upserts[-1][1]["status"], "queued")
        self.assertEqual(upserts[-1][1]["stage"], "manuscript_retry_after_quality_gate")
        self.assertEqual(upserts[-1][1]["experiment_run_id"], 13)

    def test_recover_blocked_manuscript_jobs_skips_active_retry(self):
        rows = [
            {
                "deep_insight_id": 4,
                "experiment_run_id": 335,
                "manuscript_status": "manuscript_blocked",
                "auto_status": "queued",
                "auto_stage": "manuscript_retry_after_quality_gate",
            }
        ]

        with (
            mock.patch.object(auto_research.db, "fetchall", return_value=rows),
            mock.patch.object(auto_research, "_upsert_job") as upsert,
            mock.patch.object(auto_research, "log_event"),
        ):
            recovered = auto_research.recover_blocked_manuscript_jobs()

        self.assertEqual(recovered, 0)
        upsert.assert_not_called()

    def test_process_candidate_writes_bundle_for_completed_confirmed_run(self):
        candidate = {
            "id": 26,
            "tier": 2,
            "novelty_status": "novel",
            "canonical_run_id": 12,
        }
        existing_run = {
            "id": 12,
            "status": "completed",
            "hypothesis_verdict": "confirmed",
            "effect_pct": 8.5,
            "proxy_config": '{"formal_experiment": true, "smoke_test_only": false}',
            "resource_class": "gpu_large",
        }
        upserts = []

        def _capture_upsert(insight_id, **fields):
            upserts.append((insight_id, fields))

        with (
            mock.patch.object(auto_research, "assess_experiment_route", return_value=("gpu_large", "ready")),
            mock.patch.object(auto_research, "evosci_available", return_value=False),
            mock.patch.object(auto_research, "_existing_run_for_candidate", return_value=existing_run),
            mock.patch.object(auto_research, "_auto_job_stage", return_value="manuscript_retry"),
            mock.patch.object(auto_research, "generate_submission_bundle", return_value={"bundle_ids": [44]}),
            mock.patch.object(auto_research, "schedule_benchmark_completion", return_value=False),
            mock.patch.object(auto_research, "_upsert_job", side_effect=_capture_upsert),
        ):
            auto_research._process_candidate(candidate)

        self.assertEqual(upserts[-1][0], 26)
        self.assertEqual(upserts[-1][1]["status"], "bundle_ready")
        self.assertEqual(upserts[-1][1]["stage"], "writing_submission")
        self.assertEqual(upserts[-1][1]["artifact_bundle_id"], 44)

    def test_review_retry_reforges_failed_latest_run(self):
        insight = {"id": 24, "auto_stage": "review_retry"}

        self.assertTrue(auto_research._manual_reforge_requested(insight, {"id": 10, "status": "failed"}))

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
            mock.patch.object(auto_research, "_review_pending_job_count", return_value=0),
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


    def test_process_candidate_does_not_relaunch_optional_research_stage(self):
        candidate = {
            "id": 41,
            "tier": 1,
            "novelty_status": "verifying",
            "auto_stage": "research_unavailable",
        }
        existing_run = {
            "id": 319,
            "status": "testing",
            "workdir": "/tmp/run_319",
            "proxy_config": "{\"formal_experiment\": true, \"smoke_test_only\": false}",
            "program_md": "program",
            "success_criteria": "{}",
            "resource_class": "gpu_large",
        }
        upserts = []

        def _capture_upsert(insight_id, **fields):
            upserts.append((insight_id, fields))

        with (
            mock.patch.object(auto_research, "assess_experiment_route", return_value=("gpu_large", "ready")),
            mock.patch.object(auto_research, "evosci_available", return_value=True),
            mock.patch.object(auto_research, "launch_full_research") as launch_research,
            mock.patch.object(auto_research, "_maybe_repair_preexisting_review_block", return_value=False),
            mock.patch.object(auto_research, "_existing_run_for_candidate", return_value=existing_run),
            mock.patch.object(auto_research, "_run_scaffold_ready", return_value=True),
            mock.patch.object(auto_research, "_run_is_formal", return_value=True),
            mock.patch.object(auto_research, "_upsert_job", side_effect=_capture_upsert),
            mock.patch.object(auto_research.gpu_scheduler, "start"),
            mock.patch.object(auto_research.gpu_scheduler, "queue_run", return_value=123),
            mock.patch.object(auto_research, "log_event"),
        ):
            auto_research._process_candidate(candidate)

        launch_research.assert_not_called()
        self.assertEqual(upserts[-1][1]["status"], "queued_gpu")
        self.assertEqual(upserts[-1][1]["experiment_run_id"], 319)


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

    def test_candidate_queues_skip_harness_required_and_select_execution(self):
        harness = {
            "id": 60,
            "tier": 2,
            "novelty_status": "novel",
            "auto_status": "harness_required",
            "auto_stage": "benchmark_harness_required",
        }
        runnable = {
            "id": 61,
            "tier": 2,
            "novelty_status": "novel",
            "auto_status": "eligible",
            "auto_stage": "formal_ready",
            "auto_experiment_run_id": 12,
        }

        with (
            mock.patch.object(auto_research, "_candidate_pool", return_value=[harness, runnable]),
            mock.patch.object(auto_research, "_execution_active_job_count", return_value=0),
            mock.patch.object(auto_research, "_verification_job_count", return_value=0),
            mock.patch.object(auto_research, "_research_job_count", return_value=0),
            mock.patch.object(auto_research, "_review_pending_job_count", return_value=0),
            mock.patch.object(auto_research, "evosci_available", return_value=False),
        ):
            selected, state = auto_research._select_candidate_from_queues()

        self.assertEqual(selected["id"], 61)
        self.assertEqual(state["selected_queue"], auto_research.QUEUE_EXECUTION)
        self.assertEqual(state["queue_counts"][auto_research.QUEUE_HARNESS], 1)

    def test_candidate_queues_use_review_capacity_independently_from_execution(self):
        review_candidate = {
            "id": 62,
            "tier": 2,
            "novelty_status": "novel",
            "auto_status": "queued",
            "auto_stage": "idea_ready",
        }
        execution_candidate = {
            "id": 63,
            "tier": 2,
            "novelty_status": "novel",
            "auto_status": "eligible",
            "auto_stage": "formal_ready",
            "auto_experiment_run_id": 13,
        }

        with (
            mock.patch.object(auto_research, "_candidate_pool", return_value=[review_candidate, execution_candidate]),
            mock.patch.object(auto_research, "_execution_active_job_count", return_value=0),
            mock.patch.object(auto_research, "_verification_job_count", return_value=0),
            mock.patch.object(auto_research, "_research_job_count", return_value=0),
            mock.patch.object(auto_research, "_review_pending_job_count", return_value=auto_research.MAX_PARALLEL_REVIEWS),
            mock.patch.object(auto_research, "evosci_available", return_value=False),
        ):
            selected, state = auto_research._select_candidate_from_queues()

        self.assertEqual(selected["id"], 63)
        self.assertEqual(state["selected_queue"], auto_research.QUEUE_EXECUTION)

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
            mock.patch.object(auto_research, "_review_pending_job_count", return_value=0),
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
