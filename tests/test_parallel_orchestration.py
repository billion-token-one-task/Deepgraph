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

    def test_candidate_pool_prioritizes_manuscript_retries_before_new_forge(self):
        with mock.patch.object(auto_research.db, "fetchall", return_value=[] ) as fetchall:
            auto_research._candidate_pool()

        sql = fetchall.call_args.args[0]
        params = fetchall.call_args.args[1]
        manuscript_pos = sql.index("'manuscript_retry_after_quality_gate'")
        review_pos = sql.index("'review_incomplete_reforge'")
        self.assertLess(manuscript_pos, review_pos)
        self.assertIn("arj.updated_at ASC", sql)
        self.assertIn("LIMIT ?", sql)
        self.assertEqual(params, (auto_research.AUTO_RESEARCH_CANDIDATE_POOL_LIMIT,))
        self.assertGreaterEqual(auto_research.AUTO_RESEARCH_CANDIDATE_POOL_LIMIT, 50)

    def test_harness_source_resolution_goes_to_design_repair_loop(self):
        task = {
            "dataset_refs": [
                {
                    "name": "Unresolved Benchmark",
                    "task_type": "benchmark",
                    "hf_dataset": "",
                    "generated_runner_supported": False,
                }
            ],
            "loop_state": {"status": "source_resolution_required"},
        }
        rows = [
            {
                "id": 41,
                "deep_insight_id": 29,
                "status": "harness_required",
                "benchmark_name": "Unresolved Benchmark",
                "task_plan": json.dumps(task),
                "last_error": "dataset source missing",
                "last_note": "waiting on benchmark design",
                "auto_status": "harness_required",
                "auto_stage": "benchmark_harness_required",
                "auto_last_note": "",
                "auto_last_error": "",
            }
        ]
        upserts = []
        executes = []

        def _capture_upsert(insight_id, **fields):
            upserts.append((insight_id, fields))

        def _capture_execute(sql, params=()):
            executes.append((sql, params))
            class Cursor:
                rowcount = 1
            return Cursor()

        with (
            mock.patch.object(auto_research.db, "fetchall", return_value=rows),
            mock.patch.object(auto_research.db, "execute", side_effect=_capture_execute),
            mock.patch.object(auto_research.db, "commit"),
            mock.patch.object(auto_research, "repair_experiment_plan_from_review") as repair,
            mock.patch.object(auto_research, "_upsert_job", side_effect=_capture_upsert),
            mock.patch.object(auto_research, "log_event"),
        ):
            queued = auto_research.repair_benchmark_harness_design_jobs(limit=1)

        self.assertEqual(queued, 1)
        repair.assert_not_called()
        self.assertEqual(upserts[-1][0], 29)
        self.assertEqual(upserts[-1][1]["status"], "queued")
        self.assertEqual(upserts[-1][1]["stage"], auto_research.BENCHMARK_HARNESS_DESIGN_REPAIR_STAGE)
        self.assertIn("auto_repair:benchmark_harness", upserts[-1][1]["last_note"])
        self.assertTrue(any(auto_research.BENCHMARK_HARNESS_DESIGN_REPAIR_QUEUED_STATUS in str(params) for _, params in executes))

    def test_dataset_materialization_harness_row_does_not_queue_design_repair(self):
        task = {
            "dataset_refs": [
                {
                    "name": "LongMemEval",
                    "hf_dataset": "xiaowu0162/longmemeval-cleaned",
                    "requires_harness": True,
                    "generated_runner_supported": False,
                }
            ],
            "loop_state": {"status": "dataset_materialization_required"},
        }
        rows = [
            {
                "id": 42,
                "deep_insight_id": 78,
                "status": "harness_required",
                "benchmark_name": "LongMemEval",
                "task_plan": json.dumps(task),
                "last_error": "dataset materialization pending",
                "last_note": "waiting on dataset fetch",
                "auto_status": "harness_required",
                "auto_stage": "benchmark_harness_required",
                "auto_last_note": "",
                "auto_last_error": "",
            }
        ]

        with (
            mock.patch.object(auto_research.db, "fetchall", return_value=rows),
            mock.patch.object(auto_research.db, "execute") as execute,
            mock.patch.object(auto_research.db, "commit"),
            mock.patch.object(auto_research, "_upsert_job") as upsert,
            mock.patch.object(auto_research, "log_event"),
        ):
            queued = auto_research.repair_benchmark_harness_design_jobs(limit=1)

        self.assertEqual(queued, 0)
        execute.assert_not_called()
        upsert.assert_not_called()


    def test_harness_design_repair_worker_requeues_review_after_llm_repair(self):
        task = {
            "dataset_refs": [
                {
                    "name": "Unresolved Benchmark",
                    "task_type": "benchmark",
                    "hf_dataset": "",
                    "generated_runner_supported": False,
                }
            ],
            "loop_state": {"status": "source_resolution_required"},
        }
        row = {
            "id": 41,
            "deep_insight_id": 29,
            "status": auto_research.BENCHMARK_HARNESS_DESIGN_REPAIR_QUEUED_STATUS,
            "benchmark_name": "Unresolved Benchmark",
            "task_plan": json.dumps(task),
            "last_error": "",
            "last_note": "[auto_repair:benchmark_harness attempt=1/1] queued repair",
            "auto_status": "review_pending",
            "auto_stage": auto_research.BENCHMARK_HARNESS_DESIGN_REPAIR_STAGE,
            "auto_last_note": "",
            "auto_last_error": "",
        }
        upserts = []
        executes = []

        def _capture_upsert(insight_id, **fields):
            upserts.append((insight_id, fields))

        def _capture_execute(sql, params=()):
            executes.append((sql, params))
            class Cursor:
                rowcount = 1
            return Cursor()

        with (
            mock.patch.object(auto_research, "_benchmark_harness_design_repair_row", return_value=row),
            mock.patch.object(auto_research.db, "execute", side_effect=_capture_execute),
            mock.patch.object(auto_research.db, "commit"),
            mock.patch.object(auto_research, "repair_experiment_plan_from_review", return_value={
                "status": "repaired",
                "attempt": 1,
                "repair_summary": "Pinned official benchmark source.",
                "llm_repair_used": True,
            }) as repair,
            mock.patch.object(auto_research, "_upsert_job", side_effect=_capture_upsert),
            mock.patch.object(auto_research, "log_event"),
        ):
            repaired = auto_research._run_benchmark_harness_design_repair_job(29)

        self.assertTrue(repaired)
        repair.assert_called_once()
        self.assertEqual(repair.call_args.kwargs["attempt"], 1)
        judgement = repair.call_args.kwargs["judgement"]
        self.assertIn("Pre-execution benchmark harness gate", judgement["summary"])
        self.assertEqual(upserts[-1][0], 29)
        self.assertEqual(upserts[-1][1]["status"], "queued")
        self.assertEqual(upserts[-1][1]["stage"], "experiment_review_repair")
        self.assertIn("Requeued pre-execution review", upserts[-1][1]["last_note"])
        self.assertTrue(any(auto_research.BENCHMARK_HARNESS_DESIGN_REPAIRED_STATUS in str(params) for _, params in executes))

    def test_harness_design_repair_candidate_uses_repair_queue(self):
        decision = auto_research._candidate_queue_decision({
            "id": 29,
            "auto_status": "queued",
            "auto_stage": auto_research.BENCHMARK_HARNESS_DESIGN_REPAIR_STAGE,
            "novelty_status": "unchecked",
        })

        self.assertEqual(decision.queue, auto_research.QUEUE_REPAIR)
        self.assertTrue(decision.runnable)

    def test_repair_lane_does_not_consume_review_capacity(self):
        with (
            mock.patch.object(auto_research, "_execution_active_job_count", return_value=0),
            mock.patch.object(auto_research, "_verification_job_count", return_value=0),
            mock.patch.object(auto_research, "_research_job_count", return_value=0),
            mock.patch.object(auto_research, "_review_pending_job_count", return_value=0),
            mock.patch.object(auto_research, "_repair_pending_job_count", return_value=auto_research.MAX_PARALLEL_REPAIRS),
            mock.patch.object(auto_research, "_active_queue_worker_count", return_value=0),
        ):
            counts = auto_research._queue_active_counts()

        self.assertEqual(counts[auto_research.QUEUE_REVIEW], 0)
        self.assertEqual(counts[auto_research.QUEUE_REPAIR], auto_research.MAX_PARALLEL_REPAIRS)
        self.assertTrue(auto_research._queue_has_capacity(auto_research.QUEUE_REVIEW, counts))
        self.assertFalse(auto_research._queue_has_capacity(auto_research.QUEUE_REPAIR, counts))

    def test_start_candidate_worker_tracks_repair_queue_separately(self):
        class FakeThread:
            def __init__(self, *args, **kwargs):
                pass

            def start(self):
                pass

        candidate = {"id": 91, "auto_status": "queued", "auto_stage": auto_research.BENCHMARK_HARNESS_DESIGN_REPAIR_STAGE}
        with auto_research._active_queue_worker_lock:
            auto_research._active_queue_workers.pop(91, None)
        try:
            with (
                mock.patch.object(auto_research, "_claim_review_candidate", return_value=True),
                mock.patch.object(auto_research.threading, "Thread", FakeThread),
            ):
                started = auto_research._start_candidate_worker(candidate, auto_research.QUEUE_REPAIR)

            self.assertTrue(started)
            with auto_research._active_queue_worker_lock:
                self.assertEqual(auto_research._active_queue_workers[91], auto_research.QUEUE_REPAIR)
        finally:
            with auto_research._active_queue_worker_lock:
                auto_research._active_queue_workers.pop(91, None)

    def test_select_candidate_uses_review_when_repair_lane_is_full(self):
        repair = {
            "id": 92,
            "tier": 2,
            "novelty_status": "novel",
            "auto_status": "queued",
            "auto_stage": auto_research.BENCHMARK_HARNESS_DESIGN_REPAIR_STAGE,
        }
        review = {
            "id": 93,
            "tier": 2,
            "novelty_status": "novel",
            "auto_status": "queued",
            "auto_stage": "idea_ready",
        }

        with (
            mock.patch.object(auto_research, "_candidate_pool", return_value=[repair, review]),
            mock.patch.object(auto_research, "_queue_active_counts", return_value={
                auto_research.QUEUE_EXECUTION: 0,
                auto_research.QUEUE_VERIFICATION: 0,
                auto_research.QUEUE_RESEARCH: 0,
                auto_research.QUEUE_REVIEW: 0,
                auto_research.QUEUE_REPAIR: auto_research.MAX_PARALLEL_REPAIRS,
            }),
            mock.patch.object(auto_research, "evosci_available", return_value=False),
        ):
            selected, state = auto_research._select_candidate_from_queues()

        self.assertEqual(selected["id"], 93)
        self.assertEqual(state["selected_queue"], auto_research.QUEUE_REVIEW)

    def test_archives_inactive_harness_rows_with_closed_auto_job(self):
        rows = [
            {
                "id": 5,
                "deep_insight_id": 7,
                "auto_status": "completed",
                "auto_stage": "closed_loop_complete",
            }
        ]
        executes = []

        def _capture_execute(sql, params=()):
            executes.append((sql, params))
            class Cursor:
                rowcount = 1
            return Cursor()

        with (
            mock.patch.object(auto_research.db, "fetchall", return_value=rows),
            mock.patch.object(auto_research.db, "execute", side_effect=_capture_execute),
            mock.patch.object(auto_research.db, "commit") as commit,
            mock.patch.object(auto_research, "log_event") as log_event,
        ):
            archived = auto_research.archive_inactive_benchmark_harness_jobs(limit=10)

        self.assertEqual(archived, 1)
        commit.assert_called_once()
        self.assertIn("auto_job_closed_archived", executes[0][0])
        self.assertIn("completed/closed_loop_complete", executes[0][1][0])
        log_event.assert_called_once()

    def test_process_harness_jobs_only_scans_active_harness_auto_jobs(self):
        with mock.patch.object(auto_research.db, "fetchall", return_value=[]) as fetchall:
            recovered = auto_research.process_benchmark_harness_jobs(limit=3)

        self.assertEqual(recovered, 0)
        sql = fetchall.call_args.args[0]
        self.assertIn("COALESCE(arj.status, '')='harness_required'", sql)

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

    def test_refresh_running_completed_run_waits_for_open_manuscript(self):
        job = {
            "deep_insight_id": 31,
            "status": "running_gpu",
            "stage": "gpu_scheduler",
            "experiment_run_id": 8,
            "resource_class": "gpu_large",
            "assigned_worker": "gpu1",
        }
        run = {
            "id": 8,
            "status": "completed",
            "resource_class": "gpu_large",
            "hypothesis_verdict": "confirmed",
            "effect_pct": 7.5,
        }

        def _fake_fetchall(sql, params=()):
            if "FROM auto_research_jobs" in sql:
                return [job]
            if "FROM manuscript_runs" in sql:
                return [{"manuscript_status": "drafting", "bundle_status": None}]
            return []

        def _fake_fetchone(sql, params=()):
            if "submission_bundle_id" in sql:
                return {"status": "completed", "submission_bundle_id": None}
            return run

        with (
            mock.patch.object(auto_research.db, "fetchall", side_effect=_fake_fetchall),
            mock.patch.object(auto_research.db, "fetchone", side_effect=_fake_fetchone),
            mock.patch.object(auto_research, "_upsert_job") as upsert,
            mock.patch.object(auto_research, "apply_experiment_finished_deep") as finish,
        ):
            auto_research._refresh_running_jobs()

        finish.assert_not_called()
        completed_calls = [call for call in upsert.call_args_list if call.kwargs.get("stage") == "closed_loop_complete"]
        self.assertEqual(completed_calls, [])
        self.assertEqual(upsert.call_args.kwargs["status"], "running_gpu")
        self.assertEqual(upsert.call_args.kwargs["stage"], "gpu_scheduler")
        self.assertIn("waiting for manuscript", upsert.call_args.kwargs["last_note"])

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
            mock.patch.object(auto_research, "_repair_pending_job_count", return_value=0),
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
            mock.patch.object(auto_research, "archive_inactive_benchmark_harness_jobs", return_value=0),
            mock.patch.object(auto_research, "recover_partially_supported_harness_jobs", return_value=0),
            mock.patch.object(auto_research, "repair_benchmark_harness_design_jobs", return_value=0),
            mock.patch.object(auto_research, "process_benchmark_harness_jobs", return_value=0),
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

        self.assertEqual(upserts[-1][1]["status"], "queued")
        self.assertEqual(upserts[-1][1]["stage"], "manuscript_retry_after_quality_gate")
        self.assertIn("Submission bundle failed", upserts[-1][1]["last_note"])

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

    def test_existing_run_lookup_prefers_auto_job_experiment_run_id(self):
        with mock.patch.object(
            auto_research.db,
            "fetchone",
            return_value={"id": 9, "status": "completed"},
        ) as fetchone:
            run = auto_research._existing_run_for_candidate(
                {"id": 21, "auto_experiment_run_id": 9, "canonical_run_id": 7}
            )

        self.assertEqual(run["id"], 9)
        self.assertEqual(fetchone.call_count, 1)
        self.assertEqual(fetchone.call_args.args[1], (9,))

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
            mock.patch.object(auto_research, "_upsert_job") as upsert,
            mock.patch.object(auto_research, "_queue_benchmark_completion_run", return_value=True) as queue_completion,
            mock.patch.object(auto_research, "log_event"),
        ):
            auto_research._process_candidate(candidate)

        queue_completion.assert_called_once_with(25, existing_run, "gpu_large")
        self.assertNotIn("research_unavailable", [call.kwargs.get("stage") for call in upsert.call_args_list])

    def test_process_candidate_routes_cpu_benchmark_completion_to_gpu_large(self):
        candidate = {
            "id": 25,
            "tier": 2,
            "novelty_status": "novel",
            "canonical_run_id": 11,
            "auto_stage": auto_research.BENCHMARK_COMPLETION_STAGE,
            "auto_resource_class": "cpu",
        }
        existing_run = {
            "id": 11,
            "status": "completed",
            "proxy_config": '{"formal_experiment": true, "smoke_test_only": false}',
            "resource_class": "cpu",
        }

        with (
            mock.patch.object(auto_research, "assess_experiment_route", return_value=("cpu", "ready")),
            mock.patch.object(auto_research, "evosci_available", return_value=False),
            mock.patch.object(auto_research, "_existing_run_for_candidate", return_value=existing_run),
            mock.patch.object(auto_research, "_auto_job_stage", return_value=auto_research.BENCHMARK_COMPLETION_STAGE),
            mock.patch.object(auto_research, "_upsert_job") as upsert,
            mock.patch.object(auto_research, "_queue_benchmark_completion_run", return_value=True) as queue_completion,
            mock.patch.object(auto_research, "log_event"),
        ):
            auto_research._process_candidate(candidate)

        queue_completion.assert_called_once_with(25, existing_run, "gpu_large")
        self.assertNotIn("research_unavailable", [call.kwargs.get("stage") for call in upsert.call_args_list])

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
            mock.patch.object(auto_research, "_manuscript_retry_blocker", return_value=None),
            mock.patch.object(auto_research, "_upsert_job", side_effect=_capture_upsert),
            mock.patch.object(auto_research, "log_event"),
        ):
            recovered = auto_research.recover_blocked_manuscript_jobs()

        self.assertEqual(recovered, 1)
        self.assertEqual(upserts[-1][0], 8)
        self.assertEqual(upserts[-1][1]["status"], "queued")
        self.assertEqual(upserts[-1][1]["stage"], "manuscript_retry_after_quality_gate")
        self.assertEqual(upserts[-1][1]["experiment_run_id"], 13)

    def test_invalid_manuscript_retry_marks_blocked_manuscript_stale(self):
        with (
            mock.patch.object(auto_research.db, "execute") as execute,
            mock.patch.object(auto_research, "_submission_grade_run_for_insight", return_value=None),
            mock.patch.object(auto_research, "_upsert_job") as upsert,
            mock.patch.object(auto_research, "log_event"),
        ):
            auto_research._block_invalid_manuscript_retry_run(4, 335, "Experiment run status=superseded is not valid for manuscript retry.")

        sql, params = execute.call_args.args
        self.assertIn("UPDATE manuscript_runs", sql)
        self.assertIn("status='stale'", sql)
        self.assertEqual(params, (335,))
        upsert.assert_called_once()
        self.assertEqual(upsert.call_args.kwargs["status"], "completed")
        self.assertEqual(upsert.call_args.kwargs["stage"], "closed_loop_complete")

    def test_invalid_manuscript_retry_switches_to_submission_grade_replacement(self):
        replacement = {"id": 10, "resource_class": "gpu_small"}
        with (
            mock.patch.object(auto_research.db, "execute") as execute,
            mock.patch.object(auto_research, "_submission_grade_run_for_insight", return_value=replacement) as find_replacement,
            mock.patch.object(auto_research, "_upsert_job") as upsert,
            mock.patch.object(auto_research, "log_event") as log_event,
        ):
            auto_research._block_invalid_manuscript_retry_run(4, 335, "Experiment run status=superseded is not valid for manuscript retry.")

        sql, params = execute.call_args.args
        self.assertIn("UPDATE manuscript_runs", sql)
        self.assertEqual(params, (335,))
        find_replacement.assert_called_once_with(4, exclude_run_id=335)
        upsert.assert_called_once()
        self.assertEqual(upsert.call_args.kwargs["status"], "queued")
        self.assertEqual(upsert.call_args.kwargs["stage"], "manuscript_retry_after_quality_gate")
        self.assertEqual(upsert.call_args.kwargs["experiment_run_id"], 10)
        self.assertEqual(upsert.call_args.kwargs["resource_class"], "gpu_small")
        self.assertIn("switching from invalid run 335", upsert.call_args.kwargs["last_note"])
        self.assertEqual(log_event.call_args.args[0], "auto_research")

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
            mock.patch.object(auto_research, "_repair_pending_job_count", return_value=0),
            mock.patch.object(auto_research, "evosci_available", return_value=False),
        ):
            selected = auto_research._next_candidate()

        self.assertEqual(selected["id"], 31)

    def test_active_job_count_still_includes_researching_jobs(self):
        with (
            mock.patch.object(auto_research, "_execution_active_job_count", return_value=1),
            mock.patch.object(auto_research, "_verification_job_count", return_value=2),
            mock.patch.object(auto_research, "_research_job_count", return_value=3),
            mock.patch.object(auto_research, "_review_pending_job_count", return_value=0),
            mock.patch.object(auto_research, "_repair_pending_job_count", return_value=0),
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

    def test_candidate_selection_prioritizes_benchmark_completion_over_manuscript_repair(self):
        repair = {
            "id": 8,
            "tier": 2,
            "novelty_status": "novel",
            "auto_status": "queued",
            "auto_stage": "manuscript_retry_after_quality_gate",
            "auto_experiment_run_id": 13,
        }
        completion = {
            "id": 61,
            "tier": 2,
            "novelty_status": "novel",
            "auto_status": "queued",
            "auto_stage": auto_research.BENCHMARK_COMPLETION_STAGE,
            "auto_experiment_run_id": 358,
        }

        with (
            mock.patch.object(auto_research, "_candidate_pool", return_value=[repair, completion]),
            mock.patch.object(auto_research, "_execution_active_job_count", return_value=0),
            mock.patch.object(auto_research, "_verification_job_count", return_value=0),
            mock.patch.object(auto_research, "_research_job_count", return_value=0),
            mock.patch.object(auto_research, "_review_pending_job_count", return_value=0),
            mock.patch.object(auto_research, "_repair_pending_job_count", return_value=0),
            mock.patch.object(auto_research, "evosci_available", return_value=False),
        ):
            selected, state = auto_research._select_candidate_from_queues()

        self.assertEqual(selected["id"], 61)
        self.assertEqual(state["selected_queue"], auto_research.QUEUE_EXECUTION)
        self.assertIn("benchmark completion", state["decision"])

    def test_harness_partial_recovery_skips_llm_for_unresolved_custom_harness(self):
        row = {
            "id": 80,
            "experimental_plan": json.dumps(
                {
                    "generated_runner_supported": False,
                    "benchmark_targets": [
                        {"name": "BIRD", "task_type": "text_to_sql", "requires_harness": True}
                    ],
                }
            ),
            "proposed_method": json.dumps({"name": "SQL Repair"}),
        }

        with mock.patch.object(auto_research, "_ensure_real_benchmark_plan") as ensure:
            repaired = auto_research._repair_harness_plan_for_supported_subset(row)

        ensure.assert_not_called()
        self.assertIsNone(repaired)

    def test_harness_partial_recovery_uses_only_concrete_generated_runner_subset(self):
        row = {
            "id": 81,
            "experimental_plan": json.dumps(
                {
                    "generated_runner_supported": False,
                    "benchmark_targets": [
                        {
                            "name": "GSM8K",
                            "hf_dataset": "openai/gsm8k",
                            "task_type": "math_qa",
                        },
                        {
                            "name": "BIRD",
                            "task_type": "text_to_sql",
                            "requires_harness": True,
                        },
                    ],
                }
            ),
            "proposed_method": json.dumps({"name": "Reasoning Repair"}),
        }

        with mock.patch.object(auto_research, "_ensure_real_benchmark_plan") as ensure:
            repaired = auto_research._repair_harness_plan_for_supported_subset(row)

        ensure.assert_not_called()
        self.assertTrue(repaired["generated_runner_supported"])
        self.assertEqual([target["name"] for target in repaired["benchmark_targets"]], ["GSM8K"])
        self.assertEqual(repaired["deferred_benchmark_targets"], ["BIRD"])

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
            mock.patch.object(auto_research, "_repair_pending_job_count", return_value=0),
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
            mock.patch.object(auto_research, "_repair_pending_job_count", return_value=0),
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
            mock.patch.object(auto_research, "_repair_pending_job_count", return_value=0),
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
